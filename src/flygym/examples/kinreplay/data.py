from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from flygym.compose.fly.anatomy import JointDOF, PASSIVE_TARSAL_LINKS


class Aymanns2022Dataset:
    DATA_DIR = Path(
        "~/projects/poseforge/bulk_data/kinematic_prior/aymanns2022/trials/"
    ).expanduser()
    FPS = 100
    CHILD_LINK_TO_AYMANNS_JOINT_NAME = {
        "coxa": "ThC",
        "trochanterfemur": "CTr",
        "tibia": "FTi",
        "tarsus1": "TiTa",
    }

    def __init__(
        self, joint_dofs: Sequence[JointDOF], trials_glob: str = "*.pkl"
    ) -> None:
        self.joint_dofs = joint_dofs
        self.dataframes = self._load_dataframes(trials_glob, joint_dofs=joint_dofs)

    @classmethod
    def _select_columns_by_joint_dofs(
        cls, dataframe: pd.DataFrame, joint_dofs: Sequence[JointDOF]
    ) -> list[str]:
        columns = ["Prediction", "Probability walking"]
        column_names_aymanns2flygym = {
            "Prediction": "behavior",
            "Probability walking": "prob_walking",
        }

        for joint_dof in joint_dofs:
            if not joint_dof.child.is_leg():
                raise ValueError("Only leg joints are tracked in Aymanns et al., 2022.")
            leg = joint_dof.child.pos

            if joint_dof.child.link in PASSIVE_TARSAL_LINKS:
                column_name = (
                    f"Angle__{leg}_leg_{joint_dof.child.link}_{joint_dof.axis.value}"
                )
                dataframe[column_name] = 0
            else:
                joint_name = cls.CHILD_LINK_TO_AYMANNS_JOINT_NAME[joint_dof.child.link]
                column_name = (
                    f"Angle__{leg.upper()}_leg_{joint_name}_{joint_dof.axis.value}"
                )

            columns.append(column_name)
            column_names_aymanns2flygym[column_name] = joint_dof.name

        df_selected = dataframe[columns]
        df_selected = df_selected.rename(columns=column_names_aymanns2flygym)
        return df_selected

    @classmethod
    def _load_dataframes(
        cls, trials_glob: str, joint_dofs: Sequence[JointDOF]
    ) -> dict[str, pd.DataFrame]:
        paths = sorted(cls.DATA_DIR.glob(trials_glob))
        if not paths:
            raise FileNotFoundError(f"No trial files found in {cls.DATA_DIR}")

        dataframes = {}
        for path in paths:
            df = pd.read_pickle(path)
            df = cls._select_columns_by_joint_dofs(df, joint_dofs)
            dataframes[path.stem] = df

        return dataframes

    def get_behavior_bouts(
        self,
        behavior: str,
        interpolate_freq_to: int | None = None,
        n_max: int | None = None,
        min_bout_duration_s: float = 1.0,
        mask_filter_window_s: float = 0.1,
        fixed_duration: bool = False,
    ):
        bout_arrays = []
        for _, df in self.dataframes.items():
            bout_arrays.extend(
                self._get_behavior_bouts_one_trial(
                    df,
                    behavior,
                    interpolate_freq_to=interpolate_freq_to,
                    n_max=(n_max - len(bout_arrays)) if n_max is not None else None,
                    min_bout_duration_s=min_bout_duration_s,
                    mask_filter_window_s=mask_filter_window_s,
                    fixed_duration=fixed_duration,
                )
            )
            if n_max is not None and len(bout_arrays) >= n_max:
                break
        assert n_max is None or len(bout_arrays) == n_max

        return bout_arrays

    def _get_behavior_bouts_one_trial(
        self,
        dataframe: pd.DataFrame,
        behavior: str,
        interpolate_freq_to: int | None,
        n_max: int | None,
        min_bout_duration_s: float,
        mask_filter_window_s: float,
        fixed_duration: bool,
    ):
        mask_filter_window_frames = (int(mask_filter_window_s * self.FPS) // 2) * 2 + 1
        min_bout_duration_frames = int(min_bout_duration_s * self.FPS)
        walking_mask = dataframe["behavior"].values == behavior
        walking_mask = _denoise_bool_timeseries(
            walking_mask,
            window=mask_filter_window_frames,
            min_true_period=min_bout_duration_frames,
        )
        walking_period_masks = _get_connected_component_masks_1d(walking_mask)

        bout_arrays = []
        for mask in walking_period_masks:
            cols_without_behavior = [
                c for c in dataframe.columns if c.startswith("joint-")
            ]
            kinprior_arr = dataframe.loc[mask, cols_without_behavior].values
            if fixed_duration:
                kinprior_arr = kinprior_arr[:min_bout_duration_frames, :]

            if interpolate_freq_to is not None:
                interp_factor_float = interpolate_freq_to / self.FPS
                interp_factor_int = int(round(interp_factor_float))
                if not np.isclose(interp_factor_float, interp_factor_int):
                    raise ValueError(
                        f"`interpolate_freq_to` ({interpolate_freq_to}) must be an "
                        f"integer multiple of the original frequency ({self.FPS})."
                    )
                kinprior_arr = _interpolate_array_along_one_axis(
                    kinprior_arr, interp_factor=interp_factor_int, axis=0, kind="linear"
                )

            bout_arrays.append(kinprior_arr)
            if n_max is not None and len(bout_arrays) == n_max:
                break

        return bout_arrays


def _denoise_bool_timeseries(
    ts: np.ndarray, window: int, min_true_period: int
) -> np.ndarray:
    ts = np.asarray(ts, dtype=bool)
    assert ts.ndim == 1
    assert window % 2 == 1 and window >= 1
    assert min_true_period >= 1

    # 1) median/majority filter
    s = np.convolve(ts.astype(np.int32), np.ones(window, dtype=np.int32), mode="same")
    xf = s > (window // 2)

    # 2) run detection (robust)
    y = xf.astype(np.uint8)
    d = np.diff(y.astype(np.int8))

    starts = np.flatnonzero(d == 1) + 1  # transition 0->1 at i means start at i+1
    ends = (
        np.flatnonzero(d == -1) + 1
    )  # transition 1->0 at i means end at i+1 (exclusive)

    # Handle runs that touch the boundaries
    if y[0]:
        starts = np.r_[0, starts]
    if y[-1]:
        ends = np.r_[ends, y.size]

    # Now starts/ends should match
    if starts.size != ends.size:
        # Extremely defensive fallback (shouldn't happen now)
        n = min(starts.size, ends.size)
        starts, ends = starts[:n], ends[:n]

    lengths = ends - starts

    out = np.zeros_like(xf, dtype=bool)
    for a, b in zip(
        starts[lengths >= min_true_period], ends[lengths >= min_true_period]
    ):
        out[a:b] = True
    return out


def _get_connected_component_masks_1d(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 1

    if mask.size == 0:
        return []

    y = mask.astype(np.uint8)
    d = np.diff(y.astype(np.int8))

    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1

    if y[0]:
        starts = np.r_[0, starts]
    if y[-1]:
        ends = np.r_[ends, y.size]

    comps = []
    for a, b in zip(starts, ends):
        m = np.zeros_like(mask, dtype=bool)
        m[a:b] = True
        comps.append(m)

    return comps


def _interpolate_array_along_one_axis(
    array: np.ndarray, interp_factor: int, axis: int = 0, kind: str = "linear"
) -> np.ndarray:
    n_in = array.shape[0]
    n_out = n_in * interp_factor
    indices_in = np.linspace(0, 1, n_in)
    indices_out = np.linspace(0, 1, n_out)
    interpolator = interp1d(indices_in, array, axis=axis, kind=kind)
    return interpolator(indices_out)

    # if __name__ == "__main__":
    #     from flygym.compose.fly.anatomy import Skeleton, AxisOrder

    #     skeleton = Skeleton(joint_preset="legs_only")
    #     joint_dofs = skeleton.iter_joint_dofs(axis_order=AxisOrder.ROLL_YAW_PITCH)
    #     dataset = Aymanns2022Dataset(joint_dofs, trials_glob="*.pkl")
    #     bouts = dataset.get_behavior_bouts("walking", interpolate_freq_to=10000, n_max=200)
    1
