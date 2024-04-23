import numpy as np
from typing import Dict, Tuple, Callable
from flygym.examples.path_integration import util


def path_integrate(
    trial_data: Dict[str, np.ndarray],
    heading_model: Callable,
    displacement_model: Callable,
    time_scale: float,
    contact_force_thr: Tuple[float, float, float],
    legs: str,
    dt: float,
):
    window_len = int(time_scale / dt)
    variables = util.extract_variables(
        trial_data,
        time_scale=time_scale,
        contact_force_thr=contact_force_thr,
        legs=legs,
    )

    # Integrate heading
    heading_diff_pred = heading_model(variables["stride_total_diff_lrdiff"])
    heading_pred = np.cumsum(heading_diff_pred / window_len)
    # Path int. not performed when not enough data is available. Start from the real
    # heading at the moment when path int. actually starts.
    hx_start, hy_start = trial_data["fly_orientation"][window_len, :]
    real_heading_start = np.arctan2(hy_start, hx_start)
    heading_pred += real_heading_start

    # Integrate displacement
    displacement_diff_pred = displacement_model(variables["stride_total_diff_lrsum"])
    displacement_diff_x_pred = np.cos(heading_pred) * displacement_diff_pred
    displacement_diff_y_pred = np.sin(heading_pred) * displacement_diff_pred
    pos_x_pred = np.cumsum(displacement_diff_x_pred / window_len)
    pos_y_pred = np.cumsum(displacement_diff_y_pred / window_len)
    pos_pred = np.concatenate([pos_x_pred[:, None], pos_y_pred[:, None]], axis=1)

    # Pad with NaN where prediction not available
    heading_pred = np.concatenate([np.full(window_len, np.nan), heading_pred])
    pos_pred = np.concatenate([np.full((window_len, 2), np.nan), pos_pred], axis=0)

    return {
        "heading_pred": heading_pred,
        "heading_actual": variables["heading"],
        "pos_pred": pos_pred,
        "pos_actual": trial_data["fly_pos"],
        "heading_diff_pred": heading_diff_pred,
        "heading_diff_actual": variables["heading_diff"],
        "displacement_diff_pred": displacement_diff_pred,
        "displacement_diff_actual": variables["forward_disp_total_diff"],
    }


class LinearModel:
    def __init__(self, coefs_all, intercept, legs):
        self.coefs = coefs_all[util.get_leg_mask(legs)][None, :]
        self.intercept = intercept

    def __call__(self, x):
        return (x * self.coefs).sum(axis=1) + self.intercept
