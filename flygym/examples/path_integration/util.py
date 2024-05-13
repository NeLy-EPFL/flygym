import numpy as np
from typing import Tuple, Dict
from scipy.signal import convolve2d


def get_leg_mask(legs: str) -> np.ndarray:
    """
    Given a list of legs, return a boolean mask indicating which legs are
    included.

    If legs == "F", the mask is np.array([True, False, False])
    If legs == "FM", the mask is np.array([True, True, False])
    ...
    """
    legs = legs.upper()
    leg_mask = np.zeros(3, dtype=bool)
    if "F" in legs:
        leg_mask[0] = True
    if "M" in legs:
        leg_mask[1] = True
    if "H" in legs:
        leg_mask[2] = True
    return leg_mask


def extract_variables(
    trial_data: Dict[str, np.ndarray],
    time_scale: float,
    contact_force_thr: Tuple[float, float, float],
    legs: str,
    dt: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """
    Extract variables used for path integration from trial data.

    Parameters
    ----------
    trial_data : Dict[str, np.ndarray]
        Dictionary containing trial data.
    time_scale : float
        Time scale for path integration.
    contact_force_thr : Tuple[float, float, float]
        Thresholds for contact forces. These are used to determine whether
        a leg is in contact with the ground.
    legs : str
        String indicating which legs are included. Can be any combination
        of "F", "M", and "H".
    dt : float, optional
        Time step of the physics simulation in the trial, by default 1e-4.
    """
    window_len = int(time_scale / dt)
    # contact force thresholds: (3,) -> (6,), for both sides
    contact_force_thr = np.array([*contact_force_thr, *contact_force_thr])

    # Proprioceptive signal ==========
    # Calculate total stride (Σstride) for each side
    stride_left = trial_data["end_effector_pos"][:, :3, 0]  # (L, 3)
    stride_right = trial_data["end_effector_pos"][:, 3:, 0]  # (L, 3)
    contact_mask = trial_data["contact_force"] > contact_force_thr[None, :]  # (L, 6)
    leg_mask = get_leg_mask(legs)
    stride_left = (stride_left * contact_mask[:, :3])[:, leg_mask]
    stride_right = (stride_right * contact_mask[:, 3:])[:, leg_mask]
    stride_total_left = np.cumsum(stride_left, axis=0)
    stride_total_right = np.cumsum(stride_right, axis=0)

    # Calculate difference in Σstride over proprioceptive time window (ΔΣstride)
    stride_total_diff_left = (
        stride_total_left[window_len:] - stride_total_left[:-window_len]
    )
    stride_total_diff_right = (
        stride_total_right[window_len:] - stride_total_right[:-window_len]
    )

    # Calculate sum and difference in ΔΣstride over two sides
    stride_total_diff_lrsum = stride_total_diff_left + stride_total_diff_right
    stride_total_diff_lrdiff = stride_total_diff_left - stride_total_diff_right

    # Descending signal ==========
    # Calculate mean DN drive over proprioceptive time window
    dn_drive = trial_data["dn_drive"]
    conv_kernel = np.ones(window_len)[:, None] / window_len  # (window_len, 1)
    mean_dn_drive = convolve2d(dn_drive, conv_kernel, mode="valid")[1:, :]

    # Same for left-right sum and difference
    sum_dn_drive = mean_dn_drive[:, 0] + mean_dn_drive[:, 1]
    diff_dn_drive = mean_dn_drive[:, 0] - mean_dn_drive[:, 1]

    # Change in locomotion state (heading & displacement) ==========
    # Calculate change in fly orientation over proprioceptive time window (Δheading)
    fly_orientation_xy = trial_data["fly_orientation"]
    heading_ts = np.arctan2(fly_orientation_xy[:, 1], fly_orientation_xy[:, 0])
    heading_diff = heading_ts[window_len:] - heading_ts[:-window_len]
    heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]

    # Same for displacement projected in the direction of fly's heading
    # Use projection formula: proj_v(u) = (u · v) / (v · v) * v where v is the fly's
    # heading vector and u is the change in position
    fly_disp_xy = np.diff(trial_data["fly_pos"], axis=0, prepend=0)
    fly_orientation_xy_norm = np.linalg.norm(fly_orientation_xy, axis=1)
    fly_orientation_xy_unit = fly_orientation_xy / fly_orientation_xy_norm[:, None]
    udotv = np.sum(fly_disp_xy * fly_orientation_xy_unit, axis=1)
    vdotv = np.sum(fly_orientation_xy_unit * fly_orientation_xy_unit, axis=1)
    forward_disp_mag = udotv / vdotv
    forward_disp_total = np.cumsum(forward_disp_mag)
    forward_disp_total_diff = (
        forward_disp_total[window_len:] - forward_disp_total[:-window_len]
    )

    return {
        "stride_total_diff_lrsum": stride_total_diff_lrsum.astype(np.float32),
        "stride_total_diff_lrdiff": stride_total_diff_lrdiff.astype(np.float32),
        "sum_dn_drive": sum_dn_drive.astype(np.float32)[:, None],
        "diff_dn_drive": diff_dn_drive.astype(np.float32)[:, None],
        "heading_diff": heading_diff.astype(np.float32),
        "forward_disp_total_diff": forward_disp_total_diff.astype(np.float32),
        "heading": heading_ts[window_len:].astype(np.float32),
    }
