import numpy as np
from flygym.examples.path_integration import util


def path_integrate(
    trial_data: dict[str, np.ndarray],
    heading_model: "LinearModel",
    displacement_model: "LinearModel",
    time_scale: float,
    contact_force_thr: tuple[float, float, float],
    legs: str,
    dt: float,
):
    """
    Perform path integration on trial data.

    Parameters
    ----------
    trial_data : dict[str, np.ndarray]
        Dictionary containing trial data.
    heading_model : LinearModel
        Model for predicting change in heading.
    displacement_model : LinearModel
        Model for predicting change in displacement.
    time_scale : float
        Time scale for path integration.
    contact_force_thr : tuple[float, float, float]
        Thresholds for contact forces. These are used to determine whether
        a leg is in contact with the ground.
    legs : str
        String indicating which legs are included. Can be any combination
        of "F", "M", and "H".
    dt : float
        Time step of the physics simulation in the trial.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the following keys:
        * "heading_pred": Predicted heading.
        * "heading_actual": Actual heading.
        * "pos_pred": Predicted position.
        * "pos_actual": Actual position.
        * "heading_diff_pred": Predicted change in heading.
        * "heading_diff_actual": Actual change in heading.
        * "displacement_diff_pred": Predicted change in displacement.
        * "displacement_diff_actual": Actual change in displacement.
    """
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
    hx_start, hy_start = trial_data["fly_orientation_xy"][window_len, :]
    real_heading_start = np.arctan2(hy_start, hx_start)
    heading_pred += real_heading_start

    # Integrate displacement
    displacement_diff_pred = displacement_model(variables["stride_total_diff_lrsum"])
    displacement_diff_x_pred = np.cos(heading_pred) * displacement_diff_pred
    displacement_diff_y_pred = np.sin(heading_pred) * displacement_diff_pred
    pos_x_pred = np.cumsum(displacement_diff_x_pred / window_len)
    pos_y_pred = np.cumsum(displacement_diff_y_pred / window_len)
    pos_x_pred += trial_data["fly_pos"][window_len, 0]
    pos_y_pred += trial_data["fly_pos"][window_len, 1]
    pos_pred = np.concatenate([pos_x_pred[:, None], pos_y_pred[:, None]], axis=1)

    # Pad with NaN where prediction not available
    padding = np.full(window_len, np.nan)
    heading_pred = np.concatenate([padding, heading_pred])
    pos_pred = np.concatenate([np.full((window_len, 2), np.nan), pos_pred], axis=0)
    heading_diff_pred = np.concatenate([padding, heading_diff_pred])
    heading_diff_actual = np.concatenate([padding, variables["heading_diff"]])
    displacement_diff_pred = np.concatenate([padding, displacement_diff_pred])
    displacement_diff_actual = np.concatenate(
        [padding, variables["forward_disp_total_diff"]]
    )

    return {
        "heading_pred": heading_pred,
        "heading_actual": trial_data["fly_orientation_angle"],
        "pos_pred": pos_pred,
        "pos_actual": trial_data["fly_pos"],
        "heading_diff_pred": heading_diff_pred,
        "heading_diff_actual": heading_diff_actual,
        "displacement_diff_pred": displacement_diff_pred,
        "displacement_diff_actual": displacement_diff_actual,
    }


class LinearModel:
    """
    Simple linear model for predicting change in heading and displacement.
    """

    def __init__(self, coefs_all, intercept, legs):
        self.coefs = coefs_all[util.get_leg_mask(legs)][None, :]
        self.intercept = intercept

    def __call__(self, x):
        return (x * self.coefs).sum(axis=1) + self.intercept
