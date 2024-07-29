from flygym.util import get_data_path
from pathlib import Path


def get_head_stabilization_model_paths() -> tuple[Path, Path]:
    """
    Get the paths to the head stabilization models.

    Returns
    -------
    Path
        Path to the head stabilization model checkpoint.
    Path
        Path to the pickle file containing joint angle scaler parameters.
    """
    model_dir = get_data_path("flygym", "data") / "trained_models/head_stabilization/"
    return (
        model_dir / "all_dofs_model.ckpt",
        model_dir / "joint_angle_scaler_params.pkl",
    )
