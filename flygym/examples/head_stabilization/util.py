from flygym.util import get_data_path
from pathlib import Path
from typing import Tuple


def get_head_stabilization_model_paths() -> Tuple[Path, Path]:
    """Get the paths to the head stabilization models."""
    model_dir = get_data_path("flygym", "data") / "trained_models/head_stabilization/"
    return (
        model_dir / "all_dofs_model.ckpt",
        model_dir / "joint_angle_scaler_params.pkl",
    )
