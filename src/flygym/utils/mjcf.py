from os import PathLike
from typing import Any

import dm_control.mjcf as mjcf
import yaml

__all__ = ["set_params_recursive", "set_mujoco_globals"]


def set_params_recursive(root: mjcf.Element, params_dict: dict[str, Any]) -> None:
    """Recursively set attributes and children on an MJCF model (in place)."""
    params_to_set = [(root, key, value) for key, value in params_dict.items()]
    while params_to_set:
        parent, key, value = params_to_set.pop(0)
        if key in parent.spec.attributes:
            setattr(parent, key, value)
        elif key in parent.spec.children:
            if not isinstance(value, dict):
                raise ValueError(f"Expected dict for key '{key}', got {type(value)}")
            child_element = parent.get_children(key)
            for child_key, child_value in value.items():
                params_to_set.append((child_element, child_key, child_value))


def set_mujoco_globals(
    mjcf_model: mjcf.RootElement, mujoco_globals_path: PathLike
) -> None:
    with open(mujoco_globals_path) as f:
        mujoco_globals = yaml.safe_load(f)
    set_params_recursive(mjcf_model, mujoco_globals)
