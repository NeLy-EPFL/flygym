import dm_control.mjcf as mjcf


def set_params_recursive(root: mjcf.Element, params_dict: dict[str, ...]) -> None:
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
