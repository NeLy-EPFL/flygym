import yaml
from flygym.common import get_data_path


def load_config():
    with open(get_data_path("flygym.mujoco", "config.yaml"), "r") as f:
        return yaml.safe_load(f)