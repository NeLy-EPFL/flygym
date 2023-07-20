import pkg_resources as _pkg_resources
from pathlib import Path as _Path
from matplotlib.pyplot import rcParams


data_path = _Path(_pkg_resources.resource_filename("flygym", "data"))
if not data_path.is_dir():
    raise FileNotFoundError(
        f"Data directory not found (expected at {data_path}). "
        "Please reinstall the package."
    )

# MuJoCo
mujoco_groundwalking_model_path = (
    data_path / "mjcf/groundwalking_nmf_mjcf_nofloor_230518__bendTarsus_scaled.xml"
)

# Isaac Gym
...

# PyBullet
...

# Pose
default_pose_path = data_path / "pose/pose_default.yaml"
stretch_pose_path = data_path / "pose/pose_stretch.yaml"
zero_pose_path = data_path / "pose/pose_zero.yaml"

# Vision
sample_visual_path = data_path / "vision/banana.jpg"
ommatidia_id_map_path = data_path / "vision/ommatidia_id_map.npy"

# Visualization
color_cycle_hex = rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle_rgb = []
for hex in color_cycle_hex:
    stripped = hex.lstrip("#")
    rgb = tuple(int(stripped[i : i + 2], 16) for i in (0, 2, 4))
    color_cycle_rgb.append(rgb)