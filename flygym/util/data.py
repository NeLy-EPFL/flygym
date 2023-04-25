import pkg_resources as _pkg_resources
from pathlib import Path as _Path


data_path = _Path(_pkg_resources.resource_filename('flygym', 'data'))
if not data_path.is_dir():
    raise FileNotFoundError(
        f'Data directory not found (expected at {data_path}). '
        'Please reinstall the package.'
    )

# MuJoCo
mujoco_groundwalking_model_path = data_path / 'mjcf/groundwalking_nmf_mjc_nofloor_230416_bendTarsus.xml'

# Isaac Gym
...

# PyBullet
...

# Pose
default_pose_path = data_path / 'pose/pose_default.yaml'
stretch_pose_path = data_path / 'pose/pose_stretch.yaml'
zero_pose_path = data_path / 'pose/pose_zero.yaml'