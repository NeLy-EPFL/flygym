import pickle as _pickle
import numpy as _np
import pkg_resources as _pkg_resources
from pathlib import Path as _Path
from flygym.util.config import leg_dofs_fused_tarsi

data_path = _Path(_pkg_resources.resource_filename('flygym', 'data'))
if not data_path.is_dir():
    raise FileNotFoundError(
        f'Data directory not found (expected at {data_path}). '
        'Please reinstall the package.'
    )

# MuJoCo
mujoco_groundwalking_model_path = (
    data_path / 'mjcf/groundwalking_nmf_mjcf_nofloor_230404.xml'
)

# Isaac Gym
isaacgym_asset_path = str(data_path)
# isaacgym_ground_walking_model_path = (
#     'mjcf/groundwalking_nmf_mjcf_nofloor_230404.xml'
# )
isaacgym_ground_walking_model_path = 'urdf/nmf_no_limits.urdf'

# PyBullet
...

# Pose
default_pose_path = data_path / 'pose/pose_default.yaml'
stretch_pose_path = data_path / 'pose/pose_stretch.yaml'


def load_kin_replay_data(
    target_timestep, actuated_joints=leg_dofs_fused_tarsi, run_time=1.0
):
    """Returns preselected kinematic data from a fly walking on a ball.

    Parameters
    ----------
    target_timestep : float
        Timestep to interpolate to.
    actuated_joints : List[np.ndarray]
        List of DoFs to actuate, by default leg_dofs_fused_tarsi
    run_time : float, optional
        Total amount of time to extract (in seconds), by default 1.0

    Returns
    -------
    np.ndarray
        Recorded kinematic data as an array of shape
        (len(actuated_joints), int(run_time / target_timestep))
    """
    data_path = _Path(_pkg_resources.resource_filename("flygym", "data"))

    with open(data_path / "behavior" / "210902_pr_fly1.pkl", "rb") as f:
        data = _pickle.load(f)
    # Interpolate 5x
    num_steps = int(run_time / target_timestep)
    data_block = _np.zeros((len(actuated_joints), num_steps))
    measure_t = _np.arange(len(data["joint_LFCoxa"])) * data["meta"]["timestep"]
    interp_t = _np.arange(num_steps) * target_timestep
    for i, joint in enumerate(actuated_joints):
        data_block[i, :] = _np.interp(interp_t, measure_t, data[joint])
    return data_block