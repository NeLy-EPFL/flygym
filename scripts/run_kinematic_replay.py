# run the kinematic replay inspired for initial neuromechfly
import argparse
from pathlib import Path
import glob
import numpy as np

from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from flygym.util.data import mujoco_groundwalking_model_path, mujoco_clean_groundwalking_model_path

random_state = np.random.RandomState(0)


def parse_args():
    """ Parse arguments from command line """
    parser = argparse.ArgumentParser()
    # parser.add_argument('-b', '--behavior', default='walking')
    # parser.add_argument('--record', dest='record', action='store_true')
    parser.add_argument('--show_collisions', dest='show_collisions', action='store_true')
    parser.add_argument('-fly', '--fly_number', default='1')
    parser.add_argument('-t', '--terrain', default='ball')
    parser.add_argument('-c', '--camera', default='left')

    return parser.parse_args()


###### Ball Config Transformations Constants ########
# Betwen pyBullet and MuJoCo, the coordinate systems are different.
thorax_MuJoCo_rel_pos = [495.64525485038757, -18.091375008225441, 1296.6440916061401]
thorax_Pybullet_rel_pos = [0.48076672828756273, -5.750451226305131e-07, 1.2107710354030132]
reg = np.polyfit(thorax_Pybullet_rel_pos, thorax_MuJoCo_rel_pos, 1)
regress = lambda x: reg[0] * x + reg[1]

# Pybullet uses units
meters = 1000
kilograms = 1000

# Offset from pybullet (alreadyin meters)
pybullet_model_offset = [0., 0, 11.2e-3]


def format_ball_terrain_config(terrain_config):
    """ Format the terrain config for the ball terrain """

    terrain_config["mass"] = 54.6e-6 * kilograms
    mjc_pos = [regress(x * meters + off) for x, off in zip(terrain_config["position"], pybullet_model_offset)]
    terrain_config["position"] = (tuple(mjc_pos), (1, 0, 0, 0))
    terrain_config["radius"] = regress(terrain_config["radius"] * meters)
    terrain_config["fly_placement"] = ((0, 0, 0), (0, np.deg2rad(terrain_config["angle"]), 0))
    terrain_config["lateral_friction"] = 1.3

    return terrain_config


def load_joint_angles(path):
    joint_angles = np.load(path, allow_pickle=True)
    return joint_angles


def main(args):
    """ Main function """

    behavior = 'walking'
    terrain = args.terrain
    camera = f"Animat/camera_{args.camera}"

    run_time = 6.0
    data_time_step = 5e-4
    time_step = 1e-4
    starting_time = 0.0

    # Paths for data
    data_path = Path.cwd() / f'flygym/data/joint_tracking/{behavior}/fly{args.fly_number}/df3d'

    angles_path = next(iter(data_path.glob('joint_angles*.pkl')))
    # velocity_path = glob.glob(data_path + '/joint_velocities*.pkl')[0]

    if behavior == 'walking' and terrain == 'ball':
        terrain_config_path = next(iter(data_path.glob('treadmill_info*.pkl')))
        terrain_config = np.load(terrain_config_path, allow_pickle=True)
        terrain_config = format_ball_terrain_config(terrain_config)
    else:
        terrain_config = {}

    # At some point replace all contacts by only relevant ones
    # Implement contact visualization

    render_config = {"camera_id": camera}

    # Setting the fixed joint angles to default values, can be altered to
    # change the appearance of the fly

    # Setting up the paths for the SDF and POSE files
    model_path = mujoco_clean_groundwalking_model_path
    output_dir = Path.cwd() / f'output/kinematic_replay/{behavior}/fly{args.fly_number}'

    # Simulation options
    sim_options = {"render_mode": 'saved',
                   "render_config": {},
                   "model_path": model_path,
                   # "actuated_joints": None,
                   "timestep": time_step,
                   "output_dir": output_dir,
                   "terrain": terrain,
                   "terrain_config": terrain_config,
                   "control": 'position',
                   # "init_pose": None,
                   "show_collisions": args.show_collisions,
                   "render_config": render_config
                   }

    nmf = NeuroMechFlyMuJoCo(**sim_options)

    run_time = 0.01
    freq = 500
    phase = 2 * np.pi * random_state.rand(len(nmf.actuators))
    amp = 0.9

    """joint_angles = load_joint_angles(angles_path, nmf.actuated_joints, data_time_step, time_step, starting_time, run_time)
    
    for i in range(len(joint_angles)):
        obs, info = nmf.step({'joints': joint_angles[i]})
        nmf.render()"""

    while nmf.curr_time <= run_time:
        joint_pos = amp * np.sin(freq * nmf.curr_time + phase)
        action = {'joints': joint_pos}
        obs, info = nmf.step(action)
        nmf.render()
    nmf.close()


if __name__ == "__main__":
    """ Main """
    # parse cli arguments
    cli_args = parse_args()
    main(cli_args)
