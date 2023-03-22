import numpy as np
import yaml
import imageio
from typing import List, Tuple, Dict, Any
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType

try:
    import mujoco
    import dm_control
    from dm_control import mjcf
except ImportError:
    raise ImportError(
        'MuJoCo prerequisites not installed. Please install the prerequisites '
        'by running `pip install flygym[mujoco]` or '
        '`pip install -e ."[mujoco]"` if installing locally.'
    )

from flygym.util.data import mujoco_groundwalking_model_path
from flygym.util.data import default_pose_path
from flygym.util.config import all_leg_dofs

_init_pose_lookup = {
    'default': default_pose_path,
}
_default_terrain_config = {
    'flat': {'size': (50_000, 50_000),
             'fly_placement': ((0, 0, 600), (0, 1, 0, 0.1))},
    # Ball data are all deduced from pybullet values ()
    'ball': {'radius': 5390.852782067457,
             'fly_placement': ((0, 0, 0), (0, 0.0698, 0)),  # angle given as euler angle
             'ball_placement': ((-98.67235483, -54.35809692, -5203.09506806), (1, 0, 0, 0)),
             'mass': 0.054599999999999996,
             'sliding_friction': 1.3,
             'torsional_friction': 0.005,  # default MuJoCo value
             'rolling_friction': 0.0001},
}
_default_render_config = {
    'saved': {'window_size': (640, 480), 'playspeed': 1.0, 'fps': 60},
}


class NeuroMechFlyMuJoCo(gym.Env):
    metadata = {
        'render_modes': ['headless', 'viewer', 'saved'],
        'terrain': ['flat', 'ball'],
        'control': ['position', 'velocity', 'torque'],
        'init_pose': ['default']
    }

    def __init__(self,
                 render_mode: str = 'saved',
                 render_config: Dict = {},
                 model_path: Path = mujoco_groundwalking_model_path,
                 actuated_joints: List = all_leg_dofs,
                 timestep: float = 0.0001,
                 output_dir: Path = None,
                 terrain: str = 'flat',
                 terrain_config: Dict = {},
                 control: str = 'position',
                 init_pose: str = 'default',
                 show_collisions: bool = False
                 ) -> None:
        self.render_mode = render_mode
        self.render_config = _default_render_config[render_mode]
        self.render_config.update(render_config)
        self.actuated_joints = actuated_joints
        self.timestep = timestep
        if output_dir is None:
            output_dir = Path.cwd() / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.terrain = terrain
        self.terrain_config = _default_terrain_config[terrain]
        self.terrain_config.update(terrain_config)
        self.control = control
        self.init_pose = init_pose
        self.show_collisions = show_collision

        # Define action and observation spaces
        num_dofs = len(actuated_joints)
        bound = np.pi if control == 'position' else np.inf
        self.action_space = {
            'joints': spaces.Box(low=-bound, high=bound, shape=(num_dofs,))
        }
        self.observation_space = {
            # joints: shape (3, num_dofs): (pos, vel, torque) of each DoF
            'joints': spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(3, num_dofs)),
            # fly position and orientation: shape (6,):
            # (x, y, z, roll, pitch, yaw) of thorax
            'fly': spaces.Box(low=-np.inf, high=np.inf, shape=(6,)),
        }

        # Load NMF model
        model = mjcf.from_path(model_path)
        model.option.timestep = timestep
        if init_pose not in self.metadata['init_pose']:
            raise ValueError(f'Invalid init_pose: {init_pose}')
        with open(_init_pose_lookup[init_pose]) as f:
            init_pose = {k: np.deg2rad(v)
                         for k, v in yaml.safe_load(f)['joints'].items()}
        init_pose = {k: v for k, v in init_pose.items() if k in actuated_joints}

        # Fix unactuated joints and define list of actuated joints
        # for joint in model.find_all('joint'):
        #     if joint.name not in actuated_joints:
        #         joint.type = 'fixed'
        self.actuators = [model.find('actuator', f'actuator_{control}_{joint}')
                          for joint in actuated_joints]

        # Set all bodies to default position (joint angle) even if the
        # joint is unactuated
        for body in model.find_all('body'):
            if (key := f'joint_{body.name}') in init_pose:
                if body.name.endswith('_yaw'):
                    rot_axis = [1, 0, 0]
                elif body.name.endswith('_roll'):
                    rot_axis = [0, 0, 1]
                else:  # pitch
                    rot_axis = [0, 1, 0]
                # replace hardcoded quaternion with axis-angle
                # (x, y, z, angle). See https://mujoco.readthedocs.io/en/stable/XMLreference.html#corientation
                del body.quat
                body.axisangle = [*rot_axis, init_pose[key]]

        # Add arena and put fly in it
        arena = mjcf.RootElement()
        if terrain == 'flat':
            ground_size = [*self.terrain_config['size'], 1]
            fly_pos, fly_angle = self.terrain_config['fly_placement']
            chequered = arena.asset.add('texture', type='2d', builtin='checker',
                                        width=300, height=300,
                                        rgb1=(.2, .3, .4), rgb2=(.3, .4, .5))
            grid = arena.asset.add('material', name='grid', texture=chequered,
                                   texrepeat=(10, 10), reflectance=0.1)
            arena.worldbody.add('geom', type='plane', name='ground',
                                material=grid, size=ground_size)
            spawn_site = arena.worldbody.add('site', pos=fly_pos,
                                             axisangle=fly_angle)
            spawn_site.attach(model).add('freejoint')

        elif terrain == 'ball':
            ball_radius = [self.terrain_config['radius']]
            ball_mass = self.terrain_config['mass']
            fly_pos, fly_angle = self.terrain_config['fly_placement']
            ball_pos, ball_angle = self.terrain_config['ball_placement']
            ball_frictions = [self.terrain_config["sliding_friction"], self.terrain_config["torsional_friction"],
                              self.terrain_config["rolling_friction"]]
            #Material for the ball
            chequered_cube = arena.asset.add('texture', type='cube', builtin='checker', width=300, height=300,
                                             rgb1=[0.2, 0.3, 0.4], rgb2=[0.3, 0.4, 0.5])
            grid = arena.asset.add('material', name='grid', texture=chequered_cube, texuniform=True,
                                   texrepeat=[10, 10], reflectance=0.2)
            #Build the ball
            treadmill_body = arena.worldbody.add('body', name='treadmill', pos=ball_pos, quat=ball_angle)
            treadmill_body.add('geom', name='treadmill_collision', type="sphere", size=ball_radius,
                               mass=ball_mass, friction=ball_frictions)
            treadmill_body.add('geom', name='treadmill_visual', type="sphere", size=ball_radius, material=grid)
            treadmill_body.add('joint', name='treadmill_joint', type='ball', limited="false")
            treadmill_body.add('inertial', pos=[0, 0, 0], mass=ball_mass)

            #Theter the model
            prismatic_support_1 = arena.worldbody.add('body', name='prismatic_support_1', pos=fly_pos, euler=fly_angle)
            prismatic_support_1.add('inertial', pos=[0, 0, 0], mass=0.0,
                                    fullinertia=[9.9999999999999995e-07, 9.9999999999999995e-07, 9.9999999999999995e-07,
                                                 0, 0, 0])
            spawn_site = prismatic_support_1.add('site', pos=[0, 0, 0], name="prismatic_support_1_attachment")
            spawn_site.attach(model)
        else:
            raise NotImplementedError

        arena.option.timestep = timestep
        self.physics = mjcf.Physics.from_mjcf_model(arena)
        self.curr_time = 0
        self._last_render_time = -np.inf
        self._eff_render_interval = (self.render_config['playspeed'] /
                                     self.render_config['fps'])
        self.frames = []

    def reset(self) -> Tuple[ObsType, Dict[str, Any]]:
        # return super().reset(seed=seed, options)
        self.physics.reset()
        self.curr_time = 0
        return self._get_observation(), self._get_info()

    def step(self, action: ObsType
             ) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        self.physics.bind(self.actuators).ctrl = action['joints']
        self.physics.step()
        self.curr_time += self.timestep
        return self._get_observation(), self._get_info()

    def render(self):
        if self.show_collisions:
            raise NotImplementedError
        if self.curr_time < self._last_render_time + self._eff_render_interval:
            return
        if self.render_mode == 'saved':
            width, height = self.render_config['window_size']
            img = self.physics.render(width=width, height=height)
            self.frames.append(img.copy())
        else:
            raise NotImplementedError

    def _get_observation(self) -> Tuple[ObsType, Dict[str, Any]]:
        return {
            'joints': np.zeros((3, len(self.actuated_joints))),
            'fly': np.zeros(6),
        }

    def _get_info(self):
        return {}

    def close(self):
        if self.render_mode == 'saved':
            with imageio.get_writer(self.output_dir / f'video_{self.terrain}.mp4',
                                    fps=self.render_config['fps']) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
        # Save data
        ...
