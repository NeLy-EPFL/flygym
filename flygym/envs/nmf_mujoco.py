import numpy as np
import yaml
import imageio
from typing import List, Tuple, Dict, Any
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
from scipy.spatial.transform import Rotation as R

try:
    import mujoco
    import dm_control
    from dm_control import mjcf
    from dm_control.utils import transformations
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
    'ball': {'radius': ...,
             'fly_placement': ((0, 0, ...), (0, 1, 0, ...))},
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
                 actuated_joints: List = all_leg_dofs,
                 timestep: float = 0.0001,
                 output_dir: Path = None,
                 terrain: str = 'flat',
                 terrain_config: Dict = {},
                 control: str = 'position',
                 init_pose: str = 'default',
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
            'fly': spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3)),
        }
        
        # Load NMF model
        model = mjcf.from_path(mujoco_groundwalking_model_path)
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
        
        # Add sensors
        self.joint_sensors = []
        for joint in actuated_joints:
            self.joint_sensors.extend([
                model.sensor.add('jointpos',
                                 name=f'jointpos_{joint}', joint=joint),
                model.sensor.add('jointvel',
                                 name=f'jointvel_{joint}', joint=joint),
                model.sensor.add('actuatorfrc',
                                 name=f'actuatorfrc_position_{joint}',
                                 actuator=f'actuator_position_{joint}'),
                model.sensor.add('actuatorfrc',
                                 name=f'actuatorfrc_velocity_{joint}',
                                 actuator=f'actuator_velocity_{joint}'),
                model.sensor.add('actuatorfrc',
                                 name=f'actuatorfrc_motor_{joint}',
                                 actuator=f'actuator_torque_{joint}'),
            ])
            # self.joint_sensors += [
            #     model.sensor.find('sensor', f'jointpos_{joint}'),
            #     model.sensor.find('sensor', f'jointvel_{joint}'),
            #     model.sensor.find('sensor', f'actuatorfrc_position_{joint}'),
            #     model.sensor.find('sensor', f'actuatorfrc_velocity_{joint}'),
            #     model.sensor.find('sensor', f'actuatorfrc_motor_{joint}'),
            # ]
        self.body_sensors = [
            model.sensor.add('framepos', name='thorax_pos',
                             objtype='body', objname='Thorax'),
            model.sensor.add('framelinvel', name='thorax_linvel',
                             objtype='body', objname='Thorax'),
            model.sensor.add('framequat', name='thorax_quat',
                             objtype='body', objname='Thorax'),
            model.sensor.add('frameangvel', name='thorax_angvel',
                             objtype='body', objname='Thorax')
        ]
        
        
        # Set all bodies to default position (joint angle) even if the
        # joint is unactuated
        for body in model.find_all('body'):
            if (key := f'joint_{body.name}') in init_pose:
                if body.name.endswith('_yaw'):
                    rot_axis = [1, 0, 0]
                elif body.name.endswith('_roll'):
                    rot_axis = [0, 0, 1]
                else:    # pitch
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
        if self.curr_time < self._last_render_time + self._eff_render_interval:
            return
        if self.render_mode == 'saved':
            width, height = self.render_config['window_size']
            img = self.physics.render(width=width, height=height)
            self.frames.append(img.copy())
        else:
            raise NotImplementedError
    
    
    def _get_observation(self) -> Tuple[ObsType, Dict[str, Any]]:
        # joint sensors
        joint_obs = np.zeros((3, len(self.actuated_joints)))
        joint_sensordata = self.physics.bind(self.joint_sensors).sensordata
        for i, joint in enumerate(self.actuated_joints):
            base_idx = i * 5
            # pos and vel
            joint_obs[:2, i] = joint_sensordata[base_idx:base_idx + 2]
            # torque from pos/vel/motor actuators
            joint_obs[2, i] = joint_sensordata[base_idx + 2:base_idx + 5].sum()
        joint_obs[2, :] *= 1e-9  # convert to N
        
        # fly position and orientation
        cart_pos = self.physics.bind(self.body_sensors[0]).sensordata
        cart_vel = self.physics.bind(self.body_sensors[1]).sensordata
        quat = self.physics.bind(self.body_sensors[2]).sensordata
        # ang_pos = transformations.quat_to_euler(quat)
        ang_pos = R.from_quat(quat).as_euler('xyz')  # explicitly use intrinsic
        ang_pos[0] *= -1  # flip roll??
        ang_vel = self.physics.bind(self.body_sensors[3]).sensordata
        fly_pos = np.array([cart_pos, cart_vel, ang_pos, ang_vel])
         
        return {
            'joints': joint_obs,
            'fly': fly_pos,
        }
    
    
    def _get_info(self):
        return {}
    
    
    def close(self):
        if self.render_mode == 'saved':
            with imageio.get_writer(self.output_dir / 'video.mp4',
                                    fps=self.render_config['fps']) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
        # Save data
        ...