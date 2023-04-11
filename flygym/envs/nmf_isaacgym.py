import numpy as np
import copy
import gymnasium as gym
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from gymnasium import spaces

try:
    from isaacgym import gymapi
except ImportError:
    raise ImportError(
        'Isaac Gym is not installed. Please follow the instructions on '
        'the installation page of the documentation.'
    )

from flygym.util.data import isaacgym_asset_path
from flygym.util.data import isaacgym_ground_walking_model_path
from flygym.util.data import default_pose_path, stretch_pose_path
from flygym.util.config import all_leg_dofs


_init_pose_lookup = {
    'default': default_pose_path,
    'stretch': stretch_pose_path,
}
_default_terrain_config = {
    'flat': {
        'size': (50_000, 50_000),
        'friction': (1, 0.005, 0.0001),
        'fly_pos': (0, 0, 300),
        'fly_orient': (0, 1, 0, 0.1)
    },
}
_default_physics_config = {
    'gravity': (0, 0, -9.81e3),
    'num_substeps': 2,
    'contact_offset': 0.2,  # mm
    'rest_offset': 0.1,  # mm
    'friction_offset_threshold': 4e-4,  # mm
    'solver_type': 1,  # 0: PGS (Iterative sequential impulse solver)
                       # 1: TGS (Nonlinear iterative solver, more robust but
                       #    slightly more expensive)
    'num_position_iterations': 8,  # [1, 255]
    'num_velocity_iterations': 1,  # [1, 255]
}
_default_render_config = {
    'saved': {'window_size': (640, 480), 'playspeed': 1.0, 'fps': 60},
    'viewer': {'window_size': (640, 480), 'max_playspeed': 1.0, 'fps': 60},
    'headless': {},
}


class NeuroMechFlyIsaacGym(gym.Env):
    
    def __init__(self,
                 render_mode: str = 'saved',
                 render_config: Dict[str, Any] = {},
                 actuated_joints: List = all_leg_dofs,
                 timestep: float = 0.0001,
                 output_dir: Optional[Path] = None,
                 terrain: str = 'flat',
                 terrain_config: Dict[str, Any] = {},
                 physics_config: Dict[str, Any] = {},
                 control: str = 'position',
                 init_pose: str = 'default',
                 use_gpu: bool = True,
                 compute_device_id: int = 0,
                 graphics_device_id: int = 0,
                 num_threads: int = 4) -> None:
        self.render_mode = render_mode
        self.render_config = copy.deepcopy(_default_render_config[render_mode])
        self.render_config.update(render_config)
        self.actuated_joints = actuated_joints
        self.timestep = timestep
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.terrain = terrain
        self.terrain_config = copy.deepcopy(_default_terrain_config[terrain])
        self.terrain_config.update(terrain_config)
        self.physics_config = copy.deepcopy(_default_physics_config)
        self.physics_config.update(physics_config)
        self.control = control
        self.use_gpu = use_gpu
        self.compute_device_id = compute_device_id
        self.graphics_device_id = graphics_device_id
        self.num_threads = num_threads
        self.compute_device = f'cuda:{compute_device_id}' if use_gpu else 'cpu'
        
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
            # fly: shape (4, 3):
            # 0th row: x, y, z position of the fly in arena
            # 1st row: x, y, z velocity of the fly in arena
            # 2nd row: orientation of fly around x, y, z axes
            # 3rd row: rate of change of fly orientation
            'fly': spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3)),
        }
        
        # Initialize physics simulation
        gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(*self.physics_config['gravity'])
        sim_params.substeps = self.physics_config['num_substeps']
        sim_params.dt = self.timestep
        sim_params.physx.contact_offset = self.physics_config['contact_offset']
        sim_params.physx.rest_offset = self.physics_config['rest_offset']
        sim_params.physx.friction_offset_threshold = \
            self.physics_config['friction_offset_threshold']
        sim_params.physx.solver_type = self.physics_config['solver_type']
        sim_params.physx.num_position_iterations = \
            self.physics_config['num_position_iterations']
        sim_params.physx.num_velocity_iterations = \
            self.physics_config['num_velocity_iterations']
        sim_params.physx.num_threads = num_threads
        sim_params.physx.use_gpu = use_gpu
        sim_params.use_gpu_pipeline = use_gpu
        self.sim = gym.create_sim(self.compute_device_id, graphics_device_id,
                                  gymapi.SIM_PHYSX, sim_params)
        model = gym.load_asset(self.sim, isaacgym_asset_path,
                               isaacgym_ground_walking_model_path)