import numpy as np
import copy
import yaml
import gymnasium as gym
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from gymnasium import spaces

try:
    from isaacgym import gymapi
    from isaacgym import gymtorch
except ImportError:
    raise ImportError(
        "Failed to import Isaac Gym. Please follow the instructions on the "
        "installation page of the documentation. Note that torch must be "
        "imported after isaacgym modules."
    )

import torch  # has to be imported after torch modules

from flygym.util.data import isaacgym_asset_path
from flygym.util.data import isaacgym_ground_walking_model_path
from flygym.util.data import default_pose_path, stretch_pose_path
from flygym.util.config import leg_dofs_fused_tarsi


_init_pose_lookup = {
    "default": default_pose_path,
    "stretch": stretch_pose_path,
}
_default_terrain_config = {
    "flat": {
        "size": (50_000, 50_000),
        "friction": (1, 0.005, 0.0001),
        "fly_pos": (0, 0, 300),
        "fly_orient": (0, 1, 0, 0.1),
    },
}
_default_physics_config = {
    "gravity": (0, 0, -9.81e3),
    "num_substeps": 2,
    "contact_offset": 0.2,  # mm
    "rest_offset": 0.1,  # mm
    "friction_offset_threshold": 4e-4,  # mm
    "solver_type": 1,  # 0: PGS (Iterative sequential impulse solver)
    # 1: TGS (Nonlinear iterative solver, more robust but
    #    slightly more expensive)
    "num_position_iterations": 8,  # [1, 255]
    "num_velocity_iterations": 1,  # [1, 255]
}
_default_render_config = {
    "saved": {"window_size": (640, 480), "playspeed": 1.0, "fps": 60},
    "viewer": {"window_size": (640, 480), "max_playspeed": 1.0, "fps": 60},
    "headless": {},
}


class NeuroMechFlyIsaacGym(gym.Env):
    def __init__(
        self,
        num_envs: int = 1,
        render_mode: str = "viewer",
        render_config: Dict[str, Any] = {},
        actuated_joints: List = leg_dofs_fused_tarsi,
        timestep: float = 0.0001,
        output_dir: Optional[Path] = None,
        terrain: str = "flat",
        terrain_config: Dict[str, Any] = {},
        physics_config: Dict[str, Any] = {},
        control: str = "position",
        init_pose: str = "default",
        use_gpu: bool = True,
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        num_threads: int = 4,
    ) -> None:
        self.num_envs = num_envs
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
        self.compute_device = f"cuda:{compute_device_id}" if use_gpu else "cpu"

        # Define action and observation spaces
        num_dofs = len(actuated_joints)
        bound = np.pi if control == "position" else np.inf
        self.action_space = {
            "joints": spaces.Box(low=-bound, high=bound, shape=(num_dofs,))
        }
        self.observation_space = {
            # joints: shape (3, num_dofs): (pos, vel, torque) of each DoF
            "joints": spaces.Box(low=-np.inf, high=np.inf, shape=(3, num_dofs)),
            # fly: shape (4, 3):
            # 0th row: x, y, z position of the fly in arena
            # 1st row: x, y, z velocity of the fly in arena
            # 2nd row: orientation of fly around x, y, z axes
            # 3rd row: rate of change of fly orientation
            "fly": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 3)),
        }

        # Initialize physics simulation
        self.ig_gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(*self.physics_config["gravity"])
        sim_params.substeps = self.physics_config["num_substeps"]
        sim_params.dt = self.timestep
        sim_params.physx.contact_offset = self.physics_config["contact_offset"]
        sim_params.physx.rest_offset = self.physics_config["rest_offset"]
        sim_params.physx.friction_offset_threshold = self.physics_config[
            "friction_offset_threshold"
        ]
        sim_params.physx.solver_type = self.physics_config["solver_type"]
        sim_params.physx.num_position_iterations = self.physics_config[
            "num_position_iterations"
        ]
        sim_params.physx.num_velocity_iterations = self.physics_config[
            "num_velocity_iterations"
        ]
        sim_params.physx.num_threads = num_threads
        sim_params.physx.use_gpu = use_gpu
        sim_params.use_gpu_pipeline = use_gpu
        self.sim = self.ig_gym.create_sim(
            self.compute_device_id,
            graphics_device_id,
            gymapi.SIM_PHYSX,
            sim_params,
        )

        # Set up viewer
        if render_mode == "viewer":
            self.viewer = self.ig_gym.create_viewer(
                self.sim, gymapi.CameraProperties()
            )

        # Initial root pose
        if init_pose == "default":
            pose_file = default_pose_path
        elif init_pose == "stretch":
            pose_file = stretch_pose_path
        else:
            pose_file = init_pose
        with open(pose_file) as f:
            init_pose_config = yaml.safe_load(f)["joints"]
        init_pose_config = {
            k: np.deg2rad(v) for k, v in init_pose_config.items()
        }

        # Add assets and prepare simulation
        self._make_ground()
        self.envs, self.actors, self._dof2idx = self._make_envs(num_envs)
        self._idx2dof = {v: k for k, v in self._dof2idx.items()}
        self.ig_gym.prepare_sim(self.sim)

        # Calculate initial state for each joint
        num_dofs = len(self.actuated_joints)
        self._default_dof_pos = torch.tensor(
            [init_pose_config[self._idx2dof[i]] for i in range(num_dofs)],
            dtype=torch.float32,
            device=self.compute_device,
        )
        self._default_dof_vel = torch.zeros(
            num_dofs, dtype=torch.float32, device=self.compute_device
        )

        # Get handles to state variables (root, DoFs, sensors)
        _dof_state = self.ig_gym.acquire_dof_state_tensor(self.sim)
        _root_state = self.ig_gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)
        self.dof_pos = self.dof_state.view(self.num_envs, num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, num_dofs, 2)[..., 1]
        print(f"dof_pos: {self.dof_pos.shape}")
        print(f"dof_state: {self.dof_state.shape}")

        # Create buffers to track status of each env
        self.obs_buffer = ...
        self.reward_buffer = ...
        self.reset_buffer = ...
        self.time_buffer = ...

    def _make_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
        self.ig_gym.add_ground(self.sim, plane_params)

    def _make_envs(self, num_envs):
        # Load assets
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        nmf_asset = self.ig_gym.load_asset(
            self.sim,
            isaacgym_asset_path,
            isaacgym_ground_walking_model_path,
            asset_options,
        )

        # Define initial root pose
        root_pose = gymapi.Transform()
        root_pose.p = gymapi.Vec3(0, 0, 5)
        self._default_root_state = torch.tensor(
            [
                root_pose.p.x,
                root_pose.p.y,
                root_pose.p.z,
                root_pose.r.x,
                root_pose.r.y,
                root_pose.r.z,
                root_pose.r.w,
                *[0, 0, 0, 0, 0, 0],  # lin & ang vel set to 0
            ],
            device=self.compute_device,
        )
        # Set up env grid, create envs with actors
        spacing = 3
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        envs_per_row = int(np.sqrt(num_envs))

        envs = []
        actors = []
        for i in range(num_envs):
            env = self.ig_gym.create_env(
                self.sim, env_lower, env_upper, envs_per_row
            )
            actor = self.ig_gym.create_actor(
                env, nmf_asset, root_pose, f"nmf-{i:04d}", i, 1, 0
            )
            envs.append(env)
            actors.append(actor)

            dof_props = self.ig_gym.get_actor_dof_properties(env, actor)
            dof_props["driveMode"] = gymapi.DOF_MODE_POS
            dof_props["stiffness"].fill(10000)
            dof_props["damping"].fill(50)
            self.ig_gym.set_actor_dof_properties(env, actor, dof_props)

        # Get dictionary mapping joint names to DoF indices - this should be
        # the same for all envs
        actor_dof_dict = self.ig_gym.get_actor_dof_dict(env, actor)

        return envs, actors, actor_dof_dict

    def reset(self, env_ids=None):
        if env_ids is None:  # Reset all envs by default
            env_ids = torch.arange(self.num_envs)
        if len(env_ids) == 0:
            return
        env_ids_int32 = env_ids.to(
            dtype=torch.int32, device=self.compute_device
        )
        print(f"Resetting envs {env_ids_int32}...")

        # Reset root pose
        self.root_state[env_ids, :] = self._default_root_state
        self.ig_gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        # Reset DoFs
        self.dof_pos[env_ids, :] = self._default_dof_pos
        self.dof_vel[env_ids, :] = self._default_dof_vel
        self.ig_gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
    
    def step(self, action):
        ...

    def _test_simulate(self):
        t_idx = 0
        # while not self.ig_gym.query_viewer_has_closed(self.viewer):
        for i in range(500):
            self.ig_gym.simulate(self.sim)
            self.ig_gym.fetch_results(self.sim, True)
            # update viewer
            if self.render_mode == "viewer":
                self.ig_gym.step_graphics(self.sim)
                self.ig_gym.draw_viewer(self.viewer, self.sim, False)
                self.ig_gym.sync_frame_time(self.sim)  # make it real time
            t_idx += 1


if __name__ == "__main__":
    env = NeuroMechFlyIsaacGym(num_envs=4)
    env.reset()
    print("ok")
    env._test_simulate()
    env.reset()
    env._test_simulate()
