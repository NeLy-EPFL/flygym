import numpy as np
from enum import Enum
from typing import Tuple, Union, Optional, Callable
from dm_control import mjcf

from flygym.arena import BaseArena
from flygym.examples.turning_controller import HybridTurningNMF


class WalkingState(Enum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3


class PathIntegrationArena(BaseArena):
    def __init__(self):
        super().__init__()
        self.friction = (1, 0.005, 0.0001)

        # Set up floor
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.7, 0.7, 0.7),
            rgb2=(0.8, 0.8, 0.8),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(10, 10),
            reflectance=0.1,
            rgba=(1.0, 1.0, 1.0, 1.0),
        )
        self.root_element.worldbody.add(
            "geom",
            name="floor",
            type="box",
            size=(300, 300, 1),
            pos=(0, 0, -1),
            material=grid,
        )

        # Add marker at origin
        self.root_element.worldbody.add(
            "geom",
            name="origin_marker",
            type="sphere",
            size=(1,),
            pos=(0, 0, 5),
            rgba=(1, 0, 0, 1),
        )

        # Add birdeye camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(0, 0, 150),
            euler=(0, 0, 0),
            fovy=60,
        )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def update_cam_pos(self, physics: mjcf.Physics, fly_pos: np.ndarray) -> None:
        physics.bind(self.birdeye_cam).pos[:2] = fly_pos / 2


class RandomExplorationController:
    def __init__(
        self,
        dt: float,
        forward_dn_drive: Tuple[float, float] = (1.0, 1.0),
        left_turn_dn_drive: Tuple[float, float] = (-0.4, 1.2),
        right_turn_dn_drive: Tuple[float, float] = (1.2, -0.4),
        turn_duration_mean: float = 0.4,
        turn_duration_std: float = 0.1,
        lambda_turn: float = 1.0,
        seed: int = 0,
        init_time: float = 0.1,
    ) -> None:
        self.random_state = np.random.RandomState(seed)
        self.dt = dt
        self.init_time = init_time
        self.curr_time = 0.0
        self.curr_state: WalkingState = WalkingState.FORWARD
        self._curr_turn_duration: Union[None, float] = None

        # DN drives
        self.dn_drives = {
            WalkingState.FORWARD: np.array(forward_dn_drive),
            WalkingState.TURN_LEFT: np.array(left_turn_dn_drive),
            WalkingState.TURN_RIGHT: np.array(right_turn_dn_drive),
        }

        # Turning related parameters
        self.turn_duration_mean = turn_duration_mean
        self.turn_duration_std = turn_duration_std
        self.lambda_turn = lambda_turn

    def step(self):
        # Upon spawning, just walk straight for a bit (init_time) for things to settle
        if self.curr_time < self.init_time:
            self.curr_time += self.dt
            return WalkingState.FORWARD, self.dn_drives[WalkingState.FORWARD]

        # Forward -> turn transition
        if self.curr_state == WalkingState.FORWARD:
            p_nochange = np.exp(-self.lambda_turn * self.dt)
            if self.random_state.rand() > p_nochange:
                # decide turn duration and direction
                self._curr_turn_duration = self.random_state.normal(
                    self.turn_duration_mean, self.turn_duration_std
                )
                self.curr_state = self.random_state.choice(
                    [WalkingState.TURN_LEFT, WalkingState.TURN_RIGHT]
                )
                self.curr_state_start_time = self.curr_time

        # Turn -> forward transition
        if self.curr_state in (WalkingState.TURN_LEFT, WalkingState.TURN_RIGHT):
            if self.curr_time - self.curr_state_start_time > self._curr_turn_duration:
                self.curr_state = WalkingState.FORWARD
                self.curr_state_start_time = self.curr_time

        self.curr_time += self.dt
        return self.curr_state, self.dn_drives[self.curr_state]


class PathIntegrationNMF(HybridTurningNMF):
    def __init__(
        self,
        time_scale: float = 0.1,
        do_path_integration: bool = True,
        heading_model: Optional[Callable] = None,
        displacement_model: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if time_scale is not None:
            self.comparsion_window_steps = int(time_scale / self.timestep)
            self.comparsion_window = self.comparsion_window_steps * self.timestep
        self.do_path_integration = do_path_integration
        self.heading_model = heading_model
        self.displacement_model = displacement_model
        if do_path_integration:
            if heading_model is None or displacement_model is None:
                raise ValueError(
                    "``heading_model`` and ``displacement_model`` "
                    "must be provided when ``do_path_integration`` is True."
                )

        self._last_end_effector_pos: Union[None, np.ndarray] = None
        self.total_stride_lengths_hist = []
        self.heading_estimate_hist = []
        self.pos_estimate_hist = []

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Update camera position
        self.arena.update_cam_pos(self.physics, obs["fly"][0, :2])

        # Calculate stride since last step per leg
        ee_pos_rel = self.absolute_to_relative_pos(
            obs["end_effectors"][:, :2], obs["fly"][0, :2], obs["fly_orientation"][:2]
        )
        if self._last_end_effector_pos is None:
            ee_diff_unmasked = np.zeros_like(ee_pos_rel)
        else:
            ee_diff_unmasked = ee_pos_rel - self._last_end_effector_pos
        ee_diff_masked = ee_diff_unmasked * info["adhesion"][:, None]
        self._last_end_effector_pos = ee_pos_rel

        # Update total stride length per leg
        last_total_stride_lengths = (
            self.total_stride_lengths_hist[-1] if self.total_stride_lengths_hist else 0
        )
        self.total_stride_lengths_hist.append(
            last_total_stride_lengths + ee_diff_masked[:, 0]
        )

        # Update path integration if enabled
        if self.do_path_integration:
            # Estimate change in heading and position in the past step
            if len(self.total_stride_lengths_hist) > self.comparsion_window_steps:
                stride_diff = (
                    self.total_stride_lengths_hist[-1]
                    - self.total_stride_lengths_hist[-self.comparsion_window_steps - 1]
                )
                lr_asymmetry = stride_diff[:3].sum() - stride_diff[3:].sum()
                stride_total = stride_diff.sum()

                # Estimate Δheading
                heading_diff = self.heading_model(lr_asymmetry)
                heading_diff /= self.comparsion_window_steps

                # Estimate ||Δposition|| in the direction of the fly's heading
                forward_displacement_diff = self.displacement_model(stride_total)
                forward_displacement_diff /= self.comparsion_window_steps
            else:
                heading_diff = 0  # no update when not enough data
                forward_displacement_diff = 0

            # Integrate heading and position estimates
            last_heading_estimate = (
                self.heading_estimate_hist[-1] if self.heading_estimate_hist else 0
            )
            curr_heading_estimate = last_heading_estimate + heading_diff
            self.heading_estimate_hist.append(curr_heading_estimate)
            vec_disp_estimate = np.array(
                [
                    np.cos(curr_heading_estimate) * forward_displacement_diff,
                    np.sin(curr_heading_estimate) * forward_displacement_diff,
                ]
            )
            last_pos_estimate = (
                self.pos_estimate_hist[-1] if self.pos_estimate_hist else np.zeros(2)
            )
            curr_pos_estimate = last_pos_estimate + vec_disp_estimate
            self.pos_estimate_hist.append(curr_pos_estimate)
        else:
            self.heading_estimate_hist.append(None)
            self.pos_estimate_hist.append(None)

        # Write path-integration-related variables to info for debugging/analysis
        info["ee_diff_masked"] = ee_diff_masked
        info["ee_diff_unmasked"] = ee_diff_unmasked
        info["total_stride_lengths"] = self.total_stride_lengths_hist[-1]
        info["heading_estimate"] = self.heading_estimate_hist[-1]
        info["position_estimate"] = self.pos_estimate_hist[-1]

        return obs, reward, terminated, truncated, info

    @staticmethod
    def absolute_to_relative_pos(
        pos: np.ndarray, base_pos: np.ndarray, heading: np.ndarray
    ) -> np.ndarray:
        rel_pos = pos - base_pos
        heading = heading / np.linalg.norm(heading)
        angle = np.arctan2(heading[1], heading[0])
        rot_matrix = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )
        pos_rotated = np.dot(rel_pos, rot_matrix.T)
        return pos_rotated
