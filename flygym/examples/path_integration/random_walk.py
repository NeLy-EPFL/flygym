import numpy as np
import pickle
import cv2
from enum import Enum
from tqdm import trange
from pathlib import Path
from typing import Tuple, Union, Optional, Callable
from dm_control import mjcf

from flygym import Fly, Camera
from flygym.arena import BaseArena, FlatTerrain, BlocksTerrain
from flygym.simulation import SingleFlySimulation
from flygym.util import get_data_path
from flygym.examples.turning_controller import HybridTurningNMF


def get_walking_icons():
    icons_dir = get_data_path("flygym", "data") / "etc/locomotion_icons"
    icons = {}
    for key in ["forward", "left", "right", "stop"]:
        icon_path = icons_dir / f"{key}.png"
        icons[key] = cv2.imread(str(icon_path), cv2.IMREAD_UNCHANGED)
    return {
        WalkingState.FORWARD: icons["forward"],
        WalkingState.TURN_LEFT: icons["left"],
        WalkingState.TURN_RIGHT: icons["right"],
        WalkingState.STOP: icons["stop"],
    }


def add_icon_to_image(image, icon):
    sel = image[: icon.shape[0], -icon.shape[1] :, :]
    mask = icon[:, :, 3] > 0
    sel[mask] = icon[mask, :3]


def add_heading_to_image(
    image, real_heading, estimated_heading, real_position, estimated_position
):
    cv2.putText(
        image,
        f"Real position: ({int(real_position[0]): 3d}, {int(real_position[1]): 3d})",
        org=(20, 390),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    cv2.putText(
        image,
        (
            f"Estm position: ({int(estimated_position[0]): 3d}, "
            f"{int(estimated_position[1]): 3d})"
        ),
        org=(20, 410),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    cv2.putText(
        image,
        f"Real heading: {int(np.rad2deg(real_heading)): 4d} deg",
        org=(20, 430),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=1,
    )
    cv2.putText(
        image,
        f"Estm heading: {int(np.rad2deg(estimated_heading)): 4d} deg",
        org=(20, 450),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        lineType=cv2.LINE_AA,
        thickness=1,
    )


class PathIntegrationArena(BaseArena):
    def __init__(self):
        super().__init__()
        self.friction = (1, 0.005, 0.0001)

        # Set up floor
        floor_material = self.root_element.asset.add(
            "material",
            name="floor_material",
            reflectance=0.0,
            shininess=0.0,
            specular=0.0,
            rgba=[0.8, 0.8, 0.8, 1],
        )
        self.root_element.worldbody.add(
            "geom",
            name="floor",
            type="box",
            size=(300, 300, 1),
            pos=(0, 0, -1),
            material=floor_material,
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


class WalkingState(Enum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3


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


class HomingController:
    def __init__(
        self,
        forward_dn_drive: Tuple[float, float] = (1.0, 1.0),
        right_turn_dn_drive: Tuple[float, float] = (1.2, -0.4),
        min_turn_angle: float = np.deg2rad(5),
        max_turn_angle: float = np.deg2rad(20),
        w_turn: float = 0.1,
        w_stop: float = 0.5,
        dt: float = 1e-4,
        accepted_radius: float = 1.0,
    ):
        # Parse parameters
        self.forward_dn_drive = np.array(forward_dn_drive)
        self.right_turn_dn_diff = np.array(right_turn_dn_drive) - self.forward_dn_drive
        self.min_turn_angle = min_turn_angle
        self.gain = 1 / max_turn_angle
        self.w_turn_steps = int(w_turn / dt)
        self.w_stop_steps = int(w_stop / dt)
        self.dt = dt
        self.accepted_radius = accepted_radius

        # State tracking
        self._heading_hist = []
        self._dist_to_home_hist = []
        self._last_walking_state = None
        self._heading_tgt_buffer = 0

    def step(self, curr_heading: float, curr_pos: np.ndarray):
        # Make decision on stopping
        dist_to_home = np.linalg.norm(curr_pos)
        self._dist_to_home_hist.append(dist_to_home)
        dist_to_home_sel = self._dist_to_home_hist[-self.w_stop_steps :]
        if len(dist_to_home_sel) >= self.w_stop_steps:
            delta_t = -self.dt * np.arange(self.w_stop_steps)
            k, _ = np.polyfit(delta_t, dist_to_home_sel, 1)
            print(f"dist_to_home: {dist_to_home:.2f}, k: {k:.2f}")
            if k > 0 and dist_to_home < self.accepted_radius:
                return WalkingState.STOP, np.zeros(2)

        # Make decision on turning
        self._heading_hist.append(curr_heading)
        print(f"        {curr_heading}")
        curr_heading_smoothed = np.mean(self._heading_hist[-self.w_turn_steps :])
        if (
            self._last_walking_state is None
            or self._last_walking_state == WalkingState.STOP
        ):
            self._heading_tgt_buffer = np.arctan2(-curr_pos[1], -curr_pos[0])
        # heading_tgt = np.arctan2(-curr_pos[1], -curr_pos[0])
        print(
            f"heading_tgt: {self._heading_tgt_buffer:.2f}, curr_heading: {curr_heading_smoothed:.2f}"
        )
        heading_diff = wrap_angle(self._heading_tgt_buffer - curr_heading_smoothed)
        # print(f"heading_diff: {heading_diff:.2f}")
        if np.abs(heading_diff) < self.min_turn_angle:
            heading_diff = 0
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
            print("+++++++++++++++++++++++++++++++++++++++++++++")
        elif heading_diff > 0 and self._last_walking_state == WalkingState.TURN_LEFT:
            # if intended turn direction suddenly jumps to RIGHT during a LEFT turn,
            # it's because the angle has jumped by 2π. Continue the LEFT turn.
            heading_diff -= 2 * np.pi
        elif heading_diff < 0 and self._last_walking_state == WalkingState.TURN_RIGHT:
            # if intended turn direction suddenly jumps to LEFT during a RIGHT turn,
            # it's because the angle has jumped by 2π. Continue the RIGHT turn.
            heading_diff += 2 * np.pi
        turn_signal_1d = np.clip(self.gain * heading_diff, -1, 1)
        dn_drive = self.forward_dn_drive + self.right_turn_dn_diff * turn_signal_1d
        if turn_signal_1d > 0:
            walking_state = WalkingState.TURN_RIGHT
        elif turn_signal_1d < 0:
            walking_state = WalkingState.TURN_LEFT
        else:
            walking_state = WalkingState.FORWARD
        self._last_walking_state = walking_state
        return walking_state, dn_drive


class PathIntegrationNMF(HybridTurningNMF):
    def __init__(
        self,
        comparison_window: float = 0.1,
        do_path_integration: bool = True,
        heading_model: Optional[Callable] = None,
        displacement_model: Optional[Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.comparsion_window_steps = int(comparison_window / self.timestep)
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

        # Calculate stride since last step per leg
        ee_pos_rel = self.absolute_to_relative_pos(
            obs["end_effectors"][:, :2], obs["fly"][0, :2], obs["fly_orientation"][:2]
        )
        if self._last_end_effector_pos is None:
            ee_diff = np.zeros_like(ee_pos_rel)
        else:
            ee_diff = ee_pos_rel - self._last_end_effector_pos
        ee_diff *= info["adhesion"][:, None]
        self._last_end_effector_pos = ee_pos_rel

        # Update total stride length per leg
        last_total_stride_lengths = (
            self.total_stride_lengths_hist[-1] if self.total_stride_lengths_hist else 0
        )
        self.total_stride_lengths_hist.append(last_total_stride_lengths + ee_diff[:, 0])

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

        # Write path-integration-related variables to info for debugging/analysis
        info["ee_diff"] = ee_diff
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


def wrap_angle(angle):
    """Wrap angle to [-π, π]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def run_simulation(
    seed: int = 0,
    explore_time: float = 30.0,
    max_homing_time: float = 0.0,
    live_display: bool = False,
    comparison_window: float = 0.1,
    heading_model: Optional[Callable] = None,
    displacement_model: Optional[Callable] = None,
    output_dir: Optional[Path] = None,
):
    icons = get_walking_icons()
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        # actuator_kp=20,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(0, 0, 0.25),
    )
    arena = PathIntegrationArena()
    # cam = Camera(
    #     fly=fly, camera_id="Animat/camera_left", play_speed=0.1, timestamp_text=True
    # )
    cam = Camera(fly=fly, camera_id="birdeye_cam", play_speed=0.5, timestamp_text=True)
    sim = PathIntegrationNMF(
        comparison_window=comparison_window,
        heading_model=heading_model,
        displacement_model=displacement_model,
        fly=fly,
        arena=arena,
        cameras=[cam],
        timestep=1e-4,
    )

    random_exploration_controller = RandomExplorationController(
        dt=sim.timestep, lambda_turn=2, seed=seed
    )
    homing_controller = HomingController(dt=sim.timestep)

    obs, info = sim.reset(0)
    obs_hist, info_hist = [], []
    _real_heading_buffer = []
    _estimated_heading_buffer = []
    for i in trange(int(explore_time + max_homing_time / sim.timestep)):
        if i * sim.timestep <= explore_time:
            walking_state, dn_drive = random_exploration_controller.step()
        else:
            walking_state, dn_drive = homing_controller.step(
                info["heading_estimate"], info["position_estimate"]
            )
            if walking_state == WalkingState.STOP:
                break
        obs, reward, terminated, truncated, info = sim.step(dn_drive)
        rendered_img = sim.render()[0]

        # Get real heading
        orientation_x, orientation_y = obs["fly_orientation"][:2]
        real_heading = np.arctan2(orientation_y, orientation_x)
        _real_heading_buffer.append(real_heading)

        # Get estimated heading
        if heading_model is not None:
            estimated_heading = info["heading_estimate"]
            _estimated_heading_buffer.append(wrap_angle(estimated_heading))

        if rendered_img is not None:
            # Add walking state icon
            add_icon_to_image(rendered_img, icons[walking_state])

            # Add heading info
            if heading_model is not None:
                real_heading = np.mean(_real_heading_buffer)
                estimated_heading = np.mean(_estimated_heading_buffer)
                _real_heading_buffer = []
                _estimated_heading_buffer = []
                add_heading_to_image(
                    rendered_img,
                    real_heading=real_heading,
                    estimated_heading=estimated_heading,
                    real_position=obs["fly"][0, :2],
                    estimated_position=info["position_estimate"],
                )

            # Display rendered image live
            if live_display:
                cv2.imshow("rendered_img", rendered_img[:, :, ::-1])
                cv2.waitKey(1)

        obs_hist.append(obs)
        info_hist.append(info)

    # Save data if output_dir is provided
    if output_dir is not None:
        path_stem = Path(output_dir) / f"simulation_seed{seed}"
        cam.save_video(path_stem.with_suffix(".mp4"))
        with open(path_stem.with_suffix(".pkl"), "wb") as f:
            pickle.dump({"obs_hist": obs_hist, "info_hist": info_hist}, f)


if __name__ == "__main__":
    from datetime import datetime

    # for i in range(10):
    #     print(f"{datetime.now()}: Running random walk {i}...")
    #     run_simulation(seed=i, run_time=20, live_display=False)

    # # Initial exploration to establish proprioception-heading relationship with
    # run_simulation(seed=0, run_time=20, live_display=True)

    # Try homing with established models
    heading_integration_model = lambda x: 0.12 * x
    displacement_integration_model = lambda x: -0.31 * x - 0.1
    run_simulation(
        comparison_window=0.1,
        heading_model=heading_integration_model,
        displacement_model=displacement_integration_model,
        seed=0,
        explore_time=5,
        max_homing_time=20,
        live_display=True,
        output_dir=Path("outputs/path_integration_homing"),
    )
