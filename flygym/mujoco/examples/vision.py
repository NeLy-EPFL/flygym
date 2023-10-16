import numpy as np
import dm_control.mjcf as mjcf
from gymnasium import spaces
from tqdm import trange
from gymnasium.utils.env_checker import check_env

from flygym.mujoco import Parameters
from flygym.mujoco.arena import BaseArena
from flygym.mujoco.examples.turning_controller import HybridTurningNMF
from flygym.mujoco.vision.visualize import save_video_with_vision_insets


class MovingObjArena(BaseArena):
    """Flat terrain with a hovering moving object.

    Attributes
    ----------
    arena : mjcf.RootElement
        The arena object that the terrain is built on.
    ball_pos : Tuple[float,float,float]
        The position of the floating object in the arena.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of the terrain in (x, y) dimensions.
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    obj_radius : float
        Radius of the spherical floating object in mm.
    obj_spawn_pos : Tuple[float,float,float]
        Initial position of the object, by default (0, 2, 1).
    move_direction : str
        Which way the ball moves toward first. Can be "left", "right", or
        "random".
    move_speed : float
        Speed of the moving object.
    """

    def __init__(
        self,
        size=(300, 300),
        friction=(1, 0.005, 0.0001),
        obj_radius=1,
        init_ball_pos=(5, 0),
        move_speed=9,
        move_direction="right",
    ):
        self.init_ball_pos = (*init_ball_pos, obj_radius)
        self.ball_pos = np.array(self.init_ball_pos, dtype="float32")
        self.friction = friction
        self.move_speed = move_speed
        self.curr_time = 0
        self.move_direction = move_direction
        if move_direction == "left":
            self.y_mult = 1
        elif move_direction == "right":
            self.y_mult = -1
        elif move_direction == "random":
            self.y_mult = np.random.choice([-1, 1])
        else:
            raise ValueError("Invalid move_direction")

        self.root_element = mjcf.RootElement()

        # Add ground
        ground_size = [*size, 1]
        chequered = self.root_element.asset.add(
            "texture",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.4, 0.4, 0.4),
            rgb2=(0.5, 0.5, 0.5),
        )
        grid = self.root_element.asset.add(
            "material",
            name="grid",
            texture=chequered,
            texrepeat=(60, 60),
            reflectance=0.1,
        )
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground",
            material=grid,
            size=ground_size,
            friction=friction,
        )
        self.root_element.worldbody.add("body", name="b_plane")

        # Add ball
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        self.root_element.worldbody.add(
            "body", name="ball_mocap", mocap=True, pos=self.ball_pos, gravcomp=1
        )
        self.object_body = self.root_element.find("body", "ball_mocap")
        self.object_body.add(
            "geom",
            name="ball",
            type="sphere",
            size=(obj_radius, obj_radius),
            rgba=(0.0, 0.0, 0.0, 1),
            material=obstacle,
        )

        # Add camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(15, 0, 35),
            euler=(0, 0, 0),
            fovy=45,
        )
        self.birdeye_cam_zoom = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam_zoom",
            mode="fixed",
            pos=(15, 0, 20),
            euler=(0, 0, 0),
            fovy=45,
        )

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def step(self, dt, physics):
        heading_vec = np.array([1, 2 * np.cos(self.curr_time * 3) * self.y_mult])
        heading_vec /= np.linalg.norm(heading_vec)
        self.ball_pos[:2] += self.move_speed * heading_vec * dt
        physics.bind(self.object_body).mocap_pos = self.ball_pos
        self.curr_time += dt

    def reset(self, physics):
        if self.move_direction == "random":
            self.y_mult = np.random.choice([-1, 1])
        self.curr_time = 0
        self.ball_pos = np.array(self.init_ball_pos, dtype="float32")
        physics.bind(self.object_body).mocap_pos = self.ball_pos


class VisualTaxis(HybridTurningNMF):
    def __init__(self, obj_threshold=0.15, decision_interval=0.05, **kwargs):
        super().__init__(**kwargs)

        self.obj_threshold = obj_threshold
        self.decision_interval = decision_interval
        self.num_substeps = int(self.decision_interval / self.timestep)
        self.visual_inputs_hist = []

        self.coms = np.empty((self.retina.num_ommatidia_per_eye, 2))
        for i in range(self.retina.num_ommatidia_per_eye):
            mask = self.retina.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)

        self.observation_space = spaces.Box(0, 1, shape=(6,))

    def step(self, control_signal):
        for _ in range(self.num_substeps):
            raw_obs, _, _, _, _ = super().step(control_signal)
            render_res = super().render()
            if render_res is not None:
                # record visual inputs too because they will be played in the video
                self.visual_inputs_hist.append(raw_obs["vision"].copy())
        visual_features = self._process_visual_observation(raw_obs)
        return visual_features, 0, False, False, {}

    def _process_visual_observation(self, raw_obs):
        features = np.zeros((2, 3))
        for i, ommatidia_readings in enumerate(raw_obs["vision"]):
            is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold
            is_obj_coords = self.coms[is_obj]
            if is_obj_coords.shape[0] > 0:
                features[i, :2] = is_obj_coords.mean(axis=0)
            features[i, 2] = is_obj_coords.shape[0]
        features[:, 0] /= self.retina.nrows  # normalize y_center
        features[:, 1] /= self.retina.ncols  # normalize x_center
        features[:, 2] /= self.retina.num_ommatidia_per_eye  # normalize area
        return features.flatten().astype("float32")

    def reset(self, seed=0, **kwargs):
        raw_obs, _ = super().reset(seed=seed)
        self.visual_inputs_hist = []
        return self._process_visual_observation(raw_obs), {}


def calc_ipsilateral_speed(deviation, is_found):
    if not is_found:
        return 1.0
    else:
        return np.clip(1 - deviation * 3, 0.2, 1.0)


if __name__ == "__main__":
    obj_threshold = 0.15
    decision_interval = 0.05
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    arena = MovingObjArena()
    sim_params = Parameters(
        render_camera="birdeye_cam",
        render_playspeed=0.5,
        render_window_size=(800, 608),
        enable_adhesion=True,
        enable_vision=True,
    )
    nmf = VisualTaxis(
        obj_threshold=obj_threshold,
        decision_interval=decision_interval,
        sim_params=sim_params,
        arena=arena,
        contact_sensor_placements=contact_sensor_placements,
    )
    check_env(nmf)

    num_substeps = int(decision_interval / sim_params.timestep)

    obs_hist = []
    deviations_hist = []
    control_signal_hist = []
    raw_visual_hist = []

    obs, _ = nmf.reset()
    for i in trange(70):
        left_deviation = 1 - obs[1]
        right_deviation = obs[4]
        left_found = obs[2] > 0.01
        right_found = obs[5] > 0.01
        if not left_found:
            left_deviation = np.nan
        if not right_found:
            right_deviation = np.nan
        control_signal = np.array(
            [
                calc_ipsilateral_speed(left_deviation, left_found),
                calc_ipsilateral_speed(right_deviation, right_found),
            ]
        )

        obs, _, _, _, _ = nmf.step(control_signal)
        obs_hist.append(obs)
        raw_visual_hist.append(nmf._curr_visual_input.copy())
        deviations_hist.append([left_deviation, right_deviation])
        control_signal_hist.append(control_signal)

    nmf.save_video("./outputs/object_following.mp4")

    save_video_with_vision_insets(
        nmf, "./outputs/object_following_with_retina_images.mp4", nmf.visual_inputs_hist
    )
