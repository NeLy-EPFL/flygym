import numpy as np
import dm_control.mjcf as mjcf
from gymnasium import spaces
from tqdm import trange
from gymnasium.utils.env_checker import check_env

from flygym.camera import Camera
from flygym.arena import BaseArena
from flygym.examples.turning_controller import HybridTurningNMF
from flygym.vision import save_video_with_vision_insets


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
    init_ball_pos : Tuple[float,float]
        Initial position of the object, by default (5, 0).
    move_speed : float
        Speed of the moving object. By default 10.
    move_direction : str
        Which way the ball moves toward first. Can be "left", "right", or
        "random". By default "right".
    lateral_magnitude : float
        Magnitude of the lateral movement of the object as a multiplier of
        forward velocity. For example, when ``lateral_magnitude`` is 1, the
        object moves at a heading (1, 1) when its movement is the most
        lateral. By default 2.
    """

    def __init__(
        self,
        size=(300, 300),
        friction=(1, 0.005, 0.0001),
        obj_radius=1,
        init_ball_pos=(5, 0),
        move_speed=10,
        move_direction="right",
        lateral_magnitude=2,
    ):
        super().__init__()

        self.init_ball_pos = (*init_ball_pos, obj_radius)
        self.ball_pos = np.array(self.init_ball_pos, dtype="float32")
        self.friction = friction
        self.move_speed = move_speed
        self.curr_time = 0
        self.move_direction = move_direction
        self.lateral_magnitude = lateral_magnitude
        if move_direction == "left":
            self.y_mult = 1
        elif move_direction == "right":
            self.y_mult = -1
        elif move_direction == "random":
            self.y_mult = np.random.choice([-1, 1])
        else:
            raise ValueError("Invalid move_direction")

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
        heading_vec = np.array(
            [1, self.lateral_magnitude * np.cos(self.curr_time * 3) * self.y_mult]
        )
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
    def __init__(
        self, camera: Camera, obj_threshold=0.15, decision_interval=0.05, **kwargs
    ):
        super().__init__(cameras=[camera], **kwargs)

        self.obj_threshold = obj_threshold
        self.decision_interval = decision_interval
        self.num_substeps = int(self.decision_interval / self.timestep)
        self.visual_inputs_hist = []

        self.coms = np.empty((self.fly.retina.num_ommatidia_per_eye, 2))
        for i in range(self.fly.retina.num_ommatidia_per_eye):
            mask = self.fly.retina.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)

        self.observation_space = spaces.Box(0, 1, shape=(6,))

    def step(self, control_signal):
        vision_inputs = []
        for _ in range(self.num_substeps):
            raw_obs, _, _, _, info = super().step(control_signal)
            if info["vision_updated"]:
                vision_inputs.append(raw_obs["vision"])
            render_res = super().render()[0]
            if render_res is not None:
                # record visual inputs too because they will be played in the video
                self.visual_inputs_hist.append(raw_obs["vision"].copy())
        median_vision_input = np.median(vision_inputs, axis=0)
        visual_features = self._process_visual_observation(median_vision_input)
        return visual_features, 0, False, False, {}

    def _process_visual_observation(self, vision_input):
        features = np.zeros((2, 3))
        for i, ommatidia_readings in enumerate(vision_input):
            is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold
            is_obj_coords = self.coms[is_obj]
            if is_obj_coords.shape[0] > 0:
                features[i, :2] = is_obj_coords.mean(axis=0)
            features[i, 2] = is_obj_coords.shape[0]
        features[:, 0] /= self.fly.retina.nrows  # normalize y_center
        features[:, 1] /= self.fly.retina.ncols  # normalize x_center
        features[:, 2] /= self.fly.retina.num_ommatidia_per_eye  # normalize area
        return features.ravel().astype("float32")

    def reset(self, seed=0, **kwargs):
        raw_obs, _ = super().reset(seed=seed)
        self.visual_inputs_hist = []
        return self._process_visual_observation(raw_obs["vision"]), {}


def calc_ipsilateral_speed(deviation, is_found):
    if not is_found:
        return 1.0
    else:
        return np.clip(1 - deviation * 3, 0.4, 1.2)


if __name__ == "__main__":
    from flygym import Fly, Camera

    obj_threshold = 0.2
    decision_interval = 0.025
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    arena = MovingObjArena()
    fly = Fly(
        contact_sensor_placements=contact_sensor_placements,
        enable_adhesion=True,
        enable_vision=True,
        head_stabilization_kp=1000,
    )
    cam = Camera(
        fly=fly,
        camera_id="birdeye_cam",
        play_speed=0.5,
        window_size=(800, 608),
    )
    sim = VisualTaxis(
        fly=fly,
        camera=cam,
        obj_threshold=obj_threshold,
        decision_interval=decision_interval,
        arena=arena,
        intrinsic_freqs=np.ones(6) * 9,
    )
    check_env(sim)

    num_substeps = int(decision_interval / sim.timestep)

    obs_hist = []
    deviations_hist = []
    control_signal_hist = []
    raw_visual_hist = []

    obs, _ = sim.reset()
    for i in trange(140):
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

        obs, _, _, _, _ = sim.step(control_signal)
        obs_hist.append(obs)
        raw_visual_hist.append(sim.fly._curr_visual_input.copy())
        deviations_hist.append([left_deviation, right_deviation])
        control_signal_hist.append(control_signal)

    cam.save_video("./outputs/object_following.mp4")

    save_video_with_vision_insets(
        sim,
        cam,
        "./outputs/object_following_with_retina_images.mp4",
        sim.visual_inputs_hist,
    )
