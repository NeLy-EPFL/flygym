import numpy as np
import dm_control.mjcf as mjcf
from gymnasium import spaces
from tqdm import trange
from gymnasium.utils.env_checker import check_env

from flygym.camera import Camera
from flygym.vision import save_video_with_vision_insets
from flygym.examples.locomotion import HybridTurningController
from flygym.examples.vision import MovingObjArena


class VisualTaxis(HybridTurningController):
    """
    A simple visual taxis task where the fly has to follow a moving object.

    Notes
    -----
    Please refer to the `"MPD Task Specifications" page
    <https://neuromechfly.org/api_ref/mdp_specs.html#simple-object-following-visualtaxis>`_
    of the API references for the detailed specifications of the action
    space, the observation space, the reward, the "terminated" and
    "truncated" flags, and the "info" dictionary.
    """

    def __init__(
        self, camera: Camera, obj_threshold=0.15, decision_interval=0.05, **kwargs
    ):
        """
        Parameters
        ----------
        camera : Camera
            The camera to be used for rendering.
        obj_threshold : float
            The threshold for object detection. Minimum and maximum
            brightness values are 0 and 1. If an ommatidium's intensity
            reading is below this value, then it is considered that this
            ommatidium is seeing the object.
        decision_interval : float
            The interval between updates of descending drives, in seconds.
        kwargs
            Additional keyword arguments to be passed to
            ``HybridTurningController.__init__``.
        """
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
        """
        Step the simulation forward in time. Note that this method is to be
        called every time the descending steering signals are updated. This
        typically includes many forward steps of the physics simulation.

        Parameters
        ----------
        control_signal : array_like
            The control signal to apply to the simulation.

        Returns
        -------
        visual_features : array_like
            The preprocessed visual features extracted from the observation.
        reward : float
            The reward obtained from the current step.
        terminated : bool
            Whether the episode is terminated or not. Always False.
        truncated : bool
            Whether the episode is truncated or not. Always False.
        info : dict
            Additional information about the step.
        """
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
        """See `HybridTurningController.reset`."""
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
        neck_kp=1000,
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
