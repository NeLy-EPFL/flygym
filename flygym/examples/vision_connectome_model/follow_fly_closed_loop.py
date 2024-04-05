import pickle
import torch
import numpy as np
import flyvision
from torch import Tensor
from pathlib import Path
from tqdm import trange
from flygym import Fly, Camera
from flygym.examples.turning_controller import HybridTurningNMF
from flyvision.utils.activity_utils import LayerActivity

from flygym.examples.vision_connectome_model import (
    RealTimeVisionNetworkView,
    RetinaMapper,
    MovingFlyArena,
    visualize_vision,
)
from flygym.examples.head_stabilization import HeadStabilizationInferenceWrapper


torch.manual_seed(0)


class NMFRealisticVison(HybridTurningNMF):
    def __init__(self, vision_network_dir=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if vision_network_dir is None:
            vision_network_dir = flyvision.results_dir / "opticflow/000/0000"
        vision_network_view = RealTimeVisionNetworkView(vision_network_dir)
        self.vision_network = vision_network_view.init_network(chkpt="best_chkpt")
        self.retina_mapper = RetinaMapper()
        self._vision_network_initialized = False
        self._nn_activities_buffer = None

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # If first frame, initialize vision network
        if not self._vision_network_initialized:
            self._initialize_vision_network(obs["vision"])

        # Step vision network if updated
        if info["vision_updated"] or self._nn_activities_buffer is None:
            nn_activities, nn_activities_arr = self._get_visual_nn_activities(
                obs["vision"]
            )
            self._nn_activities_buffer = nn_activities
            self._nn_activities_arr_buffer = nn_activities_arr

        obs["nn_activities"] = self._nn_activities_buffer
        obs["nn_activities_arr"] = self._nn_activities_arr_buffer
        return obs, reward, terminated, truncated, info

    def close(self):
        self.vision_network.cleanup_step_by_step_simulation()
        self._vision_network_initialized = False
        return super().close()

    def reset(self, *args, **kwargs):
        if self._vision_network_initialized:
            self.vision_network.cleanup_step_by_step_simulation()
            self._vision_network_initialized = False
        obs, info = super().reset(*args, **kwargs)
        self._initialize_vision_network(obs["vision"])
        nn_activities, nn_activities_arr = self._get_visual_nn_activities(obs["vision"])
        self._nn_activities_buffer = nn_activities
        self._nn_activities_arr_buffer = nn_activities_arr
        obs["nn_activities"] = self._nn_activities_buffer
        obs["nn_activities_arr"] = self._nn_activities_arr_buffer
        return obs, info

    def _initialize_vision_network(self, vision_obs):
        vision_obs_grayscale = vision_obs.max(axis=-1)
        visual_input = self.retina_mapper.flygym_to_flyvis(vision_obs_grayscale)
        visual_input = Tensor(visual_input).to(flyvision.device)
        initial_state = self.vision_network.fade_in_state(
            t_fade_in=1.0,
            dt=1 / self.fly.vision_refresh_rate,
            initial_frames=visual_input.unsqueeze(1),
        )
        self.vision_network.setup_step_by_step_simulation(
            dt=1 / self.fly.vision_refresh_rate,
            initial_state=initial_state,
            as_states=False,
            num_samples=2,
        )
        self._initial_state = initial_state
        self._vision_network_initialized = True

    def _get_visual_nn_activities(self, vision_obs):
        vision_obs_grayscale = vision_obs.max(axis=-1)
        visual_input = self.retina_mapper.flygym_to_flyvis(vision_obs_grayscale)
        visual_input = Tensor(visual_input).to(flyvision.device)
        nn_activities_arr = self.vision_network.forward_one_step(visual_input)
        nn_activities_arr = nn_activities_arr.cpu().numpy()
        nn_activities = LayerActivity(
            nn_activities_arr,
            self.vision_network.connectome,
            keepref=True,
            use_central=False,
        )
        return nn_activities, nn_activities_arr


if __name__ == "__main__":
    enable_head_stabilization = True
    output_dir = Path("./outputs/closed_loop_control/")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_time = 2  # seconds
    vision_refresh_rate = 500  # Hz
    t3_detection_threshold = 0.15

    # Setup NMF simulation
    arena = MovingFlyArena(move_speed=20, lateral_magnitude=2)
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in [
            "Tibia",
            "Tarsus1",
            "Tarsus2",
            "Tarsus3",
            "Tarsus4",
            "Tarsus5",
        ]
    ]
    fly = Fly(
        contact_sensor_placements=contact_sensor_placements,
        enable_adhesion=True,
        enable_vision=True,
        vision_refresh_rate=vision_refresh_rate,
        neck_kp=1000,
    )
    cam = Camera(
        fly=fly,
        camera_id="birdeye_cam",
        play_speed=0.2,
        window_size=(800, 608),
        fps=24,
        play_speed_text=False,
    )
    sim = NMFRealisticVison(
        fly=fly,
        cameras=[cam],
        arena=arena,
    )

    # Setup parameters for object following
    with open("outputs/replay_visual_experience/obs_hist.npy", "rb") as f:
        obs_hist = pickle.load(f)
    nn_activities_all = LayerActivity(
        np.array([obs["nn_activities_arr"] for obs in obs_hist]),
        sim.vision_network.connectome,
        keepref=True,
        use_central=False,
    )
    t3_median_response = np.median(nn_activities_all["T3"], axis=0)

    # Load head stabilization model if enabled
    if enable_head_stabilization:
        model_path = Path("outputs/head_stabilization/models/")
        head_stabilization_model = HeadStabilizationInferenceWrapper(
            model_path=model_path / "three_layer_mlp.pth",
            scaler_param_path=model_path / "joint_angle_scaler_params.pkl",
        )
    else:
        head_stabilization_model = None

    # Run simulation
    obs, info = sim.reset(seed=0)
    num_physics_steps = int(run_time / sim.timestep)
    obs_hist = []
    rendered_image_hist = []
    vision_observation_hist = []
    nn_activities_hist = []
    dn_drive = np.array([1, 1])

    for i in trange(num_physics_steps):
        if info["vision_updated"]:
            nn_activities = obs["nn_activities"]
            obj_mask = t3_median_response - nn_activities["T3"] > t3_detection_threshold
            t4a_intensity = np.mean(np.abs(nn_activities["T4a"][obj_mask]))
            t4b_intensity = np.mean(np.abs(nn_activities["T4b"][obj_mask]))
            diff = t4a_intensity - t4b_intensity
            smaller_amp = max(1 - np.abs(diff) * 3, 0.4)
            larger_amp = min(1 + np.abs(diff) * 1, 1.2)
            if diff < 0:
                dn_drive = np.array([larger_amp, smaller_amp])
            else:
                dn_drive = np.array([smaller_amp, larger_amp])
            if np.any(np.isnan(dn_drive)):
                dn_drive = np.array([1, 1])

        obs, _, _, _, info = sim.step(action=dn_drive)
        rendered_img = sim.render()[0]
        obs_hist.append(obs)
        if rendered_img is not None:
            rendered_image_hist.append(rendered_img)
            vision_observation_hist.append(obs["vision"])
            nn_activities_hist.append(obs["nn_activities"])

    # Clean up, saving data, and visualization
    cam.save_video(output_dir / "object_following.mp4")
    visualize_vision(
        Path(output_dir / "object_following.mp4"),
        fly.retina,
        sim.retina_mapper,
        rendered_image_hist,
        vision_observation_hist,
        nn_activities_hist,
        fps=cam.fps,
    )
