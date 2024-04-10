import pickle
import torch
import numpy as np
import flyvision
from torch import Tensor
from pathlib import Path
from tqdm import trange
from flygym import Fly, Camera
from flygym.examples.turning_controller import HybridTurningNMF
from flygym.examples.vision import MovingObjArena
from flyvision.utils.activity_utils import LayerActivity

from flygym.examples.vision_connectome_model import (
    RealTimeVisionNetworkView,
    RetinaMapper,
    MovingFlyArena,
    visualize_vision,
)


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
            vision_obs_grayscale = obs["vision"].max(axis=-1)
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

        # Step vision network if updated
        if info["vision_updated"] or self._nn_activities_buffer is None:
            vision_obs_grayscale = obs["vision"].max(axis=-1)
            visual_input = self.retina_mapper.flygym_to_flyvis(vision_obs_grayscale)
            visual_input = Tensor(visual_input).to(flyvision.device)
            nn_activities_arr = self.vision_network.forward_one_step(visual_input)
            self._nn_activities_arr_buffer = nn_activities_arr.cpu().numpy()
            self._nn_activities_buffer = LayerActivity(
                self._nn_activities_arr_buffer,
                self.vision_network.connectome,
                keepref=True,
                use_central=False,
            )

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

        return super().reset(*args, **kwargs)


if __name__ == "__main__":
    regenerate_walking = True
    output_dir = Path("./outputs/connectome_constrained_vision/")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_time = 2.0  # seconds
    vision_refresh_rate = 500  # Hz

    arena = MovingFlyArena(move_speed=25, lateral_magnitude=2)

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

    sim.reset(seed=0)

    num_physics_steps = int(run_time / sim.timestep)

    obs_hist = []
    rendered_image_hist = []
    vision_observation_hist = []
    nn_activities_hist = []

    # Main simulation loop
    for i in trange(num_physics_steps):
        obs, _, _, _, info = sim.step(action=np.array([1, 1]))
        rendered_img = sim.render()[0]
        obs_hist.append(obs)
        if rendered_img is not None:
            rendered_image_hist.append(rendered_img)
            vision_observation_hist.append(obs["vision"])
            nn_activities_hist.append(obs["nn_activities"])

    visualize_vision(
        Path(output_dir / "vision_simulation.mp4"),
        fly.retina,
        sim.retina_mapper,
        rendered_image_hist,
        vision_observation_hist,
        nn_activities_hist,
        fps=cam.fps,
    )

    obs_hist = [
        obs
        for obs, vision_updated in zip(obs_hist, fly.vision_update_mask)
        if vision_updated
    ]
    for obs in obs_hist:
        del obs["nn_activities"]
    with open(output_dir / "vision_simulation_obs_hist.npy", "wb") as f:
        pickle.dump(obs_hist, f)
