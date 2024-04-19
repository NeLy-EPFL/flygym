import flyvision
from torch import Tensor
from flygym.examples.turning_controller import HybridTurningNMF
from flyvision.utils.activity_utils import LayerActivity
from flygym.examples.vision_connectome_model import (
    RealTimeVisionNetworkView,
    RetinaMapper,
)


class NMFRealisticVision(HybridTurningNMF):
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

        info["nn_activities"] = self._nn_activities_buffer
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
        info["nn_activities"] = self._nn_activities_buffer
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
