import numpy as np
from tqdm import trange
from gymnasium.utils.env_checker import check_env

from flygym.mujoco import Parameters
from flygym.mujoco.arena import OdorArena
from flygym.mujoco.examples.turning_controller import HybridTurningNMF


if __name__ == "__main__":
    # Define the arena
    # Odor source: array of shape (num_odor_sources, 3) - xyz coords of odor sources
    odor_source = np.array([[24, 0, 1.5], [8, -4, 1.5], [16, 4, 1.5]])

    # Peak intensities: array of shape (num_odor_sources, odor_dimensions)
    # For each odor source, if the intensity is (x, 0) then the odor is in the 1st
    # dimension (in this case attractive). If it's (0, x) then it's in the 2nd dimension
    # (in this case aversive)
    peak_intensity = np.array([[1, 0], [0, 1], [0, 1]])

    # Marker colors: array of shape (num_odor_sources, 4) - RGBA values for each marker,
    # normalized to [0, 1]
    marker_colors = [[255, 127, 14], [31, 119, 180], [31, 119, 180]]
    marker_colors = np.array([[*np.array(color) / 255, 1] for color in marker_colors])

    odor_dimensions = len(peak_intensity[0])

    arena = OdorArena(
        odor_source=odor_source,
        peak_intensity=peak_intensity,
        diffuse_func=lambda x: x**-2,
        marker_colors=marker_colors,
        marker_size=0.3,
    )

    # Define the fly
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    sim_params = Parameters(
        timestep=1e-4,
        render_mode="saved",
        render_playspeed=0.5,
        render_window_size=(800, 608),
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=False,
        render_camera="birdeye_cam",
    )
    sim = HybridTurningNMF(
        sim_params=sim_params,
        arena=arena,
        spawn_pos=(0, 0, 0.2),
        contact_sensor_placements=contact_sensor_placements,
    )

    # Run the simulation
    attractive_gain = -500
    aversive_gain = 80
    decision_interval = 0.05
    run_time = 5
    num_decision_steps = int(run_time / decision_interval)
    physics_steps_per_decision_step = int(decision_interval / sim_params.timestep)

    obs_hist = []
    odor_history = []
    obs, _ = sim.reset()
    for i in trange(num_decision_steps):
        attractive_intensities = np.average(
            obs["odor_intensity"][0, :].reshape(2, 2), axis=0, weights=[9, 1]
        )
        aversive_intensities = np.average(
            obs["odor_intensity"][1, :].reshape(2, 2), axis=0, weights=[10, 0]
        )
        attractive_bias = (
            attractive_gain
            * (attractive_intensities[0] - attractive_intensities[1])
            / attractive_intensities.mean()
        )
        aversive_bias = (
            aversive_gain
            * (aversive_intensities[0] - aversive_intensities[1])
            / aversive_intensities.mean()
        )
        effective_bias = aversive_bias + attractive_bias
        effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
        assert np.sign(effective_bias_norm) == np.sign(effective_bias)

        control_signal = np.ones((2,))
        side_to_modulate = int(effective_bias_norm > 0)
        modulation_amount = np.abs(effective_bias_norm) * 0.8
        control_signal[side_to_modulate] -= modulation_amount

        for j in range(physics_steps_per_decision_step):
            obs, _, _, _, _ = sim.step(control_signal)
            rendered_img = sim.render()
            if rendered_img is not None:
                # record odor intensity too for video
                odor_history.append(obs["odor_intensity"])
            obs_hist.append(obs)

        # Stop when the fly is within 2mm of the attractive odor source
        if np.linalg.norm(obs["fly"][0, :2] - odor_source[0, :2]) < 2:
            break

    sim.save_video("./outputs/odor_taxis.mp4")
