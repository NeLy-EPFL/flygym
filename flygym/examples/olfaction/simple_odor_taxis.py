import numpy as np
from tqdm import trange

from flygym import Fly, Camera
from flygym.arena import OdorArena
from flygym.examples.locomotion import HybridTurningController


def run_simulation(
    odor_source,
    peak_odor_intensity,
    marker_colors=None,
    spawn_pos=(0, 0, 0.2),
    spawn_orientation=(0, 0, np.pi / 2),
    run_time=5,
    decision_interval=0.05,
    attractive_gain=-500,
    aversive_gain=80,
    attractive_palps_antennae_weights=(1, 9),
    aversive_palps_antennae_weights=(0, 10),
    target_pos=None,
    distance_threshold=2,
    video_path=None,
    enable_rendering: bool = False,
):
    """
    Parameters
    ----------
    odor_source : np.ndarray
        Array of shape (num_odor_sources, 3) - xyz coords of odor sources
    peak_odor_intensity : np.ndarray
        Array of shape (num_odor_sources, odor_dimensions)
        For each odor source, if the intensity is (x, 0) then the odor is in
        the 1st dimension (in this case attractive). If it's (0, x) then it's
        in the 2nd dimension (in this case aversive)
    marker_colors : np.ndarray
        Array of shape (num_odor_sources, 4) - RGBA values for each marker,
        normalized to [0, 1]
    spawn_pos : tuple[float, float, float], optional
        The (x, y, z) position in the arena defining where the fly will be
        spawned, in mm. By default (0, 0, 0.5).
    spawn_orientation : tuple[float, float, float], optional
        The spawn orientation of the fly in the Euler angle format: (x, y, z),
        where x, y, z define the rotation around x, y and z in radian. By
        default (0.0, 0.0, pi/2), which leads to a position facing the
        positive direction of the x-axis.
    run_time : float, optional
        Time to run the simulation for, by default 5
    decision_interval : float, optional
        Time between each decision step, by default 0.05
    attractive_gain : int, optional
        Gain for attractive odor, by default -500
    aversive_gain : int, optional
        Gain for aversive odor, by default 80
    attractive_palps_antennae_weights : tuple[int, int], optional
        Weights for averaging attractive intensities from maxillary palps and
        antenna, by default (1, 9)
    aversive_palps_antennae_weights : tuple[int, int], optional
        Weights for averaging aversive intensities from maxillary palps and
        antenna, by default (0, 10)
    target_pos : tuple[float, float], optional
        The (x, y) position in the arena defining the target position, in mm.
        By default the position of the first odor source.
    distance_threshold : float, optional
        Distance threshold to stop the simulation, by default 2
    video_path : str, optional
        Path to save the video, by default None. If enable_rendering is False,
        this parameter is ignored.
    enable_rendering : bool, optional
        Whether to enable rendering, by default False.
    """
    # Define the arena
    arena = OdorArena(
        odor_source=odor_source,
        peak_odor_intensity=peak_odor_intensity,
        diffuse_func=lambda x: x**-2,
        marker_colors=marker_colors,
        marker_size=0.3,
    )

    if target_pos is None:
        target_pos = odor_source[0, :2]

    # Define the fly
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    fly = Fly(
        spawn_pos=spawn_pos,
        spawn_orientation=spawn_orientation,
        contact_sensor_placements=contact_sensor_placements,
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=False,
    )

    cam = Camera(
        fly=fly,
        camera_id="birdeye_cam",
        play_speed=0.5,
        window_size=(800, 608),
    )

    sim = HybridTurningController(
        fly=fly,
        cameras=[cam] if enable_rendering else [],
        arena=arena,
        timestep=1e-4,
    )

    # Run the simulation
    num_decision_steps = int(run_time / decision_interval)
    physics_steps_per_decision_step = int(decision_interval / sim.timestep)

    obs_hist = []
    odor_history = []
    obs, _ = sim.reset()
    for _ in trange(num_decision_steps):
        attractive_intensities = np.average(
            obs["odor_intensity"][0, :].reshape(2, 2),
            axis=0,
            weights=attractive_palps_antennae_weights,
        )
        aversive_intensities = np.average(
            obs["odor_intensity"][1, :].reshape(2, 2),
            axis=0,
            weights=aversive_palps_antennae_weights,
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

            if enable_rendering and video_path is not None:
                rendered_img = sim.render()
                if rendered_img is not None:
                    # record odor intensity too for video
                    odor_history.append(obs["odor_intensity"])

            obs_hist.append(obs)

        # Stop when the fly is near the attractive odor source
        if np.linalg.norm(obs["fly"][0, :2] - target_pos) < distance_threshold:
            break

    if enable_rendering and video_path is not None:
        cam.save_video(video_path)

    return obs_hist


if __name__ == "__main__":
    # Define the arena
    # Odor source: array of shape (num_odor_sources, 3) - xyz coords of odor sources
    odor_source = np.array([[24, 0, 1.5], [8, -4, 1.5], [16, 4, 1.5]])

    # Peak intensities: array of shape (num_odor_sources, odor_dimensions)
    # For each odor source, if the intensity is (x, 0) then the odor is in the 1st
    # dimension (in this case attractive). If it's (0, x) then it's in the 2nd dimension
    # (in this case aversive)
    peak_odor_intensity = np.array([[1, 0], [0, 1], [0, 1]])

    # Marker colors: array of shape (num_odor_sources, 4) - RGBA values for each marker,
    # normalized to [0, 1]
    marker_colors = [[255, 127, 14], [31, 119, 180], [31, 119, 180]]
    marker_colors = np.array([[*np.array(color) / 255, 1] for color in marker_colors])

    run_simulation(
        odor_source,
        peak_odor_intensity,
        marker_colors,
        spawn_pos=(0, 0, 0.2),
        spawn_orientation=(0, 0, np.pi / 2),
        run_time=5,
        decision_interval=0.05,
        attractive_gain=-500,
        aversive_gain=80,
        attractive_palps_antennae_weights=(1, 9),
        aversive_palps_antennae_weights=(0, 10),
        video_path="./outputs/odor_taxis.mp4",
        enable_rendering=True,
    )
