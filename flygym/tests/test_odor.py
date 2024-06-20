import numpy as np

from flygym.arena import OdorArena
from flygym import Fly, SingleFlySimulation
from flygym.examples.locomotion import HybridTurningController


def test_odor_dimensions():
    num_sources = 5
    num_dims = 4
    odor_source = np.zeros((num_sources, 3))
    peak_odor_intensity = np.ones((num_sources, num_dims))

    # Initialize simulation
    num_steps = 100
    arena = OdorArena(odor_source=odor_source, peak_odor_intensity=peak_odor_intensity)

    fly = Fly(enable_olfaction=True)
    sim = SingleFlySimulation(fly=fly, arena=arena, cameras=[])

    # Run simulation
    obs_list = []
    for i in range(num_steps):
        joint_pos = np.zeros(len(fly.actuated_joints))
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = sim.step(action)
        obs_list.append(obs)
    sim.close()

    # Check dimensionality
    odor = np.array([obs["odor_intensity"] for obs in obs_list])
    assert odor.shape == (num_steps, num_dims, 4)


def test_odor_intensity():
    odor_source = np.array([[-3, -3, 1.5], [-3, 3, 1.5], [3, -3, 1.5], [3, 3, 1.5]])
    peak_odor_intensity = np.eye(4)

    arena = OdorArena(
        odor_source=odor_source,
        peak_odor_intensity=peak_odor_intensity,
        diffuse_func=lambda x: x**-2,
    )

    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]

    fly = Fly(
        spawn_pos=(0, 0, 0.2),
        contact_sensor_placements=contact_sensor_placements,
        enable_olfaction=True,
        enable_adhesion=True,
        draw_adhesion=False,
    )

    sim = HybridTurningController(
        fly=fly,
        arena=arena,
        cameras=[],
        timestep=1e-4,
    )

    obs, _ = sim.reset()

    # (n_dims, |{antenna, palp}|, |{L, R}|)
    odor_intensity = obs["odor_intensity"].reshape((-1, 2, 2))

    # (n_dims, |{L, R}|)
    weighted_intensity = np.average(odor_intensity, axis=1, weights=[9, 1])

    # (n_dims,)
    asym = np.diff(weighted_intensity, axis=-1)[..., 0]

    # Check that the odor intensity asymmetries have the correct signs
    assert all(np.sign(asym) == (1, -1, 1, -1))

    control_signal = np.ones((2,))
    odor_intensity_hist = []

    for _ in range(10000):
        obs = sim.step(control_signal)[0]
        odor_intensity_hist.append(obs["odor_intensity"])
        if obs["fly"][0, 0] > 2:
            break

    odor_intensity_hist = np.array(odor_intensity_hist)
    # Check that the odor intensity from behind decreases as the fly moves forward
    assert (odor_intensity_hist[-1, :2] < odor_intensity_hist[0, :2]).all()


def test_odor_sum():
    odor_source = np.tile([5, 0, 1.5], (3, 1))
    peak_odor_intensity = np.array([[2, 0], [0, 1], [0, 1]])
    arena = OdorArena(
        odor_source=odor_source,
        peak_odor_intensity=peak_odor_intensity,
        diffuse_func=lambda x: x**-2,
    )
    fly = Fly(enable_olfaction=True)
    sim = SingleFlySimulation(fly=fly, arena=arena, cameras=[])
    dim1_intensity, dim2_intensity = sim.reset()[0]["odor_intensity"]
    np.testing.assert_allclose(dim1_intensity, dim2_intensity)
