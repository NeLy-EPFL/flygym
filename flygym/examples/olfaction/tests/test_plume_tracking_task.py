import numpy as np
import pytest
from flygym import Fly, Camera, is_rendering_skipped
from flygym.examples.olfaction import OdorPlumeArena, PlumeNavigationTask
from flygym.util import get_data_path


@pytest.mark.skipif(is_rendering_skipped, reason="env['SKIP_RENDERING'] == 'true'")
def test_plume_tracking_task():
    plume_data_path = get_data_path(
        "flygym", "data/test_data/plume_tracking/plume_short.hdf5"
    )
    main_camera_name = "birdeye_cam"
    arena = OdorPlumeArena(
        plume_data_path=plume_data_path,
        main_camera_name=main_camera_name,
        plume_simulation_fps=20,
        intensity_scale_factor=1.0,
    )
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        enable_olfaction=True,
        enable_vision=False,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(40, 80, 0.25),
        spawn_orientation=(0, 0, -np.pi),
    )
    cam_params = {
        "mode": "fixed",
        "pos": (
            0.50 * arena.arena_size[0],
            0.15 * arena.arena_size[1],
            1.00 * arena.arena_size[1],
        ),
        "euler": (np.deg2rad(15), 0, 0),
        "fovy": 60,
    }
    cam = Camera(
        attachment_point=arena.root_element.worldbody,
        camera_name=main_camera_name,
        timestamp_text=False,
        camera_parameters=cam_params,
    )

    sim = PlumeNavigationTask(
        fly=fly,
        arena=arena,
        cameras=[cam],
    )
    sim_time = 0.1
    rendered_images = []
    obs_hist = []
    info_hist = []
    for i in range(int(sim_time / sim.timestep)):
        obs, _, _, _, info = sim.step(np.array([1, 1]))
        obs_hist.append(obs)
        info_hist.append(info)
        img = sim.render()[0]
        if img is not None:
            rendered_images.append(img)

    # Check format of observation and info
    expected_obs_keys = {
        "joints",
        "fly",
        "contact_forces",
        "end_effectors",
        "fly_orientation",
        "odor_intensity",
        "cardinal_vectors",
    }
    expected_info_keys = {"net_corrections", "joints", "adhesion", "flip"}
    for obs in obs_hist:
        assert set(obs.keys()) == expected_obs_keys
    for info in info_hist:
        assert set(info.keys()) == expected_info_keys

    odor_hist = np.array([obs["odor_intensity"] for obs in obs_hist])
    assert odor_hist.shape == (int(sim_time / sim.timestep), 1, 4)
    assert np.unique(odor_hist[:, 0, 0]).size >= 2

    # Check position mapping for rendering
    pos_display_sample, pos_physical_sample = arena.get_position_mapping(sim)
    assert pos_display_sample.shape == (160, 240, 2)
    assert pos_physical_sample.shape == (160, 240, 2)
    # plt.imshow(rendered_images[0])
    # plt.show()  # check if this makes sense. if so, the following is golden data
    # print(pos_display_sample.sum(), pos_physical_sample.sum())
    # print(sim.grid_idx_all.sum(), sim.grid_idx_all[90, 100])
    assert pos_display_sample.sum() == pytest.approx(20562741.464867506, rel=1e-6)
    assert pos_physical_sample.sum() == pytest.approx(7680000.0, rel=1e-6)
    assert sim.grid_idx_all.sum() == 84662700
    assert np.all(sim.grid_idx_all[90, 100] == [46, 270])
    assert not np.all(rendered_images[0] == rendered_images[-1])
