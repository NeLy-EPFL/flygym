import pytest
import pickle
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from flygym import Fly
from flygym.examples.path_integration.util import extract_variables, load_trial_data
from flygym.examples.path_integration.controller import PathIntegrationController
from flygym.examples.path_integration.arena import PathIntegrationArenaFlat
from flygym.preprogrammed import get_cpg_biases


def run_simulation_debug(
    dn_drive: np.ndarray = np.array([1.0, 1.0]),
    seed: int = 0,
    running_time: float = 0.1,
    terrain_type: str = "flat",
    gait: str = "tripod",
    output_dir: Path = None,
):
    contact_sensor_placements = [
        f"{leg}{segment}"
        for leg in ["LF", "LM", "LH", "RF", "RM", "RH"]
        for segment in ["Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5"]
    ]
    fly = Fly(
        enable_adhesion=True,
        draw_adhesion=True,
        contact_sensor_placements=contact_sensor_placements,
        spawn_pos=(0, 0, 0.25),
    )
    arena = PathIntegrationArenaFlat()
    sim = PathIntegrationController(
        phase_biases=get_cpg_biases(gait),
        fly=fly,
        arena=arena,
        cameras=[],
        timestep=1e-4,
        correction_rates={"retraction": (0, 0), "stumbling": (0, 0)},
    )

    obs, info = sim.reset(0)
    obs_hist, info_hist, action_hist = [], [], []
    _real_heading_buffer = []
    for i in range(int(running_time / sim.timestep)):
        obs, reward, terminated, truncated, info = sim.step(dn_drive)

        # Get real heading
        orientation_x, orientation_y = obs["fly_orientation"][:2]
        real_heading = np.arctan2(orientation_y, orientation_x)
        _real_heading_buffer.append(real_heading)

        obs_hist.append(obs)
        info_hist.append(info)
        action_hist.append(dn_drive)

    # Save data if output_dir is provided
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "sim_data.pkl", "wb") as f:
            data = {
                "obs_hist": obs_hist,
                "info_hist": info_hist,
                "action_hist": action_hist,
            }
            pickle.dump(data, f)


def test_left_right_sum():
    running_time = 0.2
    dt = 1e-4

    with TemporaryDirectory() as tmpdir:
        tempdir = Path(tmpdir)
        run_simulation_debug(
            dn_drive=np.array([1.0, 1.0]),
            seed=0,
            running_time=running_time,
            output_dir=tempdir,
        )

        trial_data = load_trial_data(tempdir)

    extracted_vars_all = {}
    for tag, time_scale in {"fast": 0.01, "slow": 0.04}.items():
        extracted_vars = extract_variables(
            trial_data,
            time_scale=time_scale,
            contact_force_thr=(0.5, 0.1, 0.3),
            legs="FM",
            dt=dt,
        )
        extracted_vars_all[tag] = extracted_vars

        nsteps = int((running_time - time_scale) / dt)
        assert extracted_vars["stride_total_diff_lrsum"].shape == (nsteps, 2)
        assert extracted_vars["stride_total_diff_lrdiff"].shape == (nsteps, 2)
        assert extracted_vars["sum_dn_drive"].shape == (nsteps, 1)
        assert extracted_vars["diff_dn_drive"].shape == (nsteps, 1)
        assert extracted_vars["heading_diff"].shape == (nsteps,)
        assert extracted_vars["forward_disp_total_diff"].shape == (nsteps,)
        assert extracted_vars["heading"].shape == (nsteps,)

    backstroke_mean_fast = np.mean(
        extracted_vars_all["fast"]["stride_total_diff_lrsum"][500:, :]
    )
    backstroke_std_fast = np.std(
        extracted_vars_all["fast"]["stride_total_diff_lrsum"][500:, :]
    )
    backstroke_mean_slow = np.mean(
        extracted_vars_all["slow"]["stride_total_diff_lrsum"][500:, :]
    )
    backstroke_std_slow = np.std(
        extracted_vars_all["slow"]["stride_total_diff_lrsum"][500:, :]
    )

    # check if they are both negative enough
    print(
        backstroke_mean_fast,
        backstroke_std_fast,
        backstroke_mean_slow,
        backstroke_std_slow,
    )
    assert backstroke_mean_fast < -backstroke_std_fast * 0.5  # allow more noise if fast
    assert backstroke_mean_slow < -backstroke_std_slow

    # check if fast version is about 4x smaller in backstroke sum than slow version
    print(backstroke_mean_slow / backstroke_mean_fast)
    assert backstroke_mean_slow / backstroke_mean_fast == pytest.approx(4, abs=1)


def test_left_right_sum():
    running_time = 0.2
    dt = 1e-4

    with TemporaryDirectory() as tmpdir:
        tempdir = Path(tmpdir)
        run_simulation_debug(
            dn_drive=np.array([0.2, 1.2]),
            seed=0,
            running_time=running_time,
            output_dir=tempdir,
        )

        trial_data = load_trial_data(tempdir)

    extracted_vars_all = {}
    for tag, time_scale in {"fast": 0.01, "slow": 0.04}.items():
        extracted_vars = extract_variables(
            trial_data,
            time_scale=time_scale,
            contact_force_thr=(0.5, 0.1, 0.3),
            legs="FM",
            dt=dt,
        )
        extracted_vars_all[tag] = extracted_vars

    asymm_mean_fast = np.mean(
        extracted_vars_all["fast"]["stride_total_diff_lrdiff"][500:, :]
    )
    asymm_std_fast = np.std(
        extracted_vars_all["fast"]["stride_total_diff_lrdiff"][500:, :]
    )
    asymm_mean_slow = np.mean(
        extracted_vars_all["slow"]["stride_total_diff_lrdiff"][500:, :]
    )
    asymm_std_slow = np.std(
        extracted_vars_all["slow"]["stride_total_diff_lrdiff"][500:, :]
    )

    # check if they are both negative enough
    print(asymm_mean_fast, asymm_std_fast, asymm_mean_slow, asymm_std_slow)
    assert asymm_mean_fast > 0  # allow more noise if fast
    assert asymm_mean_slow > asymm_std_slow * 0.5

    # check if fast version is about 4x smaller in backstroke sum than slow version
    print(asymm_mean_slow / asymm_mean_fast)
    assert asymm_mean_slow / asymm_mean_fast == pytest.approx(4, abs=1.5)
