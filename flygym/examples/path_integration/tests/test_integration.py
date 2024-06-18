import pytest
import pickle
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from sklearn.metrics import r2_score
from flygym import Fly
from flygym.examples.path_integration.util import extract_variables, load_trial_data
from flygym.examples.path_integration.exploration import run_simulation
from flygym.examples.path_integration.controller import PathIntegrationController
from flygym.examples.path_integration.arena import PathIntegrationArenaFlat
from flygym.examples.path_integration.model import LinearModel, path_integrate
from flygym.preprogrammed import get_cpg_biases


def run_simulation_debug(
    dn_drive: np.ndarray = np.array([1.0, 1.0]),
    seed: int = 0,
    running_time: float = 0.1,
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
        seed=seed,
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
            contact_force_thr=(0.5, 1, 3),
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
    # print(
    #     backstroke_mean_fast,
    #     backstroke_std_fast,
    #     backstroke_mean_slow,
    #     backstroke_std_slow,
    # )
    assert backstroke_mean_fast < -backstroke_std_fast * 0.5  # allow more noise if fast
    assert backstroke_mean_slow < -backstroke_std_slow

    # check if fast version is about 4x smaller in backstroke sum than slow version
    # print(backstroke_mean_slow / backstroke_mean_fast)
    assert backstroke_mean_slow / backstroke_mean_fast == pytest.approx(4, abs=1)


def test_left_right_diff():
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
            contact_force_thr=(0.5, 1, 3),
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
    # print(asymm_mean_fast, asymm_std_fast, asymm_mean_slow, asymm_std_slow)
    assert asymm_mean_fast > 0  # allow more noise if fast
    assert asymm_mean_slow > asymm_std_slow * 0.5

    # check if fast version is about 4x smaller in backstroke sum than slow version
    # print(asymm_mean_slow / asymm_mean_fast)
    assert asymm_mean_slow / asymm_mean_fast == pytest.approx(4, abs=1.5)


def test_path_integration():
    # Reference ensemble model:
    # k_fore_prop2heading    0.250722
    # k_mid_prop2heading     0.176886
    # k_hind_prop2heading    0.032141
    # b_prop2heading         0.005333
    # r2_prop2heading        0.969397
    # k_fore_prop2disp      -0.440427
    # k_mid_prop2disp       -0.382758
    # k_hind_prop2disp      -0.004039
    # b_prop2disp            0.970207
    # r2_prop2disp           0.978988
    # k_fore_dn2heading     -1.557306
    # k_mid_dn2heading      -1.557306
    # k_hind_dn2heading     -1.557306
    # b_dn2heading          -0.017561
    # r2_dn2heading          0.944227
    # k_fore_dn2disp         4.543855
    # k_mid_dn2disp          4.543855
    # k_hind_dn2disp         4.543855
    # b_dn2disp              0.200895
    # r2_dn2disp             0.919772
    # Name: (tripod, 0.64, 0.5, 1, 3, FMH), dtype: float64

    time_scale = 0.64
    running_time = 1.0
    dt = 1e-4
    heading_model = LinearModel(
        coefs_all=np.array([0.250722, 0.176886, 0.032141]),
        intercept=0.005333,
        legs="FMH",
    )
    displacement_model = LinearModel(
        coefs_all=np.array([-0.440427, -0.382758, -0.004039]),
        intercept=0.970207,
        legs="FMH",
    )

    with TemporaryDirectory() as tmpdir:
        tempdir = Path(tmpdir)
        run_simulation(
            seed=0,
            running_time=running_time,
            terrain_type="flat",
            gait="tripod",
            live_display=False,
            enable_rendering=False,
            pbar=False,
            output_dir=tempdir,
        )

        trial_data = load_trial_data(tempdir)

    path_int_res = path_integrate(
        trial_data,
        heading_model,
        displacement_model,
        time_scale,
        contact_force_thr=(0.5, 1, 3),
        legs="FMH",
        dt=dt,
    )

    expected_keys = {
        "heading_pred",
        "heading_actual",
        "pos_pred",
        "pos_actual",
        "heading_diff_pred",
        "heading_diff_actual",
        "displacement_diff_pred",
        "displacement_diff_actual",
    }
    expected_nsteps = int(running_time / dt)
    start_idx = int(time_scale / dt)  # path int. starts after this many steps

    # Check keys in the output dict
    assert set(path_int_res.keys()) == expected_keys

    # Check shapes of values in the output dict
    assert path_int_res["heading_pred"].shape == (expected_nsteps,)
    assert path_int_res["heading_actual"].shape == (expected_nsteps,)
    assert path_int_res["pos_pred"].shape == (expected_nsteps, 2)
    assert path_int_res["pos_actual"].shape == (expected_nsteps, 2)
    assert path_int_res["heading_diff_pred"].shape == (expected_nsteps,)
    assert path_int_res["heading_diff_actual"].shape == (expected_nsteps,)
    assert path_int_res["displacement_diff_pred"].shape == (expected_nsteps,)
    assert path_int_res["displacement_diff_actual"].shape == (expected_nsteps,)

    # For the actual values, the whole array should be finite
    # For the predicted values, the first few steps should be NaN (before path
    # integration doesn't start until there's enough data to make the first
    # delta heading/delta displacement prediction. This period is the same as the
    # integration time scale). After that, the values should be finite.
    assert np.isnan(path_int_res["heading_pred"][:start_idx]).all()
    assert np.isfinite(path_int_res["heading_pred"][start_idx:]).all()
    assert np.isfinite(path_int_res["heading_actual"]).all()
    assert np.isnan(path_int_res["pos_pred"][:start_idx]).all()
    assert np.isfinite(path_int_res["pos_pred"][start_idx:]).all()
    assert np.isfinite(path_int_res["pos_actual"]).all()

    # Though the position and heading in the actual simulation are defined from the
    # 0th step, the *changes* in them are *undefined* until the time scale is reached
    # even in the ground truth.
    assert np.isnan(path_int_res["heading_diff_pred"][:start_idx]).all()
    assert np.isfinite(path_int_res["heading_diff_pred"][start_idx:]).all()
    assert np.isnan(path_int_res["heading_diff_actual"][:start_idx]).all()
    assert np.isfinite(path_int_res["heading_diff_actual"][start_idx:]).all()
    assert np.isnan(path_int_res["displacement_diff_pred"][:start_idx]).all()
    assert np.isfinite(path_int_res["displacement_diff_pred"][start_idx:]).all()
    assert np.isnan(path_int_res["displacement_diff_actual"][:start_idx]).all()
    assert np.isfinite(path_int_res["displacement_diff_actual"][start_idx:]).all()

    # Check accuracy
    delta_heading_r2 = r2_score(
        np.unwrap(path_int_res["heading_diff_actual"][start_idx:]),
        np.unwrap(path_int_res["heading_diff_pred"][start_idx:]),
    )
    delta_disp_r2 = r2_score(
        path_int_res["displacement_diff_actual"][start_idx:],
        path_int_res["displacement_diff_pred"][start_idx:],
    )
    # import matplotlib.pyplot as plt
    # plt.plot(path_int_res["heading_diff_actual"][start_idx:])
    # plt.plot(path_int_res["heading_diff_pred"][start_idx:])
    # plt.show()
    # plt.plot(path_int_res["displacement_diff_actual"][start_idx:])
    # plt.plot(path_int_res["displacement_diff_pred"][start_idx:])
    # plt.show()
    # print(delta_heading_r2, delta_disp_r2)

    # r2 might be quite low because we are only looking at the beginning
    assert 0.6 < delta_heading_r2 < 1
    assert 0.5 < delta_disp_r2 < 1
