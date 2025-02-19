import numpy as np
import pytest
from flygym.examples.olfaction.simulate_plume_dataset import (
    get_simulation_parameters,
    generate_simulation_inputs,
    run_simulation,
)


def test_plume_simulation():
    # run 2 simulation steps of the plume generation and check that they match
    np.random.seed(0)
    simulation_time = 2

    (
        dt,
        arena_size,
        inflow_pos,
        inflow_radius,
        inflow_scaler,
        velocity_grid_size,
        smoke_grid_size,
        simulation_steps,
    ) = get_simulation_parameters(simulation_time)

    wind_hist, velocity, smoke, inflow = generate_simulation_inputs(
        simulation_steps,
        arena_size,
        inflow_pos,
        inflow_radius,
        inflow_scaler,
        velocity_grid_size,
        smoke_grid_size,
    )
    smoke_hist = run_simulation(wind_hist, velocity, smoke, inflow, dt, arena_size)

    assert wind_hist[0] == pytest.approx(np.array([0, 0]))
    assert wind_hist[1] == pytest.approx(np.array([0.35281047, 0.08003144]))

    assert smoke_hist[0].sum() == pytest.approx(10.15325)
    assert smoke_hist[0].std() == pytest.approx(0.003564552)
    assert smoke_hist[1].sum() == pytest.approx(20.316465)
    assert smoke_hist[1].std() == pytest.approx(0.0063055283)
