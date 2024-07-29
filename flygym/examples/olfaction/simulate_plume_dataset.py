# This notebook is partially based on the following script by  Felix Köhler:
# https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/phiflow/smoke_plume.py


import numpy as np
import h5py
from phi.torch import flow


@flow.math.jit_compile
def step(
    velocity_prev: flow.Grid,
    smoke_prev: flow.Grid,
    noise: np.ndarray,
    noise_magnitude: tuple[float, float] = (0.1, 2),
    dt: float = 1.0,
    inflow: flow.Grid = None,
) -> tuple[flow.Grid, flow.Grid]:
    """Simulate fluid dynamics by one time step.

    Parameters
    ----------
    velocity_prev : flow.Grid
        Velocity field at previous time step.
    smoke_prev : flow.Grid
        Smoke density at previous time step.
    noise : np.ndarray
        Brownian noise to be applied as external force.
    noise_magnitude : tuple[float, float], optional
        Magnitude of noise to be applied as external force in x and y
        directions, by default (0.1, 2)
    dt : float, optional
        Simulation time step, by default 1.0

    Returns
    -------
    tuple[flow.Grid, flow.Grid]
        Velocity field and smoke density at next time step.
    """
    smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt=dt) + inflow
    external_force = smoke_next * noise * noise_magnitude @ velocity_prev
    velocity_tentative = (
        flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt=dt)
        + external_force
    )
    velocity_next, pressure = flow.fluid.make_incompressible(velocity_tentative)
    return velocity_next, smoke_next


def converging_brownian_step(
    value_curr: np.ndarray,
    center: np.ndarray,
    gaussian_scale: float = 1.0,
    convergence: float = 0.5,
) -> np.ndarray:
    """Step to simulate Brownian noise with convergence towards a center.

    Parameters
    ----------
    value_curr : np.ndarray
        Current value of variables (i.e., noise) in Brownian motion.
    center : np.ndarray
        Center towards which the Brownian motion converges.
    gaussian_scale : float, optional
        Standard deviation of Gaussian noise to be added to the current
        value, by default 1.0
    convergence : float, optional
        Factor of convergence towards the center, by default 0.5.

    Returns
    -------
    np.ndarray
        Next value of variables (i.e., noise) in Brownian motion.
    """
    gaussian_center = (center - value_curr) * convergence
    value_diff = np.random.normal(
        loc=gaussian_center, scale=gaussian_scale, size=value_curr.shape
    )
    value_next = value_curr + value_diff
    return value_next


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from tqdm import trange
    from pathlib import Path

    np.random.seed(0)
    output_dir = Path("./outputs/plume_tracking/plume_dataset/")
    output_dir.mkdir(exist_ok=True, parents=True)
    simulation_time = 13000
    dt = 0.7
    arena_size = (120, 80)
    inflow_pos = (5, 40)
    inflow_radius = 1
    inflow_scaler = 0.2
    velocity_grid_size = 0.5
    smoke_grid_size = 0.25
    simulation_steps = int(simulation_time / dt)

    # Simulate Brownian noise
    curr_wind = np.zeros((2,))
    wind_hist = [curr_wind.copy()]
    for i in range(simulation_steps):
        curr_wind = converging_brownian_step(curr_wind, (0, 0), (0.2, 0.2), 1.0)
        wind_hist.append(curr_wind.copy())

    # Define simulation grids
    velocity = flow.StaggeredGrid(
        values=(1.0, 0.0),  # constant velocity field to the right
        extrapolation=flow.extrapolation.BOUNDARY,
        x=int(arena_size[0] / velocity_grid_size),
        y=int(arena_size[1] / velocity_grid_size),
        bounds=flow.Box(x=arena_size[0], y=arena_size[1]),
    )
    # choose extrapolation mode from
    # ('undefined', 'zeros', 'boundary', 'periodic', 'symmetric', 'reflect')
    smoke = flow.CenteredGrid(
        values=0.0,
        extrapolation=flow.extrapolation.BOUNDARY,
        x=int(arena_size[0] / smoke_grid_size),
        y=int(arena_size[1] / smoke_grid_size),
        bounds=flow.Box(x=arena_size[0], y=arena_size[1]),
    )
    inflow = inflow_scaler * flow.field.resample(
        flow.Sphere(x=inflow_pos[0], y=inflow_pos[1], radius=inflow_radius),
        to=smoke,
        soft=True,
    )

    # Run fluid dynamics simulation
    smoke_hist = []
    for i in trange(simulation_steps):
        velocity, smoke = step(velocity, smoke, wind_hist[i], dt=dt, inflow=inflow)
        smoke_vals = smoke.values.numpy("y,x")
        smoke_hist.append(smoke_vals)
        plt.imshow(
            smoke_vals,
            cmap="gray_r",
            origin="lower",
            vmin=0,
            vmax=0.7,
            extent=[0, arena_size[0], 0, arena_size[1]],
        )
        plt.gca().invert_yaxis()
        plt.draw()
        plt.pause(0.01)
        plt.clf()

    # Save wind history
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), tight_layout=True)
    ax.plot(wind_hist, label=("x", "y"))
    ax.legend()
    ax.set_xlabel("Time [AU]")
    ax.set_ylabel("Wind [AU]")
    ax.set_title('Brownian "wind"')
    fig.savefig(output_dir / "brownian_wind.png")

    # Save plume simulation as video
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), tight_layout=True)
    img = ax.imshow(
        smoke_hist[0],
        cmap="gray_r",
        origin="lower",
        vmin=0,
        vmax=0.7,
        extent=[0, arena_size[0], 0, arena_size[1]],
    )
    ax.invert_yaxis()

    def update(i):
        """Helper function to update the animation."""
        img.set_data(smoke_hist[i])

    animation = FuncAnimation(fig, update, frames=len(smoke_hist), repeat=False)
    animation.save(output_dir / "plume.mp4", fps=100, dpi=300, bitrate=500)

    # Save plume simulation data
    with h5py.File(output_dir / "plume.hdf5", "w") as f:
        f.create_dataset(
            "plume", data=np.stack(smoke_hist).astype(np.float16), compression="gzip"
        )
        f["inflow_pos"] = inflow_pos
        f["inflow_radius"] = [inflow_radius]
        f["inflow_scaler"] = [inflow_scaler]

    # Save short plume simulation data for testing
    with h5py.File(output_dir / "plume_short.hdf5", "w") as f:
        f.create_dataset(
            "plume",
            data=np.stack(smoke_hist[5000:5600:10]).astype(np.float16),
            compression="gzip",
        )
        f["inflow_pos"] = inflow_pos
        f["inflow_radius"] = [inflow_radius]
        f["inflow_scaler"] = [inflow_scaler]
