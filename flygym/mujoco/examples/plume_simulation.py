# This notebook is partially based on the following script by  Felix Köhler:
# https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/phiflow/smoke_plume.py


import numpy as np
from phi.torch import flow


@flow.math.jit_compile
def step(velocity_prev, smoke_prev, noise, noise_magnitude=(0.1, 2), dt=1.0):
    smoke_next = flow.advect.mac_cormack(smoke_prev, velocity_prev, dt=dt) + inflow
    external_force = smoke_next * noise * noise_magnitude @ velocity_prev
    velocity_tentative = (
        flow.advect.semi_lagrangian(velocity_prev, velocity_prev, dt=dt)
        + external_force
    )
    velocity_next, pressure = flow.fluid.make_incompressible(velocity_tentative)
    return velocity_next, smoke_next


def converging_brownian_step(value_curr, center, gaussian_scale=1, convergence=0.5):
    gaussian_center = (center - value_curr) * convergence
    value_diff = np.random.normal(
        loc=gaussian_center, scale=gaussian_scale, size=value_curr.shape
    )
    value_next = value_curr + value_diff
    return value_next


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from tqdm import tqdm, trange
    from pathlib import Path

    np.random.seed(0)
    output_dir = Path("./outputs/complex_plume")
    output_dir.mkdir(exist_ok=True, parents=True)
    simulation_time = 4900
    dt = 0.7
    arena_size = (120, 80)
    # arena_size = (120, 120)
    inflow_pos = (5, 40)
    inflow_radius = 1
    inflow_scaler = 0.2
    velocity_grid_size = 0.5
    smoke_grid_size = 0.25
    simulation_steps = int(simulation_time / dt)

    # Simulate Brownian noise (wind)
    curr_wind = np.zeros((2,))
    wind_hist = [curr_wind.copy()]
    for i in range(simulation_steps):
        curr_wind = converging_brownian_step(curr_wind, (0, 0), (0.2, 0.2), 0.4)
        wind_hist.append(curr_wind.copy())

    # Define simulation grids
    velocity = flow.StaggeredGrid(
        values=(1.0, 0.0),
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
        velocity, smoke = step(velocity, smoke, wind_hist[i], dt=dt)
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
        img.set_data(smoke_hist[i])

    animation = FuncAnimation(fig, update, frames=len(smoke_hist), repeat=False)
    animation.save(output_dir / "plume.mp4", fps=100, dpi=300, bitrate=500)

    # Save plume simulation data
    np.savez_compressed(
        output_dir / "plume.npy",
        plume=np.stack(smoke_hist),
        inflow_pos=inflow_pos,
        inflow_radius=inflow_radius,
        inflow_scaler=inflow_scaler,
    )
