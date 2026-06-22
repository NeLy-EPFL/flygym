"""Smoke test: replay experimentally recorded fly walking on the GPU (MuJoCo Warp).

This is a script version of the core of
``tutorials/3_gpu_accelerated_simulation.ipynb``. It is the GPU-accelerated
counterpart of ``run_cpu_smoketest.py``: it replays the same Spotlight kinematic
recording, but over many parallel worlds at once using ``flygym.warp``. Each world
is fed a different 0.1 s slice of the recording (wrapping around when the slices
are exhausted).

The inner loop is the *fully GPU-resident*, CUDA-graph-captured version from the
tutorial: control inputs are written by a Warp kernel from a pre-loaded angle
table, the physics step runs, and (when saving data) joint angles are recorded
into a GPU buffer -- all without per-step CPU<->GPU synchronization. The whole
loop body is captured once with ``wp.ScopedCapture`` and replayed with
``wp.capture_launch``.

It is intended as a human-friendly end-to-end test and a profiling target. Without
arguments it is a pure physics-throughput benchmark; passing ``--save-data DIR``
additionally renders the simulation (GPU batch rendering) and writes the
observation history, plots, and rendered video to ``DIR``.

Timing only starts *after* the simulation has been JIT-compiled: building the
capture graph compiles the physics and control kernels, and a single untimed
warm-up replay (plus one warm-up render, when rendering) forces any remaining
kernels -- notably the batch-render megakernel -- to compile before the clock
starts.

Example:
    uv run python scripts/dev/run_gpu_smoketest.py --save-data outputs/gpu_smoketest
"""

import sys
import argparse
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: no display needed
import matplotlib.pyplot as plt
import warp as wp

from flygym.warp import GPUSimulation
from flygym.warp.utils import check_gpu
from flygym.compose import ActuatorType
from flygym.anatomy import JointDOF, RotationAxis, BodySegment
from flygym_demo.benchmark import (
    make_model,
    ReplayTargetData,
    update_target_angles_kernel,
    increment_counter_kernel,
)


@wp.kernel
def record_joint_angles_kernel(
    qpos: wp.array2d(dtype=wp.float32),  # type: ignore  # (n_worlds, nq)
    qpos_adrs: wp.array(dtype=wp.int32),  # type: ignore  # (n_jointdofs,)
    step_counter: wp.array(dtype=wp.int32),  # type: ignore
    recorded: wp.array3d(dtype=wp.float32),  # type: ignore  # (n_steps, n_worlds, n_dofs)
):
    """Gather this step's joint angles into a pre-allocated, GPU-resident buffer.

    Mirrors ``GPUSimulation.get_joint_angles`` (which gathers the joint-DOF columns
    out of ``qpos``), but writes directly into ``recorded[step]`` so the whole thing
    stays inside the captured graph with no allocation or CPU synchronization.
    """
    world_id, dof_id = wp.tid()
    step = step_counter[0]
    recorded[step, world_id, dof_id] = qpos[world_id, qpos_adrs[dof_id]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to write the observation history, plots, and rendered "
        "video. Errors out if it already exists. If omitted, nothing is saved and "
        "rendering is skipped (pure physics-throughput benchmark).",
    )
    parser.add_argument(
        "--n-worlds",
        type=int,
        default=1000,
        help="Number of parallel worlds to simulate (default: 1000).",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=1000,
        help="Number of steps to simulate per world (default: 1000 = 0.1 s).",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1e-4,
        help="Simulation timestep in seconds (default: 1e-4).",
    )
    parser.add_argument(
        "--render-worlds",
        type=int,
        default=9,
        help="Number of worlds to render and save to video (default: 9). Only used "
        "when --save-data is given.",
    )
    return parser.parse_args()


def plot_joint_angle_tracking(
    snippet, fly, actuator_type, timegrid, target_angles, simulated_angles, leg="lf"
):
    """Compare target vs. achieved joint angles for one leg of one world."""
    fig, axes = plt.subplots(7, 1, figsize=(9, 6.5), tight_layout=True, sharex=True)
    for dof_idx, (parent_link, child_link, axis) in enumerate(snippet.dofs_per_leg):
        ax = axes[dof_idx]
        parent_name = "c_thorax" if parent_link == "thorax" else f"{leg}_{parent_link}"
        child_name = f"{leg}_{child_link}"
        target_dof = JointDOF(
            BodySegment(parent_name), BodySegment(child_name), RotationAxis(axis)
        )

        actuated_idx = fly.get_actuated_jointdofs_order(actuator_type).index(target_dof)
        ts_target = np.rad2deg(target_angles[:, actuated_idx])
        ax.plot(timegrid, ts_target, label="Target", linestyle=":", color="C0")

        dof_idx_all = fly.get_jointdofs_order().index(target_dof)
        ts_sim = np.rad2deg(simulated_angles[:, dof_idx_all])
        ax.plot(timegrid, ts_sim, label="Simulated", color="C1")

        ax.set_ylabel("Angle\n(°)")
        ax.set_title(f"{parent_link}-{child_link} {axis}", fontsize="medium")
        if dof_idx == 6:
            ax.set_xlabel("Time (s)")
        if dof_idx == 0:
            ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
    fig.suptitle(f"{leg.upper()} leg joint angles (world 0): target vs. simulated")
    return fig


def main() -> None:
    args = parse_args()

    check_gpu()

    data_dir: Path | None = args.save_data
    if data_dir is not None:
        if data_dir.exists():
            sys.exit(f"Error: output directory already exists: {data_dir}")
        data_dir.mkdir(parents=True)

    n_worlds = args.n_worlds
    sim_steps = args.sim_steps
    timestep = args.timestep
    actuator_type = ActuatorType.POSITION
    # Rendering (and hence video output) only happens when we are saving data.
    render_enabled = data_dir is not None
    record = data_dir is not None
    n_render = min(args.render_worlds, n_worlds)

    fly, world, cam = make_model()
    fly_name = fly.name

    # Build per-world target angle slices (world 0 -> first slice, world 1 -> next, ...).
    replay_data = ReplayTargetData(
        timestep, fly.get_actuated_jointdofs_order(actuator_type)
    )
    target_angles_all_worlds = replay_data.make_target_angles_all_worlds(
        n_worlds, sim_steps
    )
    n_dofs = target_angles_all_worlds.shape[-1]

    sim = GPUSimulation(world, n_worlds)
    assert sim.mj_model.opt.timestep == timestep

    renderer = None
    if render_enabled:
        renderer = sim.set_renderer(
            cam,
            playback_speed=0.2,
            output_fps=25,
            use_gpu_batch_rendering=True,
            worlds=list(range(n_render)),
        )

    # Reset to the neutral keyframe and settle. This must happen *before* the graph
    # capture, since `GPUSimulation.reset` reallocates `mjw_data` (which the captured
    # graph references).
    sim.reset()
    sim.set_leg_adhesion_states(fly_name, np.ones((n_worlds, 6), dtype=np.float32))
    sim.warmup()

    # --- Set up GPU-resident buffers for the captured loop ---
    target_angles_gpu = wp.array(target_angles_all_worlds)
    curr_target_angles_gpu = wp.zeros((n_worlds, n_dofs), dtype=wp.float32)
    step_counter = wp.array([0], dtype=wp.int32)

    n_jointdofs = len(fly.get_jointdofs_order())
    recorded_angles_gpu = None
    qpos_adrs = None
    if record:
        recorded_angles_gpu = wp.zeros(
            (sim_steps, n_worlds, n_jointdofs), dtype=wp.float32
        )
        # Internal: column indices of the fly's joint DOFs within the full qpos array,
        # the same mapping `GPUSimulation.get_joint_angles` uses.
        qpos_adrs = sim._wp_intern_qposadrs_by_fly[fly_name]

    # --- Capture the whole GPU-resident loop body once (this triggers JIT) ---
    with wp.ScopedCapture() as advance_sim_capture:
        wp.launch(
            update_target_angles_kernel,
            dim=(n_worlds, n_dofs),
            inputs=[target_angles_gpu, step_counter],
            outputs=[curr_target_angles_gpu],
        )
        sim.set_actuator_inputs(fly_name, actuator_type, curr_target_angles_gpu)
        sim.step()
        if record:
            wp.launch(
                record_joint_angles_kernel,
                dim=(n_worlds, n_jointdofs),
                inputs=[sim.mjw_data.qpos, qpos_adrs, step_counter],
                outputs=[recorded_angles_gpu],
            )
        wp.launch(increment_counter_kernel, dim=1, outputs=[step_counter])

    # --- Untimed warm-up: force any remaining JIT (e.g. the batch-render megakernel),
    # then reset the step counter and renderer so the timed run starts from step 0. ---
    print(f"Warming up (JIT compilation) {n_worlds} worlds...")
    wp.capture_launch(advance_sim_capture.graph)
    if render_enabled:
        sim.render_as_needed()
    wp.synchronize()

    step_counter.zero_()
    if render_enabled:
        renderer.reset()  # discard warm-up frame and reset render-time tracking

    # --- Timed run ---
    print(f"Simulating {sim_steps} steps across {n_worlds} worlds...")
    wp.synchronize()
    start_time = perf_counter_ns()
    for _ in range(sim_steps):
        wp.capture_launch(advance_sim_capture.graph)
        if render_enabled:
            sim.render_as_needed()
    wp.synchronize()
    end_time = perf_counter_ns()

    walltime_s = (end_time - start_time) / 1e9
    overall_throughput = n_worlds * sim_steps / walltime_s  # steps per sec
    realtime_factor = overall_throughput * timestep
    render_note = "with rendering" if render_enabled else "no rendering"
    print(
        f"Simulated {sim_steps} steps * {n_worlds} worlds in {walltime_s:.2f}s "
        f"({render_note})\n"
        f"Overall throughput: {overall_throughput:.2f} steps/s "
        f"({realtime_factor:.2f}x realtime)"
    )

    if data_dir is None:
        return

    print(f"Saving observation history, plots, and video to {data_dir}...")
    simulated_joint_angles = (
        recorded_angles_gpu.numpy()
    )  # (sim_steps, n_worlds, n_dofs)
    timegrid = np.arange(sim_steps) * timestep
    np.savez_compressed(
        data_dir / "observations.npz",
        timegrid=timegrid,
        target_angles_all_worlds=target_angles_all_worlds,
        simulated_joint_angles=simulated_joint_angles,
        jointdofs_order=np.array(
            [str(d) for d in fly.get_jointdofs_order()], dtype=object
        ),
        actuated_jointdofs_order=np.array(
            [str(d) for d in fly.get_actuated_jointdofs_order(actuator_type)],
            dtype=object,
        ),
    )

    fig = plot_joint_angle_tracking(
        replay_data.snippet,
        fly,
        actuator_type,
        timegrid,
        target_angles_all_worlds[0],
        simulated_joint_angles[:, 0, :],
    )
    fig.savefig(data_dir / "joint_angle_tracking_world0.png", dpi=120)
    plt.close(fig)

    renderer.save_video(
        world_id=list(range(n_render)),
        output_path=data_dir / "replay.mp4",
        scale=0.5,
    )
    print("Done.")


if __name__ == "__main__":
    main()
