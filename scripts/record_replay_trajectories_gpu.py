"""Demo: record kinematic trajectories on GPU, then render them to video post-hoc.

This demonstrates the trajectory recording / replay feature (issue #296): it
decouples how many worlds are *simulated* from how many are *rendered in one batch*.
Buffering full ``(n_worlds, n_cams, H, W, 3)`` RGB tensors during a large parallel run
is hopeless at thousands of worlds, so instead we record only the generalized
coordinates (``qpos``, plus mocap poses) at the render cadence, and rasterize them
afterwards in small GPU batches whose size is independent of the simulation's world
count.

It is the trajectory-recording counterpart of ``replay_behavior_gpu.py``: the same
Spotlight kinematic recording is replayed across many parallel worlds with the same
fully GPU-resident, CUDA-graph-captured inner loop, but the live batch renderer is
swapped for a ``WarpTrajectoryRecorder`` (via ``set_renderer(...,
record_trajectory_only=True)``). The recorder copies ``qpos`` off the GPU at the
render cadence instead of rendering frames.

The script then shows the full decoupled pipeline:

1. Simulate ``--n-worlds`` worlds, recording a trajectory for *every* world.
2. Save the trajectories (one ``.npz`` each) and the model (``save_xml_with_assets``)
   to disk -- they are independent artifacts; a trajectory carries no model.
3. Reload the trajectories from disk and render every world to video, on GPU
   (``render_trajectories_gpu``, in batches of ``--worlds-per-batch``) and optionally
   on CPU (``--cpu-replay``) to show that a GPU-recorded trajectory is backend-agnostic.

Example:
    uv run python scripts/record_replay_trajectories_gpu.py --output outputs/traj_demo
    uv run python scripts/record_replay_trajectories_gpu.py --output outputs/traj_demo --cpu-replay
"""

import argparse
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import warp as wp

from flygym.warp import (
    GPUSimulation,
    render_trajectories_gpu,
    modify_world_for_batch_rendering,
)
from flygym.warp.utils import check_gpu
from flygym.rendering import save_trajectories, load_trajectories, render_trajectories
from flygym.compose import ActuatorType
from flygym_demo.benchmark import (
    make_model,
    ReplayTargetData,
    update_target_angles_kernel,
    increment_counter_kernel,
)

_MODEL_SUBDIR = "model"
_TRAJ_SUBDIR = "trajectories"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/traj_demo"),
        metavar="DIR",
        help="Directory for the saved trajectories, model, and rendered videos "
        "(default: outputs/traj_demo). Reused if it already exists.",
    )
    parser.add_argument(
        "--n-worlds",
        type=int,
        default=50,
        help="Number of parallel worlds to simulate; a trajectory is recorded for "
        "every one (default: 50). Recording is cheap, but every world is rendered "
        "afterwards, so keep this modest unless you want many output videos.",
    )
    parser.add_argument(
        "--sim-steps",
        type=int,
        default=2000,
        help="Number of steps to simulate per world (default: 2000 = 0.2 s).",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1e-4,
        help="Simulation timestep in seconds (default: 1e-4).",
    )
    parser.add_argument(
        "--worlds-per-batch",
        type=int,
        default=10,
        help="Number of frames staged into one GPU render batch (default: 10). This "
        "is the render-time parallelism, decoupled from --n-worlds; larger uses more "
        "GPU memory.",
    )
    parser.add_argument(
        "--cpu-replay",
        action="store_true",
        help="Additionally replay the GPU-recorded trajectories on the CPU, to "
        "demonstrate that the recorded format is backend-agnostic.",
    )
    return parser.parse_args()


def record_trajectories(args: argparse.Namespace):
    """Run the GPU simulation, recording qpos for every world.

    Returns ``(trajectories, world, sim)``: the recorded trajectories (one per world),
    the world (kept so we can persist / re-compile the model), and the simulation
    (kept for its unmodified ``mj_model``, used for CPU replay).
    """
    n_worlds = args.n_worlds
    sim_steps = args.sim_steps
    timestep = args.timestep
    actuator_type = ActuatorType.POSITION

    fly, world, cam = make_model()
    fly_name = fly.name

    # Build per-world target angle slices (world 0 -> first slice, world 1 -> next...).
    replay_data = ReplayTargetData(
        timestep, fly.get_actuated_jointdofs_order(actuator_type)
    )
    target_angles_all_worlds = replay_data.make_target_angles_all_worlds(
        n_worlds, sim_steps
    )
    n_dofs = target_angles_all_worlds.shape[-1]

    sim = GPUSimulation(world, n_worlds, timestep=timestep)

    # Swap the live batch renderer for a recorder: it stores qpos for every world at
    # the render cadence instead of rasterizing frames. Omitting `worlds` records all.
    recorder = sim.set_renderer(
        cam,
        playback_speed=0.2,
        output_fps=25,
        record_trajectory_only=True,
    )

    # Reset to the neutral keyframe and settle. Must happen *before* the graph
    # capture, since `reset` reallocates `mjw_data` (which the captured graph holds).
    sim.reset()
    sim.set_leg_adhesion_states(fly_name, np.ones((n_worlds, 6), dtype=np.float32))
    sim.warmup()

    # GPU-resident buffers for the captured loop.
    target_angles_gpu = wp.array(target_angles_all_worlds)
    curr_target_angles_gpu = wp.zeros((n_worlds, n_dofs), dtype=wp.float32)
    step_counter = wp.array([0], dtype=wp.int32)

    # Capture the whole GPU-resident step body once (this triggers JIT). The recorder
    # reads qpos *outside* the graph (a host transfer), so it is not captured here.
    with wp.ScopedCapture() as advance_sim_capture:
        wp.launch(
            update_target_angles_kernel,
            dim=(n_worlds, n_dofs),
            inputs=[target_angles_gpu, step_counter],
            outputs=[curr_target_angles_gpu],
        )
        sim.set_actuator_inputs(fly_name, actuator_type, curr_target_angles_gpu)
        sim.step()
        wp.launch(increment_counter_kernel, dim=1, outputs=[step_counter])

    # Untimed warm-up: force any remaining JIT, then reset the counter and recorder so
    # recording starts cleanly from step 0.
    print(f"Warming up (JIT compilation) {n_worlds} worlds...")
    wp.capture_launch(advance_sim_capture.graph)
    sim.render_as_needed()
    wp.synchronize()
    step_counter.zero_()
    recorder.reset()

    print(f"Simulating {sim_steps} steps across {n_worlds} worlds (recording all)...")
    wp.synchronize()
    start_time = perf_counter_ns()
    for _ in range(sim_steps):
        wp.capture_launch(advance_sim_capture.graph)
        sim.render_as_needed()  # records qpos for every world at the cadence
    wp.synchronize()
    walltime_s = (perf_counter_ns() - start_time) / 1e9

    throughput = n_worlds * sim_steps / walltime_s
    trajectories = recorder.recorded_trajectories
    print(
        f"Simulated {sim_steps} steps * {n_worlds} worlds in {walltime_s:.2f}s "
        f"({throughput:.0f} steps/s, {throughput * timestep:.1f}x realtime).\n"
        f"Recorded {len(trajectories)} trajectories of "
        f"{trajectories[0].n_frames} frames each."
    )
    return trajectories, world, sim


def main() -> None:
    args = parse_args()
    check_gpu()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / _MODEL_SUBDIR
    traj_dir = out / _TRAJ_SUBDIR

    # --- Record ---
    trajectories, world, sim = record_trajectories(args)

    # --- Persist: trajectories and model are independent artifacts ---
    save_trajectories(trajectories, traj_dir)
    world.save_xml_with_assets(model_dir, "model.xml")
    print(
        f"Saved {len(trajectories)} trajectories to {traj_dir} and the model to "
        f"{model_dir}."
    )

    # --- Replay post-hoc, reloading the trajectories from disk ---
    trajectories = load_trajectories(traj_dir)

    if args.cpu_replay:
        # CPU replay needs no special model prep; reuse the unmodified compiled model.
        cpu_out = out / "replay_cpu"
        print(f"Rendering on CPU to {cpu_out}...")
        render_trajectories(sim.mj_model, trajectories, cpu_out)

    # GPU batch rendering needs a batch-ready model (textures stripped, overhead
    # lights added). The recorder ran against the unmodified model, but those edits
    # don't change the qpos layout, so the trajectories stay valid.
    gpu_out = out / "replay_gpu"
    print(
        f"Rendering {len(trajectories)} worlds on GPU to {gpu_out} "
        f"(batches of {args.worlds_per_batch})..."
    )
    modify_world_for_batch_rendering(world)
    batch_model = world.compile()[0]
    render_trajectories_gpu(
        batch_model, trajectories, gpu_out, worlds_per_batch=args.worlds_per_batch
    )
    print("Done.")


if __name__ == "__main__":
    main()
