"""Demo: record kinematic trajectories on GPU, then render them to video post-hoc.

This demonstrates the trajectory recording / replay feature (issue #296): it
decouples how many worlds are *simulated* from how many are *rendered*, and from how
many render *in one GPU batch*. Buffering full ``(n_worlds, n_cams, H, W, 3)`` RGB
tensors during a large parallel run is hopeless at thousands of worlds, so instead we
record only the generalized coordinates (``qpos``, plus mocap poses) at the render
cadence, then afterwards sub-select however many worlds we actually want on video and
rasterize them in small GPU batches whose size is independent of the world count.

It is the trajectory-recording counterpart of ``replay_behavior_gpu.py``: the same
Spotlight kinematic recording is replayed across many parallel worlds with the same
fully GPU-resident, CUDA-graph-captured inner loop, but the live batch renderer is
swapped for a ``WarpTrajectoryRecorder`` (via ``set_renderer(...,
record_trajectory_only=True)``). The recorder copies ``qpos`` off the GPU at the
render cadence instead of rendering frames.

The script then shows the full decoupled pipeline:

1. Simulate ``N_WORLDS`` worlds, recording a trajectory for *every* world.
2. Save the trajectories (one ``.npz`` each) and the model (``save_xml_with_assets``)
   to disk -- they are independent artifacts; a trajectory carries no model.
3. Reload the trajectories from disk, sub-select ``RENDER_WORLDS`` of them, and render
   those to video on GPU (``render_trajectories_gpu``, in batches of
   ``WORLDS_PER_BATCH``) and optionally on CPU (``CPU_REPLAY``) to show that a
   GPU-recorded trajectory is backend-agnostic.

Configure the run by editing the constants below, then::

    uv run python scripts/record_replay_trajectories_gpu.py
"""

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
from flygym.rendering import RecordedTrajectory, render_trajectories
from flygym.compose import ActuatorType
from flygym_demo.benchmark import (
    make_model,
    ReplayTargetData,
    update_target_angles_kernel,
    increment_counter_kernel,
)

# --- Configuration (edit these) -------------------------------------------------
OUTPUT_DIR = Path("outputs/traj_demo")  # trajectories, model, and rendered videos
N_WORLDS = 1000  # parallel worlds to simulate; a trajectory is recorded for each
RENDER_WORLDS = 50  # how many recorded worlds to render in the second stage
SIM_STEPS = 2000  # steps to simulate per world (2000 * 1e-4 s = 0.2 s)
TIMESTEP = 1e-4  # simulation timestep in seconds
WORLDS_PER_BATCH = 10  # GPU render batch size, decoupled from N_WORLDS
CPU_REPLAY = False  # also replay on CPU (shows the format is backend-agnostic)
# --------------------------------------------------------------------------------

_MODEL_SUBDIR = "model"
_TRAJ_SUBDIR = "trajectories"


def record_trajectories():
    """Run the GPU simulation, recording qpos for every world.

    Returns ``(trajectories, world, sim)``: the recorded trajectories (one per world),
    the world (kept so we can persist / re-compile the model), and the simulation
    (kept for its unmodified ``mj_model``, used for CPU replay).
    """
    actuator_type = ActuatorType.POSITION

    fly, world, cam = make_model()
    fly_name = fly.name

    # Build per-world target angle slices (world 0 -> first slice, world 1 -> next...).
    replay_data = ReplayTargetData(
        TIMESTEP, fly.get_actuated_jointdofs_order(actuator_type)
    )
    target_angles_all_worlds = replay_data.make_target_angles_all_worlds(
        N_WORLDS, SIM_STEPS
    )
    n_dofs = target_angles_all_worlds.shape[-1]

    sim = GPUSimulation(world, N_WORLDS, timestep=TIMESTEP)

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
    sim.set_leg_adhesion_states(fly_name, np.ones((N_WORLDS, 6), dtype=np.float32))
    sim.warmup()

    # GPU-resident buffers for the captured loop.
    target_angles_gpu = wp.array(target_angles_all_worlds)
    curr_target_angles_gpu = wp.zeros((N_WORLDS, n_dofs), dtype=wp.float32)
    step_counter = wp.array([0], dtype=wp.int32)

    # Capture the whole GPU-resident step body once (this triggers JIT). The recorder
    # reads qpos *outside* the graph (a host transfer), so it is not captured here.
    with wp.ScopedCapture() as advance_sim_capture:
        wp.launch(
            update_target_angles_kernel,
            dim=(N_WORLDS, n_dofs),
            inputs=[target_angles_gpu, step_counter],
            outputs=[curr_target_angles_gpu],
        )
        sim.set_actuator_inputs(fly_name, actuator_type, curr_target_angles_gpu)
        sim.step()
        wp.launch(increment_counter_kernel, dim=1, outputs=[step_counter])

    # Untimed warm-up: force any remaining JIT, then reset the counter and recorder so
    # recording starts cleanly from step 0.
    print(f"Warming up (JIT compilation) {N_WORLDS} worlds...")
    wp.capture_launch(advance_sim_capture.graph)
    sim.render_as_needed()
    wp.synchronize()
    step_counter.zero_()
    recorder.reset()

    print(f"Simulating {SIM_STEPS} steps across {N_WORLDS} worlds (recording all)...")
    wp.synchronize()
    start_time = perf_counter_ns()
    for _ in range(SIM_STEPS):
        wp.capture_launch(advance_sim_capture.graph)
        sim.render_as_needed()  # records qpos for every world at the cadence
    wp.synchronize()
    walltime_s = (perf_counter_ns() - start_time) / 1e9

    throughput = N_WORLDS * SIM_STEPS / walltime_s
    trajectories = recorder.recorded_trajectories
    print(
        f"Simulated {SIM_STEPS} steps * {N_WORLDS} worlds in {walltime_s:.2f}s "
        f"({throughput:.0f} steps/s, {throughput * TIMESTEP:.1f}x realtime).\n"
        f"Recorded {len(trajectories)} trajectories of "
        f"{trajectories[0].n_frames} frames each."
    )
    return trajectories, world, sim


def main() -> None:
    check_gpu()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / _MODEL_SUBDIR
    traj_dir = OUTPUT_DIR / _TRAJ_SUBDIR

    # --- Record ---
    trajectories, world, sim = record_trajectories()

    # --- Persist: each trajectory is one self-describing .npz; the model is a
    # separate artifact (a trajectory carries no model). ---
    traj_dir.mkdir(parents=True, exist_ok=True)
    for i, traj in enumerate(trajectories):
        traj.save(traj_dir / f"traj_{i:04d}.npz")
    world.save_xml_with_assets(model_dir, "model.xml")
    print(
        f"Saved {len(trajectories)} trajectories to {traj_dir} and the model to "
        f"{model_dir}."
    )

    # --- Replay post-hoc, reloading the trajectories from disk and sub-selecting ---
    traj_files = sorted(traj_dir.glob("traj_*.npz"))
    trajectories = [RecordedTrajectory.from_file(p) for p in traj_files]
    n_render = min(RENDER_WORLDS, len(trajectories))
    trajectories = trajectories[:n_render]
    print(f"Reloaded trajectories; rendering {n_render} of them.")

    if CPU_REPLAY:
        # CPU replay needs no special model prep; reuse the unmodified compiled model.
        cpu_out = OUTPUT_DIR / "replay_cpu"
        print(f"Rendering on CPU to {cpu_out}...")
        render_trajectories(sim.mj_model, trajectories, cpu_out)

    # GPU batch rendering needs a batch-ready model (textures stripped, overhead
    # lights added). The recorder ran against the unmodified model, but those edits
    # don't change the qpos layout, so the trajectories stay valid.
    gpu_out = OUTPUT_DIR / "replay_gpu"
    print(
        f"Rendering {len(trajectories)} worlds on GPU to {gpu_out} "
        f"(batches of {WORLDS_PER_BATCH})..."
    )
    modify_world_for_batch_rendering(world)
    batch_model = world.compile()[0]
    render_trajectories_gpu(
        batch_model, trajectories, gpu_out, worlds_per_batch=WORLDS_PER_BATCH
    )
    print("Done.")


if __name__ == "__main__":
    main()
