"""Demo: record kinematic trajectories on the GPU, then render them to video afterwards.

Buffering rendered frames for thousands of parallel worlds is infeasible. Instead, this
script simulates many worlds with the same CUDA-graph-captured loop as
`replay_behavior_gpu.py`, but records only the kinematic state of each world (see
`flygym.recording`). It then:

1. Saves the trajectories (one `.npz` file each) and the model to disk. These are
   independent: a trajectory does not contain the model.
2. Reloads a subset of the trajectories and renders them to video on the GPU, in
   batches whose size is independent of the number of simulated worlds.
3. Optionally renders the same trajectories on the CPU as well.

Configure the run by editing the constants below, then run:

    uv run python scripts/record_and_render_trajectories_gpu.py
"""

from pathlib import Path
from time import perf_counter_ns

import mujoco as mj
import numpy as np
import warp as wp

from flygym.compose import ActuatorType
from flygym.recording import RecordedTrajectory, render_trajectories
from flygym.warp import GPUSimulation, render_trajectories_gpu
from flygym.warp.rendering import modify_world_for_batch_rendering
from flygym.warp.utils import check_gpu
from flygym_demo.benchmark import (
    ReplayTargetData,
    increment_counter_kernel,
    make_model,
    update_target_angles_kernel,
)

OUTPUT_DIR = Path("outputs/trajectory_demo")
N_WORLDS = 1000  # number of worlds to simulate and record
N_WORLDS_TO_RENDER = 50  # number of recorded worlds to render afterwards
SIM_STEPS = 2000  # number of steps to simulate
TIMESTEP = 1e-4  # simulation timestep in seconds
WORLDS_PER_BATCH = 10  # GPU render batch size, independent of N_WORLDS
RENDER_ON_CPU = False  # also render on the CPU


def main() -> None:
    check_gpu()
    actuator_type = ActuatorType.POSITION
    fly, world, cam = make_model()

    # Each world replays a different slice of recorded fly kinematics
    replay_data = ReplayTargetData(
        TIMESTEP, fly.get_actuated_jointdofs_order(actuator_type)
    )
    target_angles = replay_data.make_target_angles_all_worlds(N_WORLDS, SIM_STEPS)
    n_dofs = target_angles.shape[-1]

    sim = GPUSimulation(world, N_WORLDS, timestep=TIMESTEP)
    recorder = sim.set_trajectory_recorder(playback_speed=0.2, output_fps=25)
    sim.reset()
    sim.set_leg_adhesion_states(fly.name, np.ones((N_WORLDS, 6), dtype=np.float32))
    sim.warmup()

    # Capture the GPU-resident step as a CUDA graph. Recording reads the state outside
    # of the graph.
    target_angles_gpu = wp.array(target_angles)
    curr_target_angles_gpu = wp.zeros((N_WORLDS, n_dofs), dtype=wp.float32)
    step_counter = wp.array([0], dtype=wp.int32)
    with wp.ScopedCapture() as capture:
        wp.launch(
            update_target_angles_kernel,
            dim=(N_WORLDS, n_dofs),
            inputs=[target_angles_gpu, step_counter],
            outputs=[curr_target_angles_gpu],
        )
        sim.set_actuator_inputs(fly.name, actuator_type, curr_target_angles_gpu)
        sim.step()
        wp.launch(increment_counter_kernel, dim=1, outputs=[step_counter])

    print(f"Simulating {SIM_STEPS} steps in {N_WORLDS} worlds...")
    wp.synchronize()
    start_time = perf_counter_ns()
    for _ in range(SIM_STEPS):
        wp.capture_launch(capture.graph)
        sim.record_as_needed()
    wp.synchronize()
    walltime_s = (perf_counter_ns() - start_time) * 1e-9
    trajectories = recorder.trajectories
    print(
        f"Done in {walltime_s:.2f} s. Recorded {len(trajectories)} trajectories of "
        f"{trajectories[0].n_frames} frames each."
    )

    # Save trajectories and model separately
    traj_dir = OUTPUT_DIR / "trajectories"
    model_dir = OUTPUT_DIR / "model"
    for traj in trajectories:
        traj.save(traj_dir / f"world{traj.world_id:04d}.npz")
    world.save_xml_with_assets(model_dir, "model.xml")
    print(f"Saved trajectories to {traj_dir} and the model to {model_dir}.")

    # Reload a subset of the trajectories and the model, and render them
    traj_paths = sorted(traj_dir.glob("*.npz"))[:N_WORLDS_TO_RENDER]
    trajectories = [RecordedTrajectory.from_file(path) for path in traj_paths]
    mj_model = mj.MjModel.from_xml_path(str(model_dir / "model.xml"))

    if RENDER_ON_CPU:
        print(f"Rendering {len(trajectories)} trajectories on the CPU...")
        render_trajectories(mj_model, trajectories, cam, OUTPUT_DIR / "videos_cpu")

    # GPU batch rendering requires a modified model. The modifications do not change
    # the qpos layout, so the recorded trajectories remain compatible.
    print(
        f"Rendering {len(trajectories)} trajectories on the GPU, "
        f"{WORLDS_PER_BATCH} frames at a time..."
    )
    modify_world_for_batch_rendering(world)
    batch_mj_model, _ = world.compile()
    render_trajectories_gpu(
        batch_mj_model,
        trajectories,
        cam,
        OUTPUT_DIR / "videos_gpu",
        worlds_per_batch=WORLDS_PER_BATCH,
    )
    print(f"Videos saved to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
