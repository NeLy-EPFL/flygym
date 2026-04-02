from pathlib import Path

import numpy as np
import warp as wp
from tqdm import trange

from flygym.warp import GPUSimulation
from flygym.compose import ActuatorType
from flygym.anatomy import JointDOF
from flygym_demo.spotlight_data import MotionSnippet
from flygym_demo.benchmark import (
    make_model,
    update_target_angles_kernel,
    increment_counter_kernel,
)

TIMESTEP = 1e-4


class ReplayTargetData:
    def __init__(self, sim_timestep: float, output_dof_order: list[JointDOF]):
        self.snippet = MotionSnippet()
        self.dof_angles = self.snippet.get_joint_angles(sim_timestep, output_dof_order)
        self.n_total_steps, self.n_dofs = self.dof_angles.shape

    def make_target_angles_all_worlds(self, n_worlds: int, sim_steps: int):
        """Prepare the input array for all worlds, each with a random offset."""
        dof_angles_all_worlds = np.zeros(
            (n_worlds, sim_steps, self.n_dofs), dtype=np.float32
        )
        for world_id in range(n_worlds):
            start_idx = np.random.randint(0, self.n_total_steps - sim_steps)
            end_idx = start_idx + sim_steps
            dof_angles_all_worlds[world_id] = self.dof_angles[start_idx:end_idx]
        return dof_angles_all_worlds


def run_simulation(replay_data: np.ndarray, output_path: Path) -> float:
    n_worlds, n_steps, n_dofs = replay_data.shape

    fly, world, cam = make_model(simplify_geom=False)
    fly_name = fly.name
    sim = GPUSimulation(world, n_worlds)
    assert sim.mj_model.opt.timestep == TIMESTEP

    renderer = sim.set_renderer(
        cam, playback_speed=0.2, output_fps=25, use_gpu_batch_rendering=True
    )

    # Turn adhesion on for all 6 legs across all worlds
    sim.set_leg_adhesion_states(fly_name, np.ones((n_worlds, 6), dtype=np.float32))
    sim.warmup()

    replay_data_gpu = wp.array(replay_data)
    curr_target_angles_gpu = wp.zeros((n_worlds, n_dofs), dtype=wp.float32)
    step_counter = wp.array([0], dtype=wp.int32)

    with wp.ScopedCapture() as advance_sim_capture:
        wp.launch(
            update_target_angles_kernel,
            dim=(n_worlds, n_dofs),
            inputs=[replay_data_gpu, step_counter],
            outputs=[curr_target_angles_gpu],
        )
        sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, curr_target_angles_gpu)
        sim.step()
        wp.launch(increment_counter_kernel, dim=1, outputs=[step_counter])

    for _ in trange(n_steps):
        wp.capture_launch(advance_sim_capture.graph)
        sim.render_as_needed()

    renderer.save_video(world_id=list(range(25)), output_path=output_path, scale=1)


if __name__ == "__main__":
    n_worlds = 25
    sim_steps = 10000
    output_path = Path("demo_output/gpu_demo_video.mp4")

    ref_fly, *_ = make_model()
    jointdofs_order = ref_fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    replay_data = ReplayTargetData(TIMESTEP, jointdofs_order)
    target_angles = replay_data.make_target_angles_all_worlds(n_worlds, sim_steps)
    run_simulation(target_angles, output_path)
