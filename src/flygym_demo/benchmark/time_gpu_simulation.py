from time import perf_counter_ns

import numpy as np
import warp as wp
import pandas as pd
from loguru import logger

from flygym.warp import GPUSimulation
from flygym.compose import Fly, ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.anatomy import Skeleton, JointPreset, ActuatedDOFPreset, AxisOrder, JointDOF
from flygym.utils.math import Rotation3D
from flygym_demo.spotlight_data import MotionSnippet


def make_model(
    joints_preset=JointPreset.LEGS_ONLY,
    actuated_dofs_preset=ActuatedDOFPreset.LEGS_ACTIVE_ONLY,
    actuator_type=ActuatorType.POSITION,
    position_gain=50.0,
    neutral_pose=KinematicPosePreset.NEUTRAL,
    spawn_position=(0, 0, 0.8),  # xyz in mm
    spawn_rotation=Rotation3D("quat", (1, 0, 0, 0)),
):

    fly = Fly()
    axis_order = AxisOrder.YAW_PITCH_ROLL

    # Add joints
    skeleton = Skeleton(axis_order=axis_order, joint_preset=joints_preset)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)

    # Add position actuators and set them to the neutral pose
    actuated_dofs_list = fly.skeleton.get_actuated_dofs_from_preset(
        actuated_dofs_preset
    )
    fly.add_actuators(
        actuated_dofs_list,
        actuator_type=actuator_type,
        kp=position_gain,
        neutral_input=neutral_pose,
    )

    # Add leg adhesion
    fly.add_leg_adhesion()

    # Add visuals
    fly.colorize()
    cam = fly.add_tracking_camera()

    # Create a world and spawn the fly
    world = FlatGroundWorld()
    world.add_fly(fly, spawn_position, spawn_rotation)

    return fly, world, cam


class ReplayTargetData:
    def __init__(self, sim_timestep: float, output_dof_order: list[JointDOF]):
        self.snippet = MotionSnippet()
        self.dof_angles = self.snippet.get_joint_angles(sim_timestep, output_dof_order)
        self.n_total_steps, self.n_dofs = self.dof_angles.shape

    def make_target_angles_all_worlds(self, n_worlds: int, sim_steps: int):
        """Prepare the input array for all worlds.
        World 0 gets 0s-0.1s, world 1 gets 0.1s-0.2s, etc.
        """
        dof_angles_all_worlds = np.zeros(
            (n_worlds, sim_steps, self.n_dofs), dtype=np.float32
        )
        n_partitions = self.n_total_steps // sim_steps
        for world_id in range(n_worlds):
            partition_idx = world_id % n_partitions
            start_idx = partition_idx * sim_steps
            end_idx = start_idx + sim_steps
            dof_angles_all_worlds[world_id] = self.dof_angles[start_idx:end_idx]
        return dof_angles_all_worlds


@wp.kernel
def update_target_angles_kernel(
    dof_angles_all_worlds_gpu: wp.array3d(dtype=wp.float32),  # type: ignore
    step_counter_gpu: wp.array(dtype=wp.int32),  # type: ignore
    curr_target_angles_gpu: wp.array2d(dtype=wp.float32),  # type: ignore
):
    world_id, actuator_id = wp.tid()
    step = step_counter_gpu[0]
    my_target = dof_angles_all_worlds_gpu[world_id, step, actuator_id]
    curr_target_angles_gpu[world_id, actuator_id] = my_target


@wp.kernel
def increment_counter_kernel(
    step_counter_gpu: wp.array(dtype=wp.int32),  # type: ignore
):
    step_counter_gpu[0] = step_counter_gpu[0] + 1


def run_simulation(replay_data: np.ndarray, enable_rendering: bool, timestep: float):
    n_worlds, n_steps, n_dofs = replay_data.shape

    fly, world, cam = make_model()
    fly_name = fly.name
    sim = GPUSimulation(world, n_worlds)
    assert sim.mj_model.opt.timestep == timestep

    if enable_rendering:
        renderer = sim.set_renderer(
            cam,
            playback_speed=0.2,
            output_fps=25,
            use_gpu_batch_rendering=True,
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

    start_time = perf_counter_ns()
    for step in range(n_steps):
        wp.capture_launch(advance_sim_capture.graph)
        if enable_rendering:
            sim.render_as_needed()
    end_time = perf_counter_ns()
    walltime_s = (end_time - start_time) / 1e9

    return walltime_s


def run_benchmark(
    enable_rendering: bool,
    min_worlds: int,
    max_worlds: int,
    factor: int,
    sim_timestep: float,
    sim_steps: int,
) -> pd.DataFrame:
    ref_fly, *_ = make_model()
    jointdofs_order = ref_fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
    replay_data = ReplayTargetData(sim_timestep, jointdofs_order)

    res_li = []

    n_worlds = min_worlds
    while True:
        target_angles = replay_data.make_target_angles_all_worlds(n_worlds, sim_steps)
        try:
            walltime = run_simulation(target_angles, enable_rendering, sim_timestep)
            logger.info(
                f"Simulated {sim_steps} steps * {n_worlds} worlds in {walltime:.2f}s"
            )
        except Exception as e:
            logger.error(f"Simulation failed for n_worlds={n_worlds}: {e}")
            walltime = np.nan
            break
        res_li.append({"n_worlds": n_worlds, "walltime_s": walltime})

        n_worlds *= factor
        if n_worlds > max_worlds:
            break

    df = pd.DataFrame(res_li)
    df["steps_per_second"] = sim_steps * df["n_worlds"] / df["walltime_s"]
    df["realtime_factor"] = df["steps_per_second"] * sim_timestep
    return df
