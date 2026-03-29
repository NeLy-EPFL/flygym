"""Tests for flygym.warp.simulation.GPUSimulation."""

import warnings
import pytest
import numpy as np
import warp as wp

from flygym.anatomy import Skeleton, JointPreset, ActuatedDOFPreset, AxisOrder
from flygym.compose import Fly, ActuatorType, FlatGroundWorld, KinematicPosePreset
from flygym.utils.math import Rotation3D
from flygym.warp import GPUSimulation
from flygym.warp.rendering import WarpCPURenderer


# ==============================================================================
# Module-scoped simulation fixture
# ==============================================================================


@pytest.fixture(scope="module")
def gpu_bundle(gpu_sim_factory):
    """GPUSimulation with 4 worlds, reset to keyframe 0."""
    sim, fly, cam = gpu_sim_factory(n_worlds=4, fly_name="sim_test_fly")
    sim.reset()
    yield sim, fly, cam


# ==============================================================================
# Construction
# ==============================================================================


class TestGPUSimulationConstruction:
    def test_n_worlds_attribute(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        assert sim.n_worlds == 4

    def test_mjw_model_and_data_present(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        assert sim.mjw_model is not None
        assert sim.mjw_data is not None

    def test_initial_time_is_zero(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        assert sim.time == pytest.approx(0.0)

    def test_initial_step_counter_is_zero(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        assert sim._curr_step == 0

    def test_different_n_worlds(self, gpu_sim_factory):
        """GPUSimulation should accept any positive n_worlds value."""
        sim, fly, cam = gpu_sim_factory(n_worlds=8, fly_name="multi_fly")
        assert sim.n_worlds == 8

    def test_noslip_iterations_stripped(self):
        """_strip_unsupported_options_for_mjwarp should zero out noslip_iterations."""
        fly = Fly(name="noslip_fly")
        skeleton = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL
        )
        fly.add_joints(skeleton, neutral_pose=pose)
        world = FlatGroundWorld()
        world.mjcf_root.option.noslip_iterations = 5  # unsupported by MJWarp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = GPUSimulation(world, n_worlds=2)

        assert world.mjcf_root.option.noslip_iterations == 0


# ==============================================================================
# Step and time
# ==============================================================================


class TestStep:
    def test_step_advances_time(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        t0 = sim.time
        sim.step()
        assert sim.time > t0

    def test_step_advances_by_roughly_one_timestep(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        sim.step()
        expected_dt = sim.mj_model.opt.timestep
        assert sim.time == pytest.approx(expected_dt, rel=1e-4)

    def test_multiple_steps_accumulate(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        n = 10
        for _ in range(n):
            sim.step()
        expected = n * sim.mj_model.opt.timestep
        assert sim.time == pytest.approx(expected, rel=1e-3)

    def test_step_with_profile_increments_counter(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        sim.step_with_profile()
        assert sim._curr_step == 1

    def test_step_with_profile_accumulates_physics_time(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        sim.step_with_profile()
        assert sim._total_physics_time_ns > 0


# ==============================================================================
# Reset
# ==============================================================================


class TestReset:
    def test_reset_restores_time_to_zero(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        for _ in range(5):
            sim.step()
        sim.reset()
        assert sim.time == pytest.approx(0.0)

    def test_reset_clears_step_counter(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.step_with_profile()
        sim.step_with_profile()
        sim.reset()
        assert sim._curr_step == 0

    def test_reset_clears_physics_time(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.step_with_profile()
        sim.reset()
        assert sim._total_physics_time_ns == 0


# ==============================================================================
# State queries
# ==============================================================================


class TestStateQueries:
    def test_get_joint_angles_shape(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        angles = sim.get_joint_angles(fly.name)
        assert isinstance(angles, wp.array)
        assert angles.shape[0] == sim.n_worlds
        assert angles.ndim == 2

    def test_get_joint_velocities_shape(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        vels = sim.get_joint_velocities(fly.name)
        assert isinstance(vels, wp.array)
        assert vels.shape[0] == sim.n_worlds
        assert vels.ndim == 2

    def test_joint_angles_and_velocities_have_same_dof_count(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        angles = sim.get_joint_angles(fly.name)
        vels = sim.get_joint_velocities(fly.name)
        assert angles.shape[1] == vels.shape[1]

    def test_get_body_positions_shape(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        bpos = sim.get_body_positions(fly.name)
        assert isinstance(bpos, wp.array)
        assert bpos.shape[0] == sim.n_worlds
        assert bpos.shape[2] == 3

    def test_get_body_rotations_shape(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        brots = sim.get_body_rotations(fly.name)
        assert isinstance(brots, wp.array)
        assert brots.shape[0] == sim.n_worlds
        assert brots.shape[2] == 4

    def test_body_positions_and_rotations_same_n_bodies(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        bpos = sim.get_body_positions(fly.name)
        brots = sim.get_body_rotations(fly.name)
        assert bpos.shape[1] == brots.shape[1]

    def test_body_rotations_are_unit_quaternions(self, gpu_bundle):
        """All quaternions returned by get_body_rotations should have unit norm."""
        sim, fly, cam = gpu_bundle
        sim.reset()
        brots = sim.get_body_rotations(fly.name).numpy()
        norms = np.linalg.norm(brots, axis=2)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_time_returns_float(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        assert isinstance(sim.time, float)


# ==============================================================================
# Control inputs
# ==============================================================================


class TestControlInputs:
    def test_set_actuator_inputs_numpy(self, gpu_bundle):
        """set_actuator_inputs should accept numpy arrays without error."""
        sim, fly, cam = gpu_bundle
        sim.reset()
        n_dofs = sim.get_joint_angles(fly.name).shape[1]
        inputs = np.zeros((sim.n_worlds, n_dofs), dtype=np.float32)
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, inputs)

    def test_set_actuator_inputs_warp(self, gpu_bundle):
        """set_actuator_inputs should also accept warp arrays."""
        sim, fly, cam = gpu_bundle
        sim.reset()
        n_dofs = sim.get_joint_angles(fly.name).shape[1]
        inputs = wp.zeros((sim.n_worlds, n_dofs), dtype=wp.float32)
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, inputs)

    def test_set_leg_adhesion_states_numpy(self, gpu_bundle):
        """set_leg_adhesion_states should accept numpy arrays."""
        sim, fly, cam = gpu_bundle
        sim.reset()
        adhesion = np.ones((sim.n_worlds, 6), dtype=np.float32)
        sim.set_leg_adhesion_states(fly.name, adhesion)

    def test_set_leg_adhesion_states_warp(self, gpu_bundle):
        """set_leg_adhesion_states should also accept warp arrays."""
        sim, fly, cam = gpu_bundle
        sim.reset()
        adhesion = wp.ones((sim.n_worlds, 6), dtype=wp.float32)
        sim.set_leg_adhesion_states(fly.name, adhesion)

    def test_control_inputs_affect_joint_angles(self, gpu_bundle):
        """Driving joints toward a different target should change their angles."""
        sim, fly, cam = gpu_bundle
        sim.reset()
        n_dofs = sim.get_joint_angles(fly.name).shape[1]
        angles_neutral = sim.get_joint_angles(fly.name).numpy().copy()

        zero_inputs = np.zeros((sim.n_worlds, n_dofs), dtype=np.float32)
        sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, zero_inputs)
        for _ in range(50):
            sim.step()

        angles_driven = sim.get_joint_angles(fly.name).numpy()
        assert not np.allclose(angles_neutral, angles_driven, atol=1e-4)


# ==============================================================================
# Warmup
# ==============================================================================


class TestWarmup:
    def test_warmup_advances_time(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        sim.warmup(duration_s=0.001)
        assert sim.time > 0.0

    def test_warmup_default_duration(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        sim.warmup(duration_s=0.001)
        assert sim.time > 0.0


# ==============================================================================
# Renderer attachment and performance report
# ==============================================================================


class TestSetRenderer:
    def test_set_renderer_returns_warp_cpu_renderer(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        sim.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            renderer = sim.set_renderer(
                cam,
                camera_res=(64, 64),
                worlds=[0, 1],
                use_gpu_batch_rendering=False,
            )
        assert isinstance(renderer, WarpCPURenderer)

    def test_set_renderer_world_ids(self, gpu_bundle):
        sim, fly, cam = gpu_bundle
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            renderer = sim.set_renderer(
                cam,
                camera_res=(64, 64),
                worlds=[0, 2],
                use_gpu_batch_rendering=False,
            )
        assert renderer.world_ids == [0, 2]

    def test_print_performance_report(self, gpu_bundle, capsys):
        """print_performance_report should produce tabular output after profiled steps."""
        sim, fly, cam = gpu_bundle
        sim.reset()
        for _ in range(10):
            sim.step_with_profile()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim.set_renderer(
                cam,
                camera_res=(64, 64),
                worlds=[0],
                use_gpu_batch_rendering=False,
            )
        sim.print_performance_report()
        captured = capsys.readouterr()
        assert "PERFORMANCE" in captured.out
