"""Smoke tests for flygym_demo.complex_terrain locomotion helpers."""

import numpy as np
import pytest

from flygym.compose import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    CPGController,
    CPGNetwork,
    HybridController,
    HybridTurningController,
    PreprogrammedSteps,
    RuleBasedController,
    apply_locomotion_action,
    dof_spec_to_jointdof,
    get_default_locomotion_dof_order,
    make_locomotion_fly,
    make_tripod_cpg_network,
)


@pytest.fixture(scope="module")
def preprogrammed_steps():
    return PreprogrammedSteps()


@pytest.fixture(scope="module")
def locomotion_sim():
    fly = make_locomotion_fly()
    world = FlatGroundWorld()
    world.add_fly(
        fly,
        spawn_position=[0, 0, 1.0],
        spawn_rotation=Rotation3D("quat", [1, 0, 0, 0]),
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)
    sim.reset()
    return sim, fly


class TestPreprogrammedSteps:
    def test_scalar_phase_returns_one_pose(self, preprogrammed_steps):
        angles = preprogrammed_steps.get_joint_angles("lf", 0.0)
        assert angles.shape == (7,)

    def test_length_one_phase_array_keeps_phase_axis(self, preprogrammed_steps):
        angles = preprogrammed_steps.get_joint_angles("lf", np.array([0.0]))
        assert angles.shape == (7, 1)

    def test_dof_helper_is_public(self, preprogrammed_steps):
        jointdof = dof_spec_to_jointdof("lf", preprogrammed_steps.dofs_per_leg[0])
        assert jointdof in get_default_locomotion_dof_order()

    def test_adhesion_matches_v1_binary_semantics(self, preprogrammed_steps):
        phases = np.zeros(6)
        adhesion = preprogrammed_steps.get_adhesion_onoff_by_phase(phases)
        assert adhesion.dtype == np.dtype(bool)
        assert set(adhesion.astype(int)).issubset({0, 1})


class TestCPGController:
    def test_shape_validation_happens_before_reset(self):
        with pytest.raises(ValueError, match="coupling_weights"):
            CPGNetwork(
                timestep=1e-4,
                intrinsic_freqs=np.ones(6),
                intrinsic_amps=np.ones(6),
                coupling_weights=np.ones((5, 5)),
                phase_biases=np.ones((6, 6)),
                convergence_coefs=np.ones(6),
            )

    def test_step_returns_locomotion_action(self, preprogrammed_steps):
        controller = CPGController(
            make_tripod_cpg_network(1e-4),
            preprogrammed_steps,
        )
        action = controller.step()
        assert action.joint_angles.shape == (42,)
        assert action.adhesion_onoff.shape == (6,)
        assert action.adhesion_onoff.dtype == np.dtype(bool)


class TestRuleBasedController:
    def test_first_step_excludes_hind_legs(self, preprogrammed_steps):
        controller = RuleBasedController(
            timestep=1e-4,
            preprogrammed_steps=preprogrammed_steps,
            seed=0,
        )
        action = controller.step()
        assert not controller.mask_is_stepping[[2, 5]].any()
        assert action.joint_angles.shape == (42,)


class TestHybridControllers:
    def test_hybrid_controller_steps(self, locomotion_sim, preprogrammed_steps):
        sim, fly = locomotion_sim
        controller = HybridController(
            timestep=sim.mj_model.opt.timestep,
            preprogrammed_steps=preprogrammed_steps,
        )
        action = controller.step(sim, fly.name)
        apply_locomotion_action(sim, fly.name, action)
        assert action.joint_angles.shape == (42,)
        assert action.adhesion_onoff.dtype == np.dtype(bool)

    def test_turning_zero_signal_keeps_nonnegative_frequency(
        self, locomotion_sim, preprogrammed_steps
    ):
        sim, fly = locomotion_sim
        controller = HybridTurningController(
            timestep=sim.mj_model.opt.timestep,
            preprogrammed_steps=preprogrammed_steps,
        )
        base_freqs = controller.cpg_network.intrinsic_freqs.copy()
        controller.step(np.zeros(2), sim, fly.name)
        np.testing.assert_array_equal(
            controller.cpg_network.intrinsic_freqs,
            base_freqs,
        )
