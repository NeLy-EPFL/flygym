from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    JointDOF,
    JointPreset,
    PASSIVE_TARSAL_LINKS,
    Skeleton,
)
from flygym.compose import ActuatorType, Fly, KinematicPosePreset
from flygym.simulation import Simulation


@dataclass(frozen=True)
class LocomotionAction:
    """Position-actuator and adhesion commands for v2 locomotion examples."""

    joint_angles: Float[np.ndarray, "n_actuated_dofs"]
    adhesion_onoff: Float[np.ndarray, "6"] | None = None


def get_default_locomotion_dof_order() -> list[JointDOF]:
    """Return the default active leg DOFs used by the locomotion examples."""
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    return skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)


def make_locomotion_fly(
    name: str = "nmf",
    *,
    joint_stiffness: float = 0.05,
    joint_damping: float = 0.06,
    passive_tarsus_stiffness: float = 7.5,
    passive_tarsus_damping: float = 1e-2,
    actuator_gain: float = 45.0,
    actuator_forcerange: tuple[float, float] = (-65.0, 65.0),
    add_adhesion: bool = True,
    adhesion_gain: float = 40.0,
    colorize: bool = False,
) -> Fly:
    """Create a standard legs-only, position-controlled fly for walking examples.

    Defaults mirror FlyGym v1 ``flygym_gymnasium.Fly`` position-control locomotion
    settings: soft actuated leg joints, stiffer passive tarsi, ``actuator_gain=45``,
    ``adhesion_gain=40``, and ``forcerange`` ±65. Higher ``actuator_gain`` (e.g.
    80–150) can increase open-loop translation but often trades off against smoother
    contact-driven motion.
    """
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
        AxisOrder.YAW_PITCH_ROLL
    )
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    fly = Fly(name=name)
    joints = fly.add_joints(
        skeleton,
        neutral_pose=neutral_pose,
        stiffness=joint_stiffness,
        damping=joint_damping,
    )
    for jointdof, joint in joints.items():
        if jointdof.child.link in PASSIVE_TARSAL_LINKS:
            joint.stiffness = passive_tarsus_stiffness
            joint.damping = passive_tarsus_damping
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        ActuatorType.POSITION,
        neutral_input=neutral_pose,
        kp=actuator_gain,
        forcerange=actuator_forcerange,
    )
    if add_adhesion:
        fly.add_leg_adhesion(gain=adhesion_gain)
    if colorize:
        fly.colorize()
    return fly


def apply_locomotion_action(
    sim: Simulation,
    fly_name: str,
    action: LocomotionAction,
    *,
    actuator_type: ActuatorType = ActuatorType.POSITION,
) -> None:
    """Write a locomotion action into a simulation without stepping physics."""
    sim.set_actuator_inputs(fly_name, actuator_type, action.joint_angles)
    if action.adhesion_onoff is not None:
        sim.set_leg_adhesion_states(fly_name, action.adhesion_onoff)
