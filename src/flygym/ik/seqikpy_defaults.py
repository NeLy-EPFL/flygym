"""Joint bounds and IK initial guess translated from SeqIKPy's default body config.

[SeqIKPy](https://nely-epfl.github.io/sequential-inverse-kinematics/) ships a
default `NeuroMechFly` body configuration
(`seqikpy.body_config.neuromechfly_body_config`) with per-DOF joint angle
bounds and an initial guess for its own staged leg optimization
(`_NMF_INITIAL_ANGLES_RAD`'s `"stage_4"` entry -- the seed SeqIKPy's `ikpy`
solver starts from when fitting the whole leg chain). That configuration only
defines explicit values for the front legs (`RF`/`LF`); we treat the
left-front (`LF`) values as canonical, since they are already expressed in
the same left-right-symmetric convention flygym uses internally
(`BaseFly.add_joints` flips the roll/yaw axis vector for the right side so a
given `qpos` value means the same physical rotation on both sides --
confirmed empirically: SeqIKPy's `RF_*` bounds/angles are exact sign-flipped
mirrors of the `LF_*` ones for roll/yaw, and identical for pitch). We apply
the same per-DOF-type values to all six legs, since SeqIKPy's public defaults
don't differentiate bounds/angles by leg row (front/middle/hind).

!!! warning "The initial guess is not a naturally-posed default"

    `seqikpy_initial_guess_qpos` reproduces SeqIKPy's own IK-optimizer seed,
    not a naturally standing/resting pose. SeqIKPy's `ikpy` chain defines its
    own zero-angle configuration as the entire leg pointing straight down
    (each segment's `origin_translation` is along local `-Z`, chained with no
    intervening bend), which is unrelated to flygym's zero-angle
    configuration (which follows directly from the rigged mesh geometry, and
    already looks like a naturally extended leg at `qpos = 0`). Verified by
    reproducing the exact `ikpy` chain SeqIKPy builds and running its forward
    kinematics on these angles: the resulting segment lengths match SeqIKPy's
    own body-size config exactly (so the values are parsed/indexed
    correctly), and the shape is a plausible bent leg in *SeqIKPy's own
    coordinate convention* -- it just isn't a good default pose to initialize
    a *flygym* fly's visual rest state from. For a naturally-posed default
    (e.g. to start a convergence video from), use `qpos = 0` or
    `flygym.compose.pose.KinematicPosePreset.NEUTRAL` instead; reserve
    `seqikpy_initial_guess_qpos` for what it actually is -- a starting guess
    for the optimizer, which does not need to look natural, only to help
    convergence.
"""

import mujoco as mj
import numpy as np

from flygym.anatomy import JointDOF

__all__ = ["seqikpy_joint_bounds", "seqikpy_initial_guess_qpos"]

# Degrees, keyed by SeqIKPy DOF type name. Taken from the LF_* entries of
# seqikpy.body_config._NMF_BOUNDS_DEG.
_BOUNDS_DEG_BY_DOF_TYPE = {
    "ThC_yaw": (-50.0, 50.0),
    "ThC_pitch": (-40.0, 60.0),
    "ThC_roll": (-50.0, 130.0),
    "CTr_pitch": (-180.0, 0.0),
    "CTr_roll": (0.0, 150.0),
    "FTi_pitch": (0.0, 170.0),
    "TiTa_pitch": (-150.0, 0.0),
}

# Radians, keyed by SeqIKPy DOF type name. Taken from the LF "stage_4" (full leg
# chain) entry of seqikpy.body_config._NMF_INITIAL_ANGLES_RAD -- the initial guess
# SeqIKPy's own optimizer uses for the complete leg chain (NOT a natural rest
# pose -- see the module docstring).
_INITIAL_GUESS_RAD_BY_DOF_TYPE = {
    "ThC_yaw": -0.45,
    "ThC_pitch": -0.07,
    "ThC_roll": 0.32,
    "CTr_pitch": -2.14,
    "CTr_roll": 1.25,
    "FTi_pitch": 1.48,
    "TiTa_pitch": 0.0,
}

# Maps flygym's (parent_link, child_link, axis) naming to SeqIKPy's DOF type names.
# SeqIKPy models exactly these seven leg DOFs per leg (thorax-coxa yaw/pitch/roll,
# coxa-trochanterfemur pitch/roll, trochanterfemur-tibia pitch, tibia-tarsus1 pitch)
# -- matching flygym's `JointPreset.ALL_BIOLOGICAL` leg DOFs up to tarsus1; tarsus2-5
# are not modeled by SeqIKPy and so are left untouched by the functions below.
_DOF_TYPE_BY_LINKS = {
    ("thorax", "coxa", "yaw"): "ThC_yaw",
    ("thorax", "coxa", "pitch"): "ThC_pitch",
    ("thorax", "coxa", "roll"): "ThC_roll",
    ("coxa", "trochanterfemur", "pitch"): "CTr_pitch",
    ("coxa", "trochanterfemur", "roll"): "CTr_roll",
    ("trochanterfemur", "tibia", "pitch"): "FTi_pitch",
    ("tibia", "tarsus1", "pitch"): "TiTa_pitch",
}


def _dof_type_for_joint(mj_model: mj.MjModel, joint_id: int) -> str | None:
    joint_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, joint_id)
    try:
        dof = JointDOF.from_name(joint_name)
    except ValueError:
        return None
    return _DOF_TYPE_BY_LINKS.get((dof.parent.link, dof.child.link, dof.axis.value))


def seqikpy_joint_bounds(mj_model: mj.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Per-qpos-element `(lower, upper)` bounds from SeqIKPy's default leg DOF ranges.

    Only affects the seven leg DOF types SeqIKPy models (see module docstring),
    applied identically to all six legs. Every other joint (e.g. tarsus2-5,
    antennae, wings, abdomen) is left unbounded (+/-inf), since SeqIKPy does not
    model them.

    Args:
        mj_model: Compiled MuJoCo model.

    Returns:
        Tuple of `(lower, upper)`, each shape `(mj_model.nq,)`, in radians.
    """
    lower = np.full(mj_model.nq, -np.inf)
    upper = np.full(mj_model.nq, np.inf)
    for joint_id in range(mj_model.njnt):
        dof_type = _dof_type_for_joint(mj_model, joint_id)
        if dof_type is None:
            continue
        qposadr = mj_model.jnt_qposadr[joint_id]
        lower_deg, upper_deg = _BOUNDS_DEG_BY_DOF_TYPE[dof_type]
        lower[qposadr] = np.radians(lower_deg)
        upper[qposadr] = np.radians(upper_deg)
    return lower, upper


def seqikpy_initial_guess_qpos(mj_model: mj.MjModel) -> np.ndarray:
    """A `qpos` vector using SeqIKPy's own IK-optimizer initial guess.

    Leg DOFs matching one of SeqIKPy's seven modeled types (see
    `seqikpy_joint_bounds`) are set to SeqIKPy's own initial guess for that
    DOF type (identical for all six legs); every other DOF is zero. Useful as
    an `initial_qpos` for `fit_qpos_to_keypoints`/`fit_qpos_trajectory_to_keypoints`.

    Not a naturally-posed default (see the module docstring) -- for that, use
    `qpos = 0` or `flygym.compose.pose.KinematicPosePreset.NEUTRAL`.

    Args:
        mj_model: Compiled MuJoCo model.

    Returns:
        `qpos` array, shape `(mj_model.nq,)`, in radians.
    """
    qpos = np.zeros(mj_model.nq)
    for joint_id in range(mj_model.njnt):
        dof_type = _dof_type_for_joint(mj_model, joint_id)
        if dof_type is None:
            continue
        qposadr = mj_model.jnt_qposadr[joint_id]
        qpos[qposadr] = _INITIAL_GUESS_RAD_BY_DOF_TYPE[dof_type]
    return qpos
