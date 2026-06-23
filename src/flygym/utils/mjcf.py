"""Helpers for building MuJoCo models with the native ``mujoco.MjSpec`` API.

FlyGym composes its models with ``mujoco.MjSpec`` (MuJoCo's native model-editing
API) rather than ``dm_control.mjcf`` (PyMJCF). MjSpec's element constructors take
typed enum values where PyMJCF accepted strings, and it has no high-level actuator
"shortcut" classes (``position``/``velocity``/...). This module provides the thin
translation layer FlyGym needs: string-to-enum lookups and an ``add_actuator``
helper that reproduces MuJoCo's actuator shortcuts.
"""

from os import PathLike
from typing import Any

import mujoco as mj
import yaml

__all__ = [
    "GEOM_TYPES",
    "JOINT_TYPES",
    "TEXTURE_TYPES",
    "CAMERA_MODES",
    "add_actuator",
    "add_texture",
    "add_material",
    "set_mujoco_globals",
]

# String -> enum lookups for the ``type`` attributes that PyMJCF accepted as
# strings but MjSpec requires as enum values.
GEOM_TYPES = {
    "plane": mj.mjtGeom.mjGEOM_PLANE,
    "hfield": mj.mjtGeom.mjGEOM_HFIELD,
    "sphere": mj.mjtGeom.mjGEOM_SPHERE,
    "capsule": mj.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mj.mjtGeom.mjGEOM_ELLIPSOID,
    "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
    "box": mj.mjtGeom.mjGEOM_BOX,
    "mesh": mj.mjtGeom.mjGEOM_MESH,
}

JOINT_TYPES = {
    "free": mj.mjtJoint.mjJNT_FREE,
    "ball": mj.mjtJoint.mjJNT_BALL,
    "slide": mj.mjtJoint.mjJNT_SLIDE,
    "hinge": mj.mjtJoint.mjJNT_HINGE,
}

TEXTURE_TYPES = {
    "2d": mj.mjtTexture.mjTEXTURE_2D,
    "cube": mj.mjtTexture.mjTEXTURE_CUBE,
    "skybox": mj.mjtTexture.mjTEXTURE_SKYBOX,
}

CAMERA_MODES = {
    "fixed": mj.mjtCamLight.mjCAMLIGHT_FIXED,
    "track": mj.mjtCamLight.mjCAMLIGHT_TRACK,
    "trackcom": mj.mjtCamLight.mjCAMLIGHT_TRACKCOM,
    "targetbody": mj.mjtCamLight.mjCAMLIGHT_TARGETBODY,
    "targetbodycom": mj.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM,
}

_BUILTIN_TYPES = {
    "none": mj.mjtBuiltin.mjBUILTIN_NONE,
    "gradient": mj.mjtBuiltin.mjBUILTIN_GRADIENT,
    "checker": mj.mjtBuiltin.mjBUILTIN_CHECKER,
    "flat": mj.mjtBuiltin.mjBUILTIN_FLAT,
}

_MARK_TYPES = {
    "none": mj.mjtMark.mjMARK_NONE,
    "edge": mj.mjtMark.mjMARK_EDGE,
    "cross": mj.mjtMark.mjMARK_CROSS,
    "random": mj.mjtMark.mjMARK_RANDOM,
}


def add_texture(spec: mj.MjSpec, **params: Any) -> mj.MjsTexture:
    """Add a texture, converting the ``type``/``builtin``/``mark`` string attributes
    (which PyMJCF accepted as strings) to MjSpec enums."""
    if "type" in params:
        params["type"] = TEXTURE_TYPES[params["type"]]
    if "builtin" in params:
        params["builtin"] = _BUILTIN_TYPES[params["builtin"]]
    if "mark" in params:
        params["mark"] = _MARK_TYPES[params["mark"]]
    return spec.add_texture(**params)


def add_material(
    spec: mj.MjSpec, *, texture: str | None = None, **params: Any
) -> mj.MjsMaterial:
    """Add a material, optionally linking a texture by name in the RGB texture role
    (PyMJCF exposed this as a single ``material.texture`` attribute)."""
    material = spec.add_material(**params)
    if texture is not None:
        material.textures[int(mj.mjtTextureRole.mjTEXROLE_RGB)] = texture
    return material


_INTEGRATORS = {
    "Euler": mj.mjtIntegrator.mjINT_EULER,
    "RK4": mj.mjtIntegrator.mjINT_RK4,
    "implicit": mj.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mj.mjtIntegrator.mjINT_IMPLICITFAST,
}

_SOLVERS = {
    "PGS": mj.mjtSolver.mjSOL_PGS,
    "CG": mj.mjtSolver.mjSOL_CG,
    "Newton": mj.mjtSolver.mjSOL_NEWTON,
}

_CONES = {
    "pyramidal": mj.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mj.mjtCone.mjCONE_ELLIPTIC,
}

_JACOBIANS = {
    "dense": mj.mjtJacobian.mjJAC_DENSE,
    "sparse": mj.mjtJacobian.mjJAC_SPARSE,
    "auto": mj.mjtJacobian.mjJAC_AUTO,
}

# Names used under ``option.flag`` in the globals YAML -> enable/disable bit.
# Both dicts are derived from the enums so they track MuJoCo's own additions and
# removals. Some features migrate between the two over versions (e.g. ``multiccd``
# and ``island`` became default-on and moved from enable bits to disable bits in
# MuJoCo 3.x); ``_apply_option_flags`` resolves the YAML state against whichever
# dict the flag currently lives in.
_ENABLE_BITS = {
    name[len("mjENBL_") :].lower(): getattr(mj.mjtEnableBit, name)
    for name in dir(mj.mjtEnableBit)
    if name.startswith("mjENBL_")
}
_DISABLE_BITS = {
    name[len("mjDSBL_") :].lower(): getattr(mj.mjtDisableBit, name)
    for name in dir(mj.mjtDisableBit)
    if name.startswith("mjDSBL_")
}


def add_actuator(
    spec: mj.MjSpec,
    kind: str,
    *,
    name: str,
    joint: str | None = None,
    body: str | None = None,
    tendon: str | None = None,
    site: str | None = None,
    forcelimited: bool | None = None,
    forcerange: tuple[float, float] | None = None,
    ctrllimited: bool | None = None,
    ctrlrange: tuple[float, float] | None = None,
    gear: float | None = None,
    kp: float | None = None,
    kv: float | None = None,
    gain: float | None = None,
    **kwargs: Any,
) -> mj.MjsActuator:
    """Add an actuator to ``spec``, reproducing MuJoCo's actuator shortcuts.

    MjSpec only exposes the low-level "general" actuator (gain/bias/dyn/trn). This
    helper expands the familiar ``motor``/``position``/``velocity``/``adhesion``/
    ``general`` shortcuts into the equivalent low-level parameters, matching what
    the XML compiler (and PyMJCF) produced. The expansions were verified against
    ``dm_control.mjcf`` compiled output.

    Exactly one transmission target (``joint``, ``body``, ``tendon`` or ``site``)
    must be given.

    Args:
        spec: The spec to add the actuator to.
        kind: Actuator shortcut name (e.g. ``"position"``, ``"motor"``).
        name: Actuator name.
        joint: Transmission target joint name (give exactly one of
            ``joint``/``body``/``tendon``/``site``).
        body: Transmission target body name.
        tendon: Transmission target tendon name.
        site: Transmission target site name.
        forcelimited: Whether actuator force is clamped to ``forcerange``.
        forcerange: Min/max actuator force.
        ctrllimited: Whether the control input is clamped to ``ctrlrange``.
        ctrlrange: Min/max control input.
        gear: Transmission gear ratio (``gear[0]``).
        kp: Position/intvelocity gain.
        kv: Position/velocity/intvelocity damping.
        gain: Gain (``gainprm[0]``) for adhesion, motor, and general actuators.
        **kwargs: Extra low-level attributes set directly on the actuator (e.g.
            ``gainprm``, ``biasprm`` for ``general``).

    Returns:
        The created ``MjsActuator``.
    """
    params: dict[str, Any] = {"name": name}

    # Transmission target.
    targets = {"joint": joint, "body": body, "tendon": tendon, "site": site}
    set_targets = {k: v for k, v in targets.items() if v is not None}
    if len(set_targets) != 1:
        raise ValueError(
            f"Exactly one transmission target required, got {list(set_targets)}."
        )
    trn, target = next(iter(set_targets.items()))
    trntype = {
        "joint": mj.mjtTrn.mjTRN_JOINT,
        "body": mj.mjtTrn.mjTRN_BODY,
        "tendon": mj.mjtTrn.mjTRN_TENDON,
        "site": mj.mjtTrn.mjTRN_SITE,
    }[trn]
    params["trntype"] = trntype
    params["target"] = target

    # Shortcut expansion (gain/bias/dyn). Verified against PyMJCF output. MjSpec
    # requires gainprm/biasprm to be length 10, so leading entries are padded.
    if kind == "motor" or kind == "general":
        gain_v = 1.0 if gain is None else gain
        params["gaintype"] = mj.mjtGain.mjGAIN_FIXED
        params["gainprm"] = _prm(gain_v)
        params["biastype"] = mj.mjtBias.mjBIAS_NONE
    elif kind == "position":
        kp_v = 1.0 if kp is None else kp
        kv_v = 0.0 if kv is None else kv
        params["gaintype"] = mj.mjtGain.mjGAIN_FIXED
        params["gainprm"] = _prm(kp_v)
        params["biastype"] = mj.mjtBias.mjBIAS_AFFINE
        params["biasprm"] = _prm(0.0, -kp_v, -kv_v)
    elif kind == "velocity":
        kv_v = 1.0 if kv is None else kv
        params["gaintype"] = mj.mjtGain.mjGAIN_FIXED
        params["gainprm"] = _prm(kv_v)
        params["biastype"] = mj.mjtBias.mjBIAS_AFFINE
        params["biasprm"] = _prm(0.0, 0.0, -kv_v)
    elif kind == "adhesion":
        gain_v = 1.0 if gain is None else gain
        params["gaintype"] = mj.mjtGain.mjGAIN_FIXED
        params["gainprm"] = _prm(gain_v)
        params["biastype"] = mj.mjtBias.mjBIAS_NONE
        if ctrlrange is None:
            ctrlrange = (0.0, 1.0)
        if ctrllimited is None:
            ctrllimited = True
    else:
        raise NotImplementedError(f"Actuator shortcut '{kind}' is not supported.")

    # Common attributes.
    if forcerange is not None:
        params["forcerange"] = list(forcerange)
    if forcelimited is not None:
        params["forcelimited"] = _limited_enum(forcelimited)
    if ctrlrange is not None:
        params["ctrlrange"] = list(ctrlrange)
    if ctrllimited is not None:
        params["ctrllimited"] = _limited_enum(ctrllimited)
    if gear is not None:
        params["gear"] = [gear, 0, 0, 0, 0, 0]

    params.update(kwargs)
    return spec.add_actuator(**params)


def _prm(*values: float) -> list[float]:
    """Pad gain/bias prm leading entries out to the length-10 array MjSpec wants."""
    return list(values) + [0.0] * (10 - len(values))


def _limited_enum(value: bool) -> int:
    """MjSpec ``*limited`` fields use the auto/false/true tri-state enum."""
    return mj.mjtLimited.mjLIMITED_TRUE if value else mj.mjtLimited.mjLIMITED_FALSE


def set_mujoco_globals(spec: mj.MjSpec, mujoco_globals_path: PathLike) -> None:
    """Load a YAML file of global MuJoCo settings and apply them to a spec.

    Handles the compiler/option/statistic/visual groups, including the cases that
    differ from a flat attribute set: ``compiler.angle`` maps to ``compiler.degree``,
    ``option.flag`` maps to the enable/disable bitmasks, ``option.integrator`` and
    ``option.solver`` are string-named enums, ``statistic`` maps to ``spec.stat``,
    and ``visual.global`` maps to ``spec.visual.global_`` (``global`` is reserved).

    Args:
        spec: The spec to update.
        mujoco_globals_path: Path to the YAML file of global parameter overrides.
    """
    with open(mujoco_globals_path) as f:
        cfg = yaml.safe_load(f)

    for group, params in cfg.items():
        if group == "compiler":
            _apply_compiler(spec.compiler, params)
        elif group == "option":
            _apply_option(spec.option, params)
        elif group == "statistic":
            _set_attrs(spec.stat, params)
        elif group == "visual":
            _apply_visual(spec.visual, params)
        elif group == "size":
            # The <size> directives (njmax, nconmax, nkey) are legacy: MuJoCo 3.x
            # allocates these dynamically, so they are no-ops and intentionally
            # ignored here.
            continue
        else:
            raise ValueError(f"Unsupported global settings group: '{group}'.")


def _apply_compiler(compiler: Any, params: dict[str, Any]) -> None:
    for key, value in params.items():
        if key == "angle":
            compiler.degree = value == "degree"
        else:
            setattr(compiler, key, _coerce(value))


def _apply_option(option: Any, params: dict[str, Any]) -> None:
    for key, value in params.items():
        if key == "integrator":
            option.integrator = _INTEGRATORS[value]
        elif key == "solver":
            option.solver = _SOLVERS[value]
        elif key == "cone":
            option.cone = _CONES[value]
        elif key == "jacobian":
            option.jacobian = _JACOBIANS[value]
        elif key == "flag":
            _apply_option_flags(option, value)
        else:
            setattr(option, key, _coerce(value))


def _apply_option_flags(option: Any, flags: dict[str, str]) -> None:
    for flag_name, state in flags.items():
        enable = state == "enable"
        if flag_name in _ENABLE_BITS and _ENABLE_BITS[flag_name] is not None:
            bit = int(_ENABLE_BITS[flag_name])
            if enable:
                option.enableflags |= bit
            else:
                option.enableflags &= ~bit
        elif flag_name in _DISABLE_BITS:
            bit = int(_DISABLE_BITS[flag_name])
            # An "enabled" feature is one that is NOT in the disable mask.
            if enable:
                option.disableflags &= ~bit
            else:
                option.disableflags |= bit
        else:
            raise ValueError(f"Unknown MuJoCo option flag: '{flag_name}'.")


def _apply_visual(visual: Any, params: dict[str, Any]) -> None:
    for key, value in params.items():
        # `global` is a Python keyword; MjSpec exposes it as `global_`.
        target = getattr(visual, "global_") if key == "global" else getattr(visual, key)
        _set_attrs(target, value)


def _set_attrs(struct: Any, params: dict[str, Any]) -> None:
    for key, value in params.items():
        setattr(struct, key, _coerce(value))


def _coerce(value: Any) -> Any:
    """Coerce YAML scalars/lists into the numeric types MjSpec fields expect.

    The globals YAML stores some numbers as strings (e.g. ``extent: '5'``); MjSpec
    fields are strongly typed, so cast where possible while leaving genuine strings
    (e.g. ``eulerseq: XYZ``) untouched.
    """
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return _num(value)
        except ValueError:
            pass
        # Space-separated numeric strings (e.g. "0.5 0.5 0.5") map to vectors.
        tokens = value.split()
        if len(tokens) > 1:
            try:
                return [_num(t) for t in tokens]
            except ValueError:
                return value
        return value
    return value


def _num(token: str) -> int | float:
    """Parse a numeric string, returning an int for integral values. MjSpec's
    integer-typed fields (e.g. ``iterations``) reject Python floats, while its
    float-typed fields accept ints, so preferring int when exact is always safe."""
    f = float(token)
    return int(f) if f.is_integer() else f
