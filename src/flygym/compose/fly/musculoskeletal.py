"""Musculoskeletal (FlyMimic) body model as a FlyGym composition element.

Unlike `NeuroMechFly` and `FlyBody`, which compose a body from meshes and YAML
rigging configs via `BaseFly`, the musculoskeletal model is a *pre-authored*
self-contained MJCF: FlyMimic's muscle-driven fly ships its own floor, lighting,
15 left-front-leg Hill-type muscles, 15 spatial tendons, and passive joint
properties. Rather than overlaying muscles onto FlyGym's composed body, the
musculoskeletal path *switches the body model*: it loads that MJCF and wraps it
so that `flygym.Simulation` and its sensor suite work unchanged.

`MusculoskeletalFly` therefore subclasses `BaseCompositionElement` directly
(not `BaseFly`) but exposes the same dicts/accessors `Simulation` reads, keyed
by the model's own MJCF element-name strings (e.g. ``"LFFemur"``,
``"joint_LFCoxa_yaw"``, ``"LFTibia_flex_93434"``), since FlyMimic's body
topology does not line up 1:1 with FlyGym's `BodySegment` names:

* ``bodyseg_to_mjcfbody``, ``bodyseg_to_mjcfgeom``
* ``jointdof_to_mjcfjoint``, ``jointdof_to_mjcfactuator_by_type``
* ``leg_to_adhesionactuator``, ``anatomicaljoint_to_mjcfsites``
* ``eyecameraname_to_mjcfcamera`` plus the ``get_*_order`` accessors.

Pair it with `MusculoskeletalWorld` (see `flygym.compose.world`), or use
`build_musculoskeletal_simulation` (defined below) for the common case. Guarded
GPU/MuJoCo-Warp helpers (`check_mjwarp_compatibility`,
`build_musculoskeletal_gpu_simulation`) live here too — they lazily import
``mujoco_warp`` so they are safe to import on machines without a GPU.
"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import dm_control.mjcf as mjcf

from flygym import assets_dir
from flygym.compose.base import BaseCompositionElement
from flygym.compose.fly.base_fly import ActuatorType

__all__ = [
    "MusculoskeletalFly",
    "MUSCULOSKELETAL_MODEL_DIR",
    "DEFAULT_MUSCULOSKELETAL_XML",
    "build_musculoskeletal_simulation",
    "MjWarpCompatibilityReport",
    "check_mjwarp_compatibility",
    "build_musculoskeletal_gpu_simulation",
]


MUSCULOSKELETAL_MODEL_DIR = assets_dir / "model/musculoskeletal"
"""Directory holding FlyMimic's musculoskeletal MJCF + meshes. (The imitation-
learning mocap clips live with the demo, in ``flygym_demo.muscle_imitation``.)"""

DEFAULT_MUSCULOSKELETAL_XML = (
    MUSCULOSKELETAL_MODEL_DIR / "best_combined_arm_damping_stiff_cvt3.xml"
)
"""FlyMimic's muscle-driven fly with passive joint stiffness + spring refs.

This is the ``arm_damping_stiff`` variant: 15 left-front-leg muscles and body
geometry, plus biologically-motivated passive joint stiffness (0.4) and
per-joint spring reference angles.
"""


# Body names whose eyes can host vision cameras, mapped to a marker color.
_EYE_BODIES = {"LEye": (0.0, 0.0, 1.0, 1.0), "REye": (1.0, 0.0, 0.0, 1.0)}


def _load_mjcf(xml_path: PathLike) -> mjcf.RootElement:
    """Parse a FlyMimic MJCF via dm_control, working around its reserved
    ``main`` default-class name (FlyMimic authors the root default as
    ``class="main"``, which dm_control disallows)."""
    xml_path = Path(xml_path)
    xml_text = xml_path.read_text()
    # dm_control reserves the default class name "main"; demote FlyMimic's
    # explicitly-named root default to the implicit (unnamed) root default.
    # Nothing in the model references class="main" by name, so inheritance is
    # unaffected.
    xml_text = xml_text.replace('<default class="main">', "<default>")
    root = mjcf.from_xml_string(xml_text, model_dir=str(xml_path.parent))
    # FlyGym's fisheye Retina renders the eye cameras at 512x450; FlyMimic's
    # model leaves the offscreen framebuffer at MuJoCo's 480px default, which
    # is too small. Enlarge it so get_raw_vision() works after add_vision().
    root.visual.get_children("global").offheight = 640
    root.visual.get_children("global").offwidth = 640
    return root


class MusculoskeletalFly(BaseCompositionElement):
    """FlyGym-compatible wrapper around FlyMimic's musculoskeletal MJCF.

    Args:
        xml_path: Path to the FlyMimic MJCF. Defaults to the bundled
            ``arm_damping_stiff`` musculoskeletal model.
        name: Logical fly name used by `Simulation` lookups.

    See the module docstring for the exposed attributes; dict keys are the
    model's own element-name strings.
    """

    def __init__(
        self,
        xml_path: PathLike = DEFAULT_MUSCULOSKELETAL_XML,
        *,
        name: str = "nmf",
    ) -> None:
        self._name = name
        self._mjcf_root = _load_mjcf(xml_path)

        # Make the simulation's reset target ("neutral") resolve: rename the
        # model's existing default pose keyframe.
        self._neutral_keyframe = self._mjcf_root.find("key", "default-pose")
        if self._neutral_keyframe is not None:
            self._neutral_keyframe.name = "neutral"
        else:
            self._neutral_keyframe = self._mjcf_root.keyframe.add(
                "key", name="neutral", time=0
            )

        # Build the tracking dicts that Simulation introspects. Like
        # `BaseFly`, ``bodyseg_to_mjcfgeom`` maps each segment to the *list* of
        # its geoms (Simulation and the sensor APIs iterate over them).
        self.bodyseg_to_mjcfbody: dict[str, mjcf.Element] = {}
        self.bodyseg_to_mjcfgeom: dict[str, list[mjcf.Element]] = {}
        for body in self._mjcf_root.find_all("body"):
            if body.name is None:
                continue
            self.bodyseg_to_mjcfbody[body.name] = body
            geoms = body.find_all("geom", immediate_children_only=True)
            if geoms:
                self.bodyseg_to_mjcfgeom[body.name] = list(geoms)

        self.jointdof_to_mjcfjoint: dict[str, mjcf.Element] = {
            j.name: j for j in self._mjcf_root.find_all("joint") if j.name is not None
        }

        self.jointdof_to_mjcfactuator_by_type = {ty: {} for ty in ActuatorType}
        for actuator in self._mjcf_root.find_all("actuator"):
            if actuator.name is None:
                continue
            ty = self._classify_actuator(actuator)
            self.jointdof_to_mjcfactuator_by_type[ty][actuator.name] = actuator

        # FlyMimic has no adhesion, no anatomical-joint sites, no eye cameras
        # by default.
        self.leg_to_adhesionactuator: dict[str, mjcf.Element] = {}
        self.anatomicaljoint_to_mjcfsites: dict[str, mjcf.Element] = {}
        self.eyecameraname_to_mjcfcamera: dict[str, mjcf.Element] = {}
        self.cameraname_to_mjcfcamera: dict[str, mjcf.Element] = {}

    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _classify_actuator(actuator: mjcf.Element) -> ActuatorType:
        """Best-effort mapping of an MJCF actuator element to an ActuatorType."""
        # Muscles are <general> with dyntype=muscle (set directly or via class).
        dyntype = getattr(actuator, "dyntype", None)
        dclass = getattr(actuator, "dclass", None)
        class_name = getattr(dclass, "dclass", None) if dclass is not None else None
        if dyntype == "muscle" or class_name == "muscle":
            return ActuatorType.MUSCLE
        tag = actuator.tag
        if tag in {ty.value for ty in ActuatorType}:
            return ActuatorType(tag)
        # <general> motors etc. fall back to MOTOR.
        return ActuatorType.MOTOR

    # ---- Fly-compatible accessors used by Simulation / the imitation env ----

    def get_bodysegs_order(self) -> list[str]:
        return list(self.bodyseg_to_mjcfbody.keys())

    def get_jointdofs_order(self) -> list[str]:
        return list(self.jointdof_to_mjcfjoint.keys())

    def get_actuated_jointdofs_order(
        self, actuator_type: "ActuatorType | str"
    ) -> list[str]:
        actuator_type = ActuatorType(actuator_type)
        return list(self.jointdof_to_mjcfactuator_by_type[actuator_type].keys())

    def get_sites_order(self) -> list[str]:
        return list(self.anatomicaljoint_to_mjcfsites.keys())

    def get_legs_order(self) -> list[str]:
        # No ground-contact leg grouping is defined for the musculoskeletal model.
        return []

    @property
    def muscle_names(self) -> list[str]:
        """Names of the muscle actuators, in MJCF order."""
        return self.get_actuated_jointdofs_order(ActuatorType.MUSCLE)

    def add_vision(
        self,
        *,
        fovy: float = 145.0,
        draw_sensor_markers: bool = False,
    ) -> dict[str, mjcf.Element]:
        """Attach left/right eye cameras to the model's eye bodies.

        Enables `Simulation.get_raw_vision` / `get_ommatidia_readouts`. Note
        that FlyGym's fisheye `Retina` is calibrated for FlyGym's own eye
        placement, so ommatidia readouts on this body are approximate.
        """
        added: dict[str, mjcf.Element] = {}
        for eye_body_name, rgba in _EYE_BODIES.items():
            body = self.bodyseg_to_mjcfbody.get(eye_body_name)
            if body is None:
                continue
            cam = body.add(
                "camera",
                name=f"{eye_body_name}_camera",
                mode="fixed",
                # Point the camera laterally outward from each eye.
                euler=(0.0, 0.0, 0.0),
                fovy=fovy,
            )
            if draw_sensor_markers:
                body.add(
                    "site",
                    name=f"{eye_body_name}_marker",
                    type="sphere",
                    size=(0.02, 0.02, 0.02),
                    rgba=rgba,
                    group=1,
                )
            added[eye_body_name] = cam
        self.eyecameraname_to_mjcfcamera.update(added)
        return added


# ---------------------------------------------------------------------------
# Simulation factories
# ---------------------------------------------------------------------------


def build_musculoskeletal_simulation(
    *,
    xml_path: PathLike = DEFAULT_MUSCULOSKELETAL_XML,
    name: str = "nmf",
    add_vision: bool = False,
):
    """Convenience: build a `MusculoskeletalFly` + `MusculoskeletalWorld` +
    `Simulation`.

    Equivalent to the standard composition flow::

        fly = MusculoskeletalFly(xml_path, name=name)
        world = MusculoskeletalWorld(fly)
        sim = Simulation(world)

    Returns:
        ``(simulation, fly)`` where ``simulation`` is a `flygym.Simulation`
        and ``fly`` is the `MusculoskeletalFly` (handy for reading
        ``muscle_names``).
    """
    from flygym.simulation import Simulation
    from flygym.compose.world.musculoskeletal import MusculoskeletalWorld

    fly = MusculoskeletalFly(xml_path, name=name)
    if add_vision:
        fly.add_vision()
    world = MusculoskeletalWorld(fly)
    sim = Simulation(world)
    return sim, fly


# ---------------------------------------------------------------------------
# GPU (MuJoCo-Warp) helpers
# ---------------------------------------------------------------------------
#
# Everything below is guarded so it imports and runs on machines *without*
# ``warp`` / ``mujoco_warp`` (e.g. macOS / no NVIDIA GPU): the compatibility
# probe reports "unavailable" instead of raising, and the GPU-sim factory
# raises a clear, actionable error only when actually called.
#
# Why a probe at all? ``mujoco_warp`` is a from-scratch GPU reimplementation of
# MuJoCo with an evolving feature set. The muscle model leans on three features
# that are not guaranteed to be ported: Hill-type muscle actuators
# (``mjDYN_MUSCLE`` + ``mjGAIN_MUSCLE``/``mjBIAS_MUSCLE``), spatial tendons, and
# joint-equality constraints (the locked right-front-leg joints). The probe runs
# ``mjw.put_model`` on the compiled model, which is where unsupported features
# surface.


@dataclass
class MjWarpCompatibilityReport:
    """Outcome of probing whether MuJoCo-Warp accepts the muscle model.

    Attributes:
        mjwarp_available: Whether ``mujoco_warp`` could be imported.
        put_model_ok: Whether ``mjw.put_model`` succeeded on the compiled
            model. ``None`` if mjwarp was unavailable (probe not run).
        error: The exception text if ``put_model`` failed, else ``None``.
        message: A human-readable summary.
    """

    mjwarp_available: bool
    put_model_ok: bool | None
    error: str | None
    message: str

    def __bool__(self) -> bool:
        """True only if mjwarp is available *and* accepted the model."""
        return bool(self.mjwarp_available and self.put_model_ok)


def check_mjwarp_compatibility(
    fly: "MusculoskeletalFly | None" = None,
    *,
    xml_path: PathLike = DEFAULT_MUSCULOSKELETAL_XML,
) -> MjWarpCompatibilityReport:
    """Probe whether MuJoCo-Warp can ingest the muscle model.

    Safe to call anywhere: if ``mujoco_warp`` is not installed (the usual case
    off a CUDA machine), it returns a report with ``mjwarp_available=False``
    rather than raising.

    Args:
        fly: A `MusculoskeletalFly` to probe. If None, a default one is built.
        xml_path: XML to use when ``fly`` is None.

    Returns:
        A `MjWarpCompatibilityReport`.
    """
    try:
        import mujoco_warp as mjw
    except ImportError:
        return MjWarpCompatibilityReport(
            mjwarp_available=False,
            put_model_ok=None,
            error=None,
            message=(
                "mujoco_warp is not installed; cannot probe GPU compatibility. "
                "This is expected without an NVIDIA CUDA GPU. Install the "
                "'[warp]' extra on a Linux+CUDA machine to enable the GPU path."
            ),
        )

    if fly is None:
        fly = MusculoskeletalFly(xml_path)
    mj_model, _ = fly.compile()
    try:
        mjw.put_model(mj_model)
    except Exception as e:  # noqa: BLE001 - we want to report any failure mode
        return MjWarpCompatibilityReport(
            mjwarp_available=True,
            put_model_ok=False,
            error=f"{type(e).__name__}: {e}",
            message=(
                "mujoco_warp is installed but rejected the muscle model. The "
                "likely culprits are muscle actuators, spatial tendons, or "
                "joint-equality constraints not yet supported by this mjwarp "
                "version. See the error field."
            ),
        )
    return MjWarpCompatibilityReport(
        mjwarp_available=True,
        put_model_ok=True,
        error=None,
        message="mujoco_warp accepted the muscle model (put_model succeeded).",
    )


def build_musculoskeletal_gpu_simulation(
    n_worlds: int,
    *,
    xml_path: PathLike = DEFAULT_MUSCULOSKELETAL_XML,
    name: str = "nmf",
    add_vision: bool = False,
    **gpu_kwargs,
):
    """Build a `GPUSimulation` of the muscle model with ``n_worlds`` parallel
    copies (for vectorized RL on a CUDA machine).

    Raises a clear `ImportError` if the ``[warp]`` extra / an NVIDIA GPU is not
    available. The non-GPU code path (`build_musculoskeletal_simulation`) is
    unaffected.

    Returns:
        ``(gpu_simulation, fly)``.
    """
    try:
        from flygym.warp.simulation import GPUSimulation
    except ImportError as e:
        raise ImportError(
            "GPU simulation requires the '[warp]' extra (warp-lang + "
            "mujoco_warp) and an NVIDIA CUDA GPU. On such a machine, install "
            "with `pip install 'flygym[warp]'`."
        ) from e
    from flygym.compose.world.musculoskeletal import MusculoskeletalWorld

    fly = MusculoskeletalFly(xml_path, name=name)
    if add_vision:
        fly.add_vision()
    world = MusculoskeletalWorld(fly)
    sim = GPUSimulation(world, n_worlds=n_worlds, **gpu_kwargs)
    return sim, fly
