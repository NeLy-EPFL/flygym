"""
!!! warning "Experimental"

    Support for the FlyMimic musculoskeletal body model is
    **experimental**. The API may change in future releases, and only
    the left-front leg is muscle-driven in the current model. Not all
    features available for the default `NeuroMechFly` model are
    currently supported (e.g. per-leg ground-contact sensors).

Musculoskeletal (FlyMimic) body model as a FlyGym composition element.

Unlike `NeuroMechFly` and `FlyBody`, which compose a body from meshes and YAML
rigging configs via `BaseFly`, the musculoskeletal model is a *pre-authored*
self-contained MJCF: FlyMimic's muscle-driven fly ships its own floor, lighting,
15 left-front-leg Hill-type muscles, 15 spatial tendons, and passive joint
properties. Rather than overlaying muscles onto FlyGym's composed body, the
musculoskeletal path *switches the body model*: it loads that MJCF into a
``mujoco.MjSpec`` (FlyGym's model-editing backend since the v2.1.0 PyMJCF ->
MjSpec migration) and wraps it so that `flygym.Simulation` and its sensor suite
work unchanged.

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
`build_musculoskeletal_simulation` (defined below) for the common case.
`build_musculoskeletal_simulation` returns a plain `flygym.Simulation` (CPU,
single world) — **not** ``flygym.warp.GPUSimulation``. Guarded GPU/MuJoCo-Warp
helpers (`check_mjwarp_compatibility`, `build_musculoskeletal_gpu_simulation`)
live here too — they lazily import ``mujoco_warp`` so they are safe to import
on machines without a GPU, and require the ``[warp]`` extra to actually run.
"""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco as mj

from flygym import assets_dir
from flygym.compose.base import BaseCompositionElement
from flygym.compose.fly.base_fly import ActuatorType
from flygym.utils.assets_lazy_loading import lazy_load_asset_dir
from flygym.utils.mjcf import CAMERA_MODES, GEOM_TYPES

if TYPE_CHECKING:
    from flygym.simulation import Simulation
    from flygym.warp.simulation import GPUSimulation

__all__ = [
    "MusculoskeletalFly",
    "MUSCULOSKELETAL_MODEL_DIR",
    "MUSCULOSKELETAL_MESH_DIR",
    "DEFAULT_MUSCULOSKELETAL_XML",
    "DEFAULT_SCENE_CAMERA",
    "build_musculoskeletal_simulation",
    "MjWarpCompatibilityReport",
    "check_mjwarp_compatibility",
    "build_musculoskeletal_gpu_simulation",
]


DEFAULT_SCENE_CAMERA = "scene"
"""Name assigned to FlyMimic's (otherwise unnamed) world camera so it can be
selected for rendering via ``Simulation.set_renderer`` — e.g. when recording a
rollout video of a trained policy."""


MUSCULOSKELETAL_MODEL_DIR = assets_dir / "model/musculoskeletal"
"""Directory holding FlyMimic's musculoskeletal MJCF. The body meshes it
references are *not* bundled here (see `MUSCULOSKELETAL_MESH_DIR`), and the
imitation-learning mocap clips live with the demo, in
``flygym_demo.muscle_imitation``."""

# Mesh path relative to the flygym_assets/ dir on the S3 bucket and local cache
# dir. FlyMimic's body meshes are large (~14 MB) so, like the FlyBody and
# fullsize NeuroMechFly meshes, they are not shipped with the package; they are
# downloaded and cached on first use (see flygym.utils.assets_lazy_loading).
MUSCULOSKELETAL_MESH_DIR = "neuromechfly_musculoskeletal_meshes_20260623a"

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


def _load_mjcf(xml_path: PathLike) -> mj.MjSpec:
    """Parse a FlyMimic MJCF into a ``mujoco.MjSpec``.

    Unlike ``dm_control.mjcf``, MuJoCo's native ``MjSpec`` does not reserve the
    ``main`` default-class name (``main`` is in fact MuJoCo's own implicit root
    default), so FlyMimic's ``<default class="main">`` loads as-is and no XML
    rewriting is needed.
    """
    xml_path = Path(xml_path)
    spec = mj.MjSpec.from_file(str(xml_path))
    # FlyGym's fisheye Retina renders the eye cameras at 512x450; FlyMimic's
    # model leaves the offscreen framebuffer at MuJoCo's 480px default, which
    # is too small. Enlarge it so get_raw_vision() works after add_vision().
    # (`global` is a Python keyword; MjSpec exposes it as `global_`.)
    spec.visual.global_.offheight = 640
    spec.visual.global_.offwidth = 640
    # FlyMimic references its meshes/textures with paths relative to the XML
    # directory (via `meshdir`/`texturedir`). MjSpec resolves these at compile
    # time, but they would not survive being copied into another spec or exported
    # (e.g. `save_xml_with_assets`), which resolve `file` against the process
    # working directory. Rewrite them to absolute paths so the model is portable.
    # The body meshes are not bundled with the package; they are pulled from S3
    # and cached on first use (see `_resolve_mesh_files`).
    _resolve_mesh_files(spec, xml_path.parent / spec.meshdir)
    _absolutize_asset_paths(spec, xml_path.parent / spec.texturedir, spec.textures)
    spec.meshdir = ""
    spec.texturedir = ""
    return spec


def _absolutize_asset_paths(spec: mj.MjSpec, basedir: Path, assets) -> None:
    """Rewrite each asset's relative ``file`` to an absolute path under ``basedir``."""
    for asset in assets:
        if asset.file and not Path(asset.file).is_absolute():
            asset.file = str((basedir / asset.file).resolve())


def _resolve_mesh_files(spec: mj.MjSpec, local_meshdir: Path) -> None:
    """Rewrite each mesh's relative ``file`` to an absolute path.

    FlyMimic's high-resolution body meshes are large, so (like the FlyBody and
    fullsize NeuroMechFly meshes) they are not bundled with the package: they
    live on the FlyGym S3 bucket and are downloaded and cached on first use, then
    resolved by file name from the cache (see `MUSCULOSKELETAL_MESH_DIR`).

    A mesh that *is* present under ``local_meshdir`` -- e.g. a custom XML that
    ships its own meshes -- is used as-is and never triggers a download, so the
    download happens only for the default (bundled-XML, remote-mesh) model.
    """
    cache_dir: Path | None = None
    for mesh in spec.meshes:
        if not mesh.file or Path(mesh.file).is_absolute():
            continue
        local = (local_meshdir / mesh.file).resolve()
        if local.is_file():
            mesh.file = str(local)
            continue
        if cache_dir is None:
            cache_dir = lazy_load_asset_dir(MUSCULOSKELETAL_MESH_DIR)
        mesh.file = str((cache_dir / Path(mesh.file).name).resolve())


class MusculoskeletalFly(BaseCompositionElement):
    """FlyGym-compatible wrapper around FlyMimic's musculoskeletal MJCF.

    The musculoskeletal body model is published in:

        Ozdil, P. G., et al. (2026). Musculoskeletal simulation of limb
        movement biomechanics in *Drosophila melanogaster*. *ICLR 2026*.
        https://arxiv.org/abs/2509.06426

    Source code for the original model: https://github.com/gizemozd/FlyMimic

    !!! info "Plain flygym — not GPU-accelerated"

        `MusculoskeletalFly` works with plain `flygym.Simulation` (CPU,
        single world). It does **not** require `flygym.warp`.

    Unlike `NeuroMechFly` and `FlyBody`, which compose a body from meshes
    and YAML rigging configs via `BaseFly`, this class loads FlyMimic's
    pre-authored self-contained MJCF and wraps it so that `Simulation` and
    its sensor suite work unchanged. Dict keys on all tracking attributes
    are the model's own MJCF element-name strings (e.g. ``"LFFemur"``,
    ``"joint_LFCoxa_yaw"``).

    Args:
        xml_path: Path to the FlyMimic MJCF. Defaults to the bundled
            ``arm_damping_stiff`` musculoskeletal model
            (`DEFAULT_MUSCULOSKELETAL_XML`).
        name: Logical fly name used by `Simulation` lookups. Defaults to
            ``"nmf"``.

    Attributes:
        bodyseg_to_mjcfbody: Maps body-segment name → MJCF body element.
        bodyseg_to_mjcfgeom: Maps body-segment name → list of MJCF geom
            elements (one entry per geom on that body).
        jointdof_to_mjcfjoint: Maps joint-DoF name → MJCF joint element.
        jointdof_to_mjcfactuator_by_type: Maps `ActuatorType` → dict of
            actuator-name → MJCF actuator element.
        leg_to_adhesionactuator: Always empty (FlyMimic has no adhesion).
        anatomicaljoint_to_mjcfsites: Always empty.
        eyecameraname_to_mjcfcamera: Camera elements added via `add_vision`;
            empty until `add_vision` is called.
        cameraname_to_mjcfcamera: All scene cameras, including the world
            camera named `DEFAULT_SCENE_CAMERA`.
        muscle_names: Names of the 15 Hill-type muscle actuators, in MJCF
            order (read-only property).
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
        self._neutral_keyframe = self._mjcf_root.key("default-pose")
        if self._neutral_keyframe is not None:
            self._neutral_keyframe.name = "neutral"
        else:
            self._neutral_keyframe = self._mjcf_root.add_key(name="neutral", time=0)

        # Build the tracking dicts that Simulation introspects. Like
        # `BaseFly`, ``bodyseg_to_mjcfgeom`` maps each segment to the *list* of
        # its geoms (Simulation and the sensor APIs iterate over them).
        # ``spec.bodies`` includes the unnamed worldbody (its MjSpec name is
        # ``"world"``); skip it so only real body segments are registered. Its
        # only direct geom is the floor, which the world exposes via
        # ``ground_geoms`` instead.
        worldbody = self._mjcf_root.worldbody
        self.bodyseg_to_mjcfbody: dict[str, mj.MjsBody] = {}
        self.bodyseg_to_mjcfgeom: dict[str, list[mj.MjsGeom]] = {}
        for body in self._mjcf_root.bodies:
            if body.name == worldbody.name or not body.name:
                continue
            self.bodyseg_to_mjcfbody[body.name] = body
            geoms = list(body.geoms)
            if geoms:
                self.bodyseg_to_mjcfgeom[body.name] = geoms

        self.jointdof_to_mjcfjoint: dict[str, mj.MjsJoint] = {
            j.name: j for j in self._mjcf_root.joints if j.name
        }

        self.jointdof_to_mjcfactuator_by_type = {ty: {} for ty in ActuatorType}
        for actuator in self._mjcf_root.actuators:
            if not actuator.name:
                continue
            ty = self._classify_actuator(actuator)
            self.jointdof_to_mjcfactuator_by_type[ty][actuator.name] = actuator

        # FlyMimic has no adhesion, no anatomical-joint sites, no eye cameras
        # by default.
        self.leg_to_adhesionactuator: dict[str, mj.MjsActuator] = {}
        self.anatomicaljoint_to_mjcfsites: dict[str, mj.MjsSite] = {}
        self.eyecameraname_to_mjcfcamera: dict[str, mj.MjsCamera] = {}

        # Register scene cameras so they can be selected for rendering. FlyMimic
        # ships a single unnamed world camera; name any unnamed camera so
        # ``Simulation.set_renderer(DEFAULT_SCENE_CAMERA)`` resolves it. (Eye
        # cameras are added later, named, via add_vision().) MjSpec gives an
        # unnamed camera an empty-string name rather than ``None``.
        self.cameraname_to_mjcfcamera: dict[str, mj.MjsCamera] = {}
        for idx, cam in enumerate(self._mjcf_root.cameras):
            if not cam.name:
                cam.name = (
                    DEFAULT_SCENE_CAMERA
                    if idx == 0
                    else f"{DEFAULT_SCENE_CAMERA}_{idx}"
                )
            self.cameraname_to_mjcfcamera[cam.name] = cam

    @property
    def mjcf_root(self) -> mj.MjSpec:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _classify_actuator(actuator: mj.MjsActuator) -> ActuatorType:
        """Best-effort mapping of an MjSpec actuator element to an ActuatorType.

        MjSpec exposes every actuator as a low-level "general" element, so there
        is no shortcut tag (``motor``/``position``/...) to read. The dynamics
        type is the reliable discriminator: muscles use ``mjDYN_MUSCLE`` (FlyMimic
        sets this on the ``"muscle"`` default class, and MjSpec resolves the class
        default onto the element). Everything else falls back to ``MOTOR``, which
        is all FlyMimic's non-muscle actuators are.
        """
        if actuator.dyntype == mj.mjtDyn.mjDYN_MUSCLE:
            return ActuatorType.MUSCLE
        return ActuatorType.MOTOR

    # ---- Fly-compatible accessors used by Simulation / the imitation env ----

    def get_bodysegs_order(self) -> list[str]:
        """Return all body-segment names in MJCF order."""
        return list(self.bodyseg_to_mjcfbody.keys())

    def get_jointdofs_order(self) -> list[str]:
        """Return all joint-DoF names in MJCF order."""
        return list(self.jointdof_to_mjcfjoint.keys())

    def get_actuated_jointdofs_order(
        self, actuator_type: "ActuatorType | str"
    ) -> list[str]:
        """Return actuator names of the given type, in MJCF order.

        Args:
            actuator_type: An `ActuatorType` value or its string name (e.g.
                ``"muscle"`` or ``ActuatorType.MUSCLE``).
        """
        actuator_type = ActuatorType(actuator_type)
        return list(self.jointdof_to_mjcfactuator_by_type[actuator_type].keys())

    def get_sites_order(self) -> list[str]:
        """Return anatomical-joint site names (always empty for this model)."""
        return list(self.anatomicaljoint_to_mjcfsites.keys())

    def get_legs_order(self) -> list[str]:
        """Return leg names (always empty — no ground-contact grouping defined)."""
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
    ) -> dict[str, mj.MjsCamera]:
        """Attach left/right eye cameras to the model's eye bodies.

        Enables `Simulation.get_raw_vision` / `get_ommatidia_readouts`. Note
        that FlyGym's fisheye `Retina` is calibrated for FlyGym's own eye
        placement, so ommatidia readouts on this body are approximate.
        """
        added: dict[str, mj.MjsCamera] = {}
        for eye_body_name, rgba in _EYE_BODIES.items():
            body = self.bodyseg_to_mjcfbody.get(eye_body_name)
            if body is None:
                continue
            cam = body.add_camera(
                name=f"{eye_body_name}_camera",
                mode=CAMERA_MODES["fixed"],
                # Point the camera laterally outward from each eye.
                euler=(0.0, 0.0, 0.0),
                fovy=fovy,
            )
            if draw_sensor_markers:
                body.add_site(
                    name=f"{eye_body_name}_marker",
                    type=GEOM_TYPES["sphere"],
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
) -> "tuple[Simulation, MusculoskeletalFly]":
    """Build a `MusculoskeletalFly` + `MusculoskeletalWorld` + `Simulation`.

    Equivalent to the standard composition flow::

        fly = MusculoskeletalFly(xml_path, name=name)
        world = MusculoskeletalWorld(fly)
        sim = Simulation(world)

    Args:
        xml_path: Path to the FlyMimic MJCF. Defaults to
            `DEFAULT_MUSCULOSKELETAL_XML`.
        name: Logical fly name. Defaults to ``"nmf"``.
        add_vision: If True, attach left/right eye cameras so
            `Simulation.get_raw_vision` / `get_ommatidia_readouts` work.
            Note that ommatidia readouts are approximate because FlyGym's
            `Retina` is calibrated for its own eye placement.

    !!! info "Plain flygym — not GPU-accelerated"

        Returns a plain `flygym.Simulation` (CPU, single world). For the
        GPU path use `build_musculoskeletal_gpu_simulation` with the
        `[warp]` extra.

    Returns:
        ``(simulation, fly)`` where *simulation* is a `flygym.Simulation`
        and *fly* is the `MusculoskeletalFly` instance (useful for
        inspecting ``fly.muscle_names`` or calling ``fly.add_vision``).
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
    **gpu_kwargs: Any,
) -> "tuple[GPUSimulation, MusculoskeletalFly]":
    """Build a `GPUSimulation` of the muscle model with *n_worlds* parallel
    copies for vectorized RL on a CUDA machine.

    Call `check_mjwarp_compatibility` first to verify that the installed
    ``mujoco_warp`` version supports the muscle model's Hill-type actuators,
    spatial tendons, and joint-equality constraints.

    Args:
        n_worlds: Number of parallel simulation worlds.
        xml_path: Path to the FlyMimic MJCF. Defaults to
            `DEFAULT_MUSCULOSKELETAL_XML`.
        name: Logical fly name. Defaults to ``"nmf"``.
        add_vision: Attach eye cameras (approximate; see
            `build_musculoskeletal_simulation`).
        **gpu_kwargs: Forwarded to `GPUSimulation`.

    Returns:
        ``(gpu_simulation, fly)`` — a `GPUSimulation` and the
        `MusculoskeletalFly`.

    Raises:
        ImportError: If the ``[warp]`` extra or an NVIDIA CUDA GPU is not
            available.
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
