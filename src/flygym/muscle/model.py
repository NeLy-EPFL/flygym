"""Load FlyMimic's musculoskeletal MJCF as a FlyGym-compatible fly + world.

Rather than overlaying muscles onto FlyGym's composed body, the muscle path
*switches the body model*: it loads FlyMimic's self-contained MJCF (which ships
its own floor, lighting, 15 left-front-leg Hill-type muscles, and passive joint
properties) and wraps it so that `flygym.Simulation` and its sensor suite work
unchanged.

The wrappers expose exactly the attributes/methods that `Simulation` reads:

* `MuscleFly`  — Fly-compatible: ``bodyseg_to_mjcfbody``, ``bodyseg_to_mjcfgeom``,
  ``jointdof_to_mjcfjoint``, ``jointdof_to_mjcfactuator_by_type``,
  ``leg_to_adhesionactuator``, ``anatomicaljoint_to_mjcfsites``,
  ``eyecameraname_to_mjcfcamera`` plus the ``get_*_order`` accessors. Dict keys
  are the model's own element-name strings (e.g. ``"LFFemur"``,
  ``"joint_LFCoxa_yaw"``, ``"LFTibia_flex_93434"``), since FlyMimic's body
  topology does not line up 1:1 with FlyGym's `BodySegment` names.
* `MuscleWorld` — minimal world: ``fly_lookup``, ``ground_geoms``,
  ``legpos_to_groundcontactsensors_by_fly``, ``world_dof_neutral_states`` and a
  ``compile()`` that yields a model carrying a keyframe named ``"neutral"``.

Use `build_muscle_simulation` for the common case.
"""

from os import PathLike
from pathlib import Path

import dm_control.mjcf as mjcf

from flygym.compose.base import BaseCompositionElement
from flygym.compose.fly import ActuatorType
from flygym.compose.world import BaseWorld
from flygym.muscle.assets import DEFAULT_MUSCLE_XML
from flygym.utils.math import Rotation3D, Vec3

__all__ = ["MuscleFly", "MuscleWorld", "build_muscle_simulation"]


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


class MuscleFly(BaseCompositionElement):
    """FlyGym-compatible wrapper around FlyMimic's musculoskeletal MJCF.

    Args:
        xml_path: Path to the FlyMimic MJCF. Defaults to the bundled
            ``arm_damping_stiff`` muscle model.
        name: Logical fly name used by `Simulation` lookups.

    Attributes mirror `flygym.compose.fly.Fly` (see module docstring), but dict
    keys are the model's own element-name strings.
    """

    def __init__(
        self,
        xml_path: PathLike = DEFAULT_MUSCLE_XML,
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

        # Build the tracking dicts that Simulation introspects.
        self.bodyseg_to_mjcfbody: dict[str, mjcf.Element] = {}
        self.bodyseg_to_mjcfgeom: dict[str, mjcf.Element] = {}
        for body in self._mjcf_root.find_all("body"):
            if body.name is None:
                continue
            self.bodyseg_to_mjcfbody[body.name] = body
            geoms = body.find_all("geom", immediate_children_only=True)
            if geoms:
                self.bodyseg_to_mjcfgeom[body.name] = geoms[0]

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
        # No ground-contact leg grouping is defined for the muscle model.
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


class MuscleWorld(BaseWorld):
    """`BaseWorld` over a self-contained `MuscleFly`.

    The FlyMimic MJCF already provides a floor, lighting, and an anchored
    thorax, so — unlike the composable FlyGym worlds — this world does not
    attach the fly into a separately-built scene. It adopts the fly's MJCF root
    directly and presents the `BaseWorld` surface that `Simulation` and
    `GPUSimulation` consume.

    Subclassing `BaseWorld` (rather than just duck-typing) keeps the type
    contract honest for the GPU path, which annotates ``world: BaseWorld``.
    """

    def __init__(self, fly: MuscleFly) -> None:
        # NOTE: intentionally skip super().__init__(): BaseWorld's initializer
        # builds a fresh empty scene (new root + skybox + neutral keyframe) and
        # expects flies to be added later via add_fly(). The muscle model is a
        # self-contained MJCF that already holds the fly, floor, and lighting,
        # so we adopt its root instead of constructing a new one.
        self._fly = fly
        self._mjcf_root = fly.mjcf_root
        self._fly_lookup = {fly.name: fly}
        # Expose the floor geom so contact-force queries can filter on ground.
        floor = fly.mjcf_root.find("geom", "floor")
        self.ground_geoms = [floor] if floor is not None else []
        # FlyMimic ships no per-leg ground-contact sensors.
        self.legpos_to_groundcontactsensors_by_fly = None
        self.world_dof_neutral_states: dict[str, list[float]] = {}

    def _attach_fly_mjcf(
        self,
        fly: MuscleFly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *args,
        **kwargs,
    ) -> mjcf.Element:
        raise NotImplementedError(
            "MuscleWorld wraps a self-contained musculoskeletal MJCF in which "
            "the fly is already present; add_fly()/attachment is not supported. "
            "Pass a MuscleFly at construction or use build_muscle_simulation()."
        )


def build_muscle_simulation(
    *,
    xml_path: PathLike = DEFAULT_MUSCLE_XML,
    name: str = "nmf",
    add_vision: bool = False,
):
    """Convenience: build a `MuscleFly` + `MuscleWorld` + `Simulation`.

    Returns:
        ``(simulation, fly)`` where ``simulation`` is a `flygym.Simulation`
        and ``fly`` is the `MuscleFly` (handy for reading ``muscle_names``).
    """
    from flygym.simulation import Simulation

    fly = MuscleFly(xml_path, name=name)
    if add_vision:
        fly.add_vision()
    world = MuscleWorld(fly)
    sim = Simulation(world)
    return sim, fly
