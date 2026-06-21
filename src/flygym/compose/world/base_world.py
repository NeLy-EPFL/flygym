from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, override

import mujoco as mj
import numpy as np

from flygym.anatomy import BaseContactBodiesPreset, ContactBodiesPreset, BodySegment
from flygym.compose.base import BaseCompositionElement
from flygym.compose.fly import BaseFly
from flygym.compose.physics import ContactParams
from flygym.utils.mjcf import add_texture, set_mujoco_globals
from flygym.utils.math import Rotation3D, Vec3
from flygym.utils.exceptions import FlyGymInternalError

__all__ = ["BaseWorld"]


_STATE_DIM_BY_JOINT_TYPE = {
    mj.mjtJoint.mjJNT_FREE: 7,
    mj.mjtJoint.mjJNT_BALL: 4,
    mj.mjtJoint.mjJNT_HINGE: 1,
    mj.mjtJoint.mjJNT_SLIDE: 1,
}


class BaseWorld(BaseCompositionElement, ABC):
    """Base class for worlds that contain environmental features that the fly can
    interact with (e.g., ground) and define how flies are attached to the world (e.g.,
    free-floating or tethered). A world can contain multiple flies that can interact
    with one another.

    Concrete subclasses typically override `__init__` to set up environmental features
    (e.g., ground plane) and `_attach_fly_mjcf` to define how flies are attached. See
    method documentation below for details.

    Attributes:
        name:
            Name of the world.
        fly_lookup:
            A dictionary mapping fly names to `Fly` objects in the world.
        mjcf_root:
            The root element of the world's MJCF model (fly MJCF models are attached to
            this root).
        world_dof_neutral_states:
            A dictionary mapping names of DoFs managed by the world (e.g., free joints
            by which flies are attached to the world) to their neutral state values.
            The neutral state is 1D for slide and hinge joints, 4D for ball joints
            (quaternion), and 7D for free joints (position + orientation).
    """

    def __init__(self, name: str) -> None:
        """Initialize the world and its underlying MJCF model.

        Concrete subclasses should call this first (i.e., `super().__init__(name)`) as
        it sets up a few essential attributes.
        """
        self._mjcf_root = mj.MjSpec()
        self._mjcf_root.modelname = name
        self._fly_lookup: dict[str, BaseFly] = {}
        self.ground_geoms: list = []
        self.legpos_to_groundcontactsensors_by_fly = None
        self.world_dof_neutral_states = {}
        self._neutral_keyframe = self.mjcf_root.add_key(name="neutral", time=0)
        self._add_skybox()

    @override
    @property
    def mjcf_root(self) -> mj.MjSpec:
        return self._mjcf_root

    @property
    def fly_lookup(self) -> dict[str, BaseFly]:
        """Lookup for `Fly` objects in the world, keyed by fly name."""
        return self._fly_lookup

    @abstractmethod
    def _attach_fly_mjcf(
        self,
        fly: BaseFly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *args,
        **kwargs,
    ) -> dict[str, list[float]]:
        """Attach the fly's MJCF root to the world MJCF model.

        Concrete subclasses should implement this method instead of overriding
        `add_fly()` directly. The `add_fly()` method handles registering the fly under
        `fly_lookup` and updating neutral states; this method is responsible only for
        connecting the fly's MJCF model to the world's MJCF model.

        Use `MjSpec.attach()` to attach the fly's MjSpec to the world. See
        `_GroundContactMixin` and `TetheredWorld` for examples. More details can be
        found in the [MuJoCo model editing documentation](https://mujoco.readthedocs.io/en/stable/python.html#model-editing).

        Returns:
            Mapping from joint name to neutral state for any world-level
            DoFs created by this attachment. Return an empty dict if the fly is
            rigidly attached (no new DoFs).
        """
        pass

    def _add_skybox(self):
        add_texture(
            self.mjcf_root,
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1=(1, 1, 1),
            rgb2=(1, 1, 1),
            width=10,
            height=10,
        )

    def add_fly(
        self,
        fly: BaseFly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Attach a fly to the world at the specified pose.

        The fly's MJCF model is merged into the world and registered under
        `fly_lookup`. Extra keyword arguments are forwarded to the subclass
        `_attach_fly_mjcf` implementation (see the specific world subclass for
        available options).

        Args:
            fly: The fly to add.
            spawn_position: Initial ``(x, y, z)`` position in mm.
            spawn_rotation: Initial orientation as a `Rotation3D` in quaternion format.
            *args: Forwarded to `_attach_fly_mjcf`.
            **kwargs: Forwarded to `_attach_fly_mjcf`.

        Raises:
            ValueError: If a fly with the same name already exists in the world.
            ValueError: If ``spawn_rotation`` is not in quaternion format.
        """
        # Register fly in the fly lookup
        if fly.name in self._fly_lookup:
            raise ValueError(f"Fly with name '{fly.name}' already exists in the world.")
        self._fly_lookup[fly.name] = fly

        # Inherit the fly's global MuJoCo settings (timestep, gravity, integrator,
        # etc.). MjSpec.attach() uses the parent (world) spec's <option>/<compiler>
        # for the compiled model and does not merge the child's, so we apply the
        # fly's globals to the world spec here.
        set_mujoco_globals(self.mjcf_root, fly.mujoco_globals_path)

        # Remove neutral keyframes that are already generated by the fly. Neutral states
        # are globally managed at the world level. A single neutral keyframe will be
        # managed by the world from now on.
        fly_neutral_keyframes = [
            k for k in fly.mjcf_root.keys if k.name == "neutral"
        ]
        for keyframe in fly_neutral_keyframes:
            fly.mjcf_root.delete(keyframe)

        # Attach the fly's MJCF root to the world MJCF model with a free joint.
        # This is an abstract method that must be implemented by concrete world classes.
        new_dofs = self._attach_fly_mjcf(
            fly, spawn_position, spawn_rotation, *args, **kwargs
        )

        # Set neutral state for the freejoint attaching the fly to the world
        # (freejoint state is in [x, y, z, qw, qx, qy, qz] format)
        if spawn_rotation.format != "quat":
            raise ValueError(
                "Freejoint neutral rotation can only be specified in quaternion format "
                f"for now. Got {spawn_rotation}."
            )

        self.world_dof_neutral_states.update(new_dofs)
        self._rebuild_neutral_keyframe()

    def _rebuild_neutral_keyframe(self):
        mj_model, _ = self.compile()
        neutral_qpos = np.zeros(mj_model.nq)
        neutral_ctrl = np.zeros(mj_model.nu)

        # Step 1: set neutral qpos for DoFs created by the world (e.g. the free joints
        # by which flies are attached). Joints are keyed by their (prefixed) name,
        # which matches the compiled model's joint names. We use the compiled `qpos0`
        # rest pose, which already composes the spawn site transform with each fly's
        # root-body offset. This matches the previous (PyMJCF) behavior, where
        # `spawn_position` positions the fly's attachment frame rather than its root
        # body directly.
        all_world_joints = {j.name: j for j in self.mjcf_root.joints}
        for joint_name in self.world_dof_neutral_states:
            joint_element = all_world_joints.get(joint_name)
            if joint_element is None:
                raise RuntimeError(
                    f"Joint '{joint_name}' not found when rebuilding neutral keyframe."
                )
            internal_jointid = mj.mj_name2id(
                mj_model, mj.mjtObj.mjOBJ_JOINT, joint_element.name
            )
            qposadr_start = mj_model.jnt_qposadr[internal_jointid]
            qposadr_end = qposadr_start + _STATE_DIM_BY_JOINT_TYPE[joint_element.type]
            neutral_qpos[qposadr_start:qposadr_end] = mj_model.qpos0[
                qposadr_start:qposadr_end
            ]

        # Step 2: handle joints and actuators belonging to flies attached to the world
        for fly_name, fly in self.fly_lookup.items():
            # Copy neutral joint angles from fly
            qpos_filled_by_fly = fly._get_neutral_qpos(mj_model)
            indices_to_fill = qpos_filled_by_fly.nonzero()
            has_conflict = np.any(~np.isclose(neutral_qpos[indices_to_fill], 0))
            if has_conflict:
                raise FlyGymInternalError(
                    f"Conflict in neutral joint angles: fly '{fly_name}' is trying "
                    "to set neutral qpos values for DoFs that already have their "
                    "neutral qpos set."
                )
            neutral_qpos[indices_to_fill] = qpos_filled_by_fly[indices_to_fill]

            # Copy neutral actuator inputs from fly
            ctrl_filled_by_fly = fly._get_neutral_ctrl(mj_model)
            indices_to_fill = ctrl_filled_by_fly.nonzero()
            has_conflict = np.any(~np.isclose(neutral_ctrl[indices_to_fill], 0))
            if has_conflict:
                raise FlyGymInternalError(
                    f"Conflict in neutral actuator inputs: fly '{fly_name}' is trying "
                    "to set neutral ctrl values for actuators that already have their "
                    "neutral ctrl set."
                )
            neutral_ctrl[indices_to_fill] = ctrl_filled_by_fly[indices_to_fill]

        self._neutral_keyframe.qpos = neutral_qpos
        self._neutral_keyframe.ctrl = neutral_ctrl


class _GroundContactMixin:
    def _attach_fly_mjcf(
        self,
        fly: BaseFly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *,
        bodysegs_with_ground_contact: (
            list[BodySegment] | ContactBodiesPreset | str
        ) = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
        ground_contact_params: ContactParams = ContactParams(),
        add_ground_contact_sensors: bool = True,
    ) -> dict[str, list[float]]:
        spawn_site = self.mjcf_root.worldbody.add_site(
            name=fly.name, pos=spawn_position, **spawn_rotation.as_kwargs()
        )
        # Attach the fly spec at the spawn site (prefixing its element names with the
        # fly name) and give it a free joint so it floats freely in the world.
        self.mjcf_root.attach(
            fly.mjcf_root, prefix=f"{fly.name}/", site=spawn_site
        )
        freejoint = fly.bodyseg_to_mjcfbody[fly.root_segment].add_freejoint(
            name=fly.name
        )

        if isinstance(bodysegs_with_ground_contact, BaseContactBodiesPreset):
            bodysegs_with_ground_contact = (
                bodysegs_with_ground_contact.to_body_segments_list()
            )
        elif isinstance(bodysegs_with_ground_contact, str):
            # Resolve the string against the fly's own contact-bodies preset enum
            # so each Fly subclass (nmf, flybody, ...) selects the right segments.
            preset = type(fly).CONTACT_BODIES_PRESET_CLASS(bodysegs_with_ground_contact)
            bodysegs_with_ground_contact = preset.to_body_segments_list()

        self._set_ground_contact(
            fly, bodysegs_with_ground_contact, ground_contact_params
        )
        if add_ground_contact_sensors:
            self._add_ground_contact_sensors(fly, bodysegs_with_ground_contact)

        neutral_state = [*spawn_position, *spawn_rotation.values]

        return {freejoint.name: neutral_state}

    def _set_ground_contact(
        self,
        fly: BaseFly,
        bodysegs_with_ground_contact: list[BodySegment],
        ground_contact_params: ContactParams,
    ) -> None:
        for body_segment in bodysegs_with_ground_contact:
            for body_geom in fly.bodyseg_to_mjcfgeom[body_segment]:
                for ground_geom in self.ground_geoms:
                    geom_name = body_geom.name
                    self.mjcf_root.add_pair(
                        geomname1=body_geom.name,
                        geomname2=ground_geom.name,
                        name=f"{geom_name}-{ground_geom.name}-ground",
                        friction=ground_contact_params.get_friction_tuple(),
                        solref=ground_contact_params.get_solref_tuple(),
                        solimp=ground_contact_params.get_solimp_tuple(),
                        margin=ground_contact_params.margin,
                    )

    def _add_ground_contact_sensors(
        self, fly: BaseFly, bodysegs_with_ground_contact: list[BodySegment]
    ) -> None:
        if len(self.ground_geoms) != 1:
            self.legpos_to_groundcontactsensors_by_fly = None
            return

        self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)
        contact_geoms_by_leg = defaultdict(list)
        for bodyseg in bodysegs_with_ground_contact:
            if bodyseg.is_leg():
                contact_geoms_by_leg[bodyseg.pos].append(bodyseg)
        for leg, contact_geoms in contact_geoms_by_leg.items():
            subtree_rootseg = _sort_legsegs_prox2dist(contact_geoms, fly.LEG_LINKS)[0]
            subtree_rootseg_body = fly.bodyseg_to_mjcfbody[subtree_rootseg]
            # MjSpec has no high-level contact-sensor shortcut, so set the low-level
            # fields directly (matching what the XML `<contact .../>` shortcut emits):
            #   - subtree1=<body>  -> objtype=XBODY, objname=<body>
            #   - geom2=<geom>     -> reftype=GEOM,  refname=<geom>
            #   - intprm = [data_bitmask, reduce, num], where the data bitmask encodes
            #     "found force torque pos normal tangent" (found=1, force=2, torque=4,
            #     dist=8, pos=16, normal=32, tangent=64 -> 1+2+4+16+32+64 = 119) and
            #     reduce "netforce" = 3.
            sensor = self.mjcf_root.add_sensor(
                name=f"ground_contact_{leg}_leg",
                type=mj.mjtSensor.mjSENS_CONTACT,
                objtype=mj.mjtObj.mjOBJ_XBODY,
                objname=subtree_rootseg_body.name,
                reftype=mj.mjtObj.mjOBJ_GEOM,
                refname=self.ground_geoms[0].name,
                intprm=[119, 3, 1],
            )
            self.legpos_to_groundcontactsensors_by_fly[fly.name][leg] = sensor


def _sort_legsegs_prox2dist(
    segments: list[BodySegment], leg_links: list[str]
) -> list[BodySegment]:
    bodyseg_linkpos_tuples = [(seg, leg_links.index(seg.link)) for seg in segments]
    bodyseg_linkpos_tuples.sort(key=lambda x: x[1])
    return [t[0] for t in bodyseg_linkpos_tuples]


def _format_name_number(value: float) -> str:
    return f"{value:.3f}".replace("-", "m").replace(".", "p")
