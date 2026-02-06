from collections.abc import Collection

import mujoco
import dm_control.mjcf as mjcf
import numpy as np

from flygym.anatomy import ContactBodiesPreset, BodySegment, JointDOF
from flygym.compose.base import BaseCompositionElement
from flygym.compose.fly import Fly
from flygym.compose.physics import ContactParams
from flygym.utils.math import Rotation3D, Vec3

__all__ = ["BaseWorld", "FlatGroundWorld", "TetheredWorld"]


_STATE_DIM_BY_JOINT_TYPE = {"free": 7, "ball": 4, "hinge": 1, "slide": 1}


class BaseWorld(BaseCompositionElement):
    def __init__(self, name: str) -> None:
        self._mjcf_root = mjcf.RootElement(model=name)
        self._fly_lookup: dict[str, Fly] = {}
        self.world_dof_neutral_state = {}
        # self._neutral_keyframe = self.mjcf_root.keyframe.add(
        #     "key", name="neutral", time=0.0
        # )

    @property
    def fly_lookup(self) -> dict[str, Fly]:
        return self._fly_lookup

    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root

    def add_fly(
        self,
        fly: Fly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *args,
        **kwargs,
    ) -> any:
        if fly.name in self._fly_lookup:
            raise ValueError(f"Fly with name '{fly.name}' already exists in the world.")
        self._fly_lookup[fly.name] = fly

        if spawn_rotation.format != "quat":
            raise ValueError(
                "Freejoint neutral rotation can only be specified in quaternion format "
                f"for now. Got {spawn_rotation}."
            )

        freejoint = self._attach_fly_mjcf(
            fly, spawn_position, spawn_rotation, *args, **kwargs
        )

        fullid = freejoint.full_identifier
        self.world_dof_neutral_state[fullid] = [*spawn_position, *spawn_rotation.values]
        self._rebuild_neutral_keyframe()

    def _rebuild_neutral_keyframe(self):
        mj_model, _ = self.compile()
        neutral_qpos = np.zeros(mj_model.nq)

        # dm_control.mjcf has trouble finding freejoints by name with
        # .find("joint", freejoint_name), but they do show up in the list of all joints
        # obtained with .find_all("joint"). So we build a mapping manually in order to
        # set the neutral pose for freejoints corresponding to fly spawns.
        all_world_joints = {
            j.full_identifier: j for j in self.mjcf_root.find_all("joint")
        }
        for joint_name, neutral_state in self.world_dof_neutral_state.items():
            joint_element = all_world_joints.get(joint_name)
            if joint_element is None:
                raise RuntimeError(
                    f"Joint '{joint_name}' not found in MJCF model when rebuilding "
                    "neutral keyframe."
                )
            joint_type = (
                "free" if joint_element.tag == "freejoint" else joint_element.type
            )
            internal_jointid = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_element.full_identifier
            )
            dofadr_start = mj_model.jnt_dofadr[internal_jointid]
            dofadr_end = dofadr_start + _STATE_DIM_BY_JOINT_TYPE[joint_type]
            neutral_qpos[dofadr_start:dofadr_end] = neutral_state

        for fly_name, fly in self.fly_lookup.items():
            for jointdof, neutral_angle in fly.jointdof_to_neutralangle.items():
                joint_element = fly.jointdof_to_mjcfjoint[jointdof]
                internal_jointid = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_element.full_identifier
                )
                dofadr_start = mj_model.jnt_qposadr[internal_jointid]
                neutral_qpos[dofadr_start] = neutral_angle

        for key_element in self.mjcf_root.keyframe.find_all("key"):
            key_element.remove()
        self.mjcf_root.keyframe.add("key", name="neutral", time=0.0, qpos=neutral_qpos)


class FlatGroundWorld(BaseWorld):
    def __init__(
        self, name: str = "flat_ground_world", *, half_size: float = 1000
    ) -> None:
        super().__init__(name=name)

        checker_texture = self.mjcf_root.asset.add(
            "texture",
            name="checker",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.3, 0.3, 0.3),
            rgb2=(0.4, 0.4, 0.4),
        )
        grid_material = self.mjcf_root.asset.add(
            "material",
            name="grid",
            texture=checker_texture,
            texrepeat=(250, 250),
            reflectance=0.2,
        )
        ground_geom = self.mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="ground_plane",
            material=grid_material,
            pos=(0, 0, 0),
            size=(half_size, half_size, 1),
            contype=0,
            conaffinity=0,
        )
        self.ground_contact_geoms = [ground_geom]

    def _attach_fly_mjcf(
        self,
        fly: Fly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *,
        bodysegs_with_ground_contact: (
            Collection[BodySegment] | ContactBodiesPreset | str
        ) = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
        ground_contact_params: ContactParams = ContactParams(),
    ) -> mjcf.Element:
        spawn_site = self.mjcf_root.worldbody.add(
            "site", name=fly.name, pos=spawn_position, **spawn_rotation.as_kwargs()
        )
        freejoint = spawn_site.attach(fly.mjcf_root).add("freejoint", name=fly.name)
        self._set_ground_contact(
            fly, bodysegs_with_ground_contact, ground_contact_params
        )
        return freejoint

    def _set_ground_contact(
        self,
        fly: Fly,
        bodysegs_with_ground_contact: (
            Collection[BodySegment] | ContactBodiesPreset | str
        ),
        ground_contact_params: ContactParams,
    ) -> None:
        if isinstance(bodysegs_with_ground_contact, ContactBodiesPreset | str):
            preset = ContactBodiesPreset(bodysegs_with_ground_contact)
            bodysegs_with_ground_contact = preset.to_body_segments_list()

        for i, ground_geom in enumerate(self.ground_contact_geoms):
            for body_segment in bodysegs_with_ground_contact:
                body_geom = fly.mjcf_root.find("geom", f"{body_segment.name}")
                ground_geom_name = (
                    f"ground{i}" if ground_geom.name is None else ground_geom.name
                )
                self.mjcf_root.contact.add(
                    "pair",
                    geom1=ground_geom,
                    geom2=body_geom,
                    name=f"{body_segment.name}-{ground_geom_name}",
                    friction=ground_contact_params.get_friction_tuple(),
                    solref=ground_contact_params.get_solref_tuple(),
                    solimp=ground_contact_params.get_solimp_tuple(),
                )


class TetheredWorld(BaseWorld):
    def __init__(self, name: str = "tethered_world") -> None:
        super().__init__(name=name)
        # don't add ground plane

    def add_fly(
        self,
        fly,
        spawn_position: Vec3 = (0, 0, 0),
        spawn_rotation: Rotation3D = Rotation3D("quat", (1, 0, 0, 0)),
    ):
        if spawn_rotation.format != "quat":
            raise ValueError("TetheredWorld only supports quaternion rotation format.")

        super().add_fly(fly)

        spawn_site = self.mjcf_root.worldbody.add(
            "site", name=fly.name, pos=spawn_position, **spawn_rotation.as_kwargs()
        )
        attachment_frame = spawn_site.attach(fly.mjcf_root)
        attachment_frame.add("freejoint", name=f"{fly.name}")
        self.mjcf_root.equality.add(
            "weld",
            body2="world",  # worldbody is called "world" in equality constraints
            body1=fly.mjcf_root.find("body", "rootbody").full_identifier,
            relpose=(*spawn_position, *spawn_rotation.values),
            solref=(2e-4, 1.0),
            solimp=(0.98, 0.99, 1e-5, 0.5, 3),
        )
