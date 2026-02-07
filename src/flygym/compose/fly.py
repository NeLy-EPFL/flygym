from pathlib import Path
from os import PathLike
from enum import Enum
from fnmatch import filter as filter_with_wildcard
from typing import Iterable

import mujoco
import numpy as np
import dm_control.mjcf as mjcf
import yaml

from flygym import assets_dir
from flygym.anatomy import BodySegment, JointDOF, Skeleton  # anatomical features
from flygym.anatomy import RotationAxis, AxisOrder  # rotation representations
from flygym.anatomy import JointPreset, ALL_SEGMENT_NAMES  # presets and constants
from flygym.compose.base import BaseCompositionElement
from flygym.utils.mjcf import set_mujoco_globals
from flygym.utils.math import Vec3, Rotation3D
from flygym.utils.exceptions import FlyGymInternalError

__all__ = ["Fly", "ActuatorType", "PoseDict"]


DEFAULT_RIGGING_CONFIG_PATH = assets_dir / "model/rigging.yaml"
DEFAULT_MUJOCO_GLOBALS_PATH = assets_dir / "model/mujoco_globals.yaml"
DEFAULT_MESH_DIR = assets_dir / "model/meshes"
DEFAULT_VISUALS_CONFIG_PATH = assets_dir / "model/visuals.yaml"


class Fly(BaseCompositionElement):
    def __init__(
        self,
        name: str = "nmf",
        *,
        rigging_config_path: PathLike = DEFAULT_RIGGING_CONFIG_PATH,
        mesh_dir: PathLike = DEFAULT_MESH_DIR,
        mujoco_globals_path: PathLike = DEFAULT_MUJOCO_GLOBALS_PATH,
        root_segment: BodySegment | str = "c_thorax",
        mirror_left2right: bool = True,
    ) -> None:
        self._name = name
        self._mjcf_root = mjcf.RootElement(model=name)
        set_mujoco_globals(self.mjcf_root, mujoco_globals_path)

        self.bodyseg_to_mjcfmesh = {}
        self.bodyseg_to_mjcfbody = {}
        self.bodyseg_to_mjcfgeom = {}
        self.jointdof_to_mjcfjoint = {}
        self.jointdof_to_mjcfactuator_by_type = {ty: {} for ty in ActuatorType}
        self.sensorname_to_mjcfsensor = {}
        self.cameraname_to_mjcfcamera = {}

        self.jointdof_to_neutralangle = {}
        self.jointdof_to_neutralaction_by_type = {ty: {} for ty in ActuatorType}

        if isinstance(root_segment, str):
            root_segment = BodySegment(root_segment)
        self.root_segment = root_segment

        self._neutral_keyframe = self.mjcf_root.keyframe.add(
            "key", name="neutral", time=0
        )

        self._add_mesh_assets(mesh_dir, mirror_left2right)
        self._add_bodies_and_geoms(rigging_config_path)

    @property
    def name(self) -> str:
        return self._name

    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root

    def get_bodysegs_order(self) -> Iterable[BodySegment]:
        return self.bodyseg_to_mjcfbody.keys()

    def get_jointdofs_order(self) -> Iterable[JointDOF]:
        return self.jointdof_to_mjcfjoint.keys()

    def get_actuated_jointdofs_order(
        self, actuator_type: "ActuatorType | str"
    ) -> Iterable[JointDOF]:
        actuator_type = ActuatorType(actuator_type)
        return self.jointdof_to_mjcfactuator_by_type[actuator_type].keys()

    def _add_mesh_assets(self, mesh_dir: PathLike, mirror_left2right: bool) -> None:
        # For numerical reasons, we simulate length in mm, not m. This changes the units
        # of other quantities as well, for example acceleration is now in mm/s^2.
        SCALE = 1000

        mesh_dir = Path(mesh_dir)
        for segment_name in ALL_SEGMENT_NAMES:
            if mirror_left2right and segment_name[0] == "r":
                mesh_to_use = f"l{segment_name[1:]}"
                y_sign = -1
            else:
                mesh_to_use = segment_name
                y_sign = 1
            mesh_path = (mesh_dir / f"{mesh_to_use}.stl").resolve()
            if not mesh_path.exists():
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            self.bodyseg_to_mjcfmesh[segment_name] = self.mjcf_root.asset.add(
                "mesh",
                name=segment_name,
                file=str(mesh_path),
                scale=(SCALE, y_sign * SCALE, SCALE),
            )

    def _add_bodies_and_geoms(self, rigging_config_path: PathLike) -> None:
        # Load rigging config
        with open(rigging_config_path) as f:
            rigging_config = yaml.safe_load(f)

        # Add root body and geom
        virtual_root = self.mjcf_root.worldbody.add("body", name="rootbody")
        body, geom = self._add_one_body_and_geom(
            virtual_root, self.root_segment, rigging_config[self.root_segment.name]
        )
        self.bodyseg_to_mjcfbody[self.root_segment] = body
        self.bodyseg_to_mjcfgeom[self.root_segment] = geom

        full_skeleton = Skeleton(
            joint_preset=JointPreset.ALL_POSSIBLE, axis_order=AxisOrder.DONTCARE
        )

        for jointdof in full_skeleton.iter_jointdofs(self.root_segment):
            if jointdof.axis != RotationAxis.PITCH:
                # Look at only 1 DoF per joint as we're still just adding bodies/geoms
                continue
            parent_body = self.bodyseg_to_mjcfbody.get(jointdof.parent)
            if parent_body is None:
                raise FlyGymInternalError("Parent not found during kinematic tree DFS")
            my_rigging_config = rigging_config.get(jointdof.child.name)
            if my_rigging_config is None:
                raise FlyGymInternalError(
                    f"Missing rigging config for body segment {jointdof.child.name}"
                )
            body, geom = self._add_one_body_and_geom(
                parent_body, jointdof.child, my_rigging_config
            )
            self.bodyseg_to_mjcfbody[jointdof.child] = body
            self.bodyseg_to_mjcfgeom[jointdof.child] = geom

    def _add_one_body_and_geom(
        self,
        parent_body: mjcf.Element,
        segment: BodySegment,
        my_rigging_config: dict[str, any],
    ) -> tuple[mjcf.Element, mjcf.Element]:
        body_element = parent_body.add(
            "body",
            name=segment.name,
            pos=my_rigging_config["pos"],
            quat=my_rigging_config["quat"],
        )
        geom_element = body_element.add(
            "geom",
            name=segment.name,
            type="mesh",
            mesh=segment.name,
            mass=my_rigging_config["mass"],
            contype=0,  # contact pairs to be added explicitly later
            conaffinity=0,  # contact pairs to be added explicitly later
        )
        return body_element, geom_element

    def colorize(
        self, visuals_config_path: PathLike = DEFAULT_VISUALS_CONFIG_PATH
    ) -> None:
        if len(self.bodyseg_to_mjcfgeom) == 0:
            raise ValueError("Must first add geoms via `_add_bodies_and_geoms`.")

        vis_sets_all, lookup = self._parse_visuals_config(visuals_config_path)

        for vis_set_name, params in vis_sets_all.items():
            material = self.mjcf_root.asset.add(
                "material", name=vis_set_name, **params["material"]
            )
            if texture_params := params.get("texture"):
                texture = self.mjcf_root.asset.add(
                    "texture", name=vis_set_name, **texture_params
                )
                material.texture = texture

        for segment, geom in self.bodyseg_to_mjcfgeom.items():
            vis_set_name = lookup[segment]
            geom.set_attributes(material=vis_set_name)

    def add_joints(
        self,
        skeleton: Skeleton,
        neutral_pose: "PoseDict | None" = None,
        *,
        stiffness: float = 10.0,
        damping: float = 0.5,
        **kwargs,
    ) -> dict[JointDOF, mjcf.Element]:
        if neutral_pose is None:
            neutral_pose = PoseDict()

        return_dict = {}
        for jointdof in skeleton.iter_jointdofs(self.root_segment):
            child_body = self.bodyseg_to_mjcfbody[jointdof.child]
            mirror_axis = jointdof.child.name[0] == "r" and (
                jointdof.axis in (RotationAxis.ROLL, RotationAxis.YAW)
            )
            neutral_angle = neutral_pose.get(jointdof.name, 0.0)
            self.jointdof_to_neutralangle[jointdof] = neutral_angle
            return_dict[jointdof] = child_body.add(
                "joint",
                name=jointdof.name,
                type="hinge",
                axis=jointdof.axis.to_vector(mirror=mirror_axis),
                stiffness=stiffness,
                damping=damping,
                springref=neutral_angle,
                **kwargs,
            )

        self.jointdof_to_mjcfjoint.update(return_dict)
        self._rebuild_neutral_keyframe()
        return return_dict

    def _rebuild_neutral_keyframe(self):
        mj_model, _ = self.compile()
        self._neutral_keyframe.qpos = self._get_neutral_qpos(mj_model)
        self._neutral_keyframe.ctrl = self._get_neutral_ctrl(mj_model)

    def _get_neutral_qpos(self, mj_model: mujoco.MjModel) -> np.ndarray:
        neutral_qpos = np.zeros(mj_model.nq)
        for jointdof, angle in self.jointdof_to_neutralangle.items():
            joint_element = self.jointdof_to_mjcfjoint[jointdof]
            internal_jointid = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_element.full_identifier
            )
            qposadr = mj_model.jnt_qposadr[internal_jointid]
            neutral_qpos[qposadr] = angle
        return neutral_qpos

    def _get_neutral_ctrl(self, mj_model: mujoco.MjModel) -> np.ndarray:
        neutral_ctrl = np.zeros(mj_model.nu)
        for ty, jointdof_to_actuator in self.jointdof_to_mjcfactuator_by_type.items():
            for jointdof, actuator in jointdof_to_actuator.items():
                internal_actuatorid = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator.full_identifier
                )
                neutral_input = self.jointdof_to_neutralaction_by_type[ty][jointdof]
                neutral_ctrl[internal_actuatorid] = neutral_input
        return neutral_ctrl

    @staticmethod
    def _parse_visuals_config(
        visuals_config_path: PathLike,
    ) -> tuple[dict[str, dict], dict[BodySegment, dict]]:
        # Load visuals config and assign vis sets to body segments
        with open(visuals_config_path) as f:
            vis_set_params_all = yaml.safe_load(f)
        all_matches_by_segname = {k: [] for k in ALL_SEGMENT_NAMES}
        for vis_set_name, vis_set_params in vis_set_params_all.items():
            apply_to = vis_set_params.get("apply_to")
            material = vis_set_params.get("material")
            if not apply_to or not material:
                raise ValueError(
                    f"Invalid visualization set: {vis_set_name}."
                    "Must specify a non-empty 'apply_to' and 'material'."
                )
            allowed_keys = {"apply_to", "material", "texture"}
            if invalid_keys := (set(vis_set_params.keys()) - allowed_keys):
                raise ValueError(
                    f"Invalid keys in visualization set {vis_set_name}: "
                    f"{invalid_keys}. Must be one of {allowed_keys}."
                )
            target_segnames = set()
            for pattern in [apply_to] if isinstance(apply_to, str) else apply_to:
                target_segnames |= set(filter_with_wildcard(ALL_SEGMENT_NAMES, pattern))
            for segname in target_segnames:
                all_matches_by_segname[segname].append(vis_set_name)
        for segname, vis_set_names in all_matches_by_segname.items():
            if len(vis_set_names) != 1:
                raise ValueError(
                    f"Zero or multiple vis sets matched for body segment {segname}: "
                    f"{vis_set_names}. Only one should apply."
                )
        lookup_by_segname = {
            BodySegment(segname): matches[0]
            for segname, matches in all_matches_by_segname.items()
        }
        return vis_set_params_all, lookup_by_segname

    def add_actuators(
        self,
        jointdofs: Iterable[JointDOF],
        actuator_type: "ActuatorType | str",
        neutral_input: dict[str, float] | None = None,
        *,
        forcelimited: bool = True,
        forcerange: tuple[float, float] = (-50.0, 50.0),
        **kwargs,
    ) -> dict[JointDOF, mjcf.Element]:
        if neutral_input is None:
            neutral_input = {}

        actuator_type = ActuatorType(actuator_type)
        return_dict = {}
        for jointdof in jointdofs:
            self.jointdof_to_neutralaction_by_type[actuator_type][jointdof] = (
                neutral_input.get(jointdof.name, 0.0)
            )
            actuator = self.mjcf_root.actuator.add(
                actuator_type.value,
                name=f"{jointdof.name}-{actuator_type.value}",
                joint=jointdof.name,
                forcelimited=forcelimited,
                forcerange=forcerange,
                ctrlrange=(-3, 3),
                **kwargs,
            )
            return_dict[jointdof] = actuator
        self.jointdof_to_mjcfactuator_by_type[actuator_type].update(return_dict)
        self._rebuild_neutral_keyframe()
        return return_dict

    def add_tracking_camera(
        self,
        name: str,
        mode: str = "track",
        pos_offset: Vec3 = (0, -7.5, 6),
        rotation: Rotation3D = Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)),
        fovy: float = 30.0,
        **kwargs,
    ) -> mjcf.Element:
        camera = self.mjcf_root.worldbody.add(
            "camera",
            name=name,
            mode=mode,
            target="rootbody",
            pos=pos_offset,
            fovy=fovy,
            **rotation.as_kwargs(),
            **kwargs,
        )
        self.cameraname_to_mjcfcamera[name] = camera
        return camera


class ActuatorType(Enum):
    MOTOR = "motor"
    POSITION = "position"
    VELOCITY = "velocity"
    INTVELOCITY = "intvelocity"
    DAMPER = "damper"
    CYLINDER = "cylinder"
    MUSCLE = "muscle"
    ADHESION = "adhesion"


class PoseDict(dict[str, float]):
    def __init__(
        self,
        *,
        joint_angles_dict: dict[str, float] | None = None,
        file_path: PathLike | None = None,
        mirror_left2right: bool = True,
    ) -> None:
        if (joint_angles_dict is not None) + (file_path is not None) != 1:
            raise ValueError(
                "Either joint_angles_dict or file_path must be provided, but not both."
            )
        if file_path is not None:
            joint_angles_dict = self._load_yaml(file_path)
        if mirror_left2right:
            joint_angles_dict = self._apply_mirroring(joint_angles_dict)
        super().__init__(joint_angles_dict)

    @staticmethod
    def _load_yaml(file_path: PathLike) -> dict[str, float]:
        with open(file_path, "r") as f:
            pose_data = yaml.safe_load(f)

        if "angle_unit" not in pose_data:
            raise ValueError("YAML file must contain 'angle_unit' key.")
        if pose_data["angle_unit"] not in ["degree", "radian"]:
            raise ValueError("angle_unit must be either 'degree' or 'radian'.")

        if "joint_angles" not in pose_data:
            raise ValueError("YAML file must contain 'joint_angles' key.")
        for k, v in pose_data["joint_angles"].items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"Joint angle for '{k}' must be a number.")

        joint_angles = pose_data["joint_angles"]
        if pose_data["angle_unit"] == "degree":
            joint_angles = {k: np.deg2rad(v) for k, v in joint_angles.items()}

        return joint_angles

    @staticmethod
    def _apply_mirroring(joint_angles_in: dict[str, float]) -> dict[str, float]:
        joint_angles_out = {}
        for joint_name, angle in joint_angles_in.items():
            joint_angles_out[joint_name] = angle

            jointdof = JointDOF.from_name(joint_name)

            if jointdof.child.name[0] == "l":
                mirror_parent = BodySegment(
                    "r" + jointdof.parent.name[1:]
                    if jointdof.parent.name[0] == "l"
                    else jointdof.parent.name
                )
                mirror_child = BodySegment("r" + jointdof.child.name[1:])
                mirror_jointdof = JointDOF(mirror_parent, mirror_child, jointdof.axis)
                if mirror_jointdof.name not in joint_angles_in:
                    # Skip if right-side joint angle explicitly provided in input
                    joint_angles_out[mirror_jointdof.name] = angle

        return joint_angles_out
