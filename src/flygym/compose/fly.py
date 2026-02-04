from pathlib import Path
from os import PathLike
from enum import Enum
from fnmatch import filter as filter_with_wildcard
from collections import defaultdict
from typing import Iterable

import dm_control.mjcf as mjcf
import yaml

from flygym import assets_dir
from flygym.anatomy import BodySegment, JointDOF, Skeleton  # anatomical features
from flygym.anatomy import RotationAxis, AxisOrder  # rotation representations
from flygym.anatomy import JointPreset, ALL_SEGMENT_NAMES  # presets and constants
from flygym.compose._base import BaseCompositionElement
from flygym.utils.mjcf import set_mujoco_globals
from flygym.utils.math import Vec3, Rotation3D

__all__ = ["Fly", "ActuatorType"]


DEFAULT_RIGGING_CONFIG_PATH = assets_dir / "model/rigging.yaml"
DEFAULT_MUJOCO_GLOBALS_PATH = assets_dir / "model/mujoco_globals.yaml"
DEFAULT_MESH_DIR = assets_dir / "model/meshes"
DEFAULT_VISUALS_CONFIG_PATH = assets_dir / "model/visuals.yaml"


class ActuatorType(Enum):
    MOTOR = "motor"
    POSITION = "position"
    VELOCITY = "velocity"
    INTVELOCITY = "intvelocity"
    DAMPER = "damper"
    CYLINDER = "cylinder"
    MUSCLE = "muscle"
    ADHESION = "adhesion"


class Fly(BaseCompositionElement):
    def __init__(
        self,
        name: str = "neuromechfly",
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
        self.jointdof_to_mjcfactuator_by_type = defaultdict(dict)
        self.sensorname_to_mjcfsensor = {}
        self.cameraname_to_mjcfcamera = {}

        if isinstance(root_segment, str):
            root_segment = BodySegment(root_segment)
        self.root_segment = root_segment

        self._add_mesh_assets(mesh_dir, mirror_left2right)
        self._add_bodies_and_geoms(rigging_config_path)

    @property
    def name(self) -> str:
        return self._name

    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root

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
            assert parent_body is not None, "Kinematic tree DFS error"
            my_rigging_config = rigging_config.get(jointdof.child.name)
            assert my_rigging_config is not None, "Missing rigging config for body"
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
        stiffness: float = 10.0,
        damping: float = 0.5,
        **kwargs,
    ) -> dict[JointDOF, mjcf.Element]:
        return_dict = {}
        for jointdof in skeleton.iter_jointdofs(self.root_segment):
            child_body = self.bodyseg_to_mjcfbody[jointdof.child]
            return_dict[jointdof] = child_body.add(
                "joint",
                name=jointdof.name,
                type="hinge",
                axis=jointdof.axis.to_vector(),
                stiffness=stiffness,
                damping=damping,
                **kwargs,
            )
        self.jointdof_to_mjcfjoint.update(return_dict)
        return return_dict

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
        actuator_type: ActuatorType | str,
        forcelimited: bool = True,
        forcerange: tuple[float, float] = (-50.0, 50.0),
        **kwargs,
    ) -> dict[JointDOF, mjcf.Element]:
        actuator_type = ActuatorType(actuator_type)
        return_dict = {}
        for jointdof in jointdofs:
            actuator = self.mjcf_root.actuator.add(
                actuator_type.value,
                name=f"{jointdof.name}-{actuator_type.value}",
                joint=jointdof.name,
                forcelimited=forcelimited,
                forcerange=forcerange,
                **kwargs,
            )
            return_dict[jointdof] = actuator
            self.jointdof_to_mjcfactuator_by_type[jointdof][actuator_type] = actuator
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
