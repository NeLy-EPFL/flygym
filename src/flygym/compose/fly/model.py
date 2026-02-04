from pathlib import Path
from os import PathLike
from enum import Enum
from fnmatch import filter as filter_with_wildcard
from typing import Iterable

import dm_control.mjcf as mjcf
import yaml

import flygym
import flygym.compose.fly.anatomy as anatomy
from flygym.compose.base import BaseFly
from flygym.compose.util import set_params_recursive
from flygym.utils.math import Vec3, Rotation3D

DEFAULT_RIGGING_CONFIG_PATH = flygym.assets_dir / "model/rigging.yaml"
DEFAULT_MUJOCO_PARAMS_PATH = flygym.assets_dir / "model/mujoco_params.yaml"
DEFAULT_MESH_DIR = flygym.assets_dir / "model/meshes"
DEFAULT_VISUALS_CONFIG_PATH = flygym.assets_dir / "model/visuals.yaml"


class ActuatorType(Enum):
    MOTOR = "motor"
    POSITION = "position"
    VELOCITY = "velocity"
    INTVELOCITY = "intvelocity"
    DAMPER = "damper"
    CYLINDER = "cylinder"
    MUSCLE = "muscle"
    ADHESION = "adhesion"


class Fly(BaseFly):
    def __init__(
        self,
        name: str = "neuromechfly",
        *,
        rigging_config_path: PathLike = DEFAULT_RIGGING_CONFIG_PATH,
        mesh_dir: PathLike = DEFAULT_MESH_DIR,
        mujoco_params_path: PathLike = DEFAULT_MUJOCO_PARAMS_PATH,
        group: int = 1,
        root_segment: anatomy.BodySegment | str = "c_thorax",
        default_class: str = "neuromechfly",
        mirror_right_from_left: bool = True,
    ) -> None:
        self.name = name
        self.group = group
        if isinstance(root_segment, str):
            root_segment = anatomy.BodySegment(root_segment)
        self.root_segment = root_segment
        self.default_class = default_class
        self.mjcf_model = mjcf.RootElement(model=name)
        self.mjcf_model.default.add("default", dclass=default_class)
        self._set_mujoco_params(mujoco_params_path)
        self._add_mesh_assets(
            mesh_dir, scale=1000, mirror_right_from_left=mirror_right_from_left
        )
        self.bodies, self.geoms = self._add_bodies_and_geoms(rigging_config_path)
        self.joints = {}
        self.actuators = {}
        self.sensors = {}

    def _set_mujoco_params(self, mujoco_params_path: PathLike) -> None:
        with open(mujoco_params_path) as f:
            mujoco_params = yaml.safe_load(f)
        set_params_recursive(self.mjcf_model, mujoco_params)

    def _add_mesh_assets(
        self,
        mesh_dir: PathLike,
        scale: float,
        mirror_right_from_left: bool,
    ) -> None:
        mesh_dir = Path(mesh_dir)
        for segment_name in anatomy.ALL_SEGMENT_NAMES:
            if mirror_right_from_left and segment_name[0] == "r":
                mesh_to_use = f"l{segment_name[1:]}"
                y_sign = -1
            else:
                mesh_to_use = segment_name
                y_sign = 1
            mesh_path = (mesh_dir / f"{mesh_to_use}.stl").resolve()
            if not mesh_path.exists():
                raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
            self.mjcf_model.asset.add(
                "mesh",
                name=segment_name,
                file=str(mesh_path),
                dclass=self.default_class,
                scale=(scale, y_sign * scale, scale),
            )

    def _add_bodies_and_geoms(
        self,
        rigging_config_path: PathLike,
    ) -> tuple[
        dict[anatomy.BodySegment, mjcf.Element], dict[anatomy.BodySegment, mjcf.Element]
    ]:
        # Load rigging config
        with open(rigging_config_path) as f:
            rigging_config = yaml.safe_load(f)

        # Add root body and geom
        virtual_root = self.mjcf_model.worldbody.add("body", name="rootbody")
        body, geom = self._add_one_body_and_geom(
            virtual_root, self.root_segment, rigging_config[self.root_segment.name]
        )
        body_lookup = {self.root_segment: body}
        geom_lookup = {self.root_segment: geom}

        # Add all other bodies and geoms
        full_skeleton = anatomy.Skeleton(joint_preset=anatomy.JointPreset.ALL_POSSIBLE)
        # The axis order doesn't matter here: we're only adding bodies/geoms, not joints
        axis_order = anatomy.AxisOrder.PITCH_ROLL_YAW
        for joint_dof in full_skeleton.iter_joint_dofs(axis_order, self.root_segment):
            if joint_dof.axis != anatomy.RotationAxis.PITCH:
                # Look at only 1 DoF per joint as we're still just adding bodies/geoms
                continue
            parent_body = body_lookup.get(joint_dof.parent)
            assert parent_body is not None, "Kinematic tree DFS error"
            my_rigging_config = rigging_config.get(joint_dof.child.name)
            assert my_rigging_config is not None, "Missing rigging config for body"
            body, geom = self._add_one_body_and_geom(
                parent_body, joint_dof.child, my_rigging_config
            )
            body_lookup[joint_dof.child] = body
            geom_lookup[joint_dof.child] = geom

        return body_lookup, geom_lookup

    def _add_one_body_and_geom(
        self,
        parent_body: mjcf.Element,
        segment: anatomy.BodySegment,
        my_rigging_config: dict[str, ...],
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
            dclass=self.default_class,
            group=self.group,
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
        if not hasattr(self, "geoms"):
            raise ValueError("Must first add geoms via `_add_bodies_and_geoms`.")

        vis_sets_all, lookup = self._parse_visuals_config(visuals_config_path)

        for vis_set_name, params in vis_sets_all.items():
            material = self.mjcf_model.asset.add(
                "material", name=vis_set_name, **params["material"]
            )
            if texture_params := params.get("texture"):
                texture = self.mjcf_model.asset.add(
                    "texture", name=vis_set_name, **texture_params
                )
                material.texture = texture

        for segment, geom in self.geoms.items():
            vis_set_name = lookup[segment]
            geom.set_attributes(material=vis_set_name)

    def add_joints(
        self,
        skeleton: anatomy.Skeleton,
        axis_order: anatomy.AxisOrderLike,
        stiffness: float = 10.0,
        damping: float = 0.5,
        **kwargs,
    ) -> None:
        for joint_dof in skeleton.iter_joint_dofs(axis_order, self.root_segment):
            child_body = self.bodies[joint_dof.child]
            self.joints[joint_dof] = child_body.add(
                "joint",
                name=joint_dof.name,
                dclass=self.default_class,
                type="hinge",
                axis=joint_dof.axis.to_vector(),
                stiffness=stiffness,
                damping=damping,
                **kwargs,
            )

    @staticmethod
    def _parse_visuals_config(
        visuals_config_path: PathLike,
    ) -> tuple[dict[str, dict], dict[anatomy.BodySegment, dict]]:
        # Load visuals config and assign vis sets to body segments
        with open(visuals_config_path) as f:
            vis_set_params_all = yaml.safe_load(f)
        all_matches_by_segname = {k: [] for k in anatomy.ALL_SEGMENT_NAMES}
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
                target_segnames |= set(
                    filter_with_wildcard(anatomy.ALL_SEGMENT_NAMES, pattern)
                )
            for segname in target_segnames:
                all_matches_by_segname[segname].append(vis_set_name)
        for segname, vis_set_names in all_matches_by_segname.items():
            if len(vis_set_names) != 1:
                raise ValueError(
                    f"Zero or multiple vis sets matched for body segment {segname}: "
                    f"{vis_set_names}. Only one should apply."
                )
        lookup_by_segname = {
            anatomy.BodySegment(segname): matches[0]
            for segname, matches in all_matches_by_segname.items()
        }
        return vis_set_params_all, lookup_by_segname

    def add_actuators(
        self,
        joint_dofs: Iterable[anatomy.JointDOF],
        actuator_type: ActuatorType | str,
        dclass: str = "neuromechfly",
        group: int = 1,
        forcelimited: bool = True,
        forcerange: tuple[float, float] = (-50.0, 50.0),
        **kwargs,
    ) -> None:
        actuator_type = ActuatorType(actuator_type)
        for joint_dof in joint_dofs:
            actuator_name = f"{joint_dof.name}-{actuator_type.value}"
            self.actuators[(joint_dof, actuator_type)] = self.mjcf_model.actuator.add(
                actuator_type.value,
                name=actuator_name,
                joint=joint_dof.name,
                dclass=dclass,
                group=group,
                forcelimited=forcelimited,
                forcerange=forcerange,
                **kwargs,
            )

    def add_tracking_camera(
        self,
        name: str,
        mode: str = "track",
        pos_offset: Vec3 = (0, -7.5, 6),
        rotation: Rotation3D = Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)),
        fovy: float = 30.0,
        **kwargs,
    ) -> None:
        self.mjcf_model.worldbody.add(
            "camera",
            name=name,
            mode=mode,
            target="rootbody",
            pos=pos_offset,
            fovy=fovy,
            **rotation.as_kwargs(),
            **kwargs,
        )

    def get_mjcf_root(self) -> mjcf.RootElement:
        return self.mjcf_model
