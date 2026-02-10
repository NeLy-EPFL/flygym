from pathlib import Path
from os import PathLike
from enum import Enum
from fnmatch import filter as filter_with_wildcard
from typing import Iterable, Any, override

import mujoco
import numpy as np
import dm_control.mjcf as mjcf
import yaml

from flygym import assets_dir
from flygym.anatomy import (
    BodySegment,
    JointDOF,
    Skeleton,
    RotationAxis,
    AxisOrder,
    JointPreset,
    ALL_SEGMENT_NAMES,
)
from flygym.compose.base import BaseCompositionElement
from flygym.compose.pose import KinematicPose
from flygym.utils.mjcf import set_mujoco_globals
from flygym.utils.math import Vec3, Rotation3D
from flygym.utils.exceptions import FlyGymInternalError

__all__ = ["Fly", "ActuatorType"]


DEFAULT_RIGGING_CONFIG_PATH = assets_dir / "model/rigging.yaml"
DEFAULT_MUJOCO_GLOBALS_PATH = assets_dir / "model/mujoco_globals.yaml"
DEFAULT_MESH_DIR = assets_dir / "model/meshes"
DEFAULT_VISUALS_CONFIG_PATH = assets_dir / "model/visuals.yaml"


class Fly(BaseCompositionElement):
    """Represents a complete fly with body segments, joints, actuators, sensors, and
    cameras. The fly is built from mesh assets and configured via config files that
    define rigging (joint positions), visuals (colors/textures), and global MuJoCo
    parameters.

    The fly uses a hierarchical body structure with a root segment (typically the
    thorax) from which all other segments branch. Joints and actuators are added
    separately after initialization to allow flexible model configurations.

    Args:
        name:
            Identifier for this fly instance.
        rigging_config_path:
            Path to YAML file defining body segment positions, orientations, and masses.
        mesh_dir:
            Directory containing STL mesh files for body segments.
        mujoco_globals_path:
            Path to YAML file with global MuJoCo parameters (timestep, gravity, etc.).
        root_segment:
            Root body segment for the kinematic tree (e.g., `c_thorax`).
        mirror_left2right:
            If True, mirror left-side meshes for right side instead of loading separate
            mesh files. This reduces asset size and ensures symmetry.

    Attributes:
        skeleton:
            Joint structure of the fly, set when add_joints() is called.
        bodyseg_to_mjcfmesh:
            Maps body segments to MJCF mesh elements.
        bodyseg_to_mjcfbody:
            Maps body segments to MJCF body elements.
        bodyseg_to_mjcfgeom:
            Maps body segments to MJCF geometry elements.
        jointdof_to_mjcfjoint:
            Maps joint DOFs to MJCF joint elements.
        jointdof_to_mjcfactuator_by_type:
            Maps actuator type to a further dictionary, which maps joint DOFs to MJCF
            actuator elements (only if the actuator exists).
        sensorname_to_mjcfsensor:
            Maps sensor names to MJCF sensor elements.
        cameraname_to_mjcfcamera:
            Maps camera names to MJCF camera elements.
        jointdof_to_neutralangle:
            Neutral (resting) angle for each joint DOF.
        jointdof_to_neutralaction_by_type:
            Neutral actuator input for each (actuator_type, joint_dof) pair. Maps
            actuator type to a further dictionary, which maps joint DOFs to their
            neutral actuator input (only if the actuator exists).
    """

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

        self.skeleton: Skeleton | None = None

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

    @override
    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._name

    def get_bodysegs_order(self) -> Iterable[BodySegment]:
        """Get the canonical order of body segments. The exact order is not important,
        but it should be respected consistently throughout. For example, during
        simulation, the fly body state returned by the simulator will be in this order.
        """
        return list(self.bodyseg_to_mjcfbody.keys())

    def get_jointdofs_order(self) -> Iterable[JointDOF]:
        """Same as `get_bodysegs_order()`, but for joint DoFs instead of body segments."""
        return list(self.jointdof_to_mjcfjoint.keys())

    def get_actuated_jointdofs_order(
        self, actuator_type: "ActuatorType | str"
    ) -> Iterable[JointDOF]:
        """Same as `get_jointdofs_order()`, but only for the subset of joint DoFs that
        are actuated by the specified actuator type. During simulation, the user should
        provide control input in this order."""
        actuator_type = ActuatorType(actuator_type)
        return list(self.jointdof_to_mjcfactuator_by_type[actuator_type].keys())

    def add_joints(
        self,
        skeleton: Skeleton,
        neutral_pose: KinematicPose | None = None,
        *,
        stiffness: float = 10.0,
        damping: float = 0.5,
        armature: float = 1e-6,
        **kwargs,
    ) -> dict[JointDOF, mjcf.Element]:
        """Add joints to the fly model based on a skeleton definition.

        Creates hinge joints connecting body segments according to the skeleton's
        kinematic tree structure. Each joint is configured with passive spring-damper
        dynamics and a neutral (resting) angle.

        Args:
            skeleton:
                Skeleton defining which joints to create and their DOFs.
            neutral_pose:
                Resting angles for joints. If provided, must match skeleton's axis
                order. If not provided, all neutral angles default to 0.
            stiffness:
                Joint stiffness (spring constant).
            damping:
                Joint damping coefficient.
            armature:
                Additional inertia added to the joint for numerical stability. Should be
                small enough to not affect dynamics.
            **kwargs:
                Additional arguments passed to MJCF joint creation. See
                `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint>`_
                for details on supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF joint elements.
        """
        if neutral_pose is None:
            neutral_angle_lookup = {}
        else:
            if not isinstance(neutral_pose, KinematicPose):
                raise ValueError(
                    "When specified, `neutral_pose` must be a `KinematicPose`."
                )
            neutral_angle_lookup = neutral_pose.get_angles_lookup(skeleton.axis_order)

        self.skeleton = skeleton

        return_dict = {}
        for jointdof in skeleton.iter_jointdofs(self.root_segment):
            child_body = self.bodyseg_to_mjcfbody[jointdof.child]
            neutral_angle = neutral_angle_lookup.get(jointdof.name, 0.0)
            self.jointdof_to_neutralangle[jointdof] = neutral_angle

            # Flip axis direction for right side's roll and yaw so that axes are defined
            # symmetrically (e.g., positive roll is always "outward").
            vec = np.array(jointdof.axis.to_vector())
            if jointdof.child.pos[0] == "r" and jointdof.axis != RotationAxis.PITCH:
                vec = -vec

            return_dict[jointdof] = child_body.add(
                "joint",
                name=jointdof.name,
                type="hinge",
                axis=vec,
                stiffness=stiffness,
                damping=damping,
                armature=armature,
                springref=neutral_angle,
                **kwargs,
            )

        self.jointdof_to_mjcfjoint.update(return_dict)
        self._rebuild_neutral_keyframe()
        return return_dict

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
        """Add actuators to specified joints.

        Creates actuators that can apply forces/torques to joints. Multiple actuator
        types can be added to the same joints.

        Args:
            jointdofs:
                Joint DOFs to actuate.
            actuator_type:
                Type of actuator (motor, position, velocity, etc.).
            neutral_input:
                Default actuator inputs. If None, defaults to 0 for all actuators. For
                position actuators, these are joint angles and therefore must match
                skeleton axis order.
            forcelimited:
                If True, actuators cannot exceed forcerange.
            forcerange:
                Force limit as a (min, max) tuple.
            **kwargs:
                Additional arguments passed to MJCF actuator creation (e.g., kp for
                position actuators, kv for velocity actuators). See
                `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator>`_
                for details on supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF actuator elements.
        """
        if neutral_input is None:
            neutral_input = {}
        if isinstance(neutral_input, KinematicPose) and (
            ActuatorType(actuator_type) == ActuatorType.POSITION
        ):
            neutral_input = neutral_input.get_angles_lookup(self.skeleton.axis_order)

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
                **kwargs,
            )
            return_dict[jointdof] = actuator
        self.jointdof_to_mjcfactuator_by_type[actuator_type].update(return_dict)
        self._rebuild_neutral_keyframe()
        return return_dict

    def colorize(
        self, visuals_config_path: PathLike = DEFAULT_VISUALS_CONFIG_PATH
    ) -> None:
        """Apply colors and textures to fly model based on a YAML configuration file."""
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

    def add_tracking_camera(
        self,
        name: str = "trackcam",
        mode: str = "track",
        pos_offset: Vec3 = (0, -7.5, 6),
        rotation: Rotation3D = Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)),
        fovy: float = 30.0,
        **kwargs,
    ) -> mjcf.Element:
        """Add a camera that tracks the fly's root body.

        See `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera>`_
        for details on supported attributes.
        """
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
        my_rigging_config: dict[str, Any],
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


class ActuatorType(Enum):
    """Actuator types supported by MuJoCo.
    See `MuJoCo XML reference <https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator>`_
    for details on each type."""

    MOTOR = "motor"
    POSITION = "position"
    VELOCITY = "velocity"
    INTVELOCITY = "intvelocity"
    DAMPER = "damper"
    CYLINDER = "cylinder"
    MUSCLE = "muscle"
    ADHESION = "adhesion"
