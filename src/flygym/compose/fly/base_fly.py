import warnings
from os import PathLike
from enum import Enum
from fnmatch import filter as filter_with_wildcard
from typing import Iterable, Any, override

import mujoco as mj
import numpy as np
import yaml

from flygym.anatomy import (
    BodySegment,
    AnatomicalJoint,
    JointDOF,
    Skeleton,
    RotationAxis,
    AxisOrder,
    JointPreset,
    ContactBodiesPreset,
    ALL_SEGMENT_NAMES,
    LEGS,
    LEG_LINKS,
)

from flygym.compose.base import BaseCompositionElement
from flygym.compose.pose import KinematicPose, KinematicPosePreset
from flygym.utils.mjcf import (
    set_mujoco_globals,
    add_actuator,
    add_material,
    add_texture,
    GEOM_TYPES,
    JOINT_TYPES,
    CAMERA_MODES,
)
from flygym.utils.math import Vec3, Rotation3D
from flygym.utils.exceptions import FlyGymInternalError

__all__ = ["BaseFly", "ActuatorType", "MeshType", "GeomFittingOption"]


class MeshType(Enum):
    """Mesh resolution to use for fly body geometry.

    Attributes:
        FULLSIZE: Original high-resolution meshes.
        SIMPLIFIED_MAX2000FACES: Simplified meshes with at most 2000 faces per
            segment. Faster to render and simulate. Used by default.
    """

    FULLSIZE = "fullsize"
    SIMPLIFIED_MAX2000FACES = "simplified_max2000faces"


class GeomFittingOption(Enum):
    """How to fit collision geometries to the mesh shapes.

    Attributes:
        UNMODIFIED: Keep the original mesh-based geometries.
        ALL_TO_CAPSULES: Replace all geometries with capsule approximations.
        CLAWS_TO_CAPSULES: Replace only tarsus5 geometries with capsules.
    """

    UNMODIFIED = "unmodified"
    ALL_TO_CAPSULES = "all_to_capsules"
    CLAWS_TO_CAPSULES = "claws_to_capsules"


class ActuatorType(Enum):
    """Actuator types supported by MuJoCo.
    See [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator)
    for details on each type."""

    MOTOR = "motor"
    POSITION = "position"
    VELOCITY = "velocity"
    INTVELOCITY = "intvelocity"
    DAMPER = "damper"
    CYLINDER = "cylinder"
    MUSCLE = "muscle"
    ADHESION = "adhesion"
    TENDON = "tendon"


class BaseFly(BaseCompositionElement):
    """Abstract base for all fly body models (e.g. `NeuroMechFly`, `FlyBody`).

    FlyGym supports multiple fly body models that can be used interchangeably within
    the same simulation API. Concrete subclasses provide model-specific geometry assets,
    anatomy classes, and parameter defaults, while all model-agnostic composition logic
    lives here. Choose the model that best fits your experiment:

    - `NeuroMechFly` — The default model, derived from micro-CT imaging. Suitable for
      most locomotion and sensorimotor experiments.
    - `FlyBody` — A biomechanically detailed model from Vaxenburg et al. (2025), with
      wing and abdomen degrees of freedom and anatomically grounded parameters.

    This class is not meant to be instantiated directly — use a concrete subclass.

    Holds the model-agnostic composition logic shared by every fly. Concrete
    subclasses supply the model identity (asset paths, anatomy classes, scale)
    and may override individual build steps.

    Represents a complete fly with body segments, joints, actuators, sensors, and
    cameras. The fly is built from mesh assets and configured via config files that
    define rigging (joint positions), visuals (colors/textures), and global MuJoCo
    parameters.

    The fly uses a hierarchical body structure with a root segment (typically the
    thorax) from which all other segments branch. Joints and actuators are added
    separately after initialization to allow flexible model configurations.

    !!! warning "PyMJCF -> MjSpec migration (v2.1.0)"

        FlyGym 2.0.3 dropped the PyMJCF backend in favour of MuJoCo's native
        ``MjSpec`` API. If you are upgrading from an earlier version, see the
        [v2.1.0 changelog](https://neuromechfly.org/changelog/#version-210)
        for breaking changes and a migration guide.

    Args:
        name:
            Identifier for this fly instance.
        rigging_config_path:
            Path to YAML file defining body segment positions, orientations, and masses.
        mesh_basedir:
            Directory containing STL mesh files for body segments.
        mujoco_globals_path:
            Path to YAML file with global MuJoCo parameters (timestep, gravity, etc.).
        root_segment:
            Root body segment for the kinematic tree (e.g., ``c_thorax``).
        mirror_left2right:
            If True, mirror left-side meshes for right side instead of loading separate
            mesh files. Reduces asset size and ensures symmetry.
        mesh_type:
            Mesh resolution to use.
        geom_fitting_option:
            How to fit collision geometries.

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
        anatomicaljoint_to_mjcfsites:
            Maps anatomical joints to MJCF site elements.
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

    # For numerical reasons, we simulate length in mm, not m. This changes the units
    # of other quantities as well, for example acceleration is now in mm/s^2.
    SCALE = 1000
    BODY_SEGMENT_CLASS = BodySegment
    JOINT_DOF_CLASS = JointDOF
    AXIS_ORDER_CLASS = AxisOrder
    BASE_SKELETON_CLASS = Skeleton
    CONTACT_BODIES_PRESET_CLASS = ContactBodiesPreset
    LEG_LINKS = LEG_LINKS

    def __init__(
        self,
        name: str = "fly",
        *,
        rigging_config_path: PathLike,
        mesh_basedir: PathLike,
        mujoco_globals_path: PathLike,
        mirror_left2right: bool,
        mesh_type: MeshType,
        vision_config_path: PathLike,
        root_segment: BodySegment | str = "c_thorax",
        geom_fitting_option: GeomFittingOption = GeomFittingOption.UNMODIFIED,
    ) -> None:
        self._name = name
        self._mjcf_root = mj.MjSpec()
        self._mjcf_root.modelname = name
        # Keep the globals path so the world can inherit the fly's physics settings:
        # MjSpec.attach() does not merge the child's <option>/<compiler> into the
        # parent, so these must be (re)applied to the world spec that gets compiled.
        self.mujoco_globals_path = mujoco_globals_path
        set_mujoco_globals(self.mjcf_root, mujoco_globals_path)

        self.skeleton: Skeleton | None = None

        self.bodyseg_to_mjcfmesh = {}
        self.bodyseg_to_mjcfbody = {}
        self.bodyseg_to_mjcfgeom = {}
        self.jointdof_to_mjcfjoint = {}
        self.jointdof_to_mjcfactuator_by_type = {ty: {} for ty in ActuatorType}
        self.leg_to_adhesionactuator = {}
        self.anatomicaljoint_to_mjcfsites = {}
        self.sensorname_to_mjcfsensor = {}
        self.cameraname_to_mjcfcamera = {}
        self.eyecameraname_to_mjcfcamera = {}

        self.jointdof_to_neutralangle = {}
        self.jointdof_to_neutralaction_by_type = {ty: {} for ty in ActuatorType}

        if isinstance(root_segment, str):
            root_segment = self.BODY_SEGMENT_CLASS(root_segment)
        self.root_segment = root_segment

        self._neutral_keyframe = self.mjcf_root.add_key(name="neutral", time=0)

        self._add_mesh_assets(mesh_basedir, mirror_left2right, mesh_type)
        self._add_bodies_and_geoms(
            rigging_config_path, geom_fitting_option, vision_config_path
        )
        self.vision_config_path = vision_config_path

    @override
    @property
    def mjcf_root(self) -> mj.MjSpec:
        return self._mjcf_root

    @property
    def name(self) -> str:
        """Name of this fly instance."""
        return self._name

    @override
    def compile(self) -> tuple[mj.MjModel, mj.MjData]:
        """Compile the fly on its own (e.g. for `preview_model` or `save_xml`).

        Disables `fusestatic` for the standalone compile. A lone fly has no free joint,
        so its root segment is a static body that the optimization would fuse into the
        worldbody -- which breaks the `track`-mode tracking camera parented to it (the
        camera would fall back to tracking the worldbody and mis-place itself relative to
        the fly). A standalone fly is only ever inspected/previewed, never simulated, so
        the lost optimization does not matter here. The setting is applied to the
        compiled copy only, leaving the live spec untouched so a world can still fuse
        the fly's other static bodies once it is attached.

        As in the base method, we always compile a *copy* rather than the live spec:
        compiling mutates the spec in place, which would invalidate the element
        references FlyGym holds.
        """
        spec = self.mjcf_root.copy()
        if spec.compiler.fusestatic:
            warnings.warn(
                "Compiling a fly model that is not attached to a world. "
                "`fusestatic` is changed to false to prevent the root body segment "
                "from being fused with the MJCF root, which would impair the placement "
                "of the tracking camera."
            )
            spec.compiler.fusestatic = False
        model = spec.compile()
        return model, mj.MjData(model)

    def get_bodysegs_order(self) -> list[BodySegment]:
        """Get the canonical order of body segments. The exact order is not important,
        but it should be respected consistently throughout. For example, during
        simulation, the fly body state returned by the simulator will be in this order.
        """
        return list(self.bodyseg_to_mjcfbody.keys())

    def get_jointdofs_order(self) -> list[JointDOF]:
        """Same as `get_bodysegs_order()`, but for joint DoFs instead of body segments."""
        return list(self.jointdof_to_mjcfjoint.keys())

    def get_actuated_jointdofs_order(
        self, actuator_type: "ActuatorType | str"
    ) -> list[JointDOF]:
        """Same as `get_jointdofs_order()`, but only for the subset of joint DoFs that
        are actuated by the specified actuator type. During simulation, the user should
        provide control input in this order."""
        actuator_type = ActuatorType(actuator_type)
        return list(self.jointdof_to_mjcfactuator_by_type[actuator_type].keys())

    def get_legs_order(self) -> list[str]:
        """Get the ordered list of leg position identifiers (same as `anatomy.LEGS`)."""
        return LEGS

    def get_pose_lookup(
        self, neutral_pose: KinematicPose | KinematicPosePreset | None
    ) -> dict[str, float]:
        """Get a lookup dictionary mapping joint DOF names to neutral angles for a given
        neutral pose."""

        if self.skeleton is None:
            raise FlyGymInternalError("Skeleton must be defined to get pose lookup.")

        if neutral_pose is None:
            return {}
        elif isinstance(neutral_pose, KinematicPose):
            return neutral_pose.joint_angles_lookup_rad
        elif isinstance(neutral_pose, KinematicPosePreset):
            neutral_pose = neutral_pose.get_pose_by_axis_order(self.skeleton.axis_order)
            return neutral_pose.joint_angles_lookup_rad
        else:
            raise ValueError(
                "When specified, `neutral_pose` must be a "
                "`KinematicPose` or `KinematicPosePreset`."
            )

    def get_sites_order(self) -> list[AnatomicalJoint]:
        """Get the canonical order of anatomical joints with associated MJCF sites.

        This is the order used by simulation site-state readout methods such as
        ``Simulation.get_site_positions``.
        """
        return list(self.anatomicaljoint_to_mjcfsites.keys())

    def add_joints(
        self,
        skeleton: Skeleton,
        neutral_pose: KinematicPose | KinematicPosePreset | None = None,
        *,
        stiffness: float = 10.0,
        damping: float = 0.5,
        armature: float = 1e-6,
        **kwargs: Any,
    ) -> dict[JointDOF, mj.MjsJoint]:
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
                [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
                for details on supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF joint elements.
        """

        self.skeleton = skeleton
        neutral_angle_lookup = self.get_pose_lookup(neutral_pose)

        return_dict = {}
        for jointdof in skeleton.iter_jointdofs(self.root_segment):
            child_body = self.bodyseg_to_mjcfbody[jointdof.child]
            neutral_angle = neutral_angle_lookup.get(jointdof.name, 0.0)
            self.jointdof_to_neutralangle[jointdof] = neutral_angle

            # Flip axis direction for right side's roll and yaw so that axes are defined
            # symmetrically (e.g., positive roll is always "outward").
            vec = np.array(jointdof.axis.to_vector())
            if jointdof.child.pos[0] == "r" and not self._is_pitch(jointdof):
                vec = -vec

            return_dict[jointdof] = child_body.add_joint(
                name=jointdof.name,
                type=JOINT_TYPES["hinge"],
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
        neutral_input: "dict[str, float] | KinematicPose | KinematicPosePreset | None" = None,
        *,
        forcelimited: bool = True,
        forcerange: tuple[float, float] = (-30.0, 30.0),
        **kwargs: Any,
    ) -> dict[JointDOF, mj.MjsActuator]:
        """Add actuators to specified joints.

        Creates actuators that can apply forces/torques to joints. Multiple actuator
        types can be added to the same joints.

        Args:
            jointdofs:
                Joint DOFs to actuate.
            actuator_type:
                Type of actuator (motor, position, velocity, etc.).
            neutral_input:
                Default actuator inputs. Accepts a ``dict`` mapping DoF names to
                values, a `KinematicPose`, or a `KinematicPosePreset`. If None,
                defaults to 0 for all actuators. For position actuators the values
                are joint angles and must match the skeleton axis order.
            forcelimited:
                If True, actuators cannot exceed forcerange.
            forcerange:
                Force limit as a (min, max) tuple.
            **kwargs:
                Additional arguments passed to MJCF actuator creation (e.g., kp for
                position actuators, kv for velocity actuators). See
                [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator)
                for details on supported attributes.

        Returns:
            Dictionary mapping JointDOF to created MJCF actuator elements.
        """
        actuator_type = ActuatorType(actuator_type)

        if actuator_type == ActuatorType.POSITION:
            neutral_input = self.get_pose_lookup(neutral_input)
        else:
            if isinstance(neutral_input, (KinematicPose, KinematicPosePreset)):
                raise ValueError(
                    "When actuator_type is not POSITION, neutral_input cannot be a "
                    "KinematicPose or KinematicPosePreset since those specify joint "
                    "angles, not actuator inputs."
                )
            else:
                neutral_input = {} if neutral_input is None else neutral_input

        return_dict = {}
        for jointdof in jointdofs:
            self.jointdof_to_neutralaction_by_type[actuator_type][jointdof] = (
                neutral_input.get(jointdof.name, 0.0)
            )
            actuator = add_actuator(
                self.mjcf_root,
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

    def add_joint_sites(
        self, anatomical_joints: list[AnatomicalJoint]
    ) -> dict[AnatomicalJoint, mj.MjsSite]:
        """Add MJCF sites at the origins of selected anatomical joints.

        Each site is placed at ``(0, 0, 0)`` in the child body frame. Since body
        origins are defined at their parent-child joint locations in this model,
        these sites track anatomical joint positions in world coordinates during
        simulation.

        Args:
            anatomical_joints: Anatomical joints to materialize as MJCF sites.

        Returns:
            Dictionary mapping each anatomical joint to its created MJCF site
            element (same entries added into ``self.anatomicaljoint_to_mjcfsites``).

        Raises:
            ValueError: If a site for a requested anatomical joint already exists.
        """
        return_dict = {}
        for joint in anatomical_joints:
            if joint in self.anatomicaljoint_to_mjcfsites:
                raise ValueError(
                    f"A site has already been added for anatomical joint '{joint.name}'."
                )
            child_body_element = self.bodyseg_to_mjcfbody[joint.child]
            site = child_body_element.add_site(
                name=joint.name,
                pos=(0, 0, 0),  # origin of child body is defined at joint to parent
            )
            return_dict[joint] = site
        self.anatomicaljoint_to_mjcfsites.update(return_dict)
        return return_dict

    def add_leg_adhesion(
        self, gain: float | dict[str, float] = 1.0
    ) -> dict[str, mj.MjsActuator]:
        """Add adhesion actuators to the tarsus5 segments of all legs.

        Adhesion actuators apply a normal attraction force, enabling the fly to grip
        surfaces. The control input per leg ranges from 0 to 1, where 0 fully
        releases adhesion and 1 applies the configured gain.

        Args:
            gain: Adhesion actuator gain. Either a single float applied to all legs,
                or a dict mapping leg position identifiers to per-leg gain values.

        Returns:
            Dict mapping leg position identifier to the created MJCF adhesion
            actuator element (same as ``self.leg_to_adhesionactuator``).

        Raises:
            ValueError: If adhesion actuators have already been added.
        """
        if len(self.leg_to_adhesionactuator) > 0:
            raise ValueError("Leg adhesion actuators have already been added.")
        for leg in LEGS:
            tarsus5 = BodySegment(f"{leg}_tarsus5")
            if isinstance(gain, dict):
                gain_this_leg = gain[leg]
            else:
                gain_this_leg = gain
            self.leg_to_adhesionactuator[leg] = add_actuator(
                self.mjcf_root,
                "adhesion",
                name=f"{tarsus5.name}-adhesion",
                body=self.bodyseg_to_mjcfbody[tarsus5].name,
                gain=gain_this_leg,
                ctrlrange=(0, 1),
            )
        return self.leg_to_adhesionactuator

    def add_vision(self, draw_sensor_markers: bool = False) -> None:
        with open(self.vision_config_path) as f:
            info = yaml.safe_load(f)

        return_dict = {}

        for sensor_name, sensor_info in info["sensors"].items():
            parent_body = self.mjcf_root.body(sensor_info["parent"])
            sensor_body = parent_body.add_body(
                name=f"{sensor_name}_body",
                pos=sensor_info["rel_pos"],
            )
            cam = sensor_body.add_camera(
                name=f"{sensor_name}_camera",
                mode=CAMERA_MODES["fixed"],
                euler=sensor_info["orientation"],
                fovy=info["fovy_per_eye"],
            )

            # Add visual markers indicating where the eye sensors are
            # The MuJoCo renderer by default renders geoms of groups 0, 1, 2.
            # By convention, group 0 is for main visual/collision bodies, group 1 is for
            # helper/mocap geoms, and group 2 is for debug geoms. So if the user wants
            # to draw sensor markers, we put them in group 1. Among the groups that are
            # invisible by default, group 3 is often used for simplified physics geoms,
            # and group 4 is often for additional stuff. So if the user doesn't want to
            # draw sensor markers, we put them in group 4.
            geom_group = 1 if draw_sensor_markers else 4
            sensor_body.add_geom(
                name=f"{sensor_name}_marker",
                type=GEOM_TYPES["sphere"],
                size=[0.06, 0, 0],
                rgba=sensor_info["marker_rgba"],
                mass=0,
                contype=0,
                conaffinity=0,
                group=geom_group,
            )

            return_dict[sensor_name] = cam

        self.eyecameraname_to_mjcfcamera.update(return_dict)

    def colorize(self, visuals_config_path: PathLike) -> None:
        """Apply colors and textures to the fly model.

        Args:
            visuals_config_path: Path to the YAML file defining per-segment material
                and texture assignments.
        """
        if len(self.bodyseg_to_mjcfgeom) == 0:
            raise ValueError("Must first add geoms via `_add_bodies_and_geoms`.")

        vis_sets_all, lookup = self._parse_visuals_config(visuals_config_path)

        for vis_set_name, params in vis_sets_all.items():
            texture_name = None
            if texture_params := params.get("texture"):
                add_texture(self.mjcf_root, name=vis_set_name, **texture_params)
                texture_name = vis_set_name
            add_material(
                self.mjcf_root,
                name=vis_set_name,
                texture=texture_name,
                **params["material"],
            )

        for _, geoms in self.bodyseg_to_mjcfgeom.items():
            for geom in geoms:
                geom_name = geom.name
                vis_set_name = lookup[geom_name]
                geom.material = vis_set_name

    def add_tracking_camera(
        self,
        name: str = "trackcam",
        mode: str = "track",
        pos_offset: Vec3 = (-0.5, -7.5, 5),
        rotation: Rotation3D = Rotation3D("xyaxes", (1, 0, 0, 0, 0.6, 0.8)),
        fovy: float = 30.0,
        **kwargs: Any,
    ) -> mj.MjsCamera:
        """Add a camera that tracks the fly's root body.

        The camera is added *inside* the root segment's body element. MuJoCo's
        ``track``/``trackcom`` modes follow the camera's parent body, so the camera
        must be a child of the fly body to follow it; a camera placed in the world
        body would stay put. ``track`` follows the body's position while keeping a
        constant orientation in the world frame (a "follow" camera that pans but does
        not rotate with the fly).

        !!! warning

            ``pos_offset`` is expressed in the root segment's (thorax) body frame, not
            in world coordinates. This differs from FlyGym versions before the MjSpec
            migration, where the tracking camera lived in the world body and the offset
            was effectively a world-frame position. The default changed accordingly,
            from ``(0, -7.5, 6)`` to ``(-0.5, -7.5, 5)``. Hard-coded ``pos_offset``
            values tuned for the old world-frame placement must be re-tuned: the root
            segment sits roughly ``(0.5, 0, 1.3)`` mm from the fly's attachment point
            (plus the spawn height) in the neutral pose, so the same offset now places
            the camera higher and shifted toward the head. The upside is that a given
            ``pos_offset`` now yields the same camera position relative to the fly in
            every world and when the fly is compiled on its own.

        Args:
            name: Camera name.
            mode: MuJoCo camera tracking mode (``"track"``, ``"trackcom"``, or
                ``"fixed"``). ``"fixed"`` rigidly attaches the camera to the body so it
                also rotates with the fly.
            pos_offset: Camera position offset from the tracked root segment in mm.
            rotation: Camera orientation as a `Rotation3D`.
            fovy: Vertical field of view in degrees.
            **kwargs: Additional attributes passed to the MJCF camera element. See
                [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera).

        Returns:
            The created MJCF camera element.
        """
        root_body = self.bodyseg_to_mjcfbody[self.root_segment]
        camera = root_body.add_camera(
            name=name,
            mode=CAMERA_MODES[mode],
            pos=pos_offset,
            fovy=fovy,
            **rotation.as_kwargs(),
            **kwargs,
        )
        self.cameraname_to_mjcfcamera[name] = camera
        return camera

    def _add_mesh_assets(
        self, mesh_basedir: PathLike, mirror_left2right: bool, mesh_type: MeshType
    ) -> None:

        # Decide which folder to load mesh files from
        mesh_dir = mesh_basedir / mesh_type.value
        mesh_fallback_dir = mesh_basedir / MeshType.FULLSIZE.value
        for d in [mesh_dir, mesh_fallback_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Mesh directory not found: {d}")

        for segment_name in ALL_SEGMENT_NAMES:
            if mirror_left2right and segment_name[0] == "r":
                mesh_to_use = f"l{segment_name[1:]}"
                y_sign = -1
            else:
                mesh_to_use = segment_name
                y_sign = 1

            mesh_path = (mesh_dir / f"{mesh_to_use}.stl").resolve()
            if not mesh_path.exists():
                mesh_path = (mesh_fallback_dir / f"{mesh_to_use}.stl").resolve()
                if not mesh_path.exists():
                    raise FileNotFoundError(
                        f"Mesh file not found for segment {segment_name}: "
                        f"tried {mesh_dir} and {mesh_fallback_dir}."
                    )

            self.bodyseg_to_mjcfmesh[segment_name] = self.mjcf_root.add_mesh(
                name=segment_name,
                file=str(mesh_path),
                scale=(self.SCALE, y_sign * self.SCALE, self.SCALE),
            )

    def _all_possible_joint_preset(self):
        return JointPreset.ALL_POSSIBLE

    def _get_base_skeleton(self) -> Skeleton:
        return self.BASE_SKELETON_CLASS(
            joint_preset=self._all_possible_joint_preset(),
            axis_order=self.AXIS_ORDER_CLASS.DONTCARE,
        )

    def _is_pitch(self, jointdof: JointDOF) -> bool:
        return jointdof.axis == RotationAxis.PITCH

    def _add_bodies_and_geoms(
        self,
        rigging_config_path: PathLike,
        geom_fitting_option: GeomFittingOption,
        vision_config_path: PathLike,
    ) -> None:
        # Load rigging config
        with open(rigging_config_path) as f:
            rigging_config = yaml.safe_load(f)

        # Load vision config to find out which geoms should be invisible to the eye
        # cameras to avoid occlusion (e.g., the eye geoms themselves)
        with open(vision_config_path) as f:
            info = yaml.safe_load(f)

        # Add root body and geom. The root can also be hidden from eye cameras if
        # requested in the vision config, so we apply the same group assignment rule
        # used for all other body segments.
        root_geom_group = 2 if self.root_segment.name in info["hidden_segments"] else 0
        body, geoms = self._add_one_body_and_geoms(
            self.mjcf_root.worldbody,
            self.root_segment,
            rigging_config[self.root_segment.name],
            geom_group=root_geom_group,
        )
        self.bodyseg_to_mjcfbody[self.root_segment] = body
        self.bodyseg_to_mjcfgeom[self.root_segment] = geoms

        # Add remaining bodies and geoms by traversing the kinematic tree defined by
        # the skeleton
        full_skeleton = self._get_base_skeleton()
        for jointdof in full_skeleton.iter_jointdofs(self.root_segment):
            if not self._is_pitch(jointdof):
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

            # If the geom should be invisible to eye cameras, we put it in group 2.
            # Otherwise, it goes in group 0. The MuJoCo renderer renders geoms in groups
            # 0, 1, 2 by default, so a default renderer renders all body geoms, but the
            # eye cameras can be configured to ignore group 2 geoms to avoid visual
            # occlusion. This makes the behavior of FlyGym less surprising to users who
            # wish to add their own renderers manually. We avoid group 1 because it's by
            # convention meant for mocap markers and helper geoms.
            # Geom group is based on parent body segment name because it's more
            # intuitive to specify the entire segment.
            geom_group = 2 if jointdof.child.name in info["hidden_segments"] else 0

            # Actually add the body and geom to the MJCF model
            body, geoms = self._add_one_body_and_geoms(
                parent_body, jointdof.child, my_rigging_config, geom_group
            )
            self.bodyseg_to_mjcfbody[jointdof.child] = body
            self.bodyseg_to_mjcfgeom[jointdof.child] = geoms

        # Optionally fit certain geoms to capsule shapes for simpler physics
        for bodyseg, mjcf_elements in self.bodyseg_to_mjcfgeom.items():
            for mjcf_element in mjcf_elements:
                if (geom_fitting_option == GeomFittingOption.ALL_TO_CAPSULES) or (
                    bodyseg.is_claw()
                    and geom_fitting_option == GeomFittingOption.CLAWS_TO_CAPSULES
                ):
                    mjcf_element.type = GEOM_TYPES["capsule"]

    def _add_one_body_and_geoms(
        self,
        parent_body: mj.MjsBody,
        segment: BodySegment,
        my_rigging_config: dict[str, Any],
        geom_group: int,
    ) -> tuple[mj.MjsBody, list[mj.MjsGeom]]:
        body_element = parent_body.add_body(
            name=segment.name,
            pos=my_rigging_config["pos"],
            quat=my_rigging_config["quat"],
        )
        geom_element = body_element.add_geom(
            name=segment.name,
            type=GEOM_TYPES["mesh"],
            meshname=segment.name,
            mass=my_rigging_config["mass"],
            contype=0,  # contact pairs to be added explicitly later
            conaffinity=0,  # contact pairs to be added explicitly later
            group=geom_group,
        )
        return body_element, [geom_element]

    def _parse_visuals_config(
        self,
        visuals_config_path: PathLike,
    ) -> tuple[dict[str, dict], dict[BodySegment, dict]]:
        # Load visuals config and assign vis sets to geometry name
        all_geom_names = [
            geom.name for geoms in self.bodyseg_to_mjcfgeom.values() for geom in geoms
        ]
        with open(visuals_config_path) as f:
            vis_set_params_all = yaml.safe_load(f)
        all_matches_by_geomname = {k: [] for k in all_geom_names}
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
            target_geomnames = set()
            for pattern in [apply_to] if isinstance(apply_to, str) else apply_to:
                target_geomnames |= set(filter_with_wildcard(all_geom_names, pattern))
            for geomname in target_geomnames:
                all_matches_by_geomname[geomname].append(vis_set_name)
        for geomname, vis_set_names in all_matches_by_geomname.items():
            if len(vis_set_names) != 1:
                raise ValueError(
                    f"Zero or multiple vis sets matched for body segment {geomname}: "
                    f"{vis_set_names}. Only one should apply."
                )
        lookup_by_geomname = {
            geomname: matches[0]
            for geomname, matches in all_matches_by_geomname.items()
        }
        return vis_set_params_all, lookup_by_geomname

    def _rebuild_neutral_keyframe(self):
        mj_model, _ = self.compile()
        self._neutral_keyframe.qpos = self._get_neutral_qpos(mj_model)
        self._neutral_keyframe.ctrl = self._get_neutral_ctrl(mj_model)

    def _get_neutral_qpos(self, mj_model: mj.MjModel) -> np.ndarray:
        neutral_qpos = np.zeros(mj_model.nq)
        for jointdof, angle in self.jointdof_to_neutralangle.items():
            joint_element = self.jointdof_to_mjcfjoint[jointdof]
            internal_jointid = mj.mj_name2id(
                mj_model, mj.mjtObj.mjOBJ_JOINT, joint_element.name
            )
            qposadr = mj_model.jnt_qposadr[internal_jointid]
            neutral_qpos[qposadr] = angle
        return neutral_qpos

    def _get_neutral_ctrl(self, mj_model: mj.MjModel) -> np.ndarray:
        neutral_ctrl = np.zeros(mj_model.nu)
        for ty, jointdof_to_actuator in self.jointdof_to_mjcfactuator_by_type.items():
            for jointdof, actuator in jointdof_to_actuator.items():
                internal_actuatorid = mj.mj_name2id(
                    mj_model, mj.mjtObj.mjOBJ_ACTUATOR, actuator.name
                )
                neutral_input = self.jointdof_to_neutralaction_by_type[ty][jointdof]
                neutral_ctrl[internal_actuatorid] = neutral_input
        return neutral_ctrl
