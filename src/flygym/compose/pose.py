from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Mapping, Sequence
from os import PathLike

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as ScipyRotation

from flygym.anatomy import AxisOrder, JointDOF, BodySegment, RotationAxis

__all__ = ["KinematicPose"]


class KinematicPose:
    """A snapshot of joint angles defining a static fly pose.

    The angles are stored in their native axis order as specified at init, and can be
    retrieved in any desired axis order via `get_angles_lookup()`. However, the
    conversion between different axis orders is not exact because 3D rotations are not
    commutative (i.e., the same DoFs chained in different orders cover different subsets
    of SO(3)). An optimization is applied to minimize the cumulative rotation error
    across all joints, but some error is inevitable.

    Args:
        path:
            Path to YAML file containing joint angles and metadata. See
            `src/flygym/assets/model/pose/neutral.yaml` for an example of the expected
            format. Either this or `joint_angles_rad_dict` must be provided, but not
            both.
        joint_angles_rad_dict:
            Dictionary mapping joint DoF names to angles in radians. Either this or
            `path` must be provided, but not both.
        axis_order:
            The AxisOrder of the provided angles if initializing from
            `joint_angles_rad_dict` (required). If initializing from `path`, this
            attribute must not be specified because the axis order will be loaded from
            the file.
        mirror_left2right:
            If True, when retrieving angles via `get_angles_lookup()`, any missing
            right-side joint DoFs will be filled in by mirroring from the left side.

    Example:

        pose = KinematicPose(path="neutral.yaml", mirror_left2right=True)
        angles_lookup = pose.get_angles_lookup(axis_order=AxisOrder.YAW_PITCH_ROLL)
    """

    def __init__(
        self,
        *,
        path: PathLike | None = None,
        joint_angles_rad_dict: dict[str, float] | None = None,
        axis_order: AxisOrder | str | Sequence[RotationAxis | str] | None = None,
        mirror_left2right: bool = True,
    ) -> None:
        # Keep this flag; apply later in get_angles_lookup for explicitness/efficiency.
        self._mirror_left2right = bool(mirror_left2right)

        if joint_angles_rad_dict is not None and path is None:
            if axis_order is None:
                raise ValueError(
                    "When initializing from `joint_angles_rad_dict`, axis_order must "
                    "also be provided."
                )
            self._native_axis_order = AxisOrder(axis_order)
            self._native_joint_angles_rad_dict = {
                k: float(v) for k, v in joint_angles_rad_dict.items()
            }
        elif path is not None and joint_angles_rad_dict is None:
            if axis_order is not None:
                raise ValueError(
                    "When initializing from `path`, `axis_order` should not be "
                    "provided because it will be loaded from the pose file."
                )
            angles_rad, input_axis_order = _load_pose_yaml(path)
            self._native_axis_order = input_axis_order
            self._native_joint_angles_rad_dict = angles_rad
        else:
            raise ValueError(
                "Either joint_angles_rad_dict or path must be provided, but not both."
            )

        # Precompute rotations only for *provided* joints (no mirroring here).
        axis_order_str = self._native_axis_order.to_letters_xyz().lower()
        self._doflist_by_anatomical_joint = _group_dofs_by_anatomical_joint(
            self._native_joint_angles_rad_dict
        )
        self._rot_by_anatomical_joint: dict[_JointKey, ScipyRotation] = {
            key: _build_intrinsic_rotation_for_joint(doflist, axis_order_str)
            for key, doflist in self._doflist_by_anatomical_joint.items()
        }

    def get_angles_lookup(
        self,
        axis_order: AxisOrder | str | Sequence[RotationAxis | str],
        degrees: bool = False,
    ) -> dict[str, float]:
        """Get a dict mapping joint DoF names to angles in the specified unit, expressed
        under the specified intrinsic Euler axis order (i.e., how hinge joints are
        chained in the body model).

        If `axis_order` is the same as how the angles were originally provided upon
        `__init__`, the natively stored angles are returned directly. Otherwise,
        rotations per DoF in the new axis order are computed as follows:

        * **All 3 DoFs present:** exact rotation converted rewritten in new axis order.
        * **2 DoFs present:** best-fit with the missing axis forced to 0.
        * **1 DoF only:** unchanged.

        If mirror_left2right was set at init, we then fill missing right-side keys by
        by mirroring from left-side. The angles are exactly the same as the left side
        (no sign flip) because rotational axis in the body model are symmetrically
        defined (e.g., positive roll are always "outward").
        """
        out_axis_order = AxisOrder(axis_order)

        # 1) Reorder / convert angles (only for provided joints)
        if out_axis_order == self._native_axis_order:
            out: dict[str, float] = dict(self._native_joint_angles_rad_dict)
        else:
            out = {}
            for key, rotation in self._rot_by_anatomical_joint.items():
                doflist = self._doflist_by_anatomical_joint[key]
                out.update(_angles_for_dofs_in_order(rotation, doflist, out_axis_order))

        # 2) Apply mirroring *here* (explicit + avoids doing it if never requested)
        if self._mirror_left2right:
            _mirror_pose_left2right_in_place(out)

        # 3) Unit conversion
        if degrees:
            out = {k: float(np.rad2deg(v)) for k, v in out.items()}

        return out


# ================================================================================
# The following helpers are generated by ChatGPT 5.2 Thinking, with some manual
# tweaking. I didn't check all the details, but the output looks correct.
# ================================================================================


def _wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rotvec_residual(R_curr: ScipyRotation, R_target: ScipyRotation) -> np.ndarray:
    """Residual in so(3): log(R_curr^{-1} R_target) as a 3-vector."""
    return (R_curr.inv() * R_target).as_rotvec()


def _finite_diff_jacobian(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite difference Jacobian for f: R^n -> R^m. Returns (m, n)."""
    x = x.astype(float, copy=True)
    y0 = np.asarray(f(x))
    J = np.zeros((y0.size, x.size), dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        y_plus = np.asarray(f(x + dx))
        y_minus = np.asarray(f(x - dx))
        J[:, i] = (y_plus - y_minus) / (2 * eps)
    return J


def _solve_constrained_intrinsic_euler(
    R_target: ScipyRotation,
    out_axis_order_str: str,
    *,
    fixed_letter: str,
    fixed_value: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Find intrinsic Euler angles 'out_axis_order_str' that best match R_target,
    while forcing one axis angle (given by fixed_letter) to fixed_value.

    Returns full 3-angle array aligned with out_axis_order_str (radians).
    """
    if fixed_letter not in out_axis_order_str:
        raise ValueError(
            f"fixed_letter={fixed_letter!r} not in out_axis_order_str={out_axis_order_str!r}"
        )

    fixed_idx = out_axis_order_str.index(fixed_letter)
    free_idx = [i for i in range(3) if i != fixed_idx]

    # optimize only 2 variables (the non-fixed angles)
    x = np.zeros(2, dtype=float)

    def unpack(x2: np.ndarray) -> np.ndarray:
        full = np.zeros(3, dtype=float)
        full[fixed_idx] = fixed_value
        full[free_idx[0]] = x2[0]
        full[free_idx[1]] = x2[1]
        return full

    def R_of(x2: np.ndarray) -> ScipyRotation:
        ang = unpack(x2)
        return ScipyRotation.from_euler(out_axis_order_str, ang, degrees=False)

    # Damped Gauss–Newton / Levenberg–Marquardt on SO(3) residual
    lam = 1e-6
    for _ in range(max_iter):
        R_curr = R_of(x)
        r = _rotvec_residual(R_curr, R_target)  # (3,)
        if float(np.dot(r, r)) < tol:
            break

        def r_of(x2: np.ndarray) -> np.ndarray:
            return _rotvec_residual(R_of(x2), R_target)

        J = _finite_diff_jacobian(r_of, x)  # (3,2)
        A = J.T @ J + lam * np.eye(2)
        b = -J.T @ r
        dx = np.linalg.solve(A, b)
        x = x + dx

        if float(np.dot(dx, dx)) < tol:
            break

    full = unpack(x)
    full = np.array([_wrap_to_pi(a) for a in full], dtype=float)
    return full


@dataclass(frozen=True)
class _JointKey:
    parent: str
    child: str


def _group_dofs_by_anatomical_joint(
    joint_angles_rad_dict: Mapping[str, float],
) -> DefaultDict[_JointKey, list[tuple[JointDOF, float]]]:
    """
    Groups per-DoF angles into per-(parent, child) anatomical joints.
    Returns mapping: (parent, child) -> [(JointDOF, angle), ...]
    """
    out: DefaultDict[_JointKey, list[tuple[JointDOF, float]]] = defaultdict(list)
    for dof_name, angle in joint_angles_rad_dict.items():
        jointdof = JointDOF.from_name(dof_name)
        key = _JointKey(jointdof.parent.name, jointdof.child.name)
        out[key].append((jointdof, float(angle)))
    return out


def _build_intrinsic_rotation_for_joint(
    doflist: Sequence[tuple[JointDOF, float]],
    axis_order_str: str,
) -> ScipyRotation:
    """
    Build intrinsic rotation for a joint from its dofs.
    Missing axes are treated as 0 (consistent with SciPy from_euler).
    """
    angles = np.zeros(3, dtype=float)
    for jointdof, angle in doflist:
        axis_letter = jointdof.axis.to_letter_xyz()  # 'x'/'y'/'z'
        angles[axis_order_str.index(axis_letter)] = angle
    return ScipyRotation.from_euler(axis_order_str, angles, degrees=False)


def _angles_for_dofs_in_order(
    rotation: ScipyRotation,
    doflist: Sequence[tuple[JointDOF, float]],
    out_axis_order: AxisOrder,
) -> dict[str, float]:
    """
    Return angle dict {dof_name: angle_rad} for exactly the dofs in doflist, but
    expressed under out_axis_order, respecting missing DoFs by fixing them to 0.

    - n_dofs=1: return the single angle
    - n_dofs=3: exact euler decomposition
    - n_dofs=2: best fit with missing axis fixed to 0 (min |missing|, then set to 0)
    """
    n_dofs = len(doflist)
    out_axis_order_str = out_axis_order.to_letters_xyz().lower()

    if n_dofs == 1:
        dof, angle = doflist[0]
        return {dof.name: float(angle)}

    dofname_by_axis = {dof.axis: dof.name for dof, _ in doflist}
    present_letters = {dof.axis.to_letter_xyz() for dof, _ in doflist}

    if n_dofs == 3:
        angles = rotation.as_euler(out_axis_order_str, degrees=False)
        out: dict[str, float] = {}
        for axis in out_axis_order.value:
            dof_name = dofname_by_axis[axis]
            idx = out_axis_order_str.index(axis.to_letter_xyz())
            out[dof_name] = float(_wrap_to_pi(angles[idx]))
        return out

    if n_dofs == 2:
        missing_letters = {"x", "y", "z"} - present_letters
        if len(missing_letters) != 1:
            raise ValueError(
                f"Expected exactly one missing axis for 2DoF joint, got {missing_letters}"
            )
        (missing_letter,) = tuple(missing_letters)

        full_angles = _solve_constrained_intrinsic_euler(
            rotation,
            out_axis_order_str,
            fixed_letter=missing_letter,
            fixed_value=0.0,
        )

        out: dict[str, float] = {}
        for axis in out_axis_order.value:
            letter = axis.to_letter_xyz()
            if letter == missing_letter:
                continue  # don't output non-existent DoF
            dof_name = dofname_by_axis[axis]
            out[dof_name] = float(full_angles[out_axis_order_str.index(letter)])
        return out

    raise ValueError(f"Unexpected number of DoFs for joint: {n_dofs}")


def _load_pose_yaml(path: PathLike) -> tuple[dict[str, float], AxisOrder]:
    with open(path, "r") as f:
        pose_data = yaml.safe_load(f)

    angle_unit = pose_data.get("angle_unit")
    if angle_unit not in ("degree", "radian"):
        raise ValueError("YAML file must contain angle_unit: 'degree' or 'radian'.")

    joint_angles = pose_data.get("joint_angles")
    if not isinstance(joint_angles, dict):
        raise ValueError("YAML file must contain 'joint_angles' mapping.")
    for k, v in joint_angles.items():
        if not isinstance(v, (int, float)):
            raise ValueError(f"Joint angle for '{k}' must be a number.")

    joint_angles = {k: float(v) for k, v in joint_angles.items()}
    if angle_unit == "degree":
        joint_angles = {k: float(np.deg2rad(v)) for k, v in joint_angles.items()}

    axis_order_raw = pose_data.get("axis_order")
    try:
        axis_order = AxisOrder(axis_order_raw)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid or missing axis_order: {axis_order_raw}")

    return joint_angles, axis_order


def _mirror_pose_left2right_in_place(joint_angles: dict[str, float]) -> None:
    """
    Mirror left-side to right-side when missing.
    Mutates dict in-place for efficiency.
    """
    # We must iterate over a snapshot because we may add new keys
    items = list(joint_angles.items())
    for joint_name, angle in items:
        jointdof = JointDOF.from_name(joint_name)
        if jointdof.child.name[0] != "l":
            continue

        mirror_parent = BodySegment(
            ("r" + jointdof.parent.name[1:])
            if jointdof.parent.name[0] == "l"
            else jointdof.parent.name
        )
        mirror_child = BodySegment("r" + jointdof.child.name[1:])
        mirror_jointdof = JointDOF(mirror_parent, mirror_child, jointdof.axis)

        if mirror_jointdof.name not in joint_angles:
            joint_angles[mirror_jointdof.name] = float(angle)
