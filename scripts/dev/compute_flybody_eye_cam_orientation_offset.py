"""Find the YAML `orientation` value for the flybody eye camera so that its
GLOBAL orientation equals the nmf eye camera GLOBAL orientation.

Chain (both flies spawned at world identity rotation):

    R_cam_global(flybody) = R_c_head_global @ R_marker_local @ R_yaml
                         = R_c_head_global @ I              @ R_yaml
                         = R_c_head_global @ R_yaml

(the marker body `l_eye_cam_body` is created with `pos=rel_pos` only, no
quat/euler, see flygym/compose/fly.py::add_vision -> so R_marker_local = I.)

We want R_cam_global(flybody) == R_cam_global(nmf), so:

    R_yaml = R_c_head_global(flybody).T @ R_cam_global(nmf)

`xmat` in MuJoCo is the GLOBAL rotation matrix for bodies and cameras alike,
which is what we need. The previous version of this script mistakenly used
R_nmf_parent_global.T @ R_nmf_cam_global -- i.e. it expressed the nmf cam in
the *nmf* parent's frame (nmf has no head body, so its cam is parented to the
thorax root). That frame is NOT the same as flybody's c_head frame because the
flybody head has a default pitch relative to its thorax. Fixing that here.

MuJoCo's `eulerseq` (set to `XYZ` in flybody/mujoco_globals.yaml) is verified
empirically to behave as scipy EXTRINSIC `xyz` (lowercase), not intrinsic
`XYZ` (uppercase). The MuJoCo-compiled cam_quat matches
`R.from_euler("xyz", yaml_vals).as_quat(...)` exactly. So when emitting the
YAML euler values we use lowercase `xyz` here.
"""

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R

from flygym import Simulation
from flygym.compose import (
    KinematicPosePreset,
    FlatGroundWorld,
    Fly,
)
from flygym.compose.fly import FlybodyFly
from flygym.anatomy import Skeleton, AxisOrder, JointPreset
from flygym.assets.model.flybody.anatomy_flybody import (
    FlybodySkeleton,
    FlybodyJointPreset,
    FlybodyAxisOrder,
    FlybodyContactBodiesPreset,
)
from flygym.utils.math import Rotation3D


SIDES = ("l", "r")
FLYBODY_HEAD_BODY = "flybody/c_head"


def _build_flybody_sim():
    fly = FlybodyFly()
    skeleton = FlybodySkeleton(
        axis_order=FlybodyAxisOrder.YAW_ROLL_PITCH,
        joint_preset=FlybodyJointPreset.ALL_BIOLOGICAL,
    )
    fly.add_joints(skeleton, KinematicPosePreset.FLYBODY_NEUTRAL)
    fly.add_tracking_camera()
    fly.add_vision()

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        (0, 0, 10.0),
        Rotation3D("quat", (1, 0, 0, 0)),
        bodysegs_with_ground_contact=FlybodyContactBodiesPreset.LEGS_ONLY,
    )
    return Simulation(world)


def _build_nmf_sim():
    fly = Fly()
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL,
        joint_preset=JointPreset.LEGS_ONLY,
    )
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    fly.add_vision()

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 10.0],
        Rotation3D(format="quat", values=[1, 0, 0, 0]),
    )
    return Simulation(world)


def _body_xmat_global(sim, body_name):
    mj.mj_forward(sim.mj_model, sim.mj_data)
    return sim.mj_data.body(body_name).xmat.reshape(3, 3)


def _cam_xmat_global(sim, cam_name):
    mj.mj_forward(sim.mj_model, sim.mj_data)
    return sim.mj_data.camera(cam_name).xmat.reshape(3, 3)


def _verify_marker_local_identity(sim, cam_full_name):
    """The chain below assumes R_marker_local == I. Verify it."""
    cam_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam_full_name)
    marker_id = int(sim.mj_model.cam_bodyid[cam_id])
    # body_quat is parent-local rotation (qw, qx, qy, qz).
    quat = sim.mj_model.body_quat[marker_id]
    assert np.allclose(quat, [1, 0, 0, 0], atol=1e-9), (
        f"Marker body {mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_BODY, marker_id)} "
        f"has a non-identity local rotation ({quat}); the chain assumption breaks."
    )


def main():
    fbody_sim = _build_flybody_sim()
    fbody_sim.reset()
    nmf_sim = _build_nmf_sim()
    nmf_sim.reset()

    R_fbody_c_head_global = _body_xmat_global(fbody_sim, FLYBODY_HEAD_BODY)

    np.set_printoptions(precision=4, suppress=True)
    print(
        f"R_c_head_global(flybody) Euler XYZ (rad): "
        f"{R.from_matrix(R_fbody_c_head_global).as_euler('xyz').round(4)}"
    )
    print(
        f"R_c_head_global(flybody) Euler XYZ (deg): "
        f"{R.from_matrix(R_fbody_c_head_global).as_euler('xyz', degrees=True).round(2)}"
    )
    print()

    yaml_suggestion = {}
    for side in SIDES:
        cam_name = f"{side}_eye_cam_camera"
        fbody_cam_full = f"flybody/{cam_name}"
        nmf_cam_full = f"nmf/{cam_name}"

        _verify_marker_local_identity(fbody_sim, fbody_cam_full)

        # Target visual orientation = nmf cam global rotation (nmf is at world identity).
        R_nmf_cam_global = _cam_xmat_global(nmf_sim, nmf_cam_full)

        # The YAML value lives in the flybody c_head frame (== marker frame, since
        # marker has identity local rotation).
        R_yaml = R_fbody_c_head_global.T @ R_nmf_cam_global
        yaml_euler = R.from_matrix(R_yaml).as_euler("xyz", degrees=False)

        # Sanity: what would the cam global look like after applying R_yaml?
        R_cam_global_predicted = R_fbody_c_head_global @ R_yaml
        residual = np.linalg.norm(R_cam_global_predicted - R_nmf_cam_global)
        assert residual < 1e-9, f"chain residual {residual} -- bug in the math"

        # Reference: what flybody currently has.
        R_fbody_cam_global_current = _cam_xmat_global(fbody_sim, fbody_cam_full)
        current_yaml = R_fbody_c_head_global.T @ R_fbody_cam_global_current
        current_euler = R.from_matrix(current_yaml).as_euler("xyz", degrees=False)

        print(f"=== Eye: {side} ===")
        print(
            f"  Current cam_global (flybody) Euler XYZ (deg): "
            f"{R.from_matrix(R_fbody_cam_global_current).as_euler('xyz', degrees=True).round(3)}"
        )
        print(
            f"  Target  cam_global (=nmf)    Euler XYZ (deg): "
            f"{R.from_matrix(R_nmf_cam_global).as_euler('xyz', degrees=True).round(3)}"
        )
        print(f"  Current YAML (in c_head frame) XYZ rad: {current_euler}")
        print(f"  TARGET  YAML (in c_head frame) XYZ rad: {yaml_euler}")
        print(f"  TARGET  YAML (in c_head frame) XYZ deg: {np.degrees(yaml_euler)}")
        print()

        yaml_suggestion[side] = yaml_euler

    print(
        "Drop these straight into flybody/vision.yaml (rad, MuJoCo eulerseq=XYZ "
        "== scipy extrinsic xyz):"
    )
    for side, vals in yaml_suggestion.items():
        print(
            f"  {side}_eye_cam.orientation: [{vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}]"
        )

    # --- End-to-end verification: predict the resulting cam_global from the
    # YAML value alone (as MuJoCo would compose it) and confirm it equals nmf. ---
    print("\nEnd-to-end check (treating YAML as scipy extrinsic xyz):")
    for side, vals in yaml_suggestion.items():
        cam_name = f"{side}_eye_cam_camera"
        R_yaml_extr = R.from_euler("xyz", vals).as_matrix()
        R_pred_global = R_fbody_c_head_global @ R_yaml_extr
        R_nmf_cam_global = _cam_xmat_global(nmf_sim, f"nmf/{cam_name}")
        err = np.degrees(R.from_matrix(R_pred_global.T @ R_nmf_cam_global).magnitude())
        print(
            f"  {side} eye -- angular error between predicted and nmf cam_global: {err:.2e} deg"
        )


if __name__ == "__main__":
    main()
