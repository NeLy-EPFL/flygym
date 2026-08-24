"""Export NeuroMechFly's kinematic tree as JSON for the flyik crate.

Each node names its own parent (root "thorax" has `parent: null`). Covers
all 6 legs, tarsus2-5 collapsed per claw. Writes to `{BODYPLAN_NAME}.json`
in the current directory.
"""

import json
from pathlib import Path

import mujoco as mj
import numpy as np

from flygym.anatomy import AxisOrder, JointPreset, Skeleton
from flygym.compose import KinematicPosePreset, NeuroMechFly

JOINT_PRESET = JointPreset.ALL_BIOLOGICAL
AXIS_ORDER = AxisOrder.YAW_PITCH_ROLL
NEUTRAL_POSE = KinematicPosePreset.NEUTRAL

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
LINK_NAMES = ["coxa", "trochanterfemur", "tibia", "tarsus1"]
# Each link's own joint, keyed by its (parent, child) link names; combined
# with a leg prefix (e.g. "lf_thorax_coxa") to form each node's global name.
JOINT_KEYS = ["thorax_coxa", "coxa_trochanterfemur", "trochanterfemur_tibia", "tibia_tarsus"]
# DoFs: thorax-coxa(3), coxa-troch(2), troch-tibia(1), tibia-tarsus(1)
EXPECTED_DOFS_PER_LINK = [3, 2, 1, 1]
# tarsus1's distal children, in order, collapsed into the leg's claw node.
INTER_TARSAL_LINKS = ["tarsus2", "tarsus3", "tarsus4", "tarsus5"]

IDENTITY_POS = [0.0, 0.0, 0.0]
IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]  # (w, x, y, z)

BODYPLAN_NAME = "neuromechfly_ypr_legs"


def compose_model() -> mj.MjModel:
    """Compose a standalone fly with joints only -- no actuators, no world."""
    fly = NeuroMechFly()
    skeleton = Skeleton(joint_preset=JOINT_PRESET, axis_order=AXIS_ORDER)
    fly.add_joints(skeleton, NEUTRAL_POSE)
    # A standalone fly has no free joint, so its root segment would otherwise get
    # fused into the worldbody, breaking the track-mode camera; set explicitly
    # instead of relying on compile()'s own auto-override (which also warns).
    fly.mjcf_root.compiler.fusestatic = False
    mj_model, _ = fly.compile()
    return mj_model


def find_body_id(model: mj.MjModel, name_suffix: str) -> int:
    """Find the single body whose name ends with `name_suffix`.

    Bodies are namespaced (e.g. "nmf/lf_coxa") by flygym's MJCF attach
    mechanism, so we match by suffix rather than assuming a fixed prefix.
    """
    matches = [i for i in range(model.nbody) if model.body(i).name.endswith(name_suffix)]
    if len(matches) != 1:
        names = [model.body(i).name for i in matches]
        raise ValueError(
            f"Expected exactly one body named '*{name_suffix}', found {len(matches)}: {names}"
        )
    return matches[0]


def find_body_joint_ids(model: mj.MjModel, body_id: int) -> list[int]:
    """Ids of all joints whose child body is `body_id`, in qpos order."""
    return sorted(j for j in range(model.njnt) if model.jnt_bodyid[j] == body_id)


def extract_link(model: mj.MjModel, body_id: int, expected_n_dofs: int) -> dict:
    """Extract one link's fixed offset and its own hinge joint DOFs."""
    joint_ids = find_body_joint_ids(model, body_id)
    if len(joint_ids) != expected_n_dofs:
        raise ValueError(
            f"Body '{model.body(body_id).name}': expected {expected_n_dofs} DOFs, "
            f"found {len(joint_ids)}: {[model.joint(j).name for j in joint_ids]}"
        )

    dofs = []
    for jid in joint_ids:
        if model.jnt_type[jid] != mj.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Joint '{model.joint(jid).name}' is not a hinge joint.")
        if not (model.jnt_pos[jid] == 0.0).all():
            raise ValueError(
                f"Joint '{model.joint(jid).name}' has nonzero anchor {model.jnt_pos[jid]} "
                "in its child body's frame; export assumes joint anchor == body origin."
            )
        name = model.joint(jid).name
        qposadr = model.jnt_qposadr[jid]
        neutral_angle = float(model.qpos_spring[qposadr])
        limits = [float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])] if model.jnt_limited[jid] else None
        if limits is not None and not (limits[0] <= neutral_angle <= limits[1]):
            raise ValueError(
                f"Joint '{name}': neutral angle {neutral_angle} outside its own model-declared "
                f"limits {limits}."
            )
        dofs.append(
            {
                "name": name,
                "type": "hinge",
                "axis": model.jnt_axis[jid].tolist(),
                "neutral": neutral_angle,
                "limits": limits,
            }
        )

    return {
        "offset_pos": model.body_pos[body_id].tolist(),
        "offset_quat": model.body_quat[body_id].tolist(),  # (w, x, y, z)
        "dofs": dofs,
    }


def estimate_terminal_offset(model: mj.MjModel, body_id: int) -> np.ndarray:
    """Approximate offset from `body_id`'s origin to its distal tip.

    `body_id`'s origin is its joint to the parent (see `extract_link`), not
    its distal end, so for a rod-like leaf segment (e.g. tarsus5) the tip is
    instead approximated as the mesh vertex farthest from the origin.
    """
    geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    if len(geom_ids) != 1 or model.geom_dataid[geom_ids[0]] < 0:
        raise ValueError(
            f"Body '{model.body(body_id).name}': expected exactly one mesh geom, "
            f"found geom ids {geom_ids}."
        )
    (geom_id,) = geom_ids
    mesh_id = model.geom_dataid[geom_id]
    vert_start = model.mesh_vertadr[mesh_id]
    vert_count = model.mesh_vertnum[mesh_id]
    verts_geom_frame = model.mesh_vert[vert_start : vert_start + vert_count].astype(float)

    geom_rotmat = np.empty(9)
    mj.mju_quat2Mat(geom_rotmat, model.geom_quat[geom_id])
    verts_body_frame = verts_geom_frame @ geom_rotmat.reshape(3, 3).T + model.geom_pos[geom_id]
    return verts_body_frame[np.argmax(np.linalg.norm(verts_body_frame, axis=1))]


def leg_claw_offset(model: mj.MjModel, leg: str) -> list[float]:
    """Fixed vector from `{leg}_tarsus1`'s origin to `{leg}_tarsus5`'s tip.

    flyik does not give tarsus2-5 DOFs of their own (see module docstring):
    this collapses that whole sub-chain into one constant offset, evaluated
    at tarsus2-5's neutral (rest) angles. The result is independent of
    tarsus1's own joint angle -- it is expressed in tarsus1's local frame, so
    at solve time flyik just rotates it along with the rest of tarsus1.
    """
    data = mj.MjData(model)
    for link_name in INTER_TARSAL_LINKS:
        body_id = find_body_id(model, f"{leg}_{link_name}")
        (joint_id,) = find_body_joint_ids(model, body_id)
        qposadr = model.jnt_qposadr[joint_id]
        data.qpos[qposadr] = model.qpos_spring[qposadr]
    mj.mj_kinematics(model, data)

    tarsus1_id = find_body_id(model, f"{leg}_tarsus1")
    tarsus5_id = find_body_id(model, f"{leg}_tarsus5")
    tip_local_to_tarsus5 = estimate_terminal_offset(model, tarsus5_id)

    tarsus5_rotmat = data.xmat[tarsus5_id].reshape(3, 3)
    tip_world = data.xpos[tarsus5_id] + tarsus5_rotmat @ tip_local_to_tarsus5

    tarsus1_rotmat = data.xmat[tarsus1_id].reshape(3, 3)
    tip_local_to_tarsus1 = tarsus1_rotmat.T @ (tip_world - data.xpos[tarsus1_id])
    return tip_local_to_tarsus1.tolist()


def build_joint_tree(model: mj.MjModel) -> list[dict]:
    """Flatten the model into flyik's tree schema: one node per JSON object,
    each naming its own parent. "thorax" is the sole root (`parent: null`);
    every leg hangs off it directly, since flyik's thorax is a free-floating
    frame with no fixed offset of its own.
    """
    joints = [
        {
            "name": "thorax",
            "parent": None,
            "offset_pos": IDENTITY_POS,
            "offset_quat": IDENTITY_QUAT,
            "dofs": [],
        }
    ]
    for leg in LEGS:
        parent_name = "thorax"
        for link_name, joint_key, expected_n_dofs in zip(LINK_NAMES, JOINT_KEYS, EXPECTED_DOFS_PER_LINK):
            body_id = find_body_id(model, f"{leg}_{link_name}")
            node_name = f"{leg}_{joint_key}"
            joints.append(
                {
                    "name": node_name,
                    "parent": parent_name,
                    **extract_link(model, body_id, expected_n_dofs),
                }
            )
            parent_name = node_name
        joints.append(
            {
                "name": f"{leg}_claw",
                "parent": parent_name,
                "offset_pos": leg_claw_offset(model, leg),
                "offset_quat": IDENTITY_QUAT,
                "dofs": [],
            }
        )
    return joints


def print_joint_tree(joints: list[dict]) -> None:
    """Print the joint hierarchy, indented by depth, with each node's DOF count."""
    by_name = {j["name"]: j for j in joints}
    children: dict[str | None, list[dict]] = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)

    def visit(name: str, depth: int) -> None:
        n_dofs = len(by_name[name]["dofs"])
        print(f"{'  ' * depth}{name} ({n_dofs} dof{'s' if n_dofs != 1 else ''})")
        for child in children.get(name, []):
            visit(child["name"], depth + 1)

    (root,) = children[None]
    visit(root["name"], 0)


def main() -> None:
    """Compose the Fly model and write its body plan as JSON."""
    model = compose_model()
    joints = build_joint_tree(model)
    json_payload = {"fixed_base": False, "x-name": BODYPLAN_NAME, "joints": joints}

    output = Path(f"{BODYPLAN_NAME}.json")
    output.write_text(json.dumps(json_payload))

    n_dofs = sum(len(j["dofs"]) for j in joints)
    print(f"Wrote body plan with {len(joints)} joints, {n_dofs} dofs to {output}")
    print_joint_tree(joints)


if __name__ == "__main__":
    main()
