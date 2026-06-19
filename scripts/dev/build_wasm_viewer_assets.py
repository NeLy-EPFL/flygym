"""Generate the static assets for the browser (WebAssembly) interactive viewer.

The page at ``docs/wasm_viewer/viewer.html`` runs the *same* NeuroMechFly model
as ``scripts/launch_interactive_viewer.py`` -- but instead of MuJoCo's native
viewer it uses MuJoCo compiled to WebAssembly (vendored under
``docs/wasm_viewer/vendor/mujoco``) and renders with Three.js. The simulation is
real (``mj_step``): the position-actuator sliders write ``data.ctrl``; a little
bar over each slider shows the actuated joint's current ``qpos``; bodies can be
dragged to apply an external force; and contacts/forces/joints/actuators can be
toggled, much like MuJoCo's own viewer.

This script produces everything that page loads, so the docs build itself stays
lightweight (it never imports flygym or mujoco -- it only serves the committed
files under ``docs/wasm_viewer/``). It is therefore run *by hand* whenever the
model or its viewer config changes. It needs ``flygym`` + ``mujoco``, e.g.::

    uv run --with flygym --with mujoco --python 3.12 \
        python scripts/dev/build_wasm_viewer_assets.py

Outputs (all under ``docs/wasm_viewer/assets/``):

``model/fly.xml`` + ``model/*.stl``
    A flattened, self-contained MJCF and the meshes it references, written by
    ``dm_control.mjcf.export_with_assets``. The browser loads this via
    ``mj_loadXML``.
``model_meta.json``
    Everything the control panel needs that is awkward to read from the WASM
    model at runtime: simulation timestep; the neutral keyframe's ``qpos`` /
    ``ctrl``; one entry per position actuator (name, driven joint + its ``qpos``
    address and slider range, neutral target, UI group/label); the group order;
    and a representative RGB per geom for the mesh colors.

The metadata is resolved against the *exported* model (reloaded standalone,
exactly as the browser sees it) and asserted to exist, so a bad mapping fails
here rather than silently in the browser.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path

import mujoco as mj
import yaml
from flygym import assets_dir
from flygym.anatomy import (
    ALL_SEGMENT_NAMES,
    ActuatedDOFPreset,
    AxisOrder,
    ContactBodiesPreset,
    JointPreset,
    Skeleton,
)
from flygym.compose import ActuatorType, FlatGroundWorld, Fly, KinematicPosePreset
from flygym.utils.math import Rotation3D

# --- repo paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs/wasm_viewer/assets"
MODEL_DIR = OUT_DIR / "model"

# --- body config: must mirror scripts/launch_interactive_viewer.py ----------
JOINT_PRESET = JointPreset.ALL_BIOLOGICAL
AXIS_ORDER = AxisOrder.YAW_PITCH_ROLL
ACTUATED_DOFS = ActuatedDOFPreset.LEGS_ACTIVE_ONLY
ACTUATOR_TYPE = ActuatorType.POSITION
ACTUATOR_POSITION_GAIN = 20.0
NEUTRAL_POSE = KinematicPosePreset.NEUTRAL
SPAWN_POSITION = (0, 0, 0.8)  # xyz in mm
SPAWN_ROTATION = Rotation3D("quat", (1, 0, 0, 0))  # wxyz quaternion
CONTACT_BODIES = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD
CTRLRANGE = (-3.14, 3.14)
# The viewer raises the global no-slip solver iterations (5 -> 300) and doubles
# the main constraint-solver iterations (100 -> 200) to keep the fly's feet from
# sliding while it is posed interactively at <1x real time.
NOSLIP_ITERATIONS = 300
SOLVER_ITERATIONS = 200
IMPRATIO = 10  # frictional vs normal constraint stiffness (MuJoCo default 1)
SLIDING_FRICTION = 5.0  # foot/ground tangential friction (was 1.0)

# --- slider grouping --------------------------------------------------------
LEG_PREFIXES = ("lf", "lm", "lh", "rf", "rm", "rh")
GROUP_ORDER = ["lf_leg", "lm_leg", "lh_leg", "rf_leg", "rm_leg", "rh_leg"]
GROUP_LABELS = {
    "lf_leg": "Left front leg",
    "lm_leg": "Left mid leg",
    "lh_leg": "Left hind leg",
    "rf_leg": "Right front leg",
    "rm_leg": "Right mid leg",
    "rh_leg": "Right hind leg",
    "other": "Other",
}


def build_model() -> mj.MjModel:
    """Compose the fly + flat-ground world exactly as the interactive-viewer
    script does, export it to a self-contained MJCF under ``MODEL_DIR``, and
    return the model reloaded standalone (what the browser's ``mj_loadXML``
    sees)."""
    fly = Fly()
    skeleton = Skeleton(joint_preset=JOINT_PRESET, axis_order=AXIS_ORDER)
    fly.add_joints(skeleton, NEUTRAL_POSE)

    actuated = skeleton.get_actuated_dofs_from_preset(ACTUATED_DOFS)
    fly.add_actuators(
        actuated,
        ACTUATOR_TYPE,
        neutral_input=NEUTRAL_POSE,
        kp=ACTUATOR_POSITION_GAIN,
        ctrlrange=CTRLRANGE,
    )
    fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
    fly.colorize()
    fly.add_tracking_camera(name="trackingcam")

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        SPAWN_POSITION,
        SPAWN_ROTATION,
        bodysegs_with_ground_contact=CONTACT_BODIES,
    )

    # Tune the contact solver for steadier interactive posing (less foot slip):
    # more iterations, a high impratio (frictional constraints made much stiffer
    # than normal ones -- the most effective anti-slip knob), and higher sliding
    # friction on the foot/ground pairs.
    world.mjcf_root.option.noslip_iterations = NOSLIP_ITERATIONS
    world.mjcf_root.option.iterations = SOLVER_ITERATIONS
    world.mjcf_root.option.impratio = IMPRATIO
    for pair in world.mjcf_root.contact.all_children():
        if pair.tag != "pair":
            continue
        fr = (
            list(pair.friction)
            if pair.friction is not None
            else [1, 1, 2e-2, 1e-4, 1e-4]
        )
        fr[0] = fr[1] = SLIDING_FRICTION  # tangential (x2)
        pair.friction = fr

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True)
    world.save_xml_with_assets(MODEL_DIR, "fly.xml")

    model = mj.MjModel.from_xml_path(str(MODEL_DIR / "fly.xml"))
    assert model.nkey >= 1, "expected a baked 'neutral' keyframe in the exported model"
    return model


def actuator_group(joint_short: str) -> str:
    """Return the UI group key for a joint name like ``c_thorax-lf_coxa-yaw``."""
    for part in joint_short.split("-"):
        for leg in LEG_PREFIXES:
            if part.startswith(leg + "_"):
                return f"{leg}_leg"
    return "other"


def build_meta(model: mj.MjModel) -> dict:
    """Metadata for the control panel: timestep, neutral keyframe, per-actuator
    slider info, group order and per-geom colors."""
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)  # the "neutral" keyframe

    actuators = []
    used_groups: set[str] = set()
    for a in range(model.nu):
        act = model.actuator(a)
        joint_id = int(act.trnid[0])
        joint = model.joint(joint_id)
        joint_short = joint.name.split("/")[-1]
        qposadr = int(model.jnt_qposadr[joint_id])
        lo, hi = (float(x) for x in act.ctrlrange)
        group = actuator_group(joint_short)
        used_groups.add(group)
        actuators.append(
            {
                "id": a,
                "name": act.name.split("/")[-1],
                "joint": joint_short,
                "jointId": joint_id,
                "qposadr": qposadr,
                "ctrlrange": [lo, hi],
                "neutral": float(data.ctrl[a]),
                "group": group,
                "label": joint_short.rsplit("-", 1)[-1]
                if "-" in joint_short
                else joint_short,
                "dofLabel": _dof_label(joint_short),
            }
        )

    groups = [
        {"key": k, "label": GROUP_LABELS[k]}
        for k in GROUP_ORDER + ["other"]
        if k in used_groups
    ]

    return {
        "nq": int(model.nq),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "timestep": float(model.opt.timestep),
        "gravity": [float(x) for x in model.opt.gravity],
        "cone": int(model.opt.cone),  # 0=pyramidal, 1=elliptic (for contact forces)
        # The viewer derives its perturbation and contact/force visual scales from
        # the model's global MuJoCo settings (visual/map + stat), exactly as the
        # native viewer does, instead of hard-coding constants.
        "map": {
            "stiffness": float(model.vis.map.stiffness),
            "stiffnessrot": float(model.vis.map.stiffnessrot),
            "force": float(model.vis.map.force),
            "torque": float(model.vis.map.torque),
        },
        "stat": {
            "extent": float(model.stat.extent),
            "meanmass": float(model.stat.meanmass),
        },
        "neutral_qpos": [float(x) for x in data.qpos],
        "neutral_ctrl": [float(x) for x in data.ctrl],
        "actuators": actuators,
        "groups": groups,
        "geom_rgba": build_geom_colors(model),
    }


def _dof_label(joint_short: str) -> str:
    """A compact human label for a slider, e.g. ``coxa · yaw``."""
    parts = joint_short.rsplit("-", 2)
    if len(parts) == 3:
        _, child, axis = parts
        seg = child.split("_", 1)[-1] if "_" in child else child
        return f"{seg} · {axis}"
    return joint_short


# --- per-geom colors (mirrors the deeperfly keypoint-viewer build) ----------
def segment_colors() -> dict[str, list[float]]:
    """Map each body segment to a representative RGBA from flygym's visuals.yaml.

    Textured materials have no flat color, so take the texture's base color
    (``rgb1``, or the mean of ``rgb1``/``rgb2`` for gradients); plain materials
    use their ``rgba``. The alpha is always the material ``rgba``'s alpha (so the
    wings keep their 0.3 transparency). Wildcards in ``apply_to`` match segment
    names as in flygym.colorize().
    """
    with open(assets_dir / "model/neuromechfly/visuals.yaml") as fh:
        vis = yaml.safe_load(fh)
    colors: dict[str, list[float]] = {}
    for params in vis.values():
        rgba = params["material"].get("rgba", [1, 1, 1, 1])
        alpha = rgba[3] if len(rgba) > 3 else 1.0
        tex = params.get("texture")
        if tex:
            rgb1 = tex.get("rgb1", [0.6, 0.6, 0.6])
            rgb = (
                [(a + b) / 2 for a, b in zip(rgb1, tex.get("rgb2", rgb1))]
                if tex.get("builtin") == "gradient"
                else rgb1
            )
        else:
            rgb = rgba[:3]
        patterns = params["apply_to"]
        for pattern in [patterns] if isinstance(patterns, str) else patterns:
            for seg in fnmatch.filter(ALL_SEGMENT_NAMES, pattern):
                colors[seg] = [round(float(c), 4) for c in (*rgb, alpha)]
    return colors


def build_geom_colors(model: mj.MjModel) -> list[list[float]]:
    """A representative RGBA per geom (matched by geom/segment name)."""
    seg_color = segment_colors()
    geom_rgba = []
    for g in range(model.ngeom):
        seg = model.geom(g).name.split("/")[-1]
        rgba = seg_color.get(seg)
        if rgba is None:  # fall back to the geom's body name
            body = model.body(int(model.geom_bodyid[g])).name.split("/")[-1]
            rgba = seg_color.get(body, [0.7, 0.7, 0.7, 1.0])
        geom_rgba.append(rgba)
    return geom_rgba


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Composing + exporting NeuroMechFly model (fly + flat ground) ...")
    model = build_model()

    print("Building model_meta.json (timestep, neutral, actuators, colors) ...")
    meta = build_meta(model)
    (OUT_DIR / "model_meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    n_stl = len(list(MODEL_DIR.glob("*.stl")))
    size_mb = sum(p.stat().st_size for p in OUT_DIR.rglob("*")) / 1e6
    print(
        f"\nDone -> {OUT_DIR.relative_to(REPO_ROOT)}\n"
        f"  model/fly.xml + {n_stl} STL meshes\n"
        f"  {len(meta['actuators'])} actuators, nq={meta['nq']}, nu={meta['nu']}, "
        f"timestep={meta['timestep']}\n"
        f"  total {size_mb:.2f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
