"""Generate the static assets for the browser (WebAssembly) NeuroMechFly game.

The page at ``wasm/game/game.html`` runs the NeuroMechFly Live game in the
browser: pilot the fly through a slalom track to a finish line at three levels of
neural abstraction (CPG / tripod gait / individual legs). Like the interactive
viewer it uses MuJoCo compiled to WebAssembly (vendored under
``wasm/shared/vendor/mujoco``) and renders with Three.js, and the simulation is
real (``mj_step`` at dt=1e-4, ~0.1x playback like the desktop game). The control
logic the desktop game gets from Python/flygym is ported to JavaScript and fed
the **baked tables** this script writes, so the browser needs no SciPy/flygym.

This script is run *by hand* whenever the model or controller config changes. It
needs ``flygym`` + ``flygym_demo`` + ``mujoco``, e.g.::

    uv run python scripts/dev/build_wasm_game_assets.py

Outputs (all under ``wasm/game/assets/``):

``model/fly.xml`` + ``model/*.stl``
    A flattened, self-contained MJCF (legs-only position-actuated fly with leg
    adhesion, in the slalom arena) and the meshes it references, written by
    ``world.save_xml_with_assets``. The browser loads this via ``mj_loadXML``.
``model_meta.json``
    Everything the game needs that is awkward to read from the WASM model at
    runtime: timestep; the neutral keyframe ``qpos`` / ``ctrl``; per position
    actuator info; the adhesion actuators; the (leg, dof) -> ctrl-index mapping;
    a representative RGB per geom; the arena finish line + spawn; the CPG
    parameters and ``leg_step_time``; and the **baked PreprogrammedSteps tables**
    (per-leg joint-angle trajectories sampled over a phase grid, neutral pose,
    and swing/stance split) that the JS port interpolates instead of SciPy
    cubic splines.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
from pathlib import Path

import mujoco as mj
import numpy as np
import yaml
from flygym import assets_dir
from flygym.anatomy import ALL_SEGMENT_NAMES, ContactBodiesPreset
from flygym.compose import ContactParams, FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym.utils.mjcf import GEOM_TYPES
from flygym_demo.complex_terrain.common import make_locomotion_fly
from flygym_demo.complex_terrain.preprogrammed import PreprogrammedSteps

# --- repo paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "wasm/game/assets"
MODEL_DIR = OUT_DIR / "model"

# --- spawn / arena ----------------------------------------------------------
SPAWN_POSITION = (0.0, 0.0, 0.5)  # xyz in mm; fly starts at the start line facing +x
SPAWN_ROTATION = Rotation3D("quat", (1, 0, 0, 0))  # wxyz identity -> body +x = world +x

# SlalomArena geometry (ported verbatim from neuromechfly-live game_arena.py).
GATE_OFFSET = -3.0
GATE_WIDTH = 10.0
GATE_HEIGHT = 5.0
GATE_SPACING = 10.0
POLE_RADIUS = 0.05
N_GATES = 5
GROUND_FRICTION = (1.0, 0.005, 0.0001)

# --- fly actuator / joint tuning --------------------------------------------
# Matched to tutorial 2 (kinematic replay): default joint stiffness/damping with
# no special passive-tarsus treatment, a high position gain, the default force
# range, and the default (weak) adhesion gain.
JOINT_STIFFNESS = 10.0
JOINT_DAMPING = 0.5
PASSIVE_TARSUS_STIFFNESS = 10.0
PASSIVE_TARSUS_DAMPING = 0.5
ACTUATOR_GAIN = 150.0
ACTUATOR_FORCERANGE = (-30.0, 30.0)
ADHESION_GAIN = 1.0

# --- CPG / controller config (from neuromechfly-live TurningController) ------
INTRINSIC_FREQ = 36.0
INTRINSIC_AMP = 6.0
CONVERGENCE_COEF = 20.0
COUPLING_STRENGTH = 10.0
LEG_STEP_TIME = 0.025  # s, tripod/single per-step duration
# legs order matches PreprogrammedSteps.legs; tripod_map LF/LH/RM -> 0, LM/RF/RH -> 1
TRIPOD_MAP = [0, 1, 0, 1, 0, 1]

# Chase-camera placement (loosely from neuromechfly-live
# Game.update_camera_to_follow_fly). Only height + distance behind the fly are
# consumed by the JS, which does its own per-frame yaw smoothing and look-at, so
# the desktop's per-physics-step smoothing / tilt are intentionally not baked.
CAMERA = {"height": 4.0, "distance": 6.5}

# Per-level ground tint. Desaturated, dark versions of the desktop game's floor
# colors (Game.state_floor_colors) -- low-saturation tinted darks so the floor
# stays easy on the eyes and the orange fly reads clearly, especially the red
# of level 3.
LEVEL_GROUND_COLORS = {
    "CPG": [0.22, 0.27, 0.15, 1.0],  # dark muted green
    "tripod": [0.16, 0.19, 0.28, 1.0],  # dark muted blue
    "single": [0.27, 0.15, 0.15, 1.0],  # dark muted red
}

# Number of phase samples for the baked PreprogrammedSteps tables. The JS port
# linearly interpolates between samples (periodic); 360 (= 1 deg) is plenty.
N_PHASE_SAMPLES = 360

_tripod_phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
_tripod_coupling_weights = (_tripod_phase_biases > 0) * COUPLING_STRENGTH


class SlalomGroundWorld(FlatGroundWorld):
    """Flat checker ground (collidable via contact pairs) plus the slalom gates.

    Start poles, five gates (red/blue, last one white = finish line) and a stored
    ``finish_line_points`` segment, ported from neuromechfly-live's ``SlalomArena``.
    Like every collidable surface in flygym v2, the poles collide with the fly
    only through *explicit contact pairs* (the fly's own geoms have
    ``contype=conaffinity=0``, so the broadphase never generates fly contacts).
    ``add_obstacle_contacts`` (called after the fly is attached) pairs each pole
    with the fly's body/leg collision geoms so the fly is physically blocked by
    the gates; the finish is still a geometric path/line crossing test in JS.
    """

    def __init__(self) -> None:
        super().__init__(name="slalom_world", half_size=100)
        wb = self.mjcf_root.worldbody
        self.pole_geoms = []

        def pole(name, x, y, half_h, rgba, z=0.0):
            # contype/conaffinity left at 0: like the ground, poles collide with
            # the fly via explicit pairs (see add_obstacle_contacts), not broadphase.
            self.pole_geoms.append(
                wb.add_geom(
                    type=GEOM_TYPES["cylinder"],
                    name=name,
                    size=[POLE_RADIUS, half_h, 0.0],
                    pos=[x, y, z],
                    rgba=rgba,
                    contype=0,
                    conaffinity=0,
                )
            )

        black = [0.0, 0.0, 0.0, 1.0]
        pole("start_pole_left", 0, -GATE_WIDTH / 2, GATE_HEIGHT / 2, black)
        pole("start_pole_right", 0, GATE_WIDTH / 2, GATE_HEIGHT / 2, black)

        self.finish_line_points = None
        for i in range(N_GATES):
            offset = 1 if i % 2 == 0 else -1
            x = (i + 1) * GATE_SPACING
            if i == N_GATES - 1:
                color = [1.0, 1.0, 1.0, 1.0]
                self.finish_line_points = [
                    [x, offset * GATE_OFFSET],
                    [x, offset * (GATE_OFFSET + GATE_WIDTH)],
                ]
            else:
                color = [1.0, 0, 0, 1.0] if i % 2 == 0 else [0, 0, 1.0, 1.0]
            half_h = GATE_HEIGHT / 2 + 0.05
            pole(f"gate{i}_inside", x, offset * GATE_OFFSET, half_h, color, z=-0.05)
            pole(
                f"gate{i}_outside",
                x,
                offset * (GATE_OFFSET + GATE_WIDTH),
                half_h,
                color,
                z=-0.05,
            )

    def add_obstacle_contacts(self, fly) -> int:
        """Add explicit contact pairs so the fly is blocked by the poles.

        Pairs every pole with the fly's ground-contact collision geoms (the same
        legs + thorax/abdomen/head set used for the floor), so the whole fly is
        solid against the gates. Must be called after ``add_fly``. Returns the
        number of pairs added.
        """
        params = ContactParams()
        segs = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD.to_body_segments_list()
        fly_geoms = [g for seg in segs for g in fly.bodyseg_to_mjcfgeom[seg]]
        n = 0
        for pole_geom in self.pole_geoms:
            for fly_geom in fly_geoms:
                self.mjcf_root.add_pair(
                    geomname1=fly_geom.name,
                    geomname2=pole_geom.name,
                    name=f"obstacle_pair_{n}",
                    friction=params.get_friction_tuple(),
                    solref=params.get_solref_tuple(),
                    solimp=params.get_solimp_tuple(),
                    margin=params.margin,
                )
                n += 1
        return n


def build_model() -> tuple[mj.MjModel, SlalomGroundWorld]:
    """Compose the legs-only position-actuated fly (with adhesion) in the slalom
    arena, export it to a self-contained MJCF under ``MODEL_DIR``, and return the
    model reloaded standalone (what the browser's ``mj_loadXML`` sees)."""
    fly = make_locomotion_fly(
        name="nmf",
        joint_stiffness=JOINT_STIFFNESS,
        joint_damping=JOINT_DAMPING,
        passive_tarsus_stiffness=PASSIVE_TARSUS_STIFFNESS,
        passive_tarsus_damping=PASSIVE_TARSUS_DAMPING,
        actuator_gain=ACTUATOR_GAIN,
        actuator_forcerange=ACTUATOR_FORCERANGE,
        add_adhesion=True,
        adhesion_gain=ADHESION_GAIN,
        colorize=True,
    )

    world = SlalomGroundWorld()
    world.add_fly(fly, SPAWN_POSITION, SPAWN_ROTATION)
    n_pairs = world.add_obstacle_contacts(fly)
    print(f"  added {n_pairs} fly<->pole contact pairs")

    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True)
    world.save_xml_with_assets(MODEL_DIR, "fly.xml")

    model = mj.MjModel.from_xml_path(str(MODEL_DIR / "fly.xml"))
    assert model.nkey >= 1, "expected a baked 'neutral' keyframe in the exported model"
    return model, world


# --- preprogrammed-step DOF spec -> (leg, dof index) ------------------------
# PreprogrammedSteps.dofs_per_leg order: each entry is (parent_link, child_link, axis).
_DOF_BY_CHILD_AXIS = {
    (child, axis): idx
    for idx, (_parent, child, axis) in enumerate(PreprogrammedSteps.dofs_per_leg)
}
_LEG_INDEX = {leg: i for i, leg in enumerate(PreprogrammedSteps.legs)}


def _parse_actuator_joint(joint_short: str):
    """('c_thorax-lf_coxa-yaw') -> (leg='lf', dof_index=2) or (None, None)."""
    parts = joint_short.rsplit("-", 2)
    if len(parts) != 3:
        return None, None
    _parent, child, axis = parts
    leg = child.split("_", 1)[0]
    child_link = child.split("_", 1)[1] if "_" in child else child
    if leg not in _LEG_INDEX:
        return None, None
    return leg, _DOF_BY_CHILD_AXIS.get((child_link, axis))


def build_meta(model: mj.MjModel, world: SlalomGroundWorld) -> dict:
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)  # the "neutral" keyframe

    # Position actuators (42 = 6 legs x 7 active DOFs) and the adhesion actuators
    # (6, one per leg), and the (leg, dof) -> ctrl-index scatter map.
    actuators = []
    ctrl_index_by_leg_dof = [[None] * 7 for _ in range(6)]
    adhesion = [None] * 6
    for a in range(model.nu):
        act = model.actuator(a)
        name = act.name.split("/")[-1]

        # Adhesion actuator (transmits to a body)
        if act.trntype == mj.mjtTrn.mjTRN_BODY:
            body = model.body(int(act.trnid[0])).name.split("/")[-1]
            leg = body.split("_", 1)[0]
            if leg in _LEG_INDEX:
                adhesion[_LEG_INDEX[leg]] = a
            continue

        joint_id = int(act.trnid[0])
        joint = model.joint(joint_id)
        joint_short = joint.name.split("/")[-1]
        leg, dof_idx = _parse_actuator_joint(joint_short)

        if leg is not None and dof_idx is not None:
            ctrl_index_by_leg_dof[_LEG_INDEX[leg]][dof_idx] = a

        lo, hi = (float(x) for x in act.ctrlrange)
        actuators.append(
            {
                "id": a,
                "name": name,
                "joint": joint_short,
                "qposadr": int(model.jnt_qposadr[joint_id]),
                "ctrlrange": [lo, hi],
                "neutral": float(data.ctrl[a]),
            }
        )

    assert all(all(idx is not None for idx in leg) for leg in ctrl_index_by_leg_dof), (
        "could not map every (leg, dof) to a position actuator"
    )
    assert all(idx is not None for idx in adhesion), "missing an adhesion actuator"

    return {
        "nq": int(model.nq),
        "nu": int(model.nu),
        "timestep": float(model.opt.timestep),
        "gravity": [float(x) for x in model.opt.gravity],
        "neutral_qpos": [float(x) for x in data.qpos],
        "neutral_ctrl": [float(x) for x in data.ctrl],
        "actuators": actuators,
        "adhesion": adhesion,
        "ctrl_index_by_leg_dof": ctrl_index_by_leg_dof,
        "geom_rgba": build_geom_colors(model),
        "arena": {
            "finish_line": world.finish_line_points,
            "spawn": list(SPAWN_POSITION),
            "n_gates": N_GATES,
            "gate_spacing": GATE_SPACING,
            "ground_half_size": 100.0,
            "level_ground_colors": LEVEL_GROUND_COLORS,
        },
        "camera": CAMERA,
        "control": {
            "leg_order": list(PreprogrammedSteps.legs),
            "tripod_map": TRIPOD_MAP,
            "leg_step_time": LEG_STEP_TIME,
            "cpg": {
                "intrinsic_freqs": [INTRINSIC_FREQ] * 6,
                "intrinsic_amps": [INTRINSIC_AMP] * 6,
                "convergence_coefs": [CONVERGENCE_COEF] * 6,
                "coupling_weights": _tripod_coupling_weights.tolist(),
                "phase_biases": _tripod_phase_biases.tolist(),
            },
        },
        "preprogrammed": bake_preprogrammed_steps(),
    }


def bake_preprogrammed_steps() -> dict:
    """Sample PreprogrammedSteps onto a phase grid for the JS port.

    For each leg: ``angles`` is an ``(N, 7)`` table of joint angles at magnitude 1
    over phases ``linspace(0, 2pi, N, endpoint=False)`` (JS wraps + lerps);
    ``neutral`` is the 7-DOF neutral pose; ``swing`` is ``[start, end]`` in phase
    units carving swing (adhesion off) out of the cycle. DOF axis is
    ``PreprogrammedSteps.dofs_per_leg`` order, matching ``ctrl_index_by_leg_dof``.
    """
    steps = PreprogrammedSteps()
    phases = np.linspace(0, 2 * np.pi, N_PHASE_SAMPLES, endpoint=False)
    out = {"n_samples": N_PHASE_SAMPLES, "legs": {}}
    for leg in steps.legs:
        # get_joint_angles at magnitude 1 -> the raw spline (7,) per phase.
        angles = np.array(
            [steps.get_joint_angles(leg, p, 1.0) for p in phases]
        )  # (N, 7)
        out["legs"][leg] = {
            "angles": [[float(v) for v in row] for row in angles],
            "neutral": [float(v) for v in steps.neutral_pos[leg].ravel()],
            "swing": [float(x) for x in steps.swing_period[leg]],
        }
    return out


# --- per-geom colors (mirrors build_wasm_viewer_assets.py) ------------------
def segment_colors() -> dict[str, list[float]]:
    """Map each body segment to a representative RGBA from flygym's visuals.yaml."""
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
    """A representative RGBA per geom: fly segments from visuals.yaml, everything
    else (the slalom poles, ground) straight from the geom's own ``rgba``."""
    seg_color = segment_colors()
    geom_rgba = []
    for g in range(model.ngeom):
        seg = model.geom(g).name.split("/")[-1]
        rgba = seg_color.get(seg)
        if rgba is None:
            body = model.body(int(model.geom_bodyid[g])).name.split("/")[-1]
            rgba = seg_color.get(body)
        if rgba is None:  # poles / ground: use the model's own geom color
            rgba = [round(float(c), 4) for c in model.geom_rgba[g]]
        geom_rgba.append(rgba)
    return geom_rgba


def verify(model: mj.MjModel) -> None:
    """Sanity check: the fly settles (no NaN/blow-up) over a short rollout."""
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    for _ in range(2000):
        mj.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos)), "model blew up during verification rollout"
    z = data.qpos[2]
    assert -1.0 < z < 5.0, f"fly z={z:.3f} mm after settling looks wrong"
    print(f"  verify: stable rollout, fly z={z:.3f} mm, ncon~{data.ncon}")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Composing + exporting NeuroMechFly game model (fly + slalom arena) ...")
    model, world = build_model()

    print("Building model_meta.json (actuators, mappings, CPG + baked steps) ...")
    meta = build_meta(model, world)
    (OUT_DIR / "model_meta.json").write_text(json.dumps(meta, separators=(",", ":")))

    print("Verifying the exported model ...")
    verify(model)

    n_stl = len(list(MODEL_DIR.glob("*.stl")))
    size_mb = sum(p.stat().st_size for p in OUT_DIR.rglob("*")) / 1e6
    print(
        f"\nDone -> {OUT_DIR.relative_to(REPO_ROOT)}\n"
        f"  model/fly.xml + {n_stl} STL meshes\n"
        f"  nu={meta['nu']} ({len(meta['actuators'])} position + "
        f"{len(meta['adhesion'])} adhesion), timestep={meta['timestep']}\n"
        f"  finish line at x={meta['arena']['finish_line'][0][0]}\n"
        f"  total {size_mb:.2f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
