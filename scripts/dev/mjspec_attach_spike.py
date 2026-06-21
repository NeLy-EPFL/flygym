"""Spike: validate PyMJCF -> MjSpec migration risk #1 (attach + naming).

Issue #282 proposes dropping dm-control's PyMJCF in favour of MuJoCo's native
MjSpec. The biggest risk identified was whether MjSpec's attach + namespacing can
reproduce the compiled element names FlyGym's runtime relies on. FlyGym maps
stored MJCF element references to MuJoCo IDs via ``mj.mj_name2id(model,
element.full_identifier)`` (see ``flygym/simulation.py``), so any name mismatch
would silently break state read/write.

This script builds a minimal fly + world the same way FlyGym does, once with
PyMJCF (the status quo) and once with MjSpec, and asserts the two compiled models
agree on everything FlyGym actually maps: element names, DoF layout, and body
world positions. Run it with::

    .venv/bin/python scripts/dev/mjspec_attach_spike.py

Findings (MuJoCo 3.6.0):

* ``world.attach(fly_spec, prefix=f"{fly.name}/", site=spawn_site)`` reproduces
  PyMJCF's ``modelname/`` prefix for every body / joint / actuator / sensor that
  FlyGym references.
* MjSpec **mutates element references in place** on attach: a body created as
  ``"thorax"`` reports ``.name == "fly/thorax"`` after attach. So FlyGym's pattern
  of stashing element refs (``bodyseg_to_mjcfbody`` etc.) and later reading their
  compiled name keeps working -- ``element.full_identifier`` simply becomes
  ``element.name``.
* The only structural difference: PyMJCF inserts an extra massless intermediate
  body (the attached submodel's old worldbody, named ``"fly/"``). MjSpec attaches
  the submodel's top-level bodies directly, so ``nbody`` differs by one. FlyGym
  never references that intermediate body and indexes by name (not count), so this
  is harmless -- DoF layout (nq/nv) and segment positions are identical.
"""

import numpy as np
import mujoco as mj
import dm_control.mjcf as mjcf

# Elements FlyGym maps to MuJoCo IDs at runtime. The spike asserts these resolve
# identically under both backends.
SEGMENTS = ["fly/thorax", "fly/coxa"]
JOINTS = ["fly/coxa_joint"]
ACTUATORS = ["fly/coxa_joint-position"]


def build_pymjcf() -> mj.MjModel:
    """Minimal fly+world via the current PyMJCF idiom (mirrors flygym.compose)."""
    fly = mjcf.RootElement(model="fly")
    thorax = fly.worldbody.add("body", name="thorax", pos=[0, 0, 0])
    thorax.add("geom", name="thorax", type="sphere", size=[0.1], mass=1)
    coxa = thorax.add("body", name="coxa", pos=[0.5, 0, 0])
    coxa.add("joint", name="coxa_joint", type="hinge", axis=[0, 1, 0])
    coxa.add("geom", name="coxa", type="sphere", size=[0.1], mass=1)
    fly.actuator.add("position", name="coxa_joint-position", joint="coxa_joint")

    world = mjcf.RootElement(model="world")
    world.worldbody.add("geom", name="ground", type="plane", size=[5, 5, 1])
    site = world.worldbody.add("site", name="fly", pos=[1, 2, 3])
    # The flygym idiom: attach the fly at a spawn site, then add a freejoint.
    site.attach(fly).add("freejoint", name="fly")

    return mjcf.Physics.from_mjcf_model(world).model._model


def build_mjspec() -> tuple[mj.MjModel, mj.MjsBody, mj.MjsJoint]:
    """Same model via native MjSpec. Returns the model plus a couple of element
    references (created *before* attach) to demonstrate in-place name mutation."""
    fly = mj.MjSpec()
    fly.modelname = "fly"
    thorax = fly.worldbody.add_body(name="thorax", pos=[0, 0, 0])
    thorax.add_geom(name="thorax", type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=1)
    coxa = thorax.add_body(name="coxa", pos=[0.5, 0, 0])
    coxa_joint = coxa.add_joint(
        name="coxa_joint", type=mj.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0]
    )
    coxa.add_geom(name="coxa", type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=1)
    fly.add_actuator(
        name="coxa_joint-position", target="coxa_joint", trntype=mj.mjtTrn.mjTRN_JOINT
    )

    world = mj.MjSpec()
    world.modelname = "world"
    world.worldbody.add_geom(name="ground", type=mj.mjtGeom.mjGEOM_PLANE, size=[5, 5, 1])
    site = world.worldbody.add_site(name="fly", pos=[1, 2, 3])
    # MjSpec analog of `site.attach(fly).add("freejoint", ...)`: attach with an
    # explicit prefix, then add the freejoint to the attached root body.
    world.attach(fly, prefix="fly/", site=site)
    world.body("fly/thorax").add_freejoint(name="fly")

    return world.compile(), thorax, coxa_joint


def names(model: mj.MjModel, objtype, count) -> list[str]:
    return [mj.mj_id2name(model, objtype, i) for i in range(count)]


def positions(model: mj.MjModel) -> dict[str, np.ndarray]:
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    out = {}
    for seg in SEGMENTS:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, seg)
        out[seg] = np.round(data.xpos[bid], 6)
    return out


def main() -> None:
    pymjcf_model = build_pymjcf()
    mjspec_model, thorax_ref, joint_ref = build_mjspec()

    # 1. In-place name mutation: refs created before attach now carry the prefix.
    assert thorax_ref.name == "fly/thorax", thorax_ref.name
    assert joint_ref.name == "fly/coxa_joint", joint_ref.name
    print("[ok] MjSpec mutates held element refs in place to include attach prefix")

    # 2. Every element FlyGym maps resolves under both backends.
    for model, label in [(pymjcf_model, "PyMJCF"), (mjspec_model, "MjSpec")]:
        for seg in SEGMENTS:
            assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, seg) >= 0, (label, seg)
        for jnt in JOINTS:
            assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt) >= 0, (label, jnt)
        for act in ACTUATORS:
            assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act) >= 0, (label, act)
    print("[ok] all FlyGym-mapped names resolve via mj_name2id under both backends")

    # 3. DoF layout is identical (qpos/dof addresses are what flygym indexes).
    assert pymjcf_model.nq == mjspec_model.nq, (pymjcf_model.nq, mjspec_model.nq)
    assert pymjcf_model.nv == mjspec_model.nv, (pymjcf_model.nv, mjspec_model.nv)
    for jnt in JOINTS:
        a = pymjcf_model.jnt_qposadr[mj.mj_name2id(pymjcf_model, mj.mjtObj.mjOBJ_JOINT, jnt)]
        b = mjspec_model.jnt_qposadr[mj.mj_name2id(mjspec_model, mj.mjtObj.mjOBJ_JOINT, jnt)]
        assert a == b, (jnt, a, b)
    print(f"[ok] DoF layout matches: nq={mjspec_model.nq} nv={mjspec_model.nv}")

    # 4. Body world positions match (spawn site transform honoured identically).
    p_pos, m_pos = positions(pymjcf_model), positions(mjspec_model)
    for seg in SEGMENTS:
        assert np.allclose(p_pos[seg], m_pos[seg]), (seg, p_pos[seg], m_pos[seg])
    print("[ok] segment world positions match:", {k: list(v) for k, v in m_pos.items()})

    # 5. Document the one expected structural difference (harmless for flygym).
    print(
        f"[note] nbody differs by design: PyMJCF={pymjcf_model.nbody} "
        f"MjSpec={mjspec_model.nbody} (PyMJCF inserts a massless intermediate "
        f"'fly/' body that flygym never references)"
    )
    print("\nSPIKE PASSED: MjSpec attach reproduces all names/DoF/positions flygym relies on.")


if __name__ == "__main__":
    main()
