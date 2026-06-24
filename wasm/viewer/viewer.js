// Interactive, *simulated* viewer for the NeuroMechFly model used by
// scripts/launch_interactive_viewer.py -- the browser counterpart of MuJoCo's
// native viewer.
//
// MuJoCo (compiled to WebAssembly by Google DeepMind, vendored under
// ../shared/vendor/mujoco/) loads the flattened MJCF written by
// scripts/dev/build_wasm_viewer_assets.py and runs the real dynamics: every
// animation frame advances the physics with `mj_step`. The position-actuator
// sliders write `data.ctrl`; a bar over each slider reads the driven joint's
// `data.qpos`; bodies can be dragged (Shift+drag) to push them via
// `data.xfrc_applied`; and contacts/forces/joints/actuators can be drawn on
// top, like the native viewer's visualization flags. Three.js does the
// rendering, reading body/geom frames straight from the solved state.

import * as THREE from 'three';
import { OrbitControls } from '../shared/vendor/three/OrbitControls.js';
import {
  loadScene, makeFailOverlay, makeStepper, makeStatsMeter,
  buildMeshes, syncMeshes, pool,
} from '../shared/scene.js';

const ASSETS = './assets';
const overlayEl = document.getElementById('overlay');
const overlayMsg = document.getElementById('overlay-msg');

const fail = makeFailOverlay(overlayEl, 'viewer');

main().catch((e) => fail('Unexpected error while starting up.', e));

async function main() {
  const { mj, model, data, meta } = await loadScene({
    assetsDir: ASSETS, xmlName: 'fly.xml',
    onStage: (msg) => { overlayMsg.textContent = msg; },
  });
  buildApp(mj, model, data, meta);
  overlayEl.classList.add('hidden');
}

// Each animation frame advances at most this many physics steps. At dt=1e-4 s
// that is ~SUBSTEPS*60 steps/s; the model is heavy, so this trades real-time
// speed for a steady frame rate (the achieved factor is shown in the corner).
const MAX_SUBSTEPS = 40;
// Drag-to-push: applied force = PERTURB_GAIN * map.stiffness * mass * disp. The
// grab->pointer displacement (mm) is clamped and the force capped at this many
// body weights, so a big drag in the slowed-down sim can't explode the solver.
const PERTURB_GAIN = 50;
const PERTURB_MAX_DISP = 2.0;
const PERTURB_MAX_WEIGHTS = 3000;
// Contact-force arrows use MuJoCo's own scale (stat.extent * map.force /
// (stat.meanmass * |g|)); at flygym's tiny map.force that is ~0.01 mm/force --
// invisible -- so multiply by this documented gain to make them readable.
const CONTACT_FORCE_GAIN = 27;
const CONTACT_FORCE_MAX_LEN = 30; // mm, clamp so a spike can't fill the screen
// Force arrows are drawn as a solid shaft (a real, thick cylinder -- WebGL lines
// can't be widened) plus a fixed-size arrowhead, so only the shaft length tracks
// the force; the head stays constant.
const CONTACT_FORCE_SHAFT_R = 0.018; // mm (shaft radius)
const CONTACT_FORCE_HEAD_LEN = 0.13; // mm (constant)
const CONTACT_FORCE_HEAD_R = 0.055;  // mm (constant)

function buildApp(mj, model, data, meta) {
  const stage = document.getElementById('stage');
  THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // MuJoCo is z-up

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
  camera.up.set(0, 0, 1);

  const center = new THREE.Vector3(0, 0, 0.9);
  camera.position.set(4.5, -4.5, 3.0);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.minDistance = 1.2;
  controls.maxDistance = 40;
  controls.enablePan = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const key = new THREE.DirectionalLight(0xffffff, 1.05); key.position.set(4, -6, 8);
  const fill = new THREE.DirectionalLight(0xffffff, 0.45); fill.position.set(-6, 4, 2);
  scene.add(key, fill);
  addGround(scene);

  const meshGroup = buildMeshes(model, meta);
  scene.add(meshGroup);

  // Visual/perturbation scales derived from the model's global MuJoCo settings
  // (visual/map + stat), the same inputs the native viewer uses.
  const gnorm = Math.hypot(meta.gravity[0], meta.gravity[1], meta.gravity[2]);
  const forceScale = CONTACT_FORCE_GAIN *
    meta.stat.extent * meta.map.force / (meta.stat.meanmass * gnorm);

  const viz = buildVisualizers(scene, forceScale, meta.cone);
  const physicsSliders = buildPhysicsSliders(mj, model);
  const sliders = buildSliders(meta, data);
  const sim = { paused: false };

  // --- wiring: simulation buttons ---
  document.getElementById('reset').addEventListener('click', () => {
    mj.mj_resetDataKeyframe(model, data, 0);
    clearPerturb();
    mj.mj_forward(model, data);
    sliders.reset();
  });
  const pauseBtn = document.getElementById('pause');
  pauseBtn.addEventListener('click', () => {
    sim.paused = !sim.paused;
    pauseBtn.textContent = sim.paused ? 'Resume' : 'Pause';
    pauseBtn.classList.toggle('primary', sim.paused);
  });

  // --- wiring: visualization toggles ---
  const flags = { contacts: false, forces: false, joints: false, actuators: false };
  const bind = (id, key) => document.getElementById(id)
    .addEventListener('change', (e) => { flags[key] = e.target.checked; });
  bind('t-contacts', 'contacts');
  bind('t-forces', 'forces');
  bind('t-joints', 'joints');
  bind('t-actuators', 'actuators');
  // --- camera reset ---
  const home = { pos: camera.position.clone(), target: controls.target.clone() };
  document.getElementById('reset-view').addEventListener('click', () => {
    camera.position.copy(home.pos);
    controls.target.copy(home.target);
    controls.update();
  });
  document.getElementById('reset-physics').addEventListener('click', () => physicsSliders.reset());

  const perturb = setupPerturb({
    renderer, camera, controls, meshGroup, model, data, scene,
    stiffness: meta.map.stiffness, gnorm,
  });
  const clearPerturb = perturb.clear;

  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  // --- main loop ---
  const dt = meta.timestep;
  const statsEl = document.getElementById('stats');
  const stepper = makeStepper(dt, MAX_SUBSTEPS);
  const statsMeter = makeStatsMeter(dt, ({ fps, rtf }) => {
    statsEl.textContent =
      `${fps.toFixed(0)} fps · ${rtf.toFixed(2)}× realtime · ${data.ncon} contacts` +
      (sim.paused ? ' · paused' : '');
  });
  let lastWall = performance.now();

  function frame() {
    requestAnimationFrame(frame);
    const now = performance.now();
    const wallDt = Math.min((now - lastWall) / 1000, 0.1); // cap catch-up
    lastWall = now;

    let nSteps = 0;
    if (!sim.paused) {
      nSteps = stepper.advance(wallDt, () => {
        perturb.apply();
        mj.mj_step(model, data);
      });
    } else {
      mj.mj_forward(model, data);
    }

    syncMeshes(meshGroup, data);
    viz.update(mj, model, data, flags);
    perturb.updateGizmo();
    controls.update();
    renderer.render(scene, camera);

    statsMeter(now / 1000, nSteps);
  }
  requestAnimationFrame(frame);
}

// Flat checker-free ground: a faint grid on the z=0 plane the fly stands on.
function addGround(scene) {
  const grid = new THREE.GridHelper(60, 60, 0x6b7280, 0x44484f);
  grid.rotateX(Math.PI / 2); // GridHelper is xz; we want it on the world xy plane
  grid.position.set(0, 0, 0);
  grid.material.opacity = 0.35; grid.material.transparent = true;
  scene.add(grid);
}

// --- on-top visualizers: contacts, contact forces, joints, actuators -------
// Each is a pool of reusable objects grown on demand and hidden when unused, so
// per-frame updates never allocate once the pools are warm.
function buildVisualizers(scene, forceScale, cone) {
  const group = new THREE.Group();
  group.renderOrder = 3;
  scene.add(group);

  const onTopMat = (color) => new THREE.MeshBasicMaterial(
    { color, depthTest: false, transparent: true });
  const sphereGeom = new THREE.SphereGeometry(1, 12, 8);

  // contact points (cyan spheres)
  const contactPool = pool(() => {
    const m = new THREE.Mesh(sphereGeom, onTopMat(0x21d4fd));
    m.renderOrder = 5; group.add(m); return m;
  });
  // contact force arrows (bright green, to stand out against the orange body):
  // a thick cylinder shaft + a fixed-size cone head (WebGL lines can't be
  // widened, and we want the head size to stay constant).
  const forceMat = new THREE.MeshBasicMaterial(
    { color: 0x76ff03, depthTest: false, transparent: true });
  const shaftGeom = new THREE.CylinderGeometry(1, 1, 1, 10); // unit, axis +Y
  const headGeom = new THREE.ConeGeometry(1, 1, 14);         // unit, axis +Y
  const forcePool = pool(() => {
    const a = new THREE.Group();
    const shaft = new THREE.Mesh(shaftGeom, forceMat);
    const head = new THREE.Mesh(headGeom, forceMat);
    shaft.renderOrder = head.renderOrder = 5;
    a.add(shaft, head); a.userData = { shaft, head };
    group.add(a); return a;
  });
  const FUP = new THREE.Vector3(0, 1, 0);
  const placeForceArrow = (a, ox, oy, oz, dir, length) => {
    const shaftLen = Math.max(1e-4, length - CONTACT_FORCE_HEAD_LEN);
    const { shaft, head } = a.userData;
    shaft.scale.set(CONTACT_FORCE_SHAFT_R, shaftLen, CONTACT_FORCE_SHAFT_R);
    shaft.position.set(0, shaftLen / 2, 0);
    head.scale.set(CONTACT_FORCE_HEAD_R, CONTACT_FORCE_HEAD_LEN, CONTACT_FORCE_HEAD_R);
    head.position.set(0, shaftLen + CONTACT_FORCE_HEAD_LEN / 2, 0);
    a.position.set(ox, oy, oz);
    a.quaternion.setFromUnitVectors(FUP, dir);
    a.visible = true;
  };
  // joint markers (yellow): a short axis through the anchor
  const jointPool = pool(() => {
    const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(),
      0.18, 0xffd54f, 0.06, 0.035);
    a.line.material.depthTest = false; a.cone.material.depthTest = false;
    a.renderOrder = 4; group.add(a); return a;
  });
  // actuator markers (magenta): arrow along the joint axis, length ∝ |force|
  const actPool = pool(() => {
    const a = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(),
      0.2, 0xe040fb, 0.06, 0.04);
    a.line.material.depthTest = false; a.cone.material.depthTest = false;
    a.renderOrder = 4; group.add(a); return a;
  });

  const v = new THREE.Vector3();

  // Contact force in the contact frame, reconstructed from data.efc_force. This
  // build's mj_contactForce doesn't write back through the JS bindings, but the
  // constraint forces are reachable directly. For the default pyramidal cone the
  // normal force is the sum of the contact's pyramid multipliers and each
  // tangent is mu_k*(f_2k - f_2k+1); for the elliptic cone efc_force already
  // holds the contact-frame force. (Verified to match mj_contactForce.)
  const contactFrameForce = (data, c, out) => {
    const adr = c.efc_address, dim = c.dim, ef = data.efc_force, mu = c.friction;
    let fn, ft1 = 0, ft2 = 0;
    if (cone === 1) { // elliptic
      fn = ef[adr]; ft1 = dim > 1 ? ef[adr + 1] : 0; ft2 = dim > 2 ? ef[adr + 2] : 0;
    } else { // pyramidal (default)
      fn = 0; const nef = 2 * (dim - 1);
      for (let k = 0; k < nef; k++) fn += ef[adr + k];
      ft1 = dim > 1 ? mu[0] * (ef[adr] - ef[adr + 1]) : 0;
      ft2 = dim > 2 ? mu[1] * (ef[adr + 2] - ef[adr + 3]) : 0;
    }
    out[0] = fn; out[1] = ft1; out[2] = ft2;
  };
  const cf = [0, 0, 0];

  function update(mj, model, data, flags) {
    // contacts + forces
    contactPool.begin(); forcePool.begin();
    if (flags.contacts || flags.forces) {
      const n = data.ncon;
      for (let i = 0; i < n; i++) {
        const c = data.contact.get(i);
        const pos = c.pos;
        if (flags.contacts) {
          const s = contactPool.next();
          s.position.set(pos[0], pos[1], pos[2]); s.scale.setScalar(0.03);
          s.visible = true;
        }
        if (flags.forces) {
          // contact-frame force [normal, tan1, tan2], rotated into world coords
          // via contact.frame (rows = contact axes expressed in world).
          contactFrameForce(data, c, cf);
          const fr = c.frame;
          const fx = cf[0] * fr[0] + cf[1] * fr[3] + cf[2] * fr[6];
          const fy = cf[0] * fr[1] + cf[1] * fr[4] + cf[2] * fr[7];
          const fz = cf[0] * fr[2] + cf[1] * fr[5] + cf[2] * fr[8];
          v.set(fx, fy, fz);
          const mag = v.length();
          if (mag > 1e-9) {
            // shaft length ∝ force (MuJoCo-derived scale); head size is constant.
            const len = Math.min(CONTACT_FORCE_MAX_LEN, mag * forceScale);
            placeForceArrow(forcePool.next(), pos[0], pos[1], pos[2],
              v.multiplyScalar(1 / mag), len);
          }
        }
      }
    }
    contactPool.end(); forcePool.end();

    // joints (all hinge joints): a short double-headed axis at the anchor
    jointPool.begin();
    if (flags.joints) {
      const xa = data.xanchor, ax = data.xaxis;
      for (let j = 0; j < model.njnt; j++) {
        if (model.jnt_type[j] !== 3) continue; // hinge only
        const a = jointPool.next();
        a.position.set(xa[3 * j], xa[3 * j + 1], xa[3 * j + 2]);
        v.set(ax[3 * j], ax[3 * j + 1], ax[3 * j + 2]);
        a.position.addScaledVector(v, -0.09);
        a.setDirection(v); a.setLength(0.18, 0.06, 0.035);
        a.visible = true;
      }
    }
    jointPool.end();

    // actuators: arrow along the driven joint axis, length ∝ |actuator force|
    actPool.begin();
    if (flags.actuators) {
      const xa = data.xanchor, ax = data.xaxis, af = data.actuator_force;
      for (let u = 0; u < model.nu; u++) {
        const j = model.actuator_trnid[2 * u]; // first transmission id = joint id
        if (j < 0 || model.jnt_type[j] !== 3) continue;
        const f = af[u];
        const a = actPool.next();
        a.position.set(xa[3 * j], xa[3 * j + 1], xa[3 * j + 2]);
        v.set(ax[3 * j], ax[3 * j + 1], ax[3 * j + 2]);
        if (f < 0) v.multiplyScalar(-1);
        a.setDirection(v);
        a.setLength(0.08 + Math.min(0.5, Math.abs(f) * 0.02), 0.05, 0.035);
        a.visible = true;
      }
    }
    actPool.end();
  }

  return { update };
}

// --- drag-to-push (Shift + left-drag a body) -------------------------------
// Apply an external wrench to the grabbed body via data.xfrc_applied (the same
// channel MuJoCo's own perturbation uses): a spring from the grab point to the
// pointer's position on a camera-facing plane, with the matching torque so the
// push acts at the grabbed point, not just the COM.
function setupPerturb(ctx) {
  const { renderer, camera, controls, meshGroup, model, data, scene, stiffness, gnorm } = ctx;
  const dom = renderer.domElement;

  // Per-body perturbation scale: abdomen/head segments are more sensitive,
  // thorax is very massive — reduce applied force so drags stay controllable.
  const bodyPerturbScale = new Float64Array(model.nbody).fill(1.0);
  for (let b = 0; b < model.nbody; b++) {
    const name = model.body(b).name;
    if (/abdomen|head/.test(name)) bodyPerturbScale[b] = 1 / 3;
    else if (/thorax/.test(name)) bodyPerturbScale[b] = 1 / 8;
  }
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const plane = new THREE.Plane(), hitPt = new THREE.Vector3(), target = new THREE.Vector3();
  const n = new THREE.Vector3();

  // gizmo: a line from grab point to the pointer target
  const gizMat = new THREE.LineBasicMaterial({ color: 0xff5252, depthTest: false, transparent: true });
  const gizGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const gizmo = new THREE.Line(gizGeom, gizMat); gizmo.renderOrder = 6; gizmo.visible = false;
  scene.add(gizmo);

  let drag = null; // { bodyId, localOffset:[3] }

  const setNDC = (e) => {
    const r = dom.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  };

  const onDown = (e) => {
    if (e.button !== 0 || !e.shiftKey) return;
    setNDC(e);
    raycaster.setFromCamera(ndc, camera);
    meshGroup.updateWorldMatrix(true, true);
    const meshes = meshGroup.userData.items.map((it) => it.mesh);
    const hit = raycaster.intersectObjects(meshes, false)[0];
    if (!hit) return;
    const bodyId = hit.object.userData.bodyId;
    if (bodyId <= 0) return; // world / fixed
    // express the grabbed point in the body's local frame
    const bx = data.xpos, bm = data.xmat, b = 3 * bodyId, r = 9 * bodyId;
    const dx = hit.point.x - bx[b], dy = hit.point.y - bx[b + 1], dz = hit.point.z - bx[b + 2];
    const localOffset = [
      bm[r] * dx + bm[r + 3] * dy + bm[r + 6] * dz,
      bm[r + 1] * dx + bm[r + 4] * dy + bm[r + 7] * dz,
      bm[r + 2] * dx + bm[r + 5] * dy + bm[r + 8] * dz,
    ];
    camera.getWorldDirection(n);
    plane.setFromNormalAndCoplanarPoint(n, hit.point);
    drag = { bodyId, localOffset, target: hit.point.clone() };
    controls.enabled = false;
    dom.style.cursor = 'grabbing';
    dom.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const onMove = (e) => {
    if (!drag) {
      // hover affordance
      if (e.shiftKey) dom.style.cursor = 'grab';
      else if (dom.style.cursor === 'grab') dom.style.cursor = '';
      return;
    }
    setNDC(e);
    raycaster.setFromCamera(ndc, camera);
    if (raycaster.ray.intersectPlane(plane, target)) drag.target.copy(target);
  };

  const onUp = (e) => {
    if (!drag) return;
    clear();
    controls.enabled = true;
    dom.style.cursor = '';
    try { dom.releasePointerCapture(e.pointerId); } catch (_) { /* not captured */ }
  };

  dom.addEventListener('pointerdown', onDown);
  dom.addEventListener('pointermove', onMove);
  dom.addEventListener('pointerup', onUp);
  dom.addEventListener('pointercancel', onUp);

  // world position of the grabbed body-local point, from the current solve
  const grabWorld = new THREE.Vector3();
  function currentGrab() {
    const bx = data.xpos, bm = data.xmat, b = 3 * drag.bodyId, r = 9 * drag.bodyId;
    const o = drag.localOffset;
    grabWorld.set(
      bx[b] + bm[r] * o[0] + bm[r + 1] * o[1] + bm[r + 2] * o[2],
      bx[b + 1] + bm[r + 3] * o[0] + bm[r + 4] * o[1] + bm[r + 5] * o[2],
      bx[b + 2] + bm[r + 6] * o[0] + bm[r + 7] * o[1] + bm[r + 8] * o[2],
    );
    return grabWorld;
  }

  const fvec = new THREE.Vector3(), rvec = new THREE.Vector3(), tvec = new THREE.Vector3();
  // Called before every physics step: refresh xfrc_applied for the grabbed body.
  function apply() {
    if (!drag) return;
    const cur = currentGrab();
    const mass = model.body_mass[drag.bodyId];
    // F = map.stiffness * mass * displacement, like MuJoCo's mouse spring, with
    // the displacement clamped and the force capped at a few body weights.
    fvec.copy(drag.target).sub(cur);
    if (fvec.length() > PERTURB_MAX_DISP) fvec.setLength(PERTURB_MAX_DISP);
    fvec.multiplyScalar(stiffness * mass * PERTURB_GAIN * bodyPerturbScale[drag.bodyId]);
    const maxF = PERTURB_MAX_WEIGHTS * mass * gnorm;
    if (fvec.length() > maxF) fvec.setLength(maxF);
    // torque about COM so the force acts at the grabbed point: tau = r x F
    const cx = data.xipos, b = drag.bodyId;
    rvec.set(cur.x - cx[3 * b], cur.y - cx[3 * b + 1], cur.z - cx[3 * b + 2]);
    tvec.crossVectors(rvec, fvec);
    const xf = data.xfrc_applied, o = 6 * b;
    xf[o] = fvec.x; xf[o + 1] = fvec.y; xf[o + 2] = fvec.z;
    xf[o + 3] = tvec.x; xf[o + 4] = tvec.y; xf[o + 5] = tvec.z;
  }

  function updateGizmo() {
    if (!drag) { gizmo.visible = false; return; }
    const cur = currentGrab();
    const a = gizmo.geometry.attributes.position;
    a.setXYZ(0, cur.x, cur.y, cur.z);
    a.setXYZ(1, drag.target.x, drag.target.y, drag.target.z);
    a.needsUpdate = true;
    gizmo.visible = true;
  }

  // Zero the wrench on the body we were dragging (and any stragglers).
  function clear() {
    if (drag) {
      const xf = data.xfrc_applied, o = 6 * drag.bodyId;
      for (let i = 0; i < 6; i++) xf[o + i] = 0;
    }
    drag = null;
    gizmo.visible = false;
  }

  return { apply, updateGizmo, clear };
}

// --- physics parameter sliders: global multipliers for stiffness / damping / kp ---
function buildPhysicsSliders(mj, model) {
  const container = document.getElementById('physics-params');
  const NGAIN = mj.mjNGAIN; // entries per actuator in gainprm / biasprm (10)

  // Snapshot original model values so the slider multiplies from the baseline.
  const nj = model.njnt, nd = model.nv, nu = model.nu;
  const origStiffness = new Float64Array(nj);
  const origDamping = new Float64Array(nd);
  const origGainprm = new Float64Array(nu);  // gainprm[0] = kp
  const origBiasprm = new Float64Array(nu);  // biasprm[1] = -kp (affine position bias)
  for (let j = 0; j < nj; j++) origStiffness[j] = model.jnt_stiffness[j];
  for (let d = 0; d < nd; d++) origDamping[d] = model.dof_damping[d];
  for (let u = 0; u < nu; u++) {
    origGainprm[u] = model.actuator_gainprm[u * NGAIN];
    origBiasprm[u] = model.actuator_biasprm[u * NGAIN + 1];
  }

  const allInputs = [];
  const makeSlider = (label, onChange) => {
    const row = document.createElement('div'); row.className = 'param-row';
    const header = document.createElement('div'); header.className = 'param-header';
    const lbl = document.createElement('span'); lbl.textContent = label;
    const valEl = document.createElement('span'); valEl.className = 'val'; valEl.textContent = '1.00×';
    header.append(lbl, valEl);
    const input = document.createElement('input');
    input.type = 'range'; input.min = 0; input.max = 5; input.step = 0.01; input.value = 1;
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      valEl.textContent = v.toFixed(2) + '×';
      onChange(v);
    });
    row.append(header, input);
    container.appendChild(row);
    allInputs.push(input);
  };

  makeSlider('Joint stiffness', (mult) => {
    for (let j = 0; j < nj; j++) model.jnt_stiffness[j] = origStiffness[j] * mult;
  });

  makeSlider('Joint damping', (mult) => {
    for (let d = 0; d < nd; d++) model.dof_damping[d] = origDamping[d] * mult;
  });

  makeSlider('Actuator kp', (mult) => {
    for (let u = 0; u < nu; u++) {
      model.actuator_gainprm[u * NGAIN] = origGainprm[u] * mult;
      model.actuator_biasprm[u * NGAIN + 1] = origBiasprm[u] * mult;
    }
  });

  return {
    reset() {
      for (const inp of allInputs) {
        inp.value = 1;
        inp.dispatchEvent(new Event('input'));
      }
    },
  };
}

// --- control panel: one slider (+ live joint bar) per position actuator -----
function buildSliders(meta, data) {
  const container = document.getElementById('groups');
  const byGroup = new Map(meta.groups.map((g) => [g.key, []]));
  for (const a of meta.actuators) (byGroup.get(a.group) || byGroup.set(a.group, []).get(a.group)).push(a);

  const SWATCH = {
    lf_leg: '#7e57c2', lm_leg: '#5c6bc0', lh_leg: '#42a5f5',
    rf_leg: '#ef5350', rm_leg: '#ff7043', rh_leg: '#ffa726', other: '#9aa0a6',
  };
  const rows = []; // { a, input, valEl, mark }
  const deg = (rad) => `${(rad * 180 / Math.PI).toFixed(0)}°`;

  for (const gr of meta.groups) {
    const acts = byGroup.get(gr.key) || [];
    if (!acts.length) continue;
    const details = document.createElement('details');
    details.open = gr.key === 'lf_leg';
    const summary = document.createElement('summary');
    const sw = document.createElement('span'); sw.className = 'swatch';
    sw.style.background = SWATCH[gr.key] || '#9aa0a6';
    summary.append(sw, document.createTextNode(`${gr.label} (${acts.length})`));
    details.appendChild(summary);

    for (const a of acts) {
      const [lo, hi] = a.ctrlrange;
      const wrap = document.createElement('div'); wrap.className = 'joint';
      const row = document.createElement('div'); row.className = 'row';
      const name = document.createElement('span'); name.textContent = a.dofLabel;
      const valEl = document.createElement('span'); valEl.className = 'val';
      row.append(name, valEl);

      const sld = document.createElement('div'); sld.className = 'sld';
      const tick = document.createElement('div'); tick.className = 'tick';
      tick.title = 'current joint angle';
      const input = document.createElement('input');
      input.type = 'range'; input.min = lo; input.max = hi; input.step = 0.001;
      input.value = a.neutral;
      input.addEventListener('input', () => {
        const v = parseFloat(input.value);
        data.ctrl[a.id] = v;
        valEl.textContent = deg(v);
      });
      valEl.textContent = deg(a.neutral);
      sld.append(tick, input);
      wrap.append(row, sld);
      details.appendChild(wrap);
      rows.push({ a, input, valEl, mark: tick, lo, hi });
    }
    container.appendChild(details);
  }

  // Per-frame: move each green tick to the actuated joint's current qpos.
  function tick() {
    for (const r of rows) {
      const q = data.qpos[r.a.qposadr];
      const t = Math.max(0, Math.min(1, (q - r.lo) / (r.hi - r.lo)));
      r.mark.style.left = `${t * 100}%`;
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  // Reset sliders + ctrl to the neutral keyframe.
  function reset() {
    for (const r of rows) {
      r.input.value = r.a.neutral;
      r.valEl.textContent = deg(r.a.neutral);
      data.ctrl[r.a.id] = r.a.neutral;
    }
  }

  return { reset };
}
