// Shared MuJoCo-WASM + Three.js plumbing used by both the interactive viewer
// (../viewer/) and the game (../game/).
//
// MuJoCo (compiled to WebAssembly by Google DeepMind, vendored under
// ./vendor/mujoco/) is loaded here; the helpers below write a flattened MJCF and
// its meshes into MuJoCo's virtual filesystem, compile the model, and turn each
// renderable MuJoCo geom into a Three.js mesh whose transform is synced from the
// solved state every frame. Three.js is vendored under ./vendor/three/ and
// reached through the page's import map ("three").

import * as THREE from 'three';
import loadMujoco from './vendor/mujoco/mujoco.js';
export { loadMujoco };

// The panel defaults to dark; when embedded in the MkDocs Material docs (same
// origin) mirror the site's light/dark palette and follow its toggle live.
export function setupTheme() {
  const root = document.documentElement;
  const apply = (dark) => root.setAttribute('data-theme', dark ? 'dark' : 'light');
  apply(true);
  try {
    const pbody = window.parent !== window ? window.parent.document.body : null;
    if (!pbody) return;
    const sync = () => apply(pbody.getAttribute('data-md-color-scheme') === 'slate');
    sync();
    new MutationObserver(sync).observe(pbody,
      { attributes: true, attributeFilter: ['data-md-color-scheme'] });
  } catch (_) { /* cross-origin / standalone: keep the dark default */ }
}

export function loadModel(mj, path) {
  if (typeof mj.mj_loadXML === 'function') return mj.mj_loadXML(path);
  if (mj.MjModel && typeof mj.MjModel.from_xml_path === 'function')
    return mj.MjModel.from_xml_path(path);
  throw new Error('no XML model-loading entry point found in the MuJoCo build');
}

// Write the MJCF and every mesh it references into MuJoCo's virtual filesystem,
// then return the in-FS path of the model. `modelBaseUrl` is where the meshes
// (and the XML, when fetched by the caller) are served from; `onProgress` is an
// optional (stage, n) callback for a loading overlay.
export async function writeModelToFS(mj, { xmlText, xmlName = 'model.xml', modelBaseUrl, onProgress }) {
  const meshFiles = [...new Set(
    [...xmlText.matchAll(/<mesh[^>]*\bfile="([^"]+)"/g)].map((m) => m[1]))];
  try { mj.FS.mkdir('/work'); } catch (_) { /* already exists */ }
  mj.FS.writeFile(`/work/${xmlName}`, xmlText);
  onProgress?.('meshes', meshFiles.length);
  await Promise.all(meshFiles.map(async (f) => {
    const buf = new Uint8Array(await fetch(`${modelBaseUrl}/${f}`).then((r) => r.arrayBuffer()));
    mj.FS.writeFile(`/work/${f}`, buf);
  }));
  return `/work/${xmlName}`;
}

// One-call load sequence shared by both apps: apply the docs theme, start the
// MuJoCo runtime, fetch the flattened MJCF + its meta, write them into MuJoCo's
// FS, compile, and return a reset + forwarded MjData. `onStage(text)` (optional)
// drives a loading overlay.
export async function loadScene({ assetsDir, xmlName = 'fly.xml', onStage }) {
  setupTheme();
  onStage?.('Loading MuJoCo (WebAssembly)…');
  const mj = await loadMujoco();

  onStage?.('Fetching the fly model…');
  const [xmlText, meta] = await Promise.all([
    fetch(`${assetsDir}/model/${xmlName}`).then((r) => r.text()),
    fetch(`${assetsDir}/model_meta.json`).then((r) => r.json()),
  ]);

  const xmlPath = await writeModelToFS(mj, {
    xmlText, xmlName, modelBaseUrl: `${assetsDir}/model`,
    onProgress: (_stage, n) => onStage?.(`Loading ${n} meshes…`),
  });

  onStage?.('Compiling the model…');
  const model = loadModel(mj, xmlPath);
  const data = new mj.MjData(model);
  mj.mj_resetDataKeyframe(model, data, 0);
  mj.mj_forward(model, data);
  return { mj, model, data, meta };
}

// --- mesh building (one Three.js mesh per renderable MuJoCo geom) -----------
// `meta.geom_rgba` (built by the asset script) gives a representative RGBA per
// geom. The returned group carries `userData.items` (mesh + geom id + body id +
// base opacity) so callers can sync transforms and recolor.
export function buildMeshes(model, meta) {
  const group = new THREE.Group();
  const items = [];
  const rgbaList = meta.geom_rgba || [];
  for (let g = 0; g < model.ngeom; g++) {
    const geometry = geometryForGeom(model, g, model.geom_type[g]);
    if (!geometry) continue; // planes/hfields handled separately
    const rgba = rgbaList[g] || [0.7, 0.7, 0.7, 1.0];
    const baseOpacity = rgba.length > 3 ? rgba[3] : 1.0;
    const color = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
    const material = new THREE.MeshStandardMaterial({
      color, roughness: 0.75, metalness: 0.0, side: THREE.DoubleSide,
      transparent: baseOpacity < 1, opacity: baseOpacity, depthWrite: baseOpacity >= 1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    mesh.userData = { kind: 'mesh', g, name: model.geom(g).name, bodyId: model.geom_bodyid[g] };
    group.add(mesh);
    items.push({ mesh, g, bodyId: model.geom_bodyid[g], baseOpacity });
  }
  group.userData.items = items;
  return group;
}

export function geometryForGeom(model, g, type) {
  // mjGEOM_PLANE=0, HFIELD=1, SPHERE=2, CAPSULE=3, ELLIPSOID=4, CYLINDER=5,
  // BOX=6, MESH=7.
  const s = (k) => model.geom_size[g * 3 + k];
  if (model.geom_dataid[g] >= 0) return meshGeometry(model, model.geom_dataid[g]);
  switch (type) {
    case 2: return new THREE.SphereGeometry(s(0), 16, 12);
    case 3: { const geo = new THREE.CapsuleGeometry(s(0), 2 * s(1), 6, 12); geo.rotateX(Math.PI / 2); return geo; }
    case 4: { const geo = new THREE.SphereGeometry(1, 16, 12); geo.scale(s(0), s(1), s(2)); return geo; }
    case 5: { const geo = new THREE.CylinderGeometry(s(0), s(0), 2 * s(1), 16); geo.rotateX(Math.PI / 2); return geo; }
    case 6: return new THREE.BoxGeometry(2 * s(0), 2 * s(1), 2 * s(2));
    default: return null; // plane / hfield
  }
}

export function meshGeometry(model, dataid) {
  const va = model.mesh_vertadr[dataid], vn = model.mesh_vertnum[dataid];
  const fa = model.mesh_faceadr[dataid], fn = model.mesh_facenum[dataid];
  const allVerts = model.mesh_vert, allFaces = model.mesh_face;
  const verts = new Float32Array(vn * 3);
  for (let i = 0; i < vn * 3; i++) verts[i] = allVerts[va * 3 + i];
  const index = new Uint32Array(fn * 3);
  for (let i = 0; i < fn * 3; i++) index[i] = allFaces[fa * 3 + i];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  geo.computeVertexNormals();
  return geo;
}

// Read the solved geom frames (data.geom_xpos / geom_xmat) into the Three.js
// meshes built by buildMeshes(). Meshes use manual matrices (matrixAutoUpdate
// off), so we write the world transform straight in.
const _m4 = new THREE.Matrix4();
export function syncMeshes(meshGroup, data) {
  const gx = data.geom_xpos, gm = data.geom_xmat;
  for (const { mesh, g } of meshGroup.userData.items) {
    _m4.set(gm[9 * g + 0], gm[9 * g + 1], gm[9 * g + 2], gx[3 * g + 0],
            gm[9 * g + 3], gm[9 * g + 4], gm[9 * g + 5], gx[3 * g + 1],
            gm[9 * g + 6], gm[9 * g + 7], gm[9 * g + 8], gx[3 * g + 2],
            0, 0, 0, 1);
    mesh.matrix.copy(_m4);
  }
}

// A tiny grow-on-demand object pool: begin() rewinds, next() hands out (and
// creates) the next object, end() hides any left over from a previous frame.
export function pool(make) {
  const items = []; let cur = 0;
  return {
    begin() { cur = 0; },
    next() { const o = items[cur] || (items[cur] = make()); cur++; return o; },
    end() { for (let i = cur; i < items.length; i++) items[i].visible = false; },
  };
}

// --- app loop helpers shared by the viewer and the game ---------------------

// Show a load error in the page's #overlay. `what` names the app ("game" /
// "viewer"); `tag` matches the page's .err markup (the game styles `#overlay p`,
// the viewer uses a div). Returns the `fail(msg, err)` handler.
export function makeFailOverlay(overlayEl, what, tag = 'div') {
  return (msg, err) => {
    console.error(err || msg);
    overlayEl.classList.remove('hidden');
    overlayEl.innerHTML =
      `<${tag} class="err"><strong>Could not load the ${what}.</strong>` +
      `<br>${msg}${err ? `<br><br><code>${String(err)}</code>` : ''}</${tag}>`;
  };
}

// Fixed-timestep accumulator with a backlog cap. Call advance() once per
// animation frame with the elapsed wall time (already scaled by any desired
// playback speed); it runs `step(i)` up to `maxSubsteps` times -- dropping any
// further backlog so a stall can't spiral -- and returns the number of steps
// taken. `step` may return false to stop early (e.g. on a finish-line crossing).
export function makeStepper(dt, maxSubsteps) {
  let acc = 0;
  return {
    advance(wallDt, step) {
      acc += wallDt;
      const want = Math.floor(acc / dt);
      const nSteps = Math.min(want, maxSubsteps);
      acc -= nSteps * dt;
      if (want > maxSubsteps) acc = 0; // fell behind: drop the backlog
      for (let i = 0; i < nSteps; i++) if (step(i) === false) return i + 1;
      return nSteps;
    },
  };
}

// Windowed fps / real-time-factor meter. Feed it the current wall-clock time (in
// seconds) and the steps taken since the last call; about every `windowSec` it
// calls report({ fps, rtf }) and starts a fresh window. Formatting is left to
// the caller (the two apps render the numbers differently).
export function makeStatsMeter(dt, report, { windowSec = 0.5 } = {}) {
  let frames = 0, steps = 0, t0;
  return (nowSec, nSteps) => {
    frames++; steps += nSteps;
    if (t0 === undefined) t0 = nowSec;
    const secs = nowSec - t0;
    if (secs > windowSec) {
      report({ fps: frames / secs, rtf: (steps * dt) / secs });
      frames = 0; steps = 0; t0 = nowSec;
    }
  };
}
