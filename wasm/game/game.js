// The NeuroMechFly Live game, in the browser. Pilot the fly through a slalom
// track to the finish line at three levels of neural abstraction:
//
//   Level 1 "CPG"    -- steer (W/A/S/D); coupled CPG oscillators coordinate all
//                       six legs automatically. Easy.
//   Level 2 "Tripod" -- drive the two tripod groups (G/H forward, F/J back).
//   Level 3 "Legs"   -- drive each of the six legs (T G B Z H N / R F V U J M).
//
// The physics is real: MuJoCo (compiled to WebAssembly, shared under
// ../shared/) runs the same legs-only position-actuated NeuroMechFly model as
// the desktop game, at dt=1e-4 with leg adhesion -- so, like the desktop game,
// it plays at well below real time (the achieved factor is shown top-right).
// The control logic ported here (CPGNetwork + PreprogrammedSteps + the three
// controllers, from neuromechfly-live / flygym) is fed the *baked* tables in
// model_meta.json, so the browser needs no SciPy. Three.js renders the solved
// state; the camera chases the fly; the finish is a geometric path/line crossing.

import * as THREE from 'three';
import {
  loadScene, makeFailOverlay, makeStepper, makeStatsMeter, buildMeshes, syncMeshes,
} from '../shared/scene.js';

const ASSETS = './assets';
const TAU = Math.PI * 2;
const overlayEl = document.getElementById('overlay');
const overlayMsg = document.getElementById('overlay-msg');

// Target playback speed: sim seconds advanced per wall second. The desktop game
// plays well below real time; we pin it at a fixed slow-motion factor (rather
// than "as fast as the machine allows") so the fly is controllable and the pace
// is the same on every machine.
const PLAYBACK_SPEED = 0.1;
// Safety cap on physics steps per animation frame. At PLAYBACK_SPEED=0.1 and
// ~60 fps a frame only needs ~17 steps, well under this; the cap just prevents a
// long stall from spiralling.
const MAX_SUBSTEPS = 60;

const LEVELS = {
  CPG:    { name: 'CPG control',        chip: '#8ace00' },
  tripod: { name: 'Tripod gait',        chip: '#4d66ff' },
  single: { name: 'Individual legs',    chip: '#ff5a5a' },
};
const HELP = {
  CPG: '<kbd>W</kbd> forward · <kbd>S</kbd> back · <kbd>A</kbd>/<kbd>D</kbd> turn · <kbd>Q</kbd> stop' +
       '<div class="row2">Switch level: <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> · Restart: <kbd>Space</kbd></div>',
  tripod: '<kbd>G</kbd>/<kbd>H</kbd> step left/right tripod forward · <kbd>F</kbd>/<kbd>J</kbd> backward' +
       '<div class="row2">Each tripod is 3 alternating legs. Switch: <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> · Restart: <kbd>Space</kbd></div>',
  single: 'Forward <kbd>T</kbd><kbd>G</kbd><kbd>B</kbd> <kbd>Z</kbd><kbd>H</kbd><kbd>N</kbd> · ' +
       'Back <kbd>R</kbd><kbd>F</kbd><kbd>V</kbd> <kbd>U</kbd><kbd>J</kbd><kbd>M</kbd>' +
       '<div class="row2">Six legs, one key each (L/R front·mid·hind). Switch: <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> · Restart: <kbd>Space</kbd></div>',
};
// Joystick hint appended to the help line while a gamepad is connected. Mirrors
// the desktop game's joystick layout (see the Gamepad class / controls.py).
const HELP_PAD = {
  CPG: '🎮 Push the stick to walk · left/right to steer',
  tripod: '🎮 Left & right tripod buttons step each tripod (forward / reverse rows)',
  single: '🎮 One button per leg — forward and reverse button rows',
};

const fail = makeFailOverlay(overlayEl, 'game', 'p');

main().catch((e) => fail('Unexpected error while starting up.', e));

async function main() {
  const { mj, model, data, meta } = await loadScene({
    assetsDir: ASSETS, xmlName: 'fly.xml',
    onStage: (msg) => { overlayMsg.textContent = msg; },
  });
  new Game(mj, model, data, meta).start();
}

// --- the ported controllers -------------------------------------------------
// One object integrates the CPG (level 1) and the per-leg / per-tripod step
// state machines (levels 2-3), then scatters the resulting joint angles +
// adhesion flags into MuJoCo's `data.ctrl` via the baked (leg, dof) map.
class Controller {
  constructor(meta) {
    this.dt = meta.timestep;
    this.legs = meta.control.leg_order;            // 6 leg names
    this.cmap = meta.ctrl_index_by_leg_dof;        // [6][7] -> ctrl index
    this.adh = meta.adhesion;                       // [6] -> adhesion ctrl index
    this.tripodMap = meta.control.tripod_map;       // [6] -> 0/1
    const cpg = meta.control.cpg;
    this.freqs0 = cpg.intrinsic_freqs.slice();      // base |freq| per leg
    this.W = cpg.coupling_weights;                  // [6][6]
    this.PB = cpg.phase_biases;                     // [6][6]
    this.conv = cpg.convergence_coefs;              // [6]
    this.phaseInc = (this.dt / meta.control.leg_step_time) * TAU;

    const pp = meta.preprogrammed;
    this.N = pp.n_samples;
    this.tab = this.legs.map((l) => pp.legs[l]);    // {angles:[N][7], neutral:[7], swing:[2]}

    // scratch
    this._cpgAmps = new Float64Array(6);
    this._cpgFreqs = new Float64Array(6);
    this._d6 = new Float64Array(6);
    this._a7 = new Float64Array(7);
    this.reset();
  }

  reset() {
    this.phases = new Float64Array(6).map(() => Math.random() * TAU); // CPG phases
    this.mags = new Float64Array(6);                                  // CPG magnitudes
    this.legPhases = new Float64Array(6);                             // single-leg
    this.stepDir = new Float64Array(6);
    this.tripodPhases = new Float64Array(2);                          // tripod groups
    this.tripodDir = new Float64Array(2);
  }

  // Joint angles for one leg at a phase / magnitude, by periodic-lerp of the
  // baked table: angle = neutral + magnitude * (table(phase) - neutral).
  _anglesInto(li, phase, mag, out) {
    const t = this.tab[li], N = this.N;
    let x = ((phase % TAU) + TAU) % TAU / TAU * N;
    const i0 = Math.floor(x) % N, i1 = (i0 + 1) % N, f = x - Math.floor(x);
    const a0 = t.angles[i0], a1 = t.angles[i1], nu = t.neutral;
    for (let d = 0; d < 7; d++) {
      const samp = a0[d] * (1 - f) + a1[d] * f;
      out[d] = nu[d] + mag * (samp - nu[d]);
    }
  }

  _adhesionOn(li, phase) {
    const [s, e] = this.tab[li].swing;             // swing = adhesion OFF
    const p = ((phase % TAU) + TAU) % TAU;
    return !(p > s && p < e);
  }

  _writeLeg(ctrl, li, phase, mag) {
    this._anglesInto(li, phase, mag, this._a7);
    const row = this.cmap[li];
    for (let d = 0; d < 7; d++) ctrl[row[d]] = this._a7[d];
    ctrl[this.adh[li]] = this._adhesionOn(li, phase) ? 1 : 0;
  }

  // Level 1: descending signal action=[gainL,gainR] modulates CPG amplitude
  // (|action|) and stepping direction (sign), then one Euler integration step.
  stepCPG(ctrl, gainL, gainR) {
    const amps = this._cpgAmps, freqs = this._cpgFreqs;
    const aL = Math.abs(gainL), aR = Math.abs(gainR);
    amps[0] = amps[1] = amps[2] = aL; amps[3] = amps[4] = amps[5] = aR;
    const sL = gainL > 0 ? 1 : -1, sR = gainR > 0 ? 1 : -1;
    for (let i = 0; i < 6; i++) freqs[i] = this.freqs0[i] * (i < 3 ? sL : sR);

    // dtheta = 2pi*freq + sum_j mags_j*W_ij*sin(theta_j - theta_i - PB_ij);  dr = conv*(amp - r)
    const ph = this.phases, mg = this.mags, dt = this.dt;
    const dph = this._d6;
    for (let i = 0; i < 6; i++) {
      let coupling = 0;
      for (let j = 0; j < 6; j++)
        coupling += mg[j] * this.W[i][j] * Math.sin(ph[j] - ph[i] - this.PB[i][j]);
      dph[i] = TAU * freqs[i] + coupling;
    }
    for (let i = 0; i < 6; i++) {
      ph[i] += dph[i] * dt;
      mg[i] += this.conv[i] * (amps[i] - mg[i]) * dt;
    }
    for (let i = 0; i < 6; i++) this._writeLeg(ctrl, i, ph[i], mg[i]);
  }

  // Shared per-step state machine for the on-demand modes: a leg/tripod at rest
  // (phase<=0) starts a step when its action is non-zero, runs the cycle to
  // completion (forward to 2pi, or backward to 0), then returns to rest.
  _advance(phaseArr, dirArr, i, act) {
    if (phaseArr[i] >= TAU || (phaseArr[i] <= 0 && dirArr[i] < 0)) {
      phaseArr[i] = 0; dirArr[i] = 0;
    } else if (phaseArr[i] <= 0) {
      if (act > 0) { phaseArr[i] += this.phaseInc; dirArr[i] = 1; }
      else if (act < 0) { phaseArr[i] = TAU - this.phaseInc; dirArr[i] = -1; }
    } else {
      phaseArr[i] += this.phaseInc * dirArr[i];
    }
  }

  // Level 3: action is a 6-vector (one trigger per leg).
  stepSingle(ctrl, action) {
    for (let i = 0; i < 6; i++) {
      this._advance(this.legPhases, this.stepDir, i, action[i]);
      this._writeLeg(ctrl, i, this.legPhases[i], 1);
    }
  }

  // Level 2: action is a 2-vector (one trigger per tripod group).
  stepTripod(ctrl, action) {
    for (let g = 0; g < 2; g++) this._advance(this.tripodPhases, this.tripodDir, g, action[g]);
    for (let i = 0; i < 6; i++)
      this._writeLeg(ctrl, i, this.tripodPhases[this.tripodMap[i]], 1);
  }
}

// --- keyboard input ---------------------------------------------------------
// CPG gains persist (set on key-down, like the desktop's prev_gain); the
// on-demand modes read the currently-held keys each substep.
class Input {
  constructor(onLevel, onRestart, onMove) {
    this.held = new Set();
    this.gainL = 0; this.gainR = 0;
    const MOVE = 'wsadqtgbzhnrfvujm';
    // Level 1 (CPG) arrow-key aliases for WASD.
    const ARROW = { arrowup: 'w', arrowdown: 's', arrowleft: 'a', arrowright: 'd' };
    addEventListener('keydown', (e) => {
      const k = ARROW[e.key.toLowerCase()] || e.key.toLowerCase();
      if (e.repeat) { e.preventDefault(); return; }
      if (k === '1' || k === 'i') return onLevel('CPG');
      if (k === '2' || k === 'o') return onLevel('tripod');
      if (k === '3' || k === 'p') return onLevel('single');
      // Restart is Space only: in Level 3 'r' is the left-front leg's "backward"
      // key (see singleAction), so it must fall through to `held` below.
      if (k === ' ') { e.preventDefault(); return onRestart(); }
      this.held.add(k);
      this._cpgKey(k);
      if (MOVE.includes(k)) { e.preventDefault(); onMove(); }
    });
    addEventListener('keyup', (e) => {
      const k = e.key.toLowerCase();
      this.held.delete(ARROW[k] || k);
    });
    addEventListener('blur', () => { this.held.clear(); });
  }

  _cpgKey(k) {
    const back = this.gainL < 0 || this.gainR < 0;
    if (k === 'w') { this.gainL = 1; this.gainR = 1; }
    else if (k === 's') { this.gainL = -1; this.gainR = -1; }
    else if (k === 'q') { this.gainL = 0; this.gainR = 0; }
    else if (k === 'a') { if (back) { this.gainR = -0.6; this.gainL = -1.2; } else { this.gainL = 0.4; this.gainR = 1.2; } }
    else if (k === 'd') { if (back) { this.gainL = -0.6; this.gainR = -1.2; } else { this.gainR = 0.4; this.gainL = 1.2; } }
  }

  resetGains() { this.gainL = 0; this.gainR = 0; }

  // Level 3: per-leg trigger from held keys (forward / backward sets).
  singleAction(out) {
    const F = 'tgbzhn', B = 'rfvujm';
    for (let i = 0; i < 6; i++)
      out[i] = this.held.has(F[i]) ? 1 : this.held.has(B[i]) ? -1 : 0;
    return out;
  }

  // Level 2: per-tripod trigger. Group 0 = G/F, group 1 = H/J.
  tripodAction(out) {
    out[0] = this.held.has('g') ? 1 : this.held.has('f') ? -1 : 0;
    out[1] = this.held.has('h') ? 1 : this.held.has('j') ? -1 : 0;
    return out;
  }
}

// --- gamepad / joystick input ----------------------------------------------
// Faithful port of the desktop game's JoystickControl (neuromechfly-live
// controls.py) onto the browser Gamepad API: the *same* raw button indices and
// the *same* CPG axis math, so the physical joystick used at outreach events
// behaves identically in the browser. Button indices are device-specific (they
// match the joystick the desktop game targets); tweak PAD if you use another.
const PAD = {
  // leg order LF, LM, LH, RF, RM, RH (== meta.control.leg_order). controls.py:
  //   joystick_buttons_order          = [10, 11, 12, 4, 5, 6]  -> step forward
  //   backward_joystick_buttons_order = [15, 14, 13, 9, 8, 7]  -> step backward
  fwdButtons:  [10, 11, 12, 4, 5, 6],
  backButtons: [15, 14, 13, 9, 8, 7],
  axisX: 0, axisY: 1,     // analog stick: X = turn, Y = forward/back (fwd = -1)
  deadzone: 0.15,         // ignore stick drift (desktop polled raw axes)
};

class Gamepad {
  constructor(onChange) {
    this.index = null;
    this._disconnected = false; // active pad was unplugged; don't auto-adopt another
    this.single = new Float64Array(6);
    this.tripod = new Float64Array(2);
    this._legs = new Float64Array(6);
    addEventListener('gamepadconnected', (e) => {
      this.index = e.gamepad.index; this._disconnected = false; onChange?.();
    });
    addEventListener('gamepaddisconnected', (e) => {
      if (this.index === e.gamepad.index) { this.index = null; this._disconnected = true; }
      onChange?.();
    });
  }

  get connected() { return this._pad() != null; }

  _pad() {
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    if (this.index != null && pads[this.index]) return pads[this.index];
    // Discover a pad already connected before page load (no 'gamepadconnected'
    // event). Skip once the active pad has been unplugged, so input doesn't
    // silently jump to a different controller the player isn't holding.
    if (!this._disconnected) {
      for (const p of pads) if (p) { this.index = p.index; return p; }
    }
    return null;
  }

  _axis(pad, i) {
    const v = pad.axes[i] || 0;
    return Math.abs(v) < PAD.deadzone ? 0 : v;
  }

  // controls.py retrieve_joystick_buttons: +1 forward / -1 backward per leg.
  _legPresses(pad) {
    const legs = this._legs; legs.fill(0);
    const down = (b) => pad.buttons[b] && pad.buttons[b].pressed;
    for (let j = 0; j < 6; j++) if (down(PAD.fwdButtons[j])) legs[j] = 1;
    for (let j = 0; j < 6; j++) if (down(PAD.backButtons[j])) legs[j] = -1;
    return legs;
  }

  // One snapshot for the active level. Returns null when no pad is connected;
  // otherwise { active, gainL, gainR } plus this.single / this.tripod filled in.
  // `active` (any input past the deadzone) is used to auto-start the countdown.
  sample(level) {
    const pad = this._pad();
    if (!pad) return null;
    this.single.fill(0); this.tripod.fill(0);
    let gainL = 0, gainR = 0, active = false;

    if (level === 'CPG') {
      // controls.py CPG branch: ||axis||/sqrt(2)*1.2 sets the forward/back
      // magnitude, axisY sign its direction (stick forward reads negative), and
      // |axisX|*0.6 subtracts from one side to turn (right: -=right, left: -=left).
      const ax = this._axis(pad, PAD.axisX), ay = this._axis(pad, PAD.axisY);
      const norm = Math.hypot(ax, ay) / Math.SQRT2 * 1.2;
      const sy = ay > 0 ? 1 : ay < 0 ? -1 : 0;
      gainL = gainR = norm * -1 * sy;
      const off = Math.abs(ax) * 0.6;
      if (ax > 0) gainR -= off; else if (ax < 0) gainL -= off;
      active = ax !== 0 || ay !== 0;
    } else {
      const legs = this._legPresses(pad);
      if (level === 'single') {                       // 6-vector, one per leg
        for (let i = 0; i < 6; i++) this.single[i] = legs[i];
      } else {                                        // tripod: LH btn -> left, RH btn -> right
        this.tripod[0] = legs[2];                     // group 0 = legs LF/LH/RM
        this.tripod[1] = legs[5];                     // group 1 = legs LM/RF/RH
      }
      for (let i = 0; i < 6; i++) if (legs[i]) { active = true; break; }
    }
    return { active, gainL, gainR };
  }
}

// A repeating greyscale checkerboard texture (white / mid-grey). Tinted by the
// ground material's per-level colour; mipmapped + anisotropic so it doesn't
// shimmer into the distance.
function makeCheckerTexture() {
  const N = 256, n = 8, s = N / n;       // 8x8 squares, 32 px each
  const c = document.createElement('canvas');
  c.width = c.height = N;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, N, N);
  ctx.fillStyle = '#9c9c9c';
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if ((i + j) & 1) ctx.fillRect(i * s, j * s, s, s);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

// --- the game ---------------------------------------------------------------
class Game {
  constructor(mj, model, data, meta) {
    this.mj = mj; this.model = model; this.data = data; this.meta = meta;
    this.dt = meta.timestep;
    this.level = 'CPG';
    this.controller = new Controller(meta);
    this.input = new Input(
      (lv) => this.setLevel(lv),
      () => this.restart(),
      () => { if (this.phase === 'ready') this._startCountdown(); });
    this.pad = new Gamepad(() => this._renderHelp());
    this._padState = null;

    this.finish = meta.arena.finish_line;          // [[x,y1],[x,y2]]
    this.phase = 'ready';                           // ready | countdown | running | finished
    this.simTime = 0;                               // sim seconds since GO
    this.prevXY = [meta.arena.spawn[0], meta.arena.spawn[1]];
    this.bodyId = this._findFlyRootBody();

    this._singleAct = new Float64Array(6);
    this._tripodAct = new Float64Array(2);
    this._stepper = makeStepper(this.dt, MAX_SUBSTEPS);
    this._statsMeter = makeStatsMeter(this.dt, ({ fps, rtf }) => {
      document.getElementById('stats').innerHTML =
        `${fps.toFixed(0)} fps · ${rtf.toFixed(2)}× realtime<br>${this.data.ncon} contacts`;
    });
    this._buildScene();
    this._wireUi();
  }

  _findFlyRootBody() {
    const m = this.model;
    for (let j = 0; j < m.njnt; j++) if (m.jnt_type[j] === 0) return m.jnt_bodyid[j]; // free joint
    return 1;
  }

  start() {
    this._resetSim();
    overlayEl.classList.add('hidden');
    this._showReady();
    requestAnimationFrame((t) => this._frame(t));
  }

  // --- scene ---------------------------------------------------------------
  _buildScene() {
    const stage = document.getElementById('stage');
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);            // MuJoCo is z-up
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    stage.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0x14161b, 45, 140);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
    this.camera.up.set(0, 0, 1);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(8, -10, 16);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4); fill.position.set(-8, 6, 6);
    this.scene.add(key, fill);

    this._buildGround();
    this._buildFinish();
    this.meshGroup = buildMeshes(this.model, this.meta);
    this.scene.add(this.meshGroup);

    addEventListener('resize', () => this._resize());
    this._resize();
  }

  _buildGround() {
    // A big checkerboard plane (the model's own ground plane isn't rendered),
    // tinted per level: the greyscale checker texture is multiplied by the
    // material colour, so the squares come out as two shades of the level colour.
    const geo = new THREE.PlaneGeometry(400, 400);
    const checker = makeCheckerTexture();
    checker.repeat.set(12, 12);          // ~4 mm squares across the 400 mm plane
    this.groundMat = new THREE.MeshStandardMaterial({ map: checker, color: 0x8ace00, roughness: 0.96 });
    const ground = new THREE.Mesh(geo, this.groundMat);
    ground.position.z = -0.02;
    this.scene.add(ground);
    this._applyLevelColor();
  }

  _applyLevelColor() {
    const c = this.meta.arena.level_ground_colors[this.level] || [0.5, 0.5, 0.5, 1];
    this.groundMat.color.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
  }

  _buildFinish() {
    // A translucent white banner spanning the finish line, plus a bright strip
    // on the ground, so the goal reads clearly from the chase camera. Built from
    // thin boxes (no fiddly plane rotations): the line runs along y at fixed x.
    const [[x, y1], [, y2]] = this.finish;
    const w = Math.abs(y2 - y1) + 0.2, yc = (y1 + y2) / 2, h = 5;
    const banner = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, w, h),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.16 }));
    banner.position.set(x, yc, h / 2);
    const strip = new THREE.Mesh(
      new THREE.BoxGeometry(0.8, w, 0.02),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.65 }));
    strip.position.set(x, yc, 0.012);
    this.scene.add(banner, strip);
  }

  _resize() {
    const w = innerWidth, h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  // --- camera follow (ported from Game.update_camera_to_follow_fly) --------
  _updateCamera(init) {
    const { height, distance } = this.meta.camera;
    // The desktop applies its 0.995 yaw smoothing per *physics step* (thousands
    // per second); we update once per animation frame, so use a per-frame factor
    // that chases the fly's heading smoothly without lagging.
    const smoothing = 0.85;
    const d = this.data, b = this.bodyId;
    const fx = d.xpos[3 * b], fy = d.xpos[3 * b + 1];
    const xm = d.xmat;                       // body x-axis (forward) in world
    const newYaw = Math.atan2(xm[9 * b + 3], xm[9 * b + 0]);
    if (init || this._yaw === undefined) this._yaw = newYaw;
    this._yaw = Math.atan2(
      smoothing * Math.sin(this._yaw) + (1 - smoothing) * Math.sin(newYaw),
      smoothing * Math.cos(this._yaw) + (1 - smoothing) * Math.cos(newYaw));
    const cy = Math.cos(this._yaw), sy = Math.sin(this._yaw);
    this.camera.position.set(fx - cy * distance, fy - sy * distance, height);
    this.camera.lookAt(fx + cy * 1.5, fy + sy * 1.5, 0.4);
  }

  // --- game state ----------------------------------------------------------
  _resetSim() {
    this.mj.mj_resetDataKeyframe(this.model, this.data, 0);
    this.controller.reset();
    this.input.resetGains();
    this.mj.mj_forward(this.model, this.data);
    this.simTime = 0;
    document.getElementById('timer').textContent = '0.00';
    const b = this.bodyId;
    this.prevXY = [this.data.xpos[3 * b], this.data.xpos[3 * b + 1]];
    this._updateCamera(true);
  }

  setLevel(level) {
    if (!LEVELS[level]) return;
    this.level = level;
    document.querySelectorAll('#levels button').forEach((bt) =>
      bt.classList.toggle('active', bt.dataset.level === level));
    document.getElementById('level-name').textContent = LEVELS[level].name;
    document.querySelector('#level .chip').style.background = LEVELS[level].chip;
    this._renderHelp();
    this._applyLevelColor();
    this._renderBest();
    this.restart();
  }

  restart() {
    this._resetSim();
    this._showReady();
  }

  _showReady() {
    this.phase = 'ready';
    overlayEl.classList.remove('hidden');
    overlayEl.innerHTML =
      `<h1>${LEVELS[this.level].name}</h1>` +
      `<p>${this._levelBlurb()}</p>` +
      `<button id="go">Start ▶</button>` +
      `<p style="font-size:12px">or press any movement key</p>`;
    document.getElementById('go').onclick = () => this._startCountdown();
  }

  _levelBlurb() {
    return {
      CPG: 'Steer with W/A/S/D. Central pattern generators coordinate all six legs for you — just point the fly through the gates to the white finish line.',
      tripod: 'Drive the two tripod groups: G/H step the left/right tripod forward, F/J backward. Alternate them to walk.',
      single: 'Drive each of the six legs individually (T G B / Z H N forward). Coordinating all six is hard — that\'s the point!',
    }[this.level];
  }

  _startCountdown() {
    this._resetSim();
    this.phase = 'countdown';
    let n = 3;
    const tick = () => {
      if (this.phase !== 'countdown') return;
      overlayEl.classList.remove('hidden');
      overlayEl.innerHTML = `<div class="big">${n > 0 ? n : 'GO!'}</div>`;
      if (n < 0) { overlayEl.classList.add('hidden'); this.phase = 'running'; this.simTime = 0; return; }
      n--; setTimeout(tick, n < 0 ? 350 : 700);
    };
    tick();
  }

  // The displayed "control time" is how long the player has been steering in
  // wall-clock terms: sim time runs at PLAYBACK_SPEED, so control = sim / speed.
  _controlTime() { return this.simTime / PLAYBACK_SPEED; }

  _finishRun() {
    this.phase = 'finished';
    const t = this._controlTime();
    const { board, youIndex } = this._updateBoard(t);
    const rows = board.map((e, i) =>
      `<div class="row ${i === youIndex ? 'you' : ''}"><span>${i + 1}.</span>` +
      `<span>${e.t.toFixed(2)} s</span></div>`).join('');
    overlayEl.classList.remove('hidden');
    overlayEl.innerHTML =
      `<h1>Finished! 🏁</h1>` +
      `<p>${LEVELS[this.level].name} — your time</p>` +
      `<div class="big" style="font-size:54px">${t.toFixed(2)} s</div>` +
      `<div class="lb"><div class="row" style="color:var(--muted)"><span>Best times</span><span></span></div>${rows}</div>` +
      `<button id="again">Play again ▶</button>`;
    document.getElementById('again').onclick = () => this._startCountdown();
    this._renderBest();
  }

  // --- leaderboard (localStorage, per level, best 5) -----------------------
  _boardKey() { return `nmf-game-ctrltime-${this.level}`; }
  _getBoard() { try { return JSON.parse(localStorage.getItem(this._boardKey())) || []; } catch { return []; } }
  _updateBoard(t) {
    const board = this._getBoard();
    const entry = { t };
    board.push(entry);
    board.sort((a, b) => a.t - b.t);
    const top = board.slice(0, 5);
    try { localStorage.setItem(this._boardKey(), JSON.stringify(top.map((e) => ({ t: e.t })))); } catch { /* ignore */ }
    return { board: top, youIndex: top.indexOf(entry) };
  }
  _renderBest() {
    const board = this._getBoard();
    document.getElementById('best').textContent =
      board.length ? `Best: ${board[0].t.toFixed(2)} s` : 'Best: —';
  }

  // Help line for the current level, with the joystick hint appended whenever a
  // gamepad is connected (re-run on connect/disconnect via the Gamepad callback).
  _renderHelp() {
    const pad = this.pad && this.pad.connected ? `<div class="row2 pad">${HELP_PAD[this.level]}</div>` : '';
    document.getElementById('help').innerHTML = HELP[this.level] + pad;
  }

  _wireUi() {
    document.querySelectorAll('#levels button').forEach((bt) =>
      bt.addEventListener('click', () => this.setLevel(bt.dataset.level)));
    this.setLevel('CPG');
  }

  // --- finish detection: does the path prev->cur cross the finish segment? --
  _crossed(cx, cy) {
    const ccw = (ax, ay, bx, by, c0, c1) => (c1 - ay) * (bx - ax) > (by - ay) * (c0 - ax);
    const [a, b] = this.finish, [px, py] = this.prevXY;
    return ccw(px, py, a[0], a[1], b[0], b[1]) !== ccw(cx, cy, a[0], a[1], b[0], b[1]) &&
           ccw(px, py, cx, cy, a[0], a[1]) !== ccw(px, py, cx, cy, b[0], b[1]);
  }

  // --- one physics substep: write ctrl from the active controller, then step -
  _physicsStep() {
    const ctrl = this.data.ctrl;
    const pad = this._padState;                       // sampled once this frame
    if (this.level === 'CPG') {
      // Stick (when engaged) overrides the persistent keyboard gains.
      let gL = this.input.gainL, gR = this.input.gainR;
      if (pad && pad.active) { gL = pad.gainL; gR = pad.gainR; }
      this.controller.stepCPG(ctrl, gL, gR);
    } else if (this.level === 'tripod') {
      const a = this.input.tripodAction(this._tripodAct);
      if (pad) for (let g = 0; g < 2; g++) if (this.pad.tripod[g]) a[g] = this.pad.tripod[g];
      this.controller.stepTripod(ctrl, a);
    } else {
      const a = this.input.singleAction(this._singleAct);
      if (pad) for (let i = 0; i < 6; i++) if (this.pad.single[i]) a[i] = this.pad.single[i];
      this.controller.stepSingle(ctrl, a);
    }
    this.mj.mj_step(this.model, this.data);
    this.simTime += this.dt;
  }

  // --- main loop -----------------------------------------------------------
  _frame(nowMs) {
    requestAnimationFrame((t) => this._frame(t));
    const now = nowMs / 1000;
    const wallDt = this._lastWall === undefined ? 0 : Math.min(now - this._lastWall, 0.1);
    this._lastWall = now;

    // Poll the joystick once per frame; engaging it on the ready screen starts
    // the run, just like pressing a movement key.
    this._padState = this.pad.sample(this.level);
    if (this.phase === 'ready' && this._padState && this._padState.active) this._startCountdown();

    let nSteps = 0;
    if (this.phase === 'running') {
      const b = this.bodyId;
      // wallDt is scaled by PLAYBACK_SPEED so the sim advances in slow motion.
      nSteps = this._stepper.advance(wallDt * PLAYBACK_SPEED, () => {
        this._physicsStep();
        const cx = this.data.xpos[3 * b], cy = this.data.xpos[3 * b + 1];
        if (this._crossed(cx, cy)) { this.prevXY = [cx, cy]; this._finishRun(); return false; }
        this.prevXY = [cx, cy];
      });
    }

    syncMeshes(this.meshGroup, this.data);
    this._updateCamera(false);
    this.renderer.render(this.scene, this.camera);

    if (this.phase === 'running') document.getElementById('timer').textContent = this._controlTime().toFixed(2);
    this._statsMeter(now, nSteps);
  }
}
