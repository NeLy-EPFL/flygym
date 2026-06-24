# Browser (WebAssembly) NeuroMechFly game

> [!NOTE]
> The standalone [NeuroMechFly Live](https://github.com/NeLy-EPFL/neuromechfly-live) repository is deprecated. This WebAssembly version is now the canonical implementation, natively included in FlyGym.

The [NeuroMechFly Live](https://github.com/NeLy-EPFL/neuromechfly-live) game, in
the browser. Pilot the fly through a slalom track to the finish line at three
levels of neural abstraction. Embedded in the docs at `docs/outreach.md`; also
opens standalone. Shares the MuJoCo-WASM + Three.js plumbing in
[`../shared/`](../shared/) with the viewer; see [`../README.md`](../README.md).

## Levels & controls

- **Level 1 — CPG** (`1`/`I`): steer with <kbd>W</kbd>/<kbd>A</kbd>/<kbd>S</kbd>/<kbd>D</kbd>
  (<kbd>Q</kbd> stop). Coupled CPG oscillators coordinate all six legs.
- **Level 2 — Tripod** (`2`/`O`): <kbd>G</kbd>/<kbd>H</kbd> step the left/right
  tripod forward, <kbd>F</kbd>/<kbd>J</kbd> backward.
- **Level 3 — Individual legs** (`3`/`P`): one key per leg — forward
  <kbd>T</kbd><kbd>G</kbd><kbd>B</kbd> <kbd>Z</kbd><kbd>H</kbd><kbd>N</kbd>,
  backward <kbd>R</kbd><kbd>F</kbd><kbd>V</kbd> <kbd>U</kbd><kbd>J</kbd><kbd>M</kbd>.

<kbd>Space</kbd> restarts (in Level 3 <kbd>R</kbd> is a leg control, so restart is
Space only). Best times per level are kept in `localStorage`.

### Joystick / gamepad

A connected gamepad is auto-detected (via the browser Gamepad API) and uses the
**same joystick model as the desktop game** (`neuromechfly-live/controls.py`):

- **Level 1 — CPG**: the analog stick drives the descending signal. Stick
  magnitude (`‖axis‖/√2 · 1.2`) sets the forward/back gain, stick Y its sign
  (forward = negative), and stick X turns by subtracting `|axisX|·0.6` from the
  inside leg's gain.
- **Levels 2–3**: the six leg buttons (forward order `[10,11,12,4,5,6]`,
  backward order `[15,14,13,9,8,7]`, for LF · LM · LH · RF · RM · RH) step each
  leg; in Tripod mode the left-hind/right-hind buttons drive the two tripods.

Button indices match the joystick the desktop game targets — retune the `PAD`
constants in `game.js` for other controllers. Joystick and keyboard work
simultaneously; engaging the stick on the start screen begins the run.

## How it works

The physics is **real**: MuJoCo (WebAssembly) runs the same legs-only,
position-actuated NeuroMechFly model as the desktop game — with leg adhesion — at
`dt=1e-4`. Like the desktop game it plays well below real time (a fixed ~0.1×,
set by `PLAYBACK_SPEED` in `game.js`); the loop
caps physics substeps per animation frame for a steady frame rate and shows the
achieved factor top-right.

The control logic that the desktop game gets from Python/flygym is **ported to
JavaScript** (`game.js`) and fed the **baked tables** in `assets/model_meta.json`,
so the browser needs no SciPy:

- **`Controller`** — the CPG network (coupled phase/amplitude oscillators, Euler
  integration) for Level 1, and the per-leg / per-tripod step state machines for
  Levels 2–3. It interpolates the baked `PreprogrammedSteps` joint-angle
  trajectories and scatters them into `data.ctrl` via the baked
  `(leg, dof) -> ctrl-index` map, plus per-leg swing/stance adhesion.
- **Camera** chases the fly (yaw-smoothed); the **finish** is a geometric
  path/line-segment crossing test against the white gate at `x = 50 mm`.

## Layout

```
game.html            stage + HUD (level, timer, controls) + overlay screens
game.js              MuJoCo-WASM loop + ported controllers + Three.js rendering
assets/              (gitignored, generated)
  model/fly.xml      flattened slalom-arena MJCF (+ *.stl meshes)
  model_meta.json    timestep, neutral keyframe, actuators + adhesion, (leg,dof)
                     -> ctrl map, CPG params, baked PreprogrammedSteps tables,
                     finish line, geom colors, camera params
```

## Rebuilding the assets

`assets/` is generated and **gitignored** (regenerated for the `gh-pages` site by
`scripts/dev/push_doc_site.sh`). Regenerate whenever the model or controller
config changes (also needed once to preview locally with `properdocs serve`):

```sh
uv run python scripts/dev/build_wasm_game_assets.py
```

It composes a legs-only position-actuated fly (with adhesion) in the slalom
arena, bakes the CPG parameters and `PreprogrammedSteps` trajectories, and
verifies the exported model with a short rollout. It needs `flygym` +
`flygym_demo` + `mujoco`.

## Attribution

- **Model** (`assets/model/`): the NeuroMechFly v2 biomechanical model, generated
  from [flygym](https://github.com/NeLy-EPFL/flygym) (Apache-2.0). Please cite the
  NeuroMechFly v2 publication (see https://neuromechfly.org/).
- Game design ported from
  [NeuroMechFly Live](https://github.com/NeLy-EPFL/neuromechfly-live) (Apache-2.0).
- MuJoCo-WASM and Three.js: see [`../README.md`](../README.md).
