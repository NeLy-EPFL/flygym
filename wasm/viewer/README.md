# Browser (WebAssembly) interactive viewer

A self-contained, in-browser version of `scripts/launch_interactive_viewer.py`.
It runs the **same** NeuroMechFly model with MuJoCo compiled to WebAssembly and
renders it with Three.js. Embedded in the docs home page (`docs/index.md`); also
opens standalone. The MuJoCo-WASM + Three.js plumbing it shares with the game
lives one level up in [`../shared/`](../shared/); see [`../README.md`](../README.md).

## Features

- **Live simulation** (`mj_step`) of the fly on flat ground.
- **Reset** to the neutral keyframe; **Pause/Resume**.
- One **slider per position actuator** (the 6 legs × 7 active DOFs). Each slider
  sets the actuator target (`data.ctrl`); a green tick over the slider shows the
  driven joint's actual angle (`data.qpos`).
- **Visualization toggles**, like MuJoCo's native viewer: contact points,
  contact-force arrows, joint axes, actuator-force arrows; plus a transparent
  mesh mode.
- **Camera**: drag to orbit, scroll to zoom, right-drag to pan.
- **Drag to push**: <kbd>Shift</kbd>+drag any body to apply an external force
  (via `data.xfrc_applied`, the same channel MuJoCo's own perturbation uses).

## Layout

```
viewer.html          control panel + stage markup and styles
viewer.js            MuJoCo-WASM simulation loop + Three.js rendering + controls
assets/              (gitignored, generated)
  model/fly.xml      flattened, self-contained MJCF (+ *.stl meshes)
  model_meta.json    timestep, neutral keyframe, per-actuator slider metadata, colors
```

The shared `../shared/scene.js` (MuJoCo load, FS write, mesh build, per-frame
geom sync) and `../shared/vendor/` (MuJoCo-WASM + Three.js) are used by both the
viewer and the game.

## Rebuilding the assets

`assets/` is generated from the live model and is **gitignored** (see
`.gitignore` here): it is 39 STL meshes + `model_meta.json` that are regenerated
on every model tweak, so it is kept out of the `main` branch's history. Instead
it is published only to the `gh-pages` branch — `scripts/dev/push_doc_site.sh`
regenerates it and `mkdocs build` bundles it into the deployed site.

Regenerate it whenever the model or its viewer config changes (also needed once
to preview locally with `mkdocs serve`):

```sh
uv run python scripts/dev/build_wasm_viewer_assets.py
```

The script mirrors the body configuration in
`scripts/launch_interactive_viewer.py`. Keep the two in sync.

`viewer.html`, `viewer.js`, and this README are committed to `main`.
`../shared/vendor/` is **not** committed — it is downloaded automatically by the
MkDocs hook in `scripts/dev/mkdocs_hooks.py` when you run `mkdocs serve` or
`mkdocs build` for the first time (see [`../README.md`](../README.md)).

## Deploying

`scripts/dev/push_doc_site.sh` is the single entry point: it offers to
regenerate the assets above, verifies `../shared/vendor/` is present, runs
`mkdocs build` (whose hook copies `wasm/` into the site), and force-pushes the
resulting `site/` to the orphan `gh-pages` branch. So the heavy assets live only
on `gh-pages`, never bloating `main`.

## Attribution

- **Model** (`assets/model/`): the NeuroMechFly v2 biomechanical model, generated
  from [flygym](https://github.com/NeLy-EPFL/flygym) (Apache-2.0). If you use it,
  please cite the NeuroMechFly v2 publication (see https://neuromechfly.org/).
- **`../shared/vendor/mujoco/`**: [MuJoCo](https://github.com/google-deepmind/mujoco)
  by Google DeepMind, compiled to WebAssembly (Apache-2.0).
- **`../shared/vendor/three/`**: [Three.js](https://threejs.org/) (MIT).
