# Browser (WebAssembly) interactive viewer

A self-contained, in-browser version of `scripts/launch_interactive_viewer.py`.
It runs the **same** NeuroMechFly model with MuJoCo compiled to WebAssembly and
renders it with Three.js. Embedded in the docs at `docs/interactive.md`; also
opens standalone.

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
assets/
  model/fly.xml      flattened, self-contained MJCF (+ *.stl meshes)
  model_meta.json    timestep, neutral keyframe, per-actuator slider metadata, colors
vendor/
  mujoco/            MuJoCo compiled to WebAssembly (mujoco.js + mujoco.wasm)
  three/             Three.js + OrbitControls
```

## Rebuilding the assets

`assets/` is generated from the live model and is **gitignored** (see
`.gitignore` here): it is 39 STL meshes + `model_meta.json` that are regenerated
on every model tweak, so it is kept out of the `main` branch's history. Instead
it is published only to the `gh-pages` branch — `scripts/dev/push_doc_site.sh`
regenerates it and `mkdocs build` bundles it into the deployed site.

Regenerate it whenever the model or its viewer config changes (also needed once
to preview locally with `mkdocs serve`):

```sh
uv run python scripts/build_wasm_viewer_assets.py
```

The script mirrors the body configuration in
`scripts/launch_interactive_viewer.py`. Keep the two in sync.

`viewer.html`, `viewer.js`, and this README are committed to `main`. `vendor/`
is **not** committed — it is downloaded automatically by the MkDocs hook in
`scripts/dev/mkdocs_hooks.py` when you run `mkdocs serve` or `mkdocs build`
for the first time. You can also fetch it manually:

```sh
uv run python scripts/dev/mkdocs_hooks.py
```

The hook downloads `@mujoco/mujoco@3.9.0` and `three@0.169.0` from the npm
registry and extracts the relevant files into `vendor/`.

## Deploying

`scripts/dev/push_doc_site.sh` is the single entry point: it offers to
regenerate the assets above, verifies `vendor/` is present, runs `mkdocs build`,
and force-pushes the resulting `site/` to the orphan `gh-pages` branch. So the
heavy viewer assets live only on `gh-pages`, never bloating `main`.

## Attribution

- **Model** (`assets/model/`): the NeuroMechFly v2 biomechanical model, generated
  from [flygym](https://github.com/NeLy-EPFL/flygym) (Apache-2.0). If you use it,
  please cite the NeuroMechFly v2 publication (see https://neuromechfly.org/).
- **`vendor/mujoco/`**: [MuJoCo](https://github.com/google-deepmind/mujoco) by
  Google DeepMind, compiled to WebAssembly (Apache-2.0).
- **`vendor/three/`**: [Three.js](https://threejs.org/) (MIT).
