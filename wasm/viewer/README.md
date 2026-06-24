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

## Rebuilding the assets

`assets/` is generated and **gitignored** (regenerated for the `gh-pages` site by
`scripts/dev/push_doc_site.sh`). Regenerate whenever the model or viewer config
changes (also needed once to preview locally with `properdocs serve`):

```sh
uv run python scripts/dev/build_wasm_viewer_assets.py
```

The script mirrors the body configuration in
`scripts/launch_interactive_viewer.py` — keep the two in sync.

## Deploying

Deployment is described in [`../README.md`](../README.md#deploying):
`scripts/dev/push_doc_site.sh` regenerates the assets above, runs `properdocs build`,
and force-pushes the site to `gh-pages` (so the heavy assets live only there).

## Attribution

- **Model** (`assets/model/`): the NeuroMechFly v2 biomechanical model, generated
  from [flygym](https://github.com/NeLy-EPFL/flygym) (Apache-2.0). Please cite the
  NeuroMechFly v2 publication (see https://neuromechfly.org/).
- MuJoCo-WASM and Three.js: see [`../README.md`](../README.md).
