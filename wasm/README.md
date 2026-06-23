# Browser (WebAssembly) apps

In-browser apps that run the **NeuroMechFly** model with [MuJoCo compiled to
WebAssembly](https://github.com/google-deepmind/mujoco) and render it with
[Three.js](https://threejs.org/) — no Python, no install. Two apps share one
vendor bundle and one helper module:

```
wasm/
  shared/
    scene.js           MuJoCo load + FS write + mesh build + per-frame geom sync
    vendor/            (gitignored) MuJoCo-WASM (mujoco.js/.wasm) + Three.js
  viewer/              interactive posing viewer (counterpart of the native viewer)
    viewer.html, viewer.js
    assets/            (gitignored, generated) flat-ground fly MJCF + meshes + meta
  game/                the NeuroMechFly Live slalom game
    game.html, game.js
    assets/            (gitignored, generated) slalom-arena MJCF + meshes + meta
                       + baked CPG / preprogrammed-step tables
```

- **Viewer** — see [`viewer/README.md`](viewer/README.md). Pose the fly with one
  slider per leg actuator; toggle contact/force/joint/actuator overlays.
- **Game** — see [`game/README.md`](game/README.md). Pilot the fly through a
  slalom track to a finish line at three levels of neural abstraction (CPG /
  tripod gait / individual legs).

## How it's wired into the docs

`wasm/` lives at the repo root, *outside* MkDocs' `docs/` dir. The MkDocs hook
[`scripts/dev/properdocs_hooks.py`](../scripts/dev/properdocs_hooks.py):

1. **Downloads `shared/vendor/`** on first build/serve (`@mujoco/mujoco@3.9.0` +
   `three@0.169.0` from npm) — heavy binaries kept out of git.
2. **Builds the `assets/`** for each app if missing
   (`scripts/dev/build_wasm_viewer_assets.py`,
   `scripts/dev/build_wasm_game_assets.py`) — generated from the live flygym
   model, also kept out of git.
3. **Copies `wasm/` into the built site** (`on_post_build`), so the docs iframes
   (`../wasm/viewer/viewer.html`, `../wasm/game/game.html`) resolve, and
   **watches `wasm/`** during `properdocs serve`.

You can fetch the vendor files manually with:

```sh
uv run python scripts/dev/properdocs_hooks.py            # vendor + assets
uv run python scripts/dev/properdocs_hooks.py --vendor-only
```

## Deploying

[`scripts/dev/push_doc_site.sh`](../scripts/dev/push_doc_site.sh) is the single
entry point: it ensures `shared/vendor/` is present, offers to regenerate the
viewer + game `assets/`, runs `properdocs build` (whose hook bundles `wasm/` into the
site), and force-pushes `site/` to the orphan `gh-pages` branch. The heavy
vendor + generated assets therefore live only on `gh-pages`, never bloating
`main`.

## Attribution

- **Models** (`*/assets/model/`): the NeuroMechFly v2 biomechanical model,
  generated from [flygym](https://github.com/NeLy-EPFL/flygym) (Apache-2.0).
  Please cite the NeuroMechFly v2 publication (see https://neuromechfly.org/).
- **`shared/vendor/mujoco/`**: [MuJoCo](https://github.com/google-deepmind/mujoco)
  by Google DeepMind, compiled to WebAssembly (Apache-2.0).
- **`shared/vendor/three/`**: [Three.js](https://threejs.org/) (MIT).
