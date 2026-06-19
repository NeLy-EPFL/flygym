---
hide:
  - toc
---

# Simulating embodied sensorimotor control with NeuroMechFly v2

FlyGym is the Python library for NeuroMechFly, a digital twin of the adult fruit fly _Drosophila melanogaster_ that can see, smell, walk over challenging terrain, and interact with the environment.

For more information, see our [NeuroMechFly v2 paper](https://www.nature.com/articles/s41592-024-02497-y.epdf).

<p align="center">
  <img src="https://raw.githubusercontent.com/NeLy-EPFL/_media/refs/heads/main/flygym/overview_video.gif" alt="overview" />
</p>


## Interact with NeuroMechFly

<iframe src="../wasm_viewer/viewer.html" title="Interactive NeuroMechFly viewer"
        style="width:100%;height:640px;border:1px solid var(--md-default-fg-color--lightest);border-radius:8px;">
</iframe>

<div style="text-align:center;margin:1.2em 0;">
<a href="../wasm_viewer/viewer.html" target="_blank" rel="noopener"
   style="display:inline-block;padding:0.75em 2.2em;font-size:1.1em;font-weight:700;background:var(--md-primary-fg-color);color:var(--md-primary-bg-color);border-radius:6px;text-decoration:none;">
Open viewer in full-screen ↗</a>
</div>

??? note "Run locally with the native MuJoCo viewer"

    If you have the `flygym` repository cloned, you can launch the native desktop
    MuJoCo viewer instead — useful for faster-than-real-time playback and
    full-resolution rendering:

    ```sh
    uv run python scripts/launch_interactive_viewer.py
    ```

    <p align="center">
      <video src="https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/mujoco_interactive_viewer.mp4" controls autoplay muted playsinline>
        MuJoCo interactive viewer (video not supported by your browser).
      </video>
    </p>

    The browser viewer's static assets are regenerated from the same model with
    `python scripts/dev/build_wasm_viewer_assets.py`.


## Key features

- **Biomechanical model:** The biomechanical model is based on a micro-CT scan of a real adult female fly (see our original NeuroMechFly publication). We have adjusted several body segments (in particular in the antennae) to better reflect the biological reality.
- **Vision:** The fly has compound eyes consisting of individual units called ommatidia arranged on a hexagonal lattice. We have simulated the visual inputs on the fly’s retinas.
- **Olfaction:** The fly has odor receptors in the antennae and the maxillary palps. We have simulated the odor inputs experienced by the fly by computing the odor/chemical intensity at these locations.
- **Hierarchical control:** The fly’s Central Nervous System consists of the brain and the Ventral Nerve Cord (VNC), a hierarchy analogous to our brain-spinal cord organization. The user can build a two-part model — one handling brain-level sensory integration and decision making and one handling VNC-level motor control — with an interface between the two consisting of descending (brain-to-VNC) and ascending (VNC-to-brain) representations.
- **Leg adhesion:** Insects have evolved specialized adhesive structures at the tips of the legs that enable locomotion on vertical walls and overhanging ceilings. We have simulated these structures in our model. The mechanism by which the fly lifts the legs during locomotion despite adhesive forces is not well understood; to abstract this, adhesion can be turned on/off during leg stance/swing.
- **Mechanosensory feedback:** The user has access to joint angles, actuator forces, contact forces, and user-defined anatomical joint-site positions.

This package is developed at the [Neuroengineering Laboratory](https://www.epfl.ch/labs/ramdya-lab/), EPFL.


## Getting Started

!!! tip "March 2026 Update"

    We introduced a new FlyGym 2.x.x API in March 2026, with a complete code rewrite and redesigned interface. This version delivers significantly improved performance:

    - **~10x speed-up** for CPU-based simulations (~2x real-time throughput)
    - **~300x speed-up** for GPU-based simulation via Warp/MJWarp (~60x real-time throughput)

    Additional improvements include:

    - Improved scene composition workflow
    - Interactive viewer
    - Simplified dependency stack

    This version is not backward compatible, and not all features from FlyGym 1.x.x are available. Feature requests can be submitted via Issues on the [GitHub repository](https://github.com/NeLy-EPFL/flygym). See more information about the changes [here](https://neuromechfly.org/migration/).

    Prefer the old version? FlyGym 1.x.x has been migrated to [`flygym-gymnasium`](https://github.com/NeLy-EPFL/flygym-gymnasium). Its documentation has been migrated to [gymnasium.neuromechfly.org](https://gymnasium.neuromechfly.org/).

For installation, see [the documentation page](https://neuromechfly.org/installation).

To get started, follow [tutorials here](https://neuromechfly.org/tutorials).