---
hide:
  - toc
---

# Interactive viewer

The NeuroMechFly model runs entirely in your browser — no install required. The
simulation is real: drag the position actuator sliders to set each leg joint's
target (the green tick shows where the joint actually is), reset to the neutral
pose, toggle the contacts/forces/joints/actuators overlays, orbit/zoom/pan the
camera, and <kbd>Shift</kbd>+drag a body to push it.

<div style="text-align:center;margin:1.2em 0;">
<a href="../wasm_viewer/viewer.html" target="_blank" rel="noopener"
   style="display:inline-block;padding:0.75em 2.2em;font-size:1.1em;font-weight:700;background:var(--md-primary-fg-color);color:var(--md-primary-bg-color);border-radius:6px;text-decoration:none;">
Open viewer in full-screen ↗</a>
</div>

<iframe src="../wasm_viewer/viewer.html" title="Interactive NeuroMechFly viewer"
        style="width:100%;height:640px;border:1px solid var(--md-default-fg-color--lightest);border-radius:8px;">
</iframe>

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
    `python scripts/build_wasm_viewer_assets.py`.
