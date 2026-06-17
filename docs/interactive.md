# Interactive viewer

You can launch a MuJoCo interactive viewer by running the following from the cloned `flygym` directory:

```sh
uv run python scripts/launch_interactive_viewer.py
```

<p align="center">
  <video src="https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/mujoco_interactive_viewer.mp4" controls autoplay muted playsinline>
    MuJoCo interactive viewer (video not supported by your browser).
  </video>
</p>

## In your browser

The same model also runs entirely in your browser, with MuJoCo compiled to
WebAssembly — no install required. The simulation is real: drag the position
actuator sliders to set each leg joint's target (the green tick shows where the
joint actually is), reset to the neutral pose, toggle the
contacts/forces/joints/actuators overlays, orbit/zoom/pan the camera, and
<kbd>Shift</kbd>+drag a body to push it.

<iframe src="../wasm_viewer/viewer.html" title="Interactive NeuroMechFly viewer"
        style="width:100%; height:640px; border:1px solid var(--md-default-fg-color--lightest); border-radius:8px;">
</iframe>

[Open the viewer full-screen ↗](../wasm_viewer/viewer.html){:target="_blank" rel="noopener"}

The viewer's static assets are regenerated from the live model with
`python scripts/build_wasm_viewer_assets.py` (which uses the same body
configuration as `scripts/launch_interactive_viewer.py`).