"""Save a moving fly and its anatomical compound-eye signals as a video.

Run ``python -m flygym_demo.compound_eye --output outputs/compound_eye/input.mp4``.
Body motion is prescribed to demonstrate visual input, rather than locomotion.
"""

import argparse
from pathlib import Path

import imageio.v2 as iio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from flygym.compose.fly import NeuroMechFly
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym.vision.compound_eye import CompoundEye
from flygym_demo.compound_eye import make_world


def render_video(output_path: Path, duration: float = 6, fps: int = 25) -> None:
    """Show the fly and each eye's signals in azimuth/elevation coordinates.

    Args:
        output_path: Destination MP4 file.
        duration: Video duration in seconds.
        fps: Frames per second.
    """
    if not np.isfinite(duration) or duration <= 0 or fps < 1:
        raise ValueError("Require a finite positive duration and positive fps.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eye = CompoundEye()
    fly = NeuroMechFly()
    fly.colorize()
    fly.add_vision(compound_eye=eye)
    camera = fly.add_tracking_camera(
        pos_offset=(0, -5.5, 7),
        rotation=Rotation3D("xyaxes", (1, 0, 0, 0, 7, 5.5)),
        fovy=35,
    )
    simulation = Simulation(make_world(fly))
    figure = Figure(figsize=(10.24, 7.2), dpi=100, layout="constrained")
    FigureCanvasAgg(figure)
    try:
        renderer = simulation.set_renderer(
            camera, camera_res=(320, 640), playback_speed=1, output_fps=fps
        )
        grid = figure.add_gridspec(2, 2)
        scene_axis = figure.add_subplot(grid[0, :])
        scene_axis.axis("off")
        scene_image = scene_axis.imshow(np.zeros((320, 640, 3), np.uint8))
        scatters = []
        for side, label in enumerate(("Left eye", "Right eye")):
            axis = figure.add_subplot(grid[1, side])
            directions = eye.directions[eye.source_indices[side, eye.valid_mask[side]]]
            azimuth = -np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]))
            elevation = np.rad2deg(np.arcsin(directions[:, 2]))
            scatters.append(
                axis.scatter(
                    azimuth,
                    elevation,
                    c=np.zeros(len(azimuth)),
                    s=16,
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
            )
            axis.set(
                title=label,
                xlim=(-180, 180),
                ylim=(-90, 90),
                xlabel="Azimuth (degrees; −left / +right)",
                ylabel="Elevation (degrees)",
            )
            axis.set_facecolor("#9aaeb8")
        origin = simulation.mj_data.mocap_pos[0].copy()
        with iio.get_writer(str(output_path), fps=fps, codec="libx264") as writer:
            for t in np.arange(int(np.ceil(duration * fps))) / fps:
                progress = t / duration
                simulation.mj_data.mocap_pos[0] = origin + [3 * progress, 0, 0]
                simulation.mj_data.mocap_quat[0] = Rotation.from_euler(
                    "z", 0.4 * np.sin(2 * np.pi * progress)
                ).as_quat(scalar_first=True)
                simulation.mj_data.time = t
                mj.mj_forward(simulation.mj_model, simulation.mj_data)
                signals = simulation.get_ommatidia_readouts(fly.name)
                for side, scatter in enumerate(scatters):
                    scatter.set_array(signals[side, eye.valid_mask[side], 0])
                simulation.render_as_needed()
                scene_image.set_data(renderer.frames[camera.name].pop())
                figure.canvas.draw()
                writer.append_data(np.asarray(figure.canvas.buffer_rgba())[..., :3])
    finally:
        simulation.close()
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/compound_eye/input.mp4")
    )
    parser.add_argument("--duration", type=float, default=6)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    render_video(args.output, duration=args.duration, fps=args.fps)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
