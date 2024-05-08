# Let's save a single snapshot from any simulation as a PNG. The MP4 video
# is compressed and unsuitable for publication-quality figures.

import numpy as np
import cv2
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree

from flygym.examples.olfaction.track_plume_closed_loop import run_simulation


plume_dataset_path = Path("./outputs/plume_tracking/plume_dataset/plume.hdf5")
output_path = Path("./outputs/plume_tracking/figs/snapshot.png")
xx, yy = np.meshgrid(np.linspace(155, 200, 10), np.linspace(57.5, 102.5, 10))
points = np.vstack((xx.flat, yy.flat)).T
initial_position = points[12]

try:
    output_dir = Path(mkdtemp())
    sim = run_simulation(
        plume_dataset_path,
        output_dir,
        seed=12,
        initial_position=initial_position,
        is_control=False,
        live_display=False,
        run_time=0.8,
    )
    img = sim.cameras[0]._frames[-1]
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
finally:
    rmtree(output_dir)
