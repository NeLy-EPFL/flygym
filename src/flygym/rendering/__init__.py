"""MuJoCo rendering: live rasterization and recorded-trajectory replay.

This package is split into:

- `flygym.rendering.live_rendering`: the real-time `Renderer` and viewer helpers.
- `flygym.rendering.recorded_trajectory`: recording ``qpos`` trajectories and
  replaying them to video on the CPU.

All public names are re-exported here for backward compatibility, so
``from flygym.rendering import Renderer`` (etc.) keeps working.
"""

from flygym.rendering.live_rendering import (
    Renderer,
    launch_interactive_viewer,
    preview_model,
)
from flygym.rendering.recorded_trajectory import (
    RecordedTrajectory,
    TrajectoryRecorder,
    render_trajectories,
)

__all__ = [
    "Renderer",
    "TrajectoryRecorder",
    "RecordedTrajectory",
    "render_trajectories",
    "launch_interactive_viewer",
    "preview_model",
]
