"""Multi-world MuJoCo-Warp rendering: live rasterization and trajectory replay.

Split into `live_rendering` (real-time GPU/CPU renderers) and `recorded_trajectory`
(GPU recording + batch replay). Public names are re-exported for backward
compatibility.
"""

from flygym.warp.rendering.base import RendererType
from flygym.warp.rendering.live_rendering import (
    WarpGPUBatchRenderer,
    WarpCPURenderer,
    modify_world_for_batch_rendering,
)
from flygym.warp.rendering.recorded_trajectory import (
    WarpTrajectoryRecorder,
    render_trajectories_gpu,
)

__all__ = [
    "RendererType",
    "WarpGPUBatchRenderer",
    "WarpCPURenderer",
    "WarpTrajectoryRecorder",
    "modify_world_for_batch_rendering",
    "render_trajectories_gpu",
]
