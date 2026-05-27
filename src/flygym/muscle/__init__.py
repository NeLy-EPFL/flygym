"""Musculoskeletal body model for FlyGym.

When muscle-based actuation is needed, FlyGym *switches the body model* to
FlyMimic's musculoskeletal MJCF (15 Hill-type muscles on the left-front leg,
plus realistic passive joint properties) instead of mutating FlyGym's default
rigid-body composition. The FlyMimic MJCF is wrapped so that
`flygym.Simulation` and its sensor APIs work unchanged.

Public objects:

* `MuscleFly`: Fly-compatible wrapper around the FlyMimic MJCF.
* `MuscleWorld`: minimal world that presents the FlyMimic scene to `Simulation`.
* `build_muscle_simulation`: one-call factory returning ``(simulation, fly)``.
* `DEFAULT_MUSCLE_XML`, `MUSCULOSKELETAL_DIR`: bundled asset locations.
"""

from flygym.muscle.assets import DEFAULT_MUSCLE_XML, MUSCULOSKELETAL_DIR
from flygym.muscle.model import MuscleFly, MuscleWorld, build_muscle_simulation
from flygym.muscle.gpu import (
    MjWarpCompatibilityReport,
    build_muscle_gpu_simulation,
    check_mjwarp_compatibility,
)

__all__ = [
    "MuscleFly",
    "MuscleWorld",
    "build_muscle_simulation",
    "DEFAULT_MUSCLE_XML",
    "MUSCULOSKELETAL_DIR",
    "MjWarpCompatibilityReport",
    "check_mjwarp_compatibility",
    "build_muscle_gpu_simulation",
]
