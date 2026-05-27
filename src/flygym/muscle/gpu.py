"""GPU (MuJoCo-Warp) helpers for the musculoskeletal model.

Everything here is guarded so it imports and runs on machines *without*
``warp`` / ``mujoco_warp`` (e.g. macOS / no NVIDIA GPU): the compatibility
probe reports "unavailable" instead of raising, and the GPU-sim factory raises
a clear, actionable error only when actually called.

Why a probe at all? `mujoco_warp` is a from-scratch GPU reimplementation of
MuJoCo with an evolving feature set. The muscle model leans on three features
that are not guaranteed to be ported: Hill-type muscle actuators
(``mjDYN_MUSCLE`` + ``mjGAIN_MUSCLE``/``mjBIAS_MUSCLE``), spatial tendons, and
joint-equality constraints (the locked right-front-leg joints). The probe runs
``mjw.put_model`` on the compiled model, which is where unsupported features
surface.
"""

from dataclasses import dataclass
from os import PathLike

from flygym.muscle.assets import DEFAULT_MUSCLE_XML
from flygym.muscle.model import MuscleFly, MuscleWorld

__all__ = [
    "MjWarpCompatibilityReport",
    "check_mjwarp_compatibility",
    "build_muscle_gpu_simulation",
]


@dataclass
class MjWarpCompatibilityReport:
    """Outcome of probing whether MuJoCo-Warp accepts the muscle model.

    Attributes:
        mjwarp_available: Whether ``mujoco_warp`` could be imported.
        put_model_ok: Whether ``mjw.put_model`` succeeded on the compiled
            model. ``None`` if mjwarp was unavailable (probe not run).
        error: The exception text if ``put_model`` failed, else ``None``.
        message: A human-readable summary.
    """

    mjwarp_available: bool
    put_model_ok: bool | None
    error: str | None
    message: str

    def __bool__(self) -> bool:
        """True only if mjwarp is available *and* accepted the model."""
        return bool(self.mjwarp_available and self.put_model_ok)


def check_mjwarp_compatibility(
    fly: MuscleFly | None = None,
    *,
    xml_path: PathLike = DEFAULT_MUSCLE_XML,
) -> MjWarpCompatibilityReport:
    """Probe whether MuJoCo-Warp can ingest the muscle model.

    Safe to call anywhere: if ``mujoco_warp`` is not installed (the usual case
    off a CUDA machine), it returns a report with ``mjwarp_available=False``
    rather than raising.

    Args:
        fly: A `MuscleFly` to probe. If None, a default one is built.
        xml_path: XML to use when ``fly`` is None.

    Returns:
        A `MjWarpCompatibilityReport`.
    """
    try:
        import mujoco_warp as mjw
    except ImportError:
        return MjWarpCompatibilityReport(
            mjwarp_available=False,
            put_model_ok=None,
            error=None,
            message=(
                "mujoco_warp is not installed; cannot probe GPU compatibility. "
                "This is expected without an NVIDIA CUDA GPU. Install the "
                "'[warp]' extra on a Linux+CUDA machine to enable the GPU path."
            ),
        )

    if fly is None:
        fly = MuscleFly(xml_path)
    mj_model, _ = fly.compile()
    try:
        mjw.put_model(mj_model)
    except Exception as e:  # noqa: BLE001 - we want to report any failure mode
        return MjWarpCompatibilityReport(
            mjwarp_available=True,
            put_model_ok=False,
            error=f"{type(e).__name__}: {e}",
            message=(
                "mujoco_warp is installed but rejected the muscle model. The "
                "likely culprits are muscle actuators, spatial tendons, or "
                "joint-equality constraints not yet supported by this mjwarp "
                "version. See the error field."
            ),
        )
    return MjWarpCompatibilityReport(
        mjwarp_available=True,
        put_model_ok=True,
        error=None,
        message="mujoco_warp accepted the muscle model (put_model succeeded).",
    )


def build_muscle_gpu_simulation(
    n_worlds: int,
    *,
    xml_path: PathLike = DEFAULT_MUSCLE_XML,
    name: str = "nmf",
    add_vision: bool = False,
    **gpu_kwargs,
):
    """Build a `GPUSimulation` of the muscle model with ``n_worlds`` parallel
    copies (for vectorized RL on a CUDA machine).

    Raises a clear `ImportError` if the ``[warp]`` extra / an NVIDIA GPU is not
    available. The non-GPU code path (`build_muscle_simulation`) is unaffected.

    Returns:
        ``(gpu_simulation, fly)``.
    """
    try:
        from flygym.warp.simulation import GPUSimulation
    except ImportError as e:
        raise ImportError(
            "GPU simulation requires the '[warp]' extra (warp-lang + "
            "mujoco_warp) and an NVIDIA CUDA GPU. On such a machine, install "
            "with `pip install 'flygym[warp]'`."
        ) from e

    fly = MuscleFly(xml_path, name=name)
    if add_vision:
        fly.add_vision()
    world = MuscleWorld(fly)
    sim = GPUSimulation(world, n_worlds=n_worlds, **gpu_kwargs)
    return sim, fly
