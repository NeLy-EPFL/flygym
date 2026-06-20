"""Mocap clip loader for FlyMimic-style imitation learning.

The bundled clip was authored against FlyMimic's musculoskeletal model, so the
tracked joints/bodies below use that model's MJCF element names (resolved at
runtime via ``mujoco.mj_name2id``).

The shipped clip is ``"0002"`` (7 joint DoFs, 225 frames) — FlyMimic's own
default. Its body trajectories (``xipos``) match the bundled MJCF's forward
kinematics to <0.01 mm, so all three reward terms are meaningful and the reward
ceiling is ~1.0.
"""

from dataclasses import dataclass
from importlib.resources import files
from os import PathLike
from pathlib import Path

import numpy as np


DEFAULT_MOCAP_DIR = Path(str(files("flygym_demo.muscle_imitation") / "assets/mocap"))
"""Directory where this demo's bundled FlyMimic mocap clips live.

The clips ship with the demo (not with FlyGym's core model assets) since they
are imitation-learning training data, not part of the body model itself.
"""

# The clip's 7 qpos columns map to these MJCF joints, in order (recovered by
# matching each column's observed range against the per-joint limits in
# FlyMimic's MJCF). Keyed by qpos width so additional clips with other layouts
# can register their own mapping. See docs/imitation_muscle.md.
TRACKED_JOINT_NAMES_BY_NCOLS: dict[int, tuple[str, ...]] = {
    7: (
        "joint_LFCoxa_yaw",
        "joint_LFCoxa_pitch",
        "joint_LFCoxa_roll",
        "joint_LFTrochanter_yaw",
        "joint_LFTrochanter_pitch",
        "joint_LFTrochanter_roll",
        "joint_LFTibia_pitch",
    ),
}

# Default mapping (the shipped 7-DoF clip), kept as a convenience constant.
TRACKED_JOINT_NAMES: tuple[str, ...] = TRACKED_JOINT_NAMES_BY_NCOLS[7]

# MJCF body names tracked by the xipos array (FlyMimic body_ids [21,22,23,27]).
TRACKED_BODY_NAMES: tuple[str, ...] = (
    "LFFemur",
    "LFTibia",
    "LFTarsus1",
    "LFTarsus5",
)


def tracked_joint_names_for_ncols(ncols: int) -> tuple[str, ...]:
    """Return the MJCF joint names matching a clip's qpos column count.

    Raises:
        ValueError: if no mapping is registered for ``ncols``.
    """
    try:
        return TRACKED_JOINT_NAMES_BY_NCOLS[ncols]
    except KeyError:
        raise ValueError(
            f"No tracked-joint mapping for qpos width {ncols}. Known widths: "
            f"{sorted(TRACKED_JOINT_NAMES_BY_NCOLS)}. Add an entry to "
            "TRACKED_JOINT_NAMES_BY_NCOLS or pass tracked_joint_names explicitly."
        )


@dataclass(frozen=True)
class MoCapClip:
    """One clip's worth of *Drosophila* left-front-leg mocap data.

    All arrays are time-first; indices align frame-by-frame so that row *t*
    of each array describes the same instant.

    Attributes:
        qpos: Joint angles of the tracked joints, shape ``(T, n_joints)``
            in radians.
        qvel: Joint velocities, shape ``(T, n_joints)`` in rad/s.
        xipos: 3-D world-frame positions of `TRACKED_BODY_NAMES`, shape
            ``(T, n_bodies, 3)`` in mm.
        xivel: 3-D world-frame velocities of the same bodies, shape
            ``(T, n_bodies, 3)`` in mm/s, or ``None`` if not available.
        n_frames: Number of time steps *T* (read-only property).
    """

    qpos: np.ndarray
    qvel: np.ndarray
    xipos: np.ndarray
    xivel: np.ndarray | None

    @property
    def n_frames(self) -> int:
        """Number of time steps in this clip."""
        return int(self.qpos.shape[0])


class MoCapDataset:
    """Lazy loader for FlyMimic-format mocap clips.

    The dataset directory must contain ``qpos/{clip}.npy``,
    ``qvel/{clip}.npy``, and ``xipos/{clip}.npy`` (and optionally
    ``xivel/{clip}.npy``).  Use `default` to access the bundled clip
    (``"0002"``).

    Args:
        clip_dir: Root directory of the dataset. Defaults to
            `DEFAULT_MOCAP_DIR` (the clips bundled with this demo).

    Raises:
        FileNotFoundError: If *clip_dir* does not exist.
    """

    def __init__(self, clip_dir: PathLike = DEFAULT_MOCAP_DIR) -> None:
        self.clip_dir = Path(clip_dir)
        if not self.clip_dir.exists():
            raise FileNotFoundError(f"Mocap dir not found: {self.clip_dir}")
        self._cache: dict[str, MoCapClip] = {}

    @classmethod
    def default(cls) -> "MoCapDataset":
        """Return a dataset pointing at the bundled mocap clips."""
        return cls(DEFAULT_MOCAP_DIR)

    def available_clips(self) -> list[str]:
        """Return stem names of all clips available in this dataset."""
        return sorted(p.stem for p in (self.clip_dir / "qpos").glob("*.npy"))

    def load(self, clip: str) -> MoCapClip:
        """Load a clip by name, caching on first access.

        Args:
            clip: Clip identifier (e.g. ``"0002"``), matching the file stems
                under ``qpos/``, ``qvel/``, and ``xipos/``.

        Returns:
            A `MoCapClip` with ``qpos``, ``qvel``, ``xipos``, and optionally
            ``xivel`` arrays.

        Raises:
            FileNotFoundError: If the required ``.npy`` files are missing.
            ValueError: If the clip's qpos width has no registered joint
                mapping in `TRACKED_JOINT_NAMES_BY_NCOLS`, or if array
                shapes are inconsistent.
        """
        if clip in self._cache:
            return self._cache[clip]
        qpos = np.load(self.clip_dir / "qpos" / f"{clip}.npy")
        qvel = np.load(self.clip_dir / "qvel" / f"{clip}.npy")
        xipos = np.load(self.clip_dir / "xipos" / f"{clip}.npy")
        xivel_path = self.clip_dir / "xivel" / f"{clip}.npy"
        xivel = np.load(xivel_path) if xivel_path.exists() else None

        if qpos.shape[1] not in TRACKED_JOINT_NAMES_BY_NCOLS:
            raise ValueError(
                f"Mocap clip '{clip}' has qpos shape {qpos.shape}; no tracked-"
                f"joint mapping for width {qpos.shape[1]} (known: "
                f"{sorted(TRACKED_JOINT_NAMES_BY_NCOLS)}). Add an entry to "
                "TRACKED_JOINT_NAMES_BY_NCOLS."
            )
        if qvel.shape != qpos.shape:
            raise ValueError(
                f"Mocap clip '{clip}': qvel shape {qvel.shape} != qpos shape "
                f"{qpos.shape}."
            )
        if xipos.ndim != 3 or xipos.shape[1] != len(TRACKED_BODY_NAMES):
            raise ValueError(
                f"Mocap clip '{clip}' has xipos shape {xipos.shape}; expected "
                f"(T, {len(TRACKED_BODY_NAMES)}, 3) to match TRACKED_BODY_NAMES."
            )

        clip_obj = MoCapClip(qpos=qpos, qvel=qvel, xipos=xipos, xivel=xivel)
        self._cache[clip] = clip_obj
        return clip_obj
