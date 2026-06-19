from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flygym_demo.complex_terrain.common import LocomotionAction
from flygym_demo.complex_terrain.hybrid_controller import (
    HybridController,
    HybridControllerObservation,
)


@dataclass
class HybridTurningController(HybridController):
    """Hybrid walking controller with side-specific CPG modulation."""

    def step(
        self,
        descending_signal: np.ndarray,
        obs: HybridControllerObservation,
    ) -> LocomotionAction:
        descending_signal = np.asarray(descending_signal, dtype=float)
        if descending_signal.shape != (2,):
            raise ValueError("descending_signal must have shape (2,).")

        self.cpg_network.intrinsic_amps = np.repeat(
            np.abs(descending_signal[:, np.newaxis]), 3, axis=1
        ).ravel()

        intrinsic_freqs = self._base_intrinsic_freqs.copy()
        intrinsic_freqs[:3] *= 1 if descending_signal[0] >= 0 else -1
        intrinsic_freqs[3:] *= 1 if descending_signal[1] >= 0 else -1
        self.cpg_network.intrinsic_freqs = intrinsic_freqs

        return super().step(obs)
