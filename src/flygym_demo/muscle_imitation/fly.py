"""Factory that builds the muscle Simulation + an imitation env."""

from __future__ import annotations

from os import PathLike

from flygym.compose import (
    DEFAULT_MUSCULOSKELETAL_XML,
    build_musculoskeletal_simulation,
)

from flygym_demo.muscle_imitation.data import MoCapDataset
from flygym_demo.muscle_imitation.env import ImitationConfig, ImitationEnv


def make_imitation_env(
    *,
    xml_path: PathLike = DEFAULT_MUSCULOSKELETAL_XML,
    name: str = "nmf",
    add_vision: bool = False,
    config: ImitationConfig | None = None,
    dataset: MoCapDataset | None = None,
) -> ImitationEnv:
    """Build a muscle `Simulation` and wrap it in an `ImitationEnv`.

    !!! info "Plain flygym — not GPU-accelerated"

        Uses `build_musculoskeletal_simulation`, which returns a plain
        `flygym.Simulation` (CPU, single world), not `flygym.warp.GPUSimulation`.

    Convenience factory equivalent to::

        sim, fly = build_musculoskeletal_simulation(xml_path=xml_path,
                                                    name=name,
                                                    add_vision=add_vision)
        env = ImitationEnv(sim, fly_name=fly.name, dataset=dataset,
                           config=config or ImitationConfig())

    Args:
        xml_path: Musculoskeletal MJCF to load. Defaults to
            `DEFAULT_MUSCULOSKELETAL_XML`.
        name: Logical fly name passed to `MusculoskeletalFly`. Default
            ``"nmf"``.
        add_vision: If ``True``, attach eye cameras so
            `Simulation.get_ommatidia_readouts` works (approximate; see
            `MusculoskeletalFly.add_vision`).
        config: `ImitationConfig` controlling reward weights, clip
            selection, and episode logic. Defaults to ``ImitationConfig()``.
        dataset: Mocap clip loader. Defaults to the bundled clips.

    Returns:
        A fully constructed `ImitationEnv` ready for training or evaluation.
    """
    sim, fly = build_musculoskeletal_simulation(
        xml_path=xml_path, name=name, add_vision=add_vision
    )
    return ImitationEnv(
        sim,
        fly_name=fly.name,
        dataset=dataset,
        config=config or ImitationConfig(),
    )
