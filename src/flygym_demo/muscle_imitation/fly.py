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
    """Build FlyMimic's musculoskeletal `Simulation` and wrap it in an
    `ImitationEnv`.

    Args:
        xml_path: Musculoskeletal MJCF to load. Defaults to the bundled
            ``arm_damping_stiff`` muscle model.
        name: Logical fly name.
        add_vision: If True, attach eye cameras so `get_ommatidia_readouts`
            works (approximate; see `MusculoskeletalFly.add_vision`).
        config: `ImitationConfig` for reward weights, clip, etc.
        dataset: A `MoCapDataset`; defaults to the bundled clips.
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
