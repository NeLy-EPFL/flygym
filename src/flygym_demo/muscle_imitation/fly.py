"""Factory that builds the muscle Simulation + an imitation env."""

from __future__ import annotations

from os import PathLike

from flygym.imitation import ImitationConfig, ImitationEnv, MoCapDataset
from flygym.muscle import DEFAULT_MUSCLE_XML, build_muscle_simulation


def make_imitation_env(
    *,
    xml_path: PathLike = DEFAULT_MUSCLE_XML,
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
            works (approximate; see `MuscleFly.add_vision`).
        config: `ImitationConfig` for reward weights, clip, etc.
        dataset: A `MoCapDataset`; defaults to the bundled clips.
    """
    sim, fly = build_muscle_simulation(
        xml_path=xml_path, name=name, add_vision=add_vision
    )
    return ImitationEnv(
        sim,
        fly_name=fly.name,
        dataset=dataset,
        config=config or ImitationConfig(),
    )
