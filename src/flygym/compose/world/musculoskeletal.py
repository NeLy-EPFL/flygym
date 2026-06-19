"""World wrapping a self-contained `MusculoskeletalFly`.

FlyMimic's musculoskeletal MJCF already provides a floor, lighting, and an
anchored thorax, so — unlike the composable FlyGym worlds (`FlatGroundWorld`
et al.) — this world does not attach the fly into a separately-built scene. It
adopts the fly's MJCF root directly and presents the `BaseWorld` surface that
`Simulation` and `GPUSimulation` consume.
"""

import dm_control.mjcf as mjcf

from flygym.compose.world.base_world import BaseWorld
from flygym.compose.fly.musculoskeletal import MusculoskeletalFly
from flygym.utils.math import Rotation3D, Vec3

__all__ = ["MusculoskeletalWorld"]


class MusculoskeletalWorld(BaseWorld):
    """`BaseWorld` over a self-contained `MusculoskeletalFly`.

    Subclassing `BaseWorld` (rather than just duck-typing) keeps the type
    contract honest for the GPU path, which annotates ``world: BaseWorld``.
    """

    def __init__(self, fly: MusculoskeletalFly) -> None:
        # NOTE: intentionally skip super().__init__(): BaseWorld's initializer
        # builds a fresh empty scene (new root + skybox + neutral keyframe) and
        # expects flies to be added later via add_fly(). The musculoskeletal
        # model is a self-contained MJCF that already holds the fly, floor, and
        # lighting, so we adopt its root instead of constructing a new one.
        self._fly = fly
        self._mjcf_root = fly.mjcf_root
        self._fly_lookup = {fly.name: fly}
        # Expose the floor geom so contact-force queries can filter on ground.
        floor = fly.mjcf_root.find("geom", "floor")
        self.ground_geoms = [floor] if floor is not None else []
        # FlyMimic ships no per-leg ground-contact sensors.
        self.legpos_to_groundcontactsensors_by_fly = None
        self.world_dof_neutral_states: dict[str, list[float]] = {}

    def _attach_fly_mjcf(
        self,
        fly: MusculoskeletalFly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D,
        *args,
        **kwargs,
    ) -> mjcf.Element:
        raise NotImplementedError(
            "MusculoskeletalWorld wraps a self-contained musculoskeletal MJCF "
            "in which the fly is already present; add_fly()/attachment is not "
            "supported. Pass a MusculoskeletalFly at construction or use "
            "build_musculoskeletal_simulation()."
        )
