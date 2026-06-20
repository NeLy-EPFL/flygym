"""
!!! warning "Experimental"

    Support for the FlyMimic musculoskeletal body model is
    **experimental**. The API may change in future releases, and only
    the left-front leg is muscle-driven in the current model. Not all
    features available for the default `NeuroMechFly` model are
    currently supported (e.g. per-leg ground-contact sensors).

World wrapping a self-contained `MusculoskeletalFly`.

FlyMimic's musculoskeletal MJCF already provides a floor, lighting, and an
anchored thorax, so — unlike the composable FlyGym worlds (`FlatGroundWorld`
et al.) — this world does not attach the fly into a separately-built scene. It
adopts the fly's MJCF root directly and presents the `BaseWorld` surface that
plain `flygym.Simulation` (CPU) consumes. `GPUSimulation` (``flygym.warp``) is
also supported but requires the ``[warp]`` extra and is not the default path.
"""

import dm_control.mjcf as mjcf

from flygym.compose.world.base_world import BaseWorld
from flygym.compose.fly.musculoskeletal import MusculoskeletalFly
from flygym.utils.math import Rotation3D, Vec3

__all__ = ["MusculoskeletalWorld"]


class MusculoskeletalWorld(BaseWorld):
    """`BaseWorld` wrapping a self-contained `MusculoskeletalFly`.

    Unlike the composable FlyGym worlds (`FlatGroundWorld` et al.), this
    world does *not* build a fresh scene and attach the fly into it.
    FlyMimic's musculoskeletal MJCF already provides a floor, lighting, and
    an anchored thorax, so `MusculoskeletalWorld` adopts the fly's MJCF root
    directly. Subclassing `BaseWorld` (rather than just duck-typing) keeps
    the type contract honest for the GPU path.

    Pair with `build_musculoskeletal_simulation` for the common case, or
    construct directly:

    ```python
    from flygym import Simulation
    from flygym.compose import MusculoskeletalFly, MusculoskeletalWorld

    fly = MusculoskeletalFly()
    world = MusculoskeletalWorld(fly)
    sim = Simulation(world)
    ```

    Args:
        fly: A `MusculoskeletalFly` whose self-contained MJCF root becomes
            the world's physics scene.

    Attributes:
        ground_geoms: The floor geom from FlyMimic's MJCF, if present (used
            by contact-force queries to filter ground hits).
        legpos_to_groundcontactsensors_by_fly: Always ``None`` — FlyMimic
            ships no per-leg ground-contact sensors.
        world_dof_neutral_states: Empty dict (no extra world DoFs).
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
