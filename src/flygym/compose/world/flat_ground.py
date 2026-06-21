from typing import override

from flygym.compose.world.base_world import _GroundContactMixin, BaseWorld

__all__ = ["FlatGroundWorld"]


class FlatGroundWorld(_GroundContactMixin, BaseWorld):
    """World with a flat infinite ground plane. Flies are free to move.

    When calling `add_fly`, the following extra keyword arguments are accepted:

    - ``bodysegs_with_ground_contact``: Body segments that collide with the ground.
      Accepts a `ContactBodiesPreset`, a preset string, or a collection of
      `BodySegment` objects. Default: ``ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD``.
    - ``ground_contact_params``: `ContactParams` for friction and contact physics.
      Default: ``ContactParams()``.
    - ``add_ground_contact_sensors``: If True, add contact force sensors for each leg.
      Default: ``True``.

    Args:
        name: Name of the world.
        half_size: Half-size of the ground plane in mm.
    """

    @override
    def __init__(
        self, name: str = "flat_ground_world", *, half_size: float = 1000
    ) -> None:
        super().__init__(name=name)

        checker_texture = self.mjcf_root.asset.add(
            "texture",
            name="checker",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.3, 0.3, 0.3),
            rgb2=(0.4, 0.4, 0.4),
        )
        grid_material = self.mjcf_root.asset.add(
            "material",
            name="grid",
            texture=checker_texture,
            texrepeat=(250, 250),
            reflectance=0.2,
        )
        self.ground_geom = self.mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="ground_plane",
            material=grid_material,
            pos=(0, 0, 0),
            size=(half_size, half_size, 1),
            contype=0,
            conaffinity=0,
        )
        self.ground_geoms = [self.ground_geom]
