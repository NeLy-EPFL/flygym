from abc import ABC, abstractmethod
from collections.abc import Collection

import dm_control.mjcf as mjcf

from flygym.anatomy import ContactBodiesPreset, BodySegment
from flygym.compose._base import BaseCompositionElement
from flygym.compose.fly import BaseFly
from flygym.compose.physics import ContactParams
from flygym.utils.math import Rotation3D, Vec3, Vec4, Vec6


__all__ = ["BaseWorld", "FlatGroundWorld"]


class BaseWorld(BaseCompositionElement, ABC):
    @property
    @abstractmethod
    def fly_lookup(self) -> dict[str, BaseFly]:
        """Dictionary mapping fly names to fly instances in the world."""
        pass

    @abstractmethod
    def add_fly(
        self,
        fly: BaseFly,
        spawn_position: Vec3,
        spawn_rotation: Rotation3D | tuple[str, Vec3 | Vec4 | Vec6],
        **kwargs,
    ) -> None:
        """Adds a fly to the world at the specified position and orientation."""
        pass


class FlatGroundWorld(BaseWorld):
    def __init__(
        self,
        name: str = "flat_ground_world",
        *,
        half_size: float = 1000,
    ) -> None:
        self._mjcf_root = mjcf.RootElement(model=name)
        self._fly_lookup: dict[str, BaseFly] = {}

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
        ground_geom = self.mjcf_root.worldbody.add(
            "geom",
            type="plane",
            name="ground_plane",
            material=grid_material,
            pos=(0, 0, 0),
            size=(half_size, half_size, 1),
            contype=0,
            conaffinity=0,
        )
        self.ground_contact_geoms = [ground_geom]

    @property
    def fly_lookup(self) -> dict[str, BaseFly]:
        return self._fly_lookup
    
    @property
    def mjcf_root(self) -> mjcf.RootElement:
        return self._mjcf_root

    def add_fly(
        self,
        fly: BaseFly,
        spawn_position: Vec3 = (0, 0, 1),
        spawn_rotation: Rotation3D = Rotation3D("quat", (1, 0, 0, 0)),
        *,
        bodysegs_with_ground_contact: (
            Collection[BodySegment] | ContactBodiesPreset | str
        ) = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
        ground_contact_params: ContactParams = ContactParams(),
    ):
        if fly.name in self._fly_lookup:
            raise ValueError(f"Fly with name '{fly.name}' already exists in the world.")
        self._fly_lookup[fly.name] = fly

        self.bodysegs_with_ground_contact = bodysegs_with_ground_contact
        self.ground_contact_params = ground_contact_params

        if not isinstance(spawn_rotation, Rotation3D):
            spawn_rotation = Rotation3D(*spawn_rotation)

        spawn_site = self.mjcf_root.worldbody.add(
            "site",
            name=fly.name,
            pos=spawn_position,
            **spawn_rotation.as_kwargs(),
        )
        spawn_site.attach(fly.mjcf_root).add("freejoint", name=f"{fly.name}")

        self._enable_ground_contact(
            fly, bodysegs_with_ground_contact, ground_contact_params
        )

    def _enable_ground_contact(
        self,
        fly: BaseFly,
        bodysegs_with_ground_contact: (
            Collection[BodySegment] | ContactBodiesPreset | str
        ),
        ground_contact_params: ContactParams,
    ) -> None:
        if isinstance(bodysegs_with_ground_contact, ContactBodiesPreset | str):
            preset = ContactBodiesPreset(bodysegs_with_ground_contact)
            bodysegs_with_ground_contact = preset.to_body_segments_list()
            
        for i, ground_geom in enumerate(self.ground_contact_geoms):
            for body_segment in bodysegs_with_ground_contact:
                body_geom = fly.mjcf_root.find("geom", f"{body_segment.name}")
                ground_geom_name = (
                    f"ground{i}" if ground_geom.name is None else ground_geom.name
                )
                self.mjcf_root.contact.add(
                    "pair",
                    geom1=ground_geom,
                    geom2=body_geom,
                    name=f"{body_segment.name}-{ground_geom_name}",
                    friction=ground_contact_params.get_friction_tuple(),
                    solref=ground_contact_params.get_solref_tuple(),
                    solimp=ground_contact_params.get_solimp_tuple(),
                )
