from collections.abc import Collection


import dm_control.mjcf as mjcf

from flygym.compose.fly.anatomy import ContactBodiesPreset, BodySegment
from flygym.compose.base import BaseWorld, BaseFly
from flygym.compose.physics import ContactParams
from flygym.utils.math import Rotation3D, Vec3


class FlatGroundWorld(BaseWorld):
    def __init__(
        self,
        name: str = "flat_ground_world",
        *,
        half_size: float = 1000,
        group: int = 0,
        dclass: str = "world_geom",
    ) -> None:
        self.mjcf_model = mjcf.RootElement(model=name)
        self.mjcf_model.default.add("default", dclass=dclass)
        checker_texture = self.mjcf_model.asset.add(
            "texture",
            name="envtexture-checker",
            type="2d",
            builtin="checker",
            width=300,
            height=300,
            rgb1=(0.3, 0.3, 0.3),
            rgb2=(0.4, 0.4, 0.4),
        )
        grid = self.mjcf_model.asset.add(
            "material",
            name="envmaterial-grid",
            texture=checker_texture,
            texrepeat=(250, 250),
            reflectance=0.2,
        )
        ground_geom = self.mjcf_model.worldbody.add(
            "geom",
            type="plane",
            name="envgeom-ground_plane",
            group=group,
            dclass=dclass,
            material=grid,
            pos=(0, 0, 0),
            size=(half_size, half_size, 1),
            contype=0,
            conaffinity=0,
        )
        self.ground_contact_geoms = [ground_geom]
        self.flies: dict[str, BaseFly] = {}

    def spawn_fly(
        self,
        fly: BaseFly,
        spawn_position: Vec3 = (0, 0, 0.7),
        spawn_rotation: Rotation3D = Rotation3D("quat", (1, 0, 0, 0)),
        *,
        body_segments_with_ground_contact: (
            Collection[BodySegment] | ContactBodiesPreset | str
        ) = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
        ground_contact_params: ContactParams = ContactParams(),
    ):
        fly_mjcf_root = fly.get_mjcf_root()
        fly_name = fly_mjcf_root.model
        spawn_site = self.mjcf_model.worldbody.add(
            "site",
            name=f"spawnsite-{fly_name}",
            pos=spawn_position,
            **spawn_rotation.as_kwargs(),
        )
        spawn_site.attach(fly_mjcf_root).add("freejoint", name=f"freejoint-{fly_name}")
        self._enable_ground_contact(
            fly, body_segments_with_ground_contact, ground_contact_params
        )
        self.flies[fly_name] = fly

    def get_mjcf_root(self) -> mjcf.RootElement:
        return self.mjcf_model

    def _enable_ground_contact(
        self,
        fly: BaseFly,
        body_segments_with_ground_contact: (
            Collection[BodySegment] | ContactBodiesPreset | str
        ),
        ground_contact_params: ContactParams,
    ) -> None:
        if isinstance(body_segments_with_ground_contact, ContactBodiesPreset | str):
            preset = ContactBodiesPreset(body_segments_with_ground_contact)
            body_segments_with_ground_contact = preset.to_body_segments_list()
        fly_mjcf_root = fly.get_mjcf_root()
        for i, ground_geom in enumerate(self.ground_contact_geoms):
            for body_segment in body_segments_with_ground_contact:
                body_geom = fly_mjcf_root.find("geom", f"geom-{body_segment.name}")
                ground_geom_name = (
                    f"groundgeom-{i}" if ground_geom.name is None else ground_geom.name
                )
                self.mjcf_model.contact.add(
                    "pair",
                    geom1=ground_geom,
                    geom2=body_geom,
                    name=f"groundcontact-{body_segment.name}-{ground_geom_name}",
                    friction=ground_contact_params.get_friction_tuple(),
                    solref=ground_contact_params.get_solref_tuple(),
                    solimp=ground_contact_params.get_solimp_tuple(),
                )
