"""A moving fly with anatomical compound-eye visual input."""

import mujoco as mj

from flygym.compose.fly import NeuroMechFly
from flygym.compose.world import TetheredWorld
from flygym.utils.math import Rotation3D


def make_world(fly: NeuroMechFly) -> TetheredWorld:
    """Place a fly in a striped corridor with overhead illumination."""
    world = TetheredWorld()
    world.mjcf_root.worldbody.add_geom(
        name="floor",
        type=mj.mjtGeom.mjGEOM_PLANE,
        size=[10, 5, 0.1],
        rgba=[0.55, 0.55, 0.55, 1],
    )
    for i in range(14):
        for side in (-1, 1):
            value = 0.85 if i % 2 else 0.12
            world.mjcf_root.worldbody.add_geom(
                name=f"wall_{side}_{i}",
                type=mj.mjtGeom.mjGEOM_BOX,
                pos=[-1.5 + i * 0.6, side * 2, 1.2],
                size=[0.3, 0.05, 1.2],
                rgba=[value, value, value, 1],
            )
    world.add_fly(fly, (0, 0, 0), Rotation3D("quat", (1, 0, 0, 0)))
    world.mjcf_root.visual.headlight.active = False
    world.mjcf_root.visual.map.shadowclip = 10
    world.mjcf_root.worldbody.add_light(
        name="scene_light",
        type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        pos=[0, 0, 5],
        dir=[0, 0, -1],
        diffuse=[0.6, 0.6, 0.6],
        castshadow=True,
    )
    # Separate ambient fill keeps illumination inside native MuJoCo shadows.
    world.mjcf_root.worldbody.add_light(
        name="ambient_fill",
        type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
        dir=[0, 0, -1],
        ambient=[0.4, 0.4, 0.4],
        diffuse=[0, 0, 0],
        specular=[0, 0, 0],
        castshadow=False,
    )
    return world
