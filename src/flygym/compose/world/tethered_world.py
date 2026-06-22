from typing import override


from flygym.compose.fly import BaseFly
from flygym.utils.math import Vec3, Rotation3D
from flygym.compose.world.base_world import BaseWorld

__all__ = [
    "TetheredWorld",
]


class TetheredWorld(BaseWorld):
    """World where the fly body is rigidly fixed in space.

    The root segment is attached to a static spawn site (no free joint) and held as a
    mocap body, so the fly's body stays put while its appendages (legs, wings, etc.)
    can still move. Useful for motor control experiments without locomotion.

    Args:
        name: Name of the world.
    """

    @override
    def __init__(self, name: str = "tethered_world") -> None:
        super().__init__(name=name)

    @override
    def _attach_fly_mjcf(
        self, fly: BaseFly, spawn_position: Vec3, spawn_rotation: Rotation3D
    ) -> set[str]:
        spawn_site = self.mjcf_root.worldbody.add_site(
            name=fly.name, pos=spawn_position, **spawn_rotation.as_kwargs()
        )
        self.mjcf_root.attach(fly.mjcf_root, prefix=f"{fly.name}/", site=spawn_site)

        # Hold the root segment fixed as a mocap body rather than via a free joint. With
        # no joint the root would be a static body that the `fusestatic` optimization
        # merges into the worldbody, which breaks the tracking camera parented to it (the
        # camera would track the worldbody and mis-place itself relative to the fly). A
        # mocap body is never fused and is held rigidly in place at zero extra DoFs --
        # exactly the tethered behavior -- while every other jointless segment is still
        # fused for performance. (A free joint + weld equality would also avoid fusing,
        # but the weld is a soft constraint: the body visibly drifts under the leg
        # reaction forces unless its stiffness is tuned, so mocap is the simpler choice.)
        fly.bodyseg_to_mjcfbody[fly.root_segment].mocap = True
        return set()
