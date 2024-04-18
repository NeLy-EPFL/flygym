from flygym.arena import BaseArena
from flygym.arena.tethered import Tethered
from flygym import Fly
import numpy as np
from typing import Tuple, Optional, Callable


class MovingFlyArena(BaseArena):
    """Flat terrain with a hovering moving fly.

    Attributes
    ----------
    fly_pos : Tuple[float,float,float]
        The position of the floating fly in the arena.

    Parameters
    ----------
    terrain_type : str
        Type of terrain. Can be "flat" or "blocks". By default "flat".
    x_range : Tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 35).
    y_range : Tuple[float, float], optional
        Same as above in y, by default (-20, 20).
    block_size : float, optional
        The side length of the rectangular blocks forming the terrain in
        mm, by default 1.3.
    height_range : Tuple[float, float], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.2, 0.2).
    rand_seed : int, optional
        Seed for generating random block heights, by default 0.
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    friction : Tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    init_fly_pos : Tuple[float,float]
        Initial position of the fly, by default (5, 0).
    move_speed : float
        Speed of the moving fly. By default 10.
    move_direction : str
        Which way the fly moves toward first. Can be "left", "right", or
        "random". By default "right".
    lateral_magnitude : float
        Magnitude of the lateral movement of the fly as a multiplier of
        forward velocity. For example, when ``lateral_magnitude`` is 1, the
        fly moves at a heading (1, 1) when its movement is the most
        lateral. By default 2.
    """

    def __init__(
        self,
        terrain_type: str = "flat",
        x_range: Optional[Tuple[float, float]] = (-10, 35),
        y_range: Optional[Tuple[float, float]] = (-20, 20),
        block_size: Optional[float] = 1.3,
        height_range: Optional[Tuple[float, float]] = (0.2, 0.2),
        rand_seed: int = 0,
        ground_alpha: float = 1,
        friction=(1, 0.005, 0.0001),
        leading_fly_height=1,
        init_fly_pos=(5, 0),
        move_speed=6,
        move_direction="right",
        lateral_magnitude=2,
    ):
        super().__init__()
        self.init_fly_pos = (*init_fly_pos, leading_fly_height)
        self.fly_pos = np.array(self.init_fly_pos, dtype="float32")
        self.friction = friction
        self.move_speed = move_speed
        self.curr_time = 0
        self.move_direction = move_direction
        self.lateral_magnitude = lateral_magnitude
        if move_direction == "left":
            self.y_mult = 1
        elif move_direction == "right":
            self.y_mult = -1
        elif move_direction == "random":
            self.y_mult = np.random.choice([-1, 1])
        else:
            raise ValueError("Invalid move_direction")

        # Add ground
        if terrain_type == "flat":
            ground_size = [300, 300, 1]
            chequered = self.root_element.asset.add(
                "texture",
                type="2d",
                builtin="checker",
                width=300,
                height=300,
                rgb1=(0.4, 0.4, 0.4),
                rgb2=(0.5, 0.5, 0.5),
            )
            grid = self.root_element.asset.add(
                "material",
                name="grid",
                texture=chequered,
                texrepeat=(60, 60),
                reflectance=0.1,
            )
            self.root_element.worldbody.add(
                "geom",
                type="plane",
                name="ground",
                material=grid,
                size=ground_size,
                friction=friction,
            )
        elif terrain_type == "blocks":
            self.x_range = x_range
            self.y_range = y_range
            self.block_size = block_size
            self.height_range = height_range
            rand_state = np.random.RandomState(rand_seed)

            x_centers = np.arange(x_range[0] + block_size / 2, x_range[1], block_size)
            y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
            for i, x_pos in enumerate(x_centers):
                for j, y_pos in enumerate(y_centers):
                    is_i_odd = i % 2 == 1
                    is_j_odd = j % 2 == 1

                    if is_i_odd != is_j_odd:
                        height = 0.1
                    else:
                        height = 0.1 + rand_state.uniform(*height_range)

                    self.root_element.worldbody.add(
                        "geom",
                        type="box",
                        size=(
                            block_size / 2 + 0.1 * block_size / 2,
                            block_size / 2 + 0.1 * block_size / 2,
                            height / 2 + block_size / 2,
                        ),
                        pos=(
                            x_pos,
                            y_pos,
                            height / 2 - block_size / 2,
                        ),
                        rgba=(0.3, 0.3, 0.3, ground_alpha),
                        friction=friction,
                    )

            self.root_element.worldbody.add("body", name="base_plane")
        else:
            raise ValueError(f"Invalid terrain '{terrain_type}'")

        # Add fly
        self._prev_pos = complex(*self.init_fly_pos[:2])

        fly = Fly().model
        fly.model = "Animat_2"

        for light in fly.find_all(namespace="light"):
            light.remove()

        spawn_site = self.root_element.worldbody.add(
            "site",
            pos=self.fly_pos,
            euler=(0, 0, 0),
        )
        self.freejoint = spawn_site.attach(fly).add("freejoint")

        # Add camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(15, 0, 35),
            euler=(0, 0, 0),
            fovy=45,
        )
        self.birdeye_cam_zoom = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam_zoom",
            mode="fixed",
            pos=(15, 0, 20),
            euler=(0, 0, 0),
            fovy=45,
        )

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def step(self, dt, physics):
        heading_vec = np.array(
            [1, self.lateral_magnitude * np.cos(self.curr_time * 3) * self.y_mult]
        )
        heading_vec /= np.linalg.norm(heading_vec)
        self.fly_pos[:2] += self.move_speed * heading_vec * dt

        curr_pos = complex(*self.fly_pos[:2])
        q = np.exp(1j * np.angle(curr_pos - self._prev_pos) / 2)
        qpos = (*self.fly_pos, q.real, 0, 0, q.imag)
        physics.bind(self.freejoint).qpos = qpos
        self._prev_pos = curr_pos

        self.curr_time += dt

    def reset(self, physics):
        self._prev_pos = complex(*self.init_fly_pos[:2])

        if self.move_direction == "random":
            self.y_mult = np.random.choice([-1, 1])
        self.curr_time = 0
        self.fly_pos = np.array(self.init_fly_pos, dtype="float32")
        physics.bind(self.object_body).mocap_pos = self.fly_pos


def get_azimuth_func(start_angle=-180, end_angle=180, duration=1, start_time=0):
    """Returns a function that takes time as input and returns the azimuth angle of a
    moving object.

    Parameters
    ----------
    start_angle : float
        Starting azimuth angle of the moving object.
    end_angle : float
        Ending azimuth angle of the moving object.
    duration : float
        Duration of the movement.
    start_time : float
        Start time of the movement.
    """

    def func(t):
        t = t - start_time
        if t < 0:
            return start_angle
        elif t > duration:
            return end_angle
        else:
            return start_angle + (end_angle - start_angle) * t / duration

    return func


class MovingBarArena(Tethered):
    def __init__(
        self,
        azimuth_func: Callable[[float], float],
        visual_angle=(10, 60),
        distance=12,
        rgba=(0, 0, 0, 1),
        **kwargs,
    ):
        """Flat or blocks terrain with a moving cylinder to simulate a
        moving bar on a circular screen.

        Parameters
        ----------
        azimuth_func : Callable[[float], float]
            Function that takes time as input and returns the azimuth angle
            of the cylinder.
        visual_angle : Tuple[float, float]
            Width and height of the cylinder in degrees.
        distance : float
            Distance from the center of the arena to the center of the
            cylinders.
        rgba : Tuple[float, float, float, float]
            Color of the cylinder.
        kwargs : dict
            Additional arguments to passed to the superclass.
        """
        super().__init__(**kwargs)

        self.azimuth_func = azimuth_func
        self.distance = distance
        self.curr_time = 0

        cylinder_material = self.root_element.asset.add(
            "material", name="cylinder", reflectance=0.1
        )

        radius = 2 * distance * np.tan(np.deg2rad(visual_angle[0] / 2))
        half_height = distance * np.tan(np.deg2rad(visual_angle[1] / 2))

        self.cylinder = self.root_element.worldbody.add(
            "body",
            name="cylinder",
            mocap=True,
            pos=self.get_pos(0),
        )

        self.cylinder.add(
            "geom",
            type="cylinder",
            size=(radius, half_height),
            rgba=rgba,
            material=cylinder_material,
        )

        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(0, 0, 25),
            euler=(0, 0, 0),
            fovy=45,
        )

    def reset(self, physics):
        """Resets the position of the cylinder."""
        self.curr_time = 0
        physics.bind(self.cylinder).mocap_pos = self.get_pos(0)

    def get_pos(self, t):
        """Returns the position of the cylinder at time t."""
        angle = np.deg2rad(self.azimuth_func(t))
        x = self.distance * np.cos(angle)
        y = self.distance * np.sin(angle)
        return x, y, 0

    def step(self, dt, physics):
        """Updates the position of the cylinder."""
        self.curr_time += dt
        physics.bind(self.cylinder).mocap_pos = self.get_pos(self.curr_time)
