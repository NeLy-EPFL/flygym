import numpy as np
from typing import Union, Optional, Callable

from flygym import Fly
from flygym.arena import BaseArena, Tethered


class MovingObjArena(BaseArena):
    """Flat terrain with a hovering moving object.

    Attributes
    ----------
    ball_pos : tuple[float,float,float]
        The position of the floating object in the arena.

    Parameters
    ----------
    size : tuple[int, int]
        The size of the terrain in (x, y) dimensions.
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    obj_radius : float
        Radius of the spherical floating object in mm.
    init_ball_pos : tuple[float,float]
        Initial position of the object, by default (5, 0).
    move_speed : float
        Speed of the moving object. By default 10.
    move_direction : str
        Which way the ball moves toward first. Can be "left", "right", or
        "random". By default "right".
    lateral_magnitude : float
        Magnitude of the lateral movement of the object as a multiplier of
        forward velocity. For example, when ``lateral_magnitude`` is 1, the
        object moves at a heading (1, 1) when its movement is the most
        lateral. By default 2.
    """

    def __init__(
        self,
        size=(300, 300),
        friction=(1, 0.005, 0.0001),
        obj_radius=1,
        init_ball_pos=(5, 0),
        move_speed=10,
        move_direction="right",
        lateral_magnitude=2,
    ):
        super().__init__()

        self.init_ball_pos = (*init_ball_pos, obj_radius)
        self.ball_pos = np.array(self.init_ball_pos, dtype="float32")
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
        ground_size = [*size, 1]
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
        self.root_element.worldbody.add("body", name="b_plane")

        # Add ball
        obstacle = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        self.root_element.worldbody.add(
            "body", name="ball_mocap", mocap=True, pos=self.ball_pos, gravcomp=1
        )
        self.object_body = self.root_element.find("body", "ball_mocap")
        self.object_body.add(
            "geom",
            name="ball",
            type="sphere",
            size=(obj_radius, obj_radius),
            rgba=(0.0, 0.0, 0.0, 1),
            material=obstacle,
        )

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
        self.ball_pos[:2] += self.move_speed * heading_vec * dt
        physics.bind(self.object_body).mocap_pos = self.ball_pos
        self.curr_time += dt

    def reset(self, physics):
        if self.move_direction == "random":
            self.y_mult = np.random.choice([-1, 1])
        self.curr_time = 0
        self.ball_pos = np.array(self.init_ball_pos, dtype="float32")
        physics.bind(self.object_body).mocap_pos = self.ball_pos


class MovingFlyArena(BaseArena):
    """Flat terrain with a hovering moving fly.

    Attributes
    ----------
    fly_pos : tuple[float,float,float]
        The position of the floating fly in the arena.

    Parameters
    ----------
    terrain_type : str
        Type of terrain. Can be "flat" or "blocks". By default "flat".
    x_range : tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 35).
    y_range : tuple[float, float], optional
        Same as above in y, by default (-20, 20).
    block_size : float, optional
        The side length of the rectangular blocks forming the terrain in
        mm, by default 1.3.
    height_range : tuple[float, float], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.2, 0.2).
    rand_seed : int, optional
        Seed for generating random block heights, by default 0.
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    init_fly_pos : tuple[float,float]
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
        x_range: Optional[tuple[float, float]] = (-10, 20),
        y_range: Optional[tuple[float, float]] = (-20, 20),
        block_size: Optional[float] = 1.3,
        height_range: Optional[tuple[float, float]] = (0.2, 0.2),
        rand_seed: int = 0,
        ground_alpha: float = 1,
        friction=(1, 0.005, 0.0001),
        leading_fly_height=0.5,
        init_fly_pos=(5, 0),
        move_speed=10,
        radius=10,
    ):
        super().__init__()
        self.init_fly_pos = (*init_fly_pos, leading_fly_height)
        self.fly_pos = np.array(self.init_fly_pos, dtype="float32")
        self.friction = friction
        self.move_speed = move_speed
        self.angular_speed = move_speed / radius
        self.curr_time = 0
        self.radius = radius
        self.terrain_type = terrain_type

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

        self.fly_pos = np.array(
            [
                self.radius * np.sin(0),
                self.radius * np.cos(0),
                self.init_fly_pos[2],
            ]
        )
        curr_pos = complex(*self.fly_pos[:2])
        q = np.exp(1j * np.angle(curr_pos - self._prev_pos) / 2)
        self._prev_pos = curr_pos

        spawn_site = self.root_element.worldbody.add(
            "site",
            pos=self.fly_pos,
            quat=(q.real, 0, 0, q.imag),
        )
        self.freejoint = spawn_site.attach(fly).add("freejoint")

        # Add camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(5, 0, 35),
            euler=(0, 0, 0),
            fovy=45,
        )
        self.birdeye_cam_zoom = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam_zoom",
            mode="fixed",
            pos=(5, 0, 20),
            euler=(0, 0, 0),
            fovy=45,
        )

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

    def step(self, dt, physics):
        theta = self.angular_speed * self.curr_time
        self.fly_pos = np.array(
            [
                self.radius * np.sin(theta),
                self.radius * np.cos(theta),
                self.init_fly_pos[2],
            ]
        )
        curr_pos = complex(*self.fly_pos[:2])
        q = np.exp(1j * np.angle(curr_pos - self._prev_pos) / 2)
        qpos = (*self.fly_pos, q.real, 0, 0, q.imag)
        physics.bind(self.freejoint).qpos = qpos
        physics.bind(self.freejoint).qvel[:] = 0
        self._prev_pos = curr_pos

        self.curr_time += dt

    def reset(self, physics):
        self._prev_pos = complex(*self.init_fly_pos[:2])

        if self.move_direction == "random":
            self.y_mult = np.random.choice([-1, 1])
        self.curr_time = 0
        self.fly_pos = np.array(self.init_fly_pos, dtype="float32")
        physics.bind(self.object_body).mocap_pos = self.fly_pos


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
        visual_angle : tuple[float, float]
            Width and height of the cylinder in degrees.
        distance : float
            Distance from the center of the arena to the center of the
            cylinders.
        rgba : tuple[float, float, float, float]
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


class ObstacleOdorArena(BaseArena):
    num_sensors = 4

    def __init__(
        self,
        terrain: BaseArena,
        obstacle_positions: np.ndarray = np.array([(7.5, 0), (12.5, 5), (17.5, -5)]),
        obstacle_colors: Union[np.ndarray, tuple] = (0, 0, 0, 1),
        obstacle_radius: float = 1,
        obstacle_height: float = 4,
        odor_source: np.ndarray = np.array([[25, 0, 2]]),
        peak_odor_intensity: np.ndarray = np.array([[1]]),
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: Optional[list[tuple[float, float, float, float]]] = None,
        marker_size: float = 0.1,
        user_camera_settings: Optional[
            tuple[tuple[float, float, float], tuple[float, float, float], float]
        ] = None,
    ):
        self.terrain_arena = terrain
        self.obstacle_positions = obstacle_positions
        self.root_element = terrain.root_element
        self.friction = terrain.friction
        self.obstacle_radius = obstacle_radius
        z_offset = terrain.get_spawn_position(np.zeros(3), np.zeros(3))[0][2]
        obstacle_colors = np.array(obstacle_colors)
        if obstacle_colors.shape == (4,):
            obstacle_colors = np.array(
                [obstacle_colors for _ in range(obstacle_positions.shape[0])]
            )
        else:
            assert obstacle_colors.shape == (obstacle_positions.shape[0], 4)

        self.odor_source = np.array(odor_source)
        self.peak_odor_intensity = np.array(peak_odor_intensity)
        self.num_odor_sources = self.odor_source.shape[0]
        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )
        self.odor_dim = self.peak_odor_intensity.shape[1]
        self.diffuse_func = diffuse_func

        # Add markers at the odor sources
        if marker_colors is None:
            rgb = np.array([255, 127, 14]) / 255
            marker_colors = [(*rgb, 1)] * self.num_odor_sources
        self.marker_colors = marker_colors
        self._odor_marker_geoms = []
        for i, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors)):
            pos = list(pos)
            pos[2] += z_offset
            marker_body = self.root_element.worldbody.add(
                "body", name=f"odor_source_marker_{i}", pos=pos, mocap=True
            )
            geom = marker_body.add(
                "geom", type="capsule", size=(marker_size, marker_size), rgba=rgba
            )
            self._odor_marker_geoms.append(geom)

        # Reshape odor source and peak intensity arrays to simplify future calculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(_odor_source_repeated, self.odor_dim, axis=1)
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.num_sensors, axis=2
        )
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(
            _peak_intensity_repeated, self.num_sensors, axis=2
        )
        self._peak_intensity_repeated = _peak_intensity_repeated

        # Add obstacles
        self.obstacle_bodies = []
        obstacle_material = self.root_element.asset.add(
            "material", name="obstacle", reflectance=0.1
        )
        self.obstacle_z_pos = z_offset + obstacle_height / 2
        for i in range(obstacle_positions.shape[0]):
            obstacle_pos = [*obstacle_positions[i, :], self.obstacle_z_pos]
            obstacle_color = obstacle_colors[i]
            obstacle_body = self.root_element.worldbody.add(
                "body", name=f"obstacle_{i}", mocap=True, pos=obstacle_pos
            )
            self.obstacle_bodies.append(obstacle_body)
            obstacle_body.add(
                "geom",
                type="cylinder",
                size=(obstacle_radius, obstacle_height / 2),
                rgba=obstacle_color,
                material=obstacle_material,
            )

        # Add monitor cameras
        self.side_cam = self.root_element.worldbody.add(
            "camera",
            name="side_cam",
            mode="fixed",
            pos=(odor_source[0, 0] / 2, -25, 10),
            euler=(np.deg2rad(75), 0, 0),
            fovy=50,
        )
        self.back_cam = self.root_element.worldbody.add(
            "camera",
            name="back_cam",
            mode="fixed",
            pos=(-9, 0, 7),
            euler=(np.deg2rad(60), 0, -np.deg2rad(90)),
            fovy=55,
        )
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(7.5, 0, 25),
            euler=(0, 0, 0),
            fovy=45,
        )
        self.birdeye_cam_origin = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam_origin",
            mode="fixed",
            pos=(0, 0, 40),
            euler=(0, 0, 0),
            fovy=50,
        )
        if user_camera_settings is not None:
            cam_pos, cam_euler, cam_fovy = user_camera_settings
            self.root_element.worldbody.add(
                "camera",
                name="user_cam",
                mode="fixed",
                pos=cam_pos,
                euler=cam_euler,
                fovy=cam_fovy,
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.terrain_arena.get_spawn_position(rel_pos, rel_angle)

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, w, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, w)
        scaling = self.diffuse_func(dist_euc)  # (n, k, w)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, w)
        return intensity.sum(axis=0)  # (k, w)

    def pre_visual_render_hook(self, physics):
        for geom, rgba in zip(self._odor_marker_geoms, self.marker_colors):
            physics.bind(geom).rgba = np.array([*rgba[:3], 0])

    def post_visual_render_hook(self, physics):
        for geom, rgba in zip(self._odor_marker_geoms, self.marker_colors):
            physics.bind(geom).rgba = np.array([*rgba[:3], 1])
