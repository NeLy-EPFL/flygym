import numpy as np
from typing import Optional, Callable
from dm_control import mjcf

from flygym.util import load_config
from .base import BaseArena


class OdorArena(BaseArena):
    """Flat terrain with an odor source.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001)
    num_sensors : int
        The number of odor sensors, by default 4: 2 antennae + 2 maxillary
        palps.
    odor_source : np.ndarray
        The position of the odor source in (x, y, z) coordinates. The shape
        of the array is (n_sources, 3).
    peak_odor_intensity : np.ndarray
        The peak intensity of the odor source. The shape of the array is
        (n_sources, n_dimensions). Note that the odor intensity can be
        multidimensional.
    num_odor_sources : int
        Number of odor sources.
    odor_dimensions : int
        Dimension of the odor space.
    diffuse_func : Callable
        The function that, given a distance from the odor source, returns
        the relative intensity of the odor. By default, this is an inverse
        square relationship.
    birdeye_cam : dm_control.mujoco.Camera
        MuJoCo camera that gives a birdeye view of the arena.
    birdeye_cam_zoom : dm_control.mujoco.Camera
         MuJoCo camera that gives a birdeye view of the arena, zoomed in
         toward the fly.

    Parameters
    ----------
    size : tuple[float, float], optional
        The size of the arena in mm, by default (300, 300).
    friction : tuple[float, float, float], optional
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    num_sensors : int, optional
        The number of odor sensors, by default 4: 2 antennae + 2 maxillary
        palps.
    odor_source : np.ndarray, optional
        The position of the odor source in (x, y, z) coordinates. The shape
        of the array is (n_sources, 3).
    peak_odor_intensity : np.ndarray, optional
        The peak intensity of the odor source. The shape of the array is
        (n_sources, n_dimensions). Note that the odor intensity can be
        multidimensional.
    diffuse_func : Callable, optional
        The function that, given a distance from the odor source, returns
        the relative intensity of the odor. By default, this is an inverse
        square relationship.
    marker_colors : list[tuple[float, float, float, float]], optional
        A list of n_sources RGBA values (each as a tuple) indicating the
        colors of the markers indicating the positions of the odor sources.
        The RGBA values should be given in the range [0, 1]. By default,
        the matplotlib color cycle is used.
    marker_size : float, optional
        The size of the odor source markers, by default 0.25.
    """

    def __init__(
        self,
        size: tuple[float, float] = (300, 300),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        num_sensors: int = 4,
        odor_source: np.ndarray = np.array([[10, 0, 0]]),
        peak_odor_intensity: np.ndarray = np.array([[1]]),
        diffuse_func: Callable = lambda x: x**-2,
        marker_colors: Optional[list[tuple[float, float, float, float]]] = None,
        marker_size: float = 0.25,
    ):
        super().__init__()

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
        self.friction = friction
        self.num_sensors = num_sensors
        self.odor_source = np.array(odor_source)
        self.peak_odor_intensity = np.array(peak_odor_intensity)
        self.num_odor_sources = self.odor_source.shape[0]
        if self.odor_source.shape[0] != self.peak_odor_intensity.shape[0]:
            raise ValueError(
                "Number of odor source locations and peak intensities must match."
            )
        self.diffuse_func = diffuse_func

        # Add birdeye camera
        self.birdeye_cam = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam",
            mode="fixed",
            pos=(self.odor_source[:, 0].max() / 2, 0, 35),
            euler=(0, 0, 0),
            fovy=45,
        )
        self.birdeye_cam_zoom = self.root_element.worldbody.add(
            "camera",
            name="birdeye_cam_zoom",
            mode="fixed",
            pos=(11, 0, 29),
            euler=(0, 0, 0),
            fovy=45,
        )

        # Add markers at the odor sources
        if marker_colors is None:
            color_cycle_rgb = load_config()["color_cycle_rgb"]
            marker_colors = []
            num_odor_sources = self.odor_source.shape[0]
            for i in range(num_odor_sources):
                rgb = np.array(color_cycle_rgb[i % num_odor_sources]) / 255
                rgba = (*rgb, 1)
                marker_colors.append(rgba)
        for i, (pos, rgba) in enumerate(zip(self.odor_source, marker_colors)):
            marker_body = self.root_element.worldbody.add(
                "body", name=f"odor_source_marker_{i}", pos=pos, mocap=True
            )
            marker_body.add(
                "geom", type="capsule", size=(marker_size, marker_size), rgba=rgba
            )

        # Reshape odor source and peak intensity arrays to simplify future calculations
        _odor_source_repeated = self.odor_source[:, np.newaxis, np.newaxis, :]
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.odor_dimensions, axis=1
        )
        _odor_source_repeated = np.repeat(
            _odor_source_repeated, self.num_sensors, axis=2
        )
        self._odor_source_repeated = _odor_source_repeated
        _peak_intensity_repeated = self.peak_odor_intensity[:, :, np.newaxis]
        _peak_intensity_repeated = np.repeat(
            _peak_intensity_repeated, self.num_sensors, axis=2
        )
        self._peak_intensity_repeated = _peak_intensity_repeated

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def get_olfaction(self, antennae_pos: np.ndarray) -> np.ndarray:
        """
        Notes
        -----
        w = 4: number of sensors (2x antennae + 2x max. palps)
        3: spatial dimensionality
        k: data dimensionality
        n: number of odor sources

        Input - odor source position: [n, 3]
        Input - sensor positions: [w, 3]
        Input - peak intensity: [n, k]
        Input - diffusion function: f(dist)

        Reshape sources to S = [n, k*, w*, 3] (* means repeated)
        Reshape sensor position to A = [n*, k*, w, 3] (* means repeated)
        Subtract, getting a Delta = [n, k, w, 3] array of rel difference
        Calculate Euclidean distance: D = [n, k, w]

        Apply pre-integrated diffusion function: S = f(D) -> [n, k, w]
        Reshape peak intensities to P = [n, k, w*]
        Apply scaling: I = P * S -> [n, k, w] element wise

        Output - Sum over the first axis: [k, w]
        """
        antennae_pos_repeated = antennae_pos[np.newaxis, np.newaxis, :, :]
        dist_3d = antennae_pos_repeated - self._odor_source_repeated  # (n, k, w, 3)
        dist_euc = np.linalg.norm(dist_3d, axis=3)  # (n, k, w)
        scaling = self.diffuse_func(dist_euc)  # (n, k, w)
        intensity = self._peak_intensity_repeated * scaling  # (n, k, w)
        return intensity.sum(axis=0)  # (k, w)

    @property
    def odor_dimensions(self) -> int:
        return self.peak_odor_intensity.shape[1]
