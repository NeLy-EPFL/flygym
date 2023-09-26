import numpy as np
import numba as nb
from typing import Optional

from flygym.common import get_data_path
from flygym.mujoco.util import load_config


class Retina:
    """
    This class handles the simulation of the fly's visual input.
    Calculation in this class is vectorized and parallelized using Numba.

    Attributes
    ----------
    ommatidia_id_map : np.ndarray
        Integer NumPy array of shape (nrows, ncols) where the value
        indicates the ID of the ommatidium (starting from 1). 0 indicates
        background (outside the hex lattice).
    num_pixels_per_ommatidia : np.ndarray
        Integer NumPy array of shape (max(ommatidia_id_map),) where the
        value of each element indicates the number of raw pixels covered
        within each ommatidia.
    pale_type_mask : np.ndarray
        Integer NumPy array of shape (max(ommatidia_id_map),) where the
        value of each element indicates whether the ommatidia is pale-type
        (1) or yellow-type (0).
    distortion_coefficient : float
        A coefficient determining the extent of fisheye effect applied to
        the raw MuJoCo camera images.
    zoom : float
        A coefficient determining the zoom level when the fisheye effect is
        applied.
    nrows : int
        The number of rows in the raw image rendered by the MuJoCo camera.
    ncols : int
        The number of columns in the raw image rendered by the MuJoCo
        camera.

    Parameters
    ----------
    ommatidia_id_map : np.ndarray
        Integer NumPy array of shape (nrows, ncols) where the value
        indicates the ID of the ommatidium (starting from 1). 0 indicates
        background (outside the hex lattice). By default, the map indicated
        in the configuration file is loaded.
    pale_type_mask : np.ndarray
        Integer NumPy array of shape (max(ommatidia_id_map),) where the
        value of each element indicates whether the ommatidia is pale-type
        (1) or yellow-type (0). By default, the mask indicated in the
        configuration file is used.
    distortion_coefficient : float
        A coefficient determining the extent of fisheye effect applied to
        the raw MuJoCo camera images. By default, the value indicated in
        the configuration file is used.
    zoom : float
        A coefficient determining the zoom level when the fisheye effect is
        applied. By default, the value indicated in the configuration file
        is used.
    nrows : int
        The number of rows in the raw image rendered by the MuJoCo camera.
        By default, the value indicated in the configuration file is used.
    ncols : int
        The number of columns in the raw image rendered by the MuJoCo
        camera. By default, the value used in the configuration file is
        used.
    """

    def __init__(
        self,
        ommatidia_id_map: Optional[np.ndarray] = None,
        pale_type_mask: Optional[np.ndarray] = None,
        distortion_coefficient: Optional[float] = None,
        zoom: Optional[float] = None,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
    ) -> None:
        # Load parameters from config file if not supplied
        config = load_config()
        data_path = get_data_path("flygym", "data")
        ommatidia_id_map_path = data_path / config["paths"]["ommatidia_id_map"]
        pale_type_mask_path = data_path / config["paths"]["canonical_pale_type_mask"]
        vision_config = config["vision"]
        if ommatidia_id_map is None:
            ommatidia_id_map = np.load(ommatidia_id_map_path)
        if pale_type_mask is None:
            pale_type_mask = np.load(pale_type_mask_path).astype(int)
        if distortion_coefficient is None:
            distortion_coefficient = vision_config["fisheye_distortion_coefficient"]
        if zoom is None:
            zoom = vision_config["fisheye_zoom"]
        if nrows is None:
            nrows = vision_config["raw_img_height_px"]
        if ncols is None:
            ncols = vision_config["raw_img_width_px"]

        self.ommatidia_id_map = ommatidia_id_map
        _unique_count = np.unique(ommatidia_id_map, return_counts=True)
        self.num_pixels_per_ommatidia = _unique_count[1][1:]
        self.pale_type_mask = pale_type_mask
        self.distortion_coefficient = distortion_coefficient
        self.zoom = zoom
        self.nrows = nrows
        self.ncols = ncols

    def raw_image_to_hex_pxls(self, raw_img: np.ndarray) -> np.ndarray:
        """Given a raw image from an eye (one camera), simulate what the
        fly would see.

        Parameters
        ----------
        raw_img : np.ndarray
            RGB image with the shape (H, W, 3) returned by the camera.

        Returns
        -------
        np.ndarray
            Our simulation of what the fly might see through its compound
            eyes. It is a (N, 2) array where the first dimension is for the
            N ommatidia, and the third dimension is for the two channels.
        """
        return self._raw_image_to_hex_pxls(
            raw_img,
            self.ommatidia_id_map,
            self.num_pixels_per_ommatidia,
            self.pale_type_mask,
        )

    def hex_pxls_to_human_readable(self, ommatidia_reading: np.ndarray) -> np.ndarray:
        """Given the intensity readings for all ommatidia in one eye,
        convert them to an (nrows, ncols) image with hexagonal blocks that
        can be visualized as a human-readable image.

        Parameters
        ----------
        ommatidia_reading : np.ndarray
            Our simulation of what the fly might see through its compound
            eyes. It is a (N, 2) array where the first dimension is for the
            N ommatidia, and the third dimension is for the two channels.

        Returns
        -------
        np.ndarray
            An (nrows, ncols) grayscale image with hexagonal blocks that
            can be visualized as a human-readable image.
        """
        return self._hex_pxls_to_human_readable(
            ommatidia_reading, self.ommatidia_id_map
        )

    def correct_fisheye(self, img: np.ndarray) -> np.ndarray:
        """
        The raw imaged rendered by the MuJoCo camera is rectilinear. This
        distorts the image and overrepresents the periphery of the field of
        view (the same angle near the periphery is reflected by a greater
        angle in the rendered image). This method applies a fisheye effect
        to make the same angle represented roughly equally anywhere within
        the field of view.

        Notes
        -----
        This implementation is based on https://github.com/Gil-Mor/iFish,
        MIT License.

        Parameters
        ----------
        img: np.ndarray
            The raw MuJoCo camera rendering as a NumPy array of shape
            (nrows, ncols, 3).

        Returns
        -------
        np.ndarray
            The corrected camera rendering as a NumPy array of shape
            (nrows, ncols, 3).

        """
        return self._correct_fisheye(
            img, self.nrows, self.ncols, self.zoom, self.distortion_coefficient
        )

    @staticmethod
    @nb.njit(parallel=True)
    def _raw_image_to_hex_pxls(
        raw_img, ommatidia_id_map, num_pixels_per_ommatidia, pale_type_mask
    ):
        vals = np.zeros((len(num_pixels_per_ommatidia), 2))
        img_arr_flat = raw_img.reshape((-1, 3))
        hex_id_map_flat = ommatidia_id_map.flatten()
        for i in nb.prange(hex_id_map_flat.size):
            hex_pxl_id = hex_id_map_flat[i] - 1
            hex_pxl_size = num_pixels_per_ommatidia[hex_pxl_id]  # num raw pxls
            if hex_pxl_id != -1:
                ch_idx = pale_type_mask[hex_pxl_id]
                vals[hex_pxl_id, ch_idx] += img_arr_flat[i, ch_idx + 1] / hex_pxl_size
        return vals / 255

    @staticmethod
    @nb.njit(parallel=True)
    def _hex_pxls_to_human_readable(ommatidia_reading, ommatidia_id_map):
        ommatidia_reading_scaled = ommatidia_reading * 255
        processed_image_flat = np.zeros(ommatidia_id_map.size, dtype=np.uint8) + 255
        hex_id_map_flat = ommatidia_id_map.flatten().astype(np.int16)
        for i in nb.prange(hex_id_map_flat.size):
            hex_pxl_id = hex_id_map_flat[i] - 1
            if hex_pxl_id != -1:
                hex_pxl_val = ommatidia_reading_scaled[hex_pxl_id, :].max()
                processed_image_flat[i] = hex_pxl_val
        return processed_image_flat.reshape(ommatidia_id_map.shape)

    @staticmethod
    @nb.njit(parallel=True)
    def _correct_fisheye(img, nrows, ncols, zoom, distortion_coefficient):
        """Based on https://github.com/Gil-Mor/iFish, MIT License."""
        dst_img = np.zeros((nrows, ncols, 3), dtype="uint8")

        # easier to calculate if we traverse x, y in dst image
        for dst_row in nb.prange(nrows):
            for dst_col in nb.prange(ncols):
                # normalize row and col to be in interval of [-1, 1] and apply zoom
                dst_row_norm = ((2 * dst_row - nrows) / nrows) / zoom
                dst_col_norm = ((2 * dst_col - ncols) / ncols) / zoom

                # get normalized row and col dist from center, +1e-6 to avoid div by 0
                dst_radius_norm = np.sqrt(dst_col_norm**2 + dst_row_norm**2)
                denom = 1 - (distortion_coefficient * (dst_radius_norm**2)) + 1e-6
                src_row_norm = dst_row_norm / denom
                src_col_norm = dst_col_norm / denom

                # convert the normalized distorted row and col back to image pixels
                src_row = int(((src_row_norm + 1) * nrows) / 2)
                src_col = int(((src_col_norm + 1) * ncols) / 2)

                # if new pixel is in bounds copy from source pixel to destination pixel
                if (
                    0 <= src_row
                    and src_row < nrows
                    and 0 <= src_col
                    and src_col < ncols
                ):
                    dst_img[dst_row][dst_col] = img[src_row][src_col]

        return dst_img
