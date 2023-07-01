import numpy as np
import numba as nb
from flygym.util.data import ommatidia_id_map_path

ommatidia_id_map = np.load(ommatidia_id_map_path)
num_pixels_per_ommatidia = np.unique(ommatidia_id_map, return_counts=True)[1][1:]


@nb.njit(parallel=True)
def raw_image_to_hex_pxls(img_arr, num_pixels_per_ommatidia, ommatidia_id_map):
    vals = np.zeros((len(num_pixels_per_ommatidia), 2))
    img_arr_flat = img_arr.reshape((-1, 3))
    hex_id_map_flat = ommatidia_id_map.flatten()
    for i in nb.prange(hex_id_map_flat.size):
        hex_pxl_id = hex_id_map_flat[i] - 1
        if hex_pxl_id != -1:
            ch_idx = hex_pxl_id % 2
            vals[hex_pxl_id, ch_idx] += (
                img_arr_flat[i, ch_idx + 1] / num_pixels_per_ommatidia[hex_pxl_id]
            )
    return vals


@nb.njit(parallel=True)
def hex_pxls_to_human_readable(vals, ommatidia_id_map):
    processed_image_flat = np.zeros(ommatidia_id_map.size, dtype=np.uint8) + 255
    hex_id_map_flat = ommatidia_id_map.flatten().astype(np.int16)
    for i in nb.prange(hex_id_map_flat.size):
        hex_pxl_id = hex_id_map_flat[i] - 1
        if hex_pxl_id != -1:
            hex_pxl_val = vals[hex_pxl_id, :].max()
            processed_image_flat[i] = hex_pxl_val
    return processed_image_flat.reshape(ommatidia_id_map.shape)
