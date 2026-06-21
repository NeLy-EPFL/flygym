"""Generate the compound eye raster maps used by flygym.
Adapted from https://github.com/NeLy-EPFL/flygym-gymnasium/blob/0b14434/scripts/generate_ommatidia_type_mask.py
and https://github.com/NeLy-EPFL/flygym-gymnasium/blob/411f03c/notebooks/retina_simulation.ipynb
"""

import numpy as np
from rasterio.features import rasterize
from shapely.geometry import Point, Polygon
import yaml

from flygym import assets_dir

RETINA_SIDE_LEN_HEX = 16  # Bounding hexagon side length in small-hex units.
OVERALL_LAYOUT = (
    "vertical"  # "vertical" makes the eye tall; "horizontal" makes it wide.
)
GRID_SCALE = 2.5
BOUND_HEX_INSET = 0.8
PALE_YELLOW_RATIO = (0.3, 0.7)
RNG_SEED = 0


def load_vision_config():
    with open(assets_dir / "model/vision.yaml") as config_file:
        return yaml.safe_load(config_file)


def calc_area_in_hexagons(side_len):
    return 3 * side_len**2 - 3 * side_len + 1


def generate_centers(retina_side_len_hex, overall_layout, grid_scale):
    centers = []
    n_rows = int(grid_scale * retina_side_len_hex)
    n_cols = int(grid_scale * retina_side_len_hex)
    for row in range(n_rows):
        offset = 0.0 if row % 2 == 0 else np.sqrt(3) / 2
        for col in range(n_cols):
            centers.append([np.sqrt(3) * col + offset, 1.5 * row])
    centers = np.array(centers, dtype=float)
    # Use an odd-sized sample to keep the median aligned to a lattice point.
    max_len = len(centers) - 1 if len(centers) % 2 == 0 else len(centers)
    centers -= np.median(centers[:max_len], axis=0)
    if overall_layout == "vertical":
        centers = centers[:, ::-1]
    return centers


def hex_vertices(overall_layout):
    verts = np.array(
        [
            (-np.sqrt(3) / 2, -0.5),
            (-np.sqrt(3) / 2, 0.5),
            (0.0, 1.0),
            (np.sqrt(3) / 2, 0.5),
            (np.sqrt(3) / 2, -0.5),
            (0.0, -1.0),
            (-np.sqrt(3) / 2, -0.5),
        ],
        dtype=float,
    )
    if overall_layout == "vertical":
        verts = verts[:, ::-1]
    return verts


def build_hex_shapes(centers, hex_verts):
    return np.array([hex_verts + center for center in centers], dtype=float)


def bounding_hexagon(retina_side_len_hex, overall_layout, inset):
    big_side_len = np.sqrt(3) * (retina_side_len_hex - inset)
    verts = np.array(
        [
            [-big_side_len / 2, -big_side_len * np.sqrt(3) / 2],
            [big_side_len / 2, -big_side_len * np.sqrt(3) / 2],
            [big_side_len, 0.0],
            [big_side_len / 2, big_side_len * np.sqrt(3) / 2],
            [-big_side_len / 2, big_side_len * np.sqrt(3) / 2],
            [-big_side_len, 0.0],
            [-big_side_len / 2, -big_side_len * np.sqrt(3) / 2],
        ],
        dtype=float,
    )
    if overall_layout == "vertical":
        verts = verts[:, ::-1]
    return verts


def select_hexes_inside(centers, hex_shapes, bound_hexagon_verts):
    big_polygon = Polygon(bound_hexagon_verts)
    is_inside = np.array([big_polygon.contains(Point(center)) for center in centers])
    return hex_shapes[is_inside]


def normalize_to_canvas(hex_shapes_inside):
    x_min, y_min = np.min(hex_shapes_inside.reshape(-1, 2), axis=0)
    x_max, y_max = np.max(hex_shapes_inside.reshape(-1, 2), axis=0)
    hex_shapes_inside = hex_shapes_inside.copy()
    hex_shapes_inside[:, :, 0] -= x_min
    hex_shapes_inside[:, :, 1] -= y_min
    return hex_shapes_inside, x_max - x_min, y_max - y_min


def compute_raw_img_width(canvas_width_au, canvas_height_au, raw_img_height_px):
    raw_img_width_px = int(canvas_width_au / canvas_height_au * raw_img_height_px) + 1
    if raw_img_width_px % 2 == 1:
        raw_img_width_px += 1
    return raw_img_width_px


def rasterize_hex_id_map(hex_shapes_inside_px, out_shape):
    polygons = [Polygon(points) for points in hex_shapes_inside_px]
    indexed_polygons = [(polygon, index + 1) for index, polygon in enumerate(polygons)]
    return rasterize(indexed_polygons, out_shape=out_shape, default_value=0)


def build_pale_mask(num_ommatidia_per_eye, pale_yellow_ratio, rng):
    pale_weight, yellow_weight = pale_yellow_ratio
    num_pale = int(num_ommatidia_per_eye * pale_weight / (pale_weight + yellow_weight))
    num_yellow = num_ommatidia_per_eye - num_pale
    mask = np.concatenate(
        [np.ones(num_pale, dtype=bool), np.zeros(num_yellow, dtype=bool)]
    )
    rng.shuffle(mask)
    return mask


def main():
    if OVERALL_LAYOUT not in {"vertical", "horizontal"}:
        raise ValueError(
            f"Unsupported OVERALL_LAYOUT: {OVERALL_LAYOUT!r}. Use 'vertical' or 'horizontal'."
        )

    vision_config = load_vision_config()
    raw_img_height_px = vision_config["raw_img_height_px"]
    expected_raw_img_width_px = vision_config["raw_img_width_px"]
    expected_num_ommatidia = vision_config["num_ommatidia_per_eye"]

    centers = generate_centers(RETINA_SIDE_LEN_HEX, OVERALL_LAYOUT, GRID_SCALE)
    hex_verts = hex_vertices(OVERALL_LAYOUT)
    hex_shapes = build_hex_shapes(centers, hex_verts)

    bound_hexagon_verts = bounding_hexagon(
        RETINA_SIDE_LEN_HEX, OVERALL_LAYOUT, BOUND_HEX_INSET
    )
    hex_shapes_inside = select_hexes_inside(centers, hex_shapes, bound_hexagon_verts)
    hex_shapes_inside, canvas_width_au, canvas_height_au = normalize_to_canvas(
        hex_shapes_inside
    )

    raw_img_width_px = compute_raw_img_width(
        canvas_width_au, canvas_height_au, raw_img_height_px
    )
    assert raw_img_width_px == expected_raw_img_width_px

    hex_shapes_inside_px = hex_shapes_inside * raw_img_height_px / canvas_height_au
    hex_id_map = rasterize_hex_id_map(
        hex_shapes_inside_px, (raw_img_height_px, raw_img_width_px)
    )

    num_ommatidia_per_eye = calc_area_in_hexagons(RETINA_SIDE_LEN_HEX)
    assert num_ommatidia_per_eye == expected_num_ommatidia

    rng = np.random.RandomState(RNG_SEED)
    pale_mask = build_pale_mask(num_ommatidia_per_eye, PALE_YELLOW_RATIO, rng)

    np.savez_compressed(
        assets_dir / "model/compound_eye.npz",
        pale_mask=pale_mask,
        ommatidia_id_map=hex_id_map,
    )


if __name__ == "__main__":
    main()
