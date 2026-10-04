"""Convert the bundled µCT specimen's hexagonal lattice from RData to JSON.

Run ``uv run --with rdata python scripts/dev/convert_compound_eye_lattice.py``.
The R reader is needed only for this development script, not for FlyGym.
The output uses zero-based per-eye lens indices, in source lens order. Missing
neighbors and their angular distances are null. Lattice coordinates are integers;
angular distances are in radians. No geometry or optical axes are changed.
"""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import rdata

from flygym import assets_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=assets_dir / "model/neuromechfly/vision/ommatidia_lattice.json",
    )
    args = parser.parse_args()
    commit = "99d2a43123db636cedb55af9ff31a59657e7d17e"
    source_path = "data/microCT/20240701_nb.RData"
    repository = "https://github.com/reiserlab/eyemap_T4"
    with urlopen(
        f"https://raw.githubusercontent.com/reiserlab/eyemap_T4/{commit}/{source_path}",
        timeout=30,
    ) as response:
        raw = response.read()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "source.RData"
        path.write_bytes(raw)
        source = rdata.read_rda(path)
    result = {
        "source": {
            "repository": repository,
            "commit": commit,
            "path": source_path,
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "index_base": 0,
        "source_order": "Per-eye lens rows in data/microCT/20240701.RData",
        "coordinate_axes": ["p", "q"],
        "neighbor_offsets": [[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1]],
        "angle_unit": "radian",
    }
    for side in ("left", "right"):
        arrays = []
        for key in ("ind_xy", "nb_ind", "nb_dist_ucl"):
            array = np.asarray(source[f"{key}_{side}"])
            array = array[np.argsort(array[:, 0])]
            np.testing.assert_array_equal(array[:, 0], np.arange(1, len(array) + 1))
            arrays.append(array[:, 1:])
        coordinates, neighbors, angles = arrays
        if not np.array_equal(np.isnan(neighbors), np.isnan(angles)):
            raise ValueError("Missing neighbor indices and angles disagree.")
        result[side] = {
            "coordinates": coordinates.astype(int).tolist(),
            "neighbors": [
                [int(value) - 1 if np.isfinite(value) else None for value in row]
                for row in neighbors
            ],
            "neighbor_angles": [
                [float(value) if np.isfinite(value) else None for value in row]
                for row in angles
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
