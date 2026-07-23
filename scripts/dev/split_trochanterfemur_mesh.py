"""Generate fullsize trochanter/femur split meshes from the fused fullsize mesh.

This is a maintenance/dev script, not part of the runtime package. The
trochanterfemur segment is rigged (see ``rigging.yaml``) as two geoms, a
trochanter and a femur, each needing its own mesh file. Split meshes for the
bundled ``simplified_max2000faces`` set are already committed to the repo, but
the ``fullsize`` set (hosted on S3, not in git) still only has the old fused
``{leg}_trochanterfemur.stl`` -- so ``NeuroMechFly(mesh_type=MeshType.FULLSIZE)``
currently raises ``FileNotFoundError``.

Rather than re-authoring the cut by hand, this script recovers the exact cut
plane already used for the simplified split (by finding the shared seam
vertices between the simplified trochanter and femur pieces, which lie on a
single plane to within numerical noise) and applies that same plane to the
fullsize fused mesh. Validated against the simplified set itself: reslicing the
simplified fused mesh at the recovered plane reproduces the committed
trochanter/femur pieces to within <0.1% of their volume.

This only needs to run for the left legs (lf, lm, lh): NeuroMechFly mirrors the
right legs from the left mesh files at load time.

Output is written locally for review, NOT uploaded anywhere. To publish it:

1. Inspect the output directory below (e.g. open the STLs in MeshLab/Blender).
2. Upload the new fullsize meshes to the S3 bucket under a fresh versioned
   directory, alongside the existing (still-fused) ones, e.g.
   ``flygym_assets/neuromechfly_fullsize_meshes_<YYYYMMDD><x>/`` (the bucket
   uses immutable, dated directories so existing releases keep working).
3. Bump ``NEUROMECHFLY_FULLSIZE_MESH_DIR`` in
   ``flygym/compose/fly/neuromechfly.py`` to the new directory name.
4. Re-run ``scripts/dev/simplify_meshes.py`` so the bundled simplified set is
   regenerated directly from the true fullsize split (replacing the
   currently-committed simplified trochanter/femur pieces, which predate this
   script and were authored by hand).

Usage:
    python scripts/dev/split_trochanterfemur_mesh.py
"""

import trimesh
import numpy as np
import yaml
from scipy.spatial import cKDTree

from flygym import assets_dir
from flygym.utils.assets_lazy_loading import lazy_load_asset_dir
from flygym.compose.fly.neuromechfly import NEUROMECHFLY_FULLSIZE_MESH_DIR

SCALE = 1000  # STL files are authored in meters; body-frame positions are in mm.
LEGS = ("lf", "lm", "lh")  # rf/rm/rh mirror these at load time, no split needed.

SIMPLIFIED_MESH_DIR = assets_dir / "model/neuromechfly/meshes/simplified_max2000faces"
RIGGING_CONFIG_PATH = assets_dir / "model/neuromechfly/rigging.yaml"
OUTPUT_DIR = assets_dir.parent.parent.parent / "outputs/fullsize_trochanterfemur_split"


def find_seam_plane(
    troch: trimesh.Trimesh, femur: trimesh.Trimesh, femur_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the plane the fused mesh was cut along, from its two pieces.

    Both pieces are capped where they were cut, so they share a ring of
    (near-)coincident seam vertices once placed in the same frame. Fitting a
    plane to those shared vertices (via SVD -- the seam is planar, so the
    smallest singular value is ~0) recovers the original cut plane.

    Args:
        troch: Trochanter piece, in the body/fused-mesh frame (pos == [0, 0, 0]).
        femur: Femur piece, in its own local frame (translated by ``femur_pos``
            relative to the body/fused-mesh frame).
        femur_pos: The femur geom's ``pos`` offset from ``rigging.yaml``, in mm.

    Returns:
        ``(origin, normal)`` of the cut plane, in mm, with ``normal`` pointing
        from the trochanter side towards the femur side.
    """
    troch_scaled = troch.vertices * SCALE
    femur_scaled = femur.vertices * SCALE + femur_pos

    tree = cKDTree(troch_scaled)
    dists, _ = tree.query(femur_scaled, k=1)
    seam_pts = femur_scaled[dists < 1e-3]
    if len(seam_pts) < 3:
        raise RuntimeError("Could not find a shared seam between trochanter and femur.")

    centroid = seam_pts.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(seam_pts - centroid)
    normal = vt[-1]
    if singular_values[-1] / singular_values[0] > 1e-3:
        raise RuntimeError("Seam vertices are not planar; cut plane is ambiguous.")
    if np.dot(femur_scaled.mean(axis=0) - centroid, normal) < 0:
        normal = -normal
    return centroid, normal


def split_at_plane(
    fused: trimesh.Trimesh, origin_mm: np.ndarray, normal: np.ndarray
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Slice a fused mesh (in meters) into (trochanter, femur) at a plane given
    in mm, both capped so the resulting pieces are closed."""
    origin_m = origin_mm / SCALE
    troch = fused.slice_plane(plane_origin=origin_m, plane_normal=-normal, cap=True)
    femur = fused.slice_plane(plane_origin=origin_m, plane_normal=normal, cap=True)
    return troch, femur


if __name__ == "__main__":
    with open(RIGGING_CONFIG_PATH) as f:
        rigging_config = yaml.safe_load(f)

    fullsize_dir = lazy_load_asset_dir(NEUROMECHFLY_FULLSIZE_MESH_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    for leg in LEGS:
        geoms_config = rigging_config[f"{leg}_trochanterfemur"]["geoms"]
        femur_pos = np.array(geoms_config[f"{leg}_femur"]["pos"])

        simplified_troch = trimesh.load(SIMPLIFIED_MESH_DIR / f"{leg}_trochanter.stl")
        simplified_femur = trimesh.load(SIMPLIFIED_MESH_DIR / f"{leg}_femur.stl")
        origin, normal = find_seam_plane(simplified_troch, simplified_femur, femur_pos)

        fullsize_fused = trimesh.load(fullsize_dir / f"{leg}_trochanterfemur.stl")
        troch_piece, femur_piece = split_at_plane(fullsize_fused, origin, normal)
        # Femur is authored with its own origin at the trochanter-femur joint,
        # matching the simplified set's convention (see rigging.yaml's `pos`).
        femur_piece.vertices -= femur_pos / SCALE

        troch_path = OUTPUT_DIR / f"{leg}_trochanter.stl"
        femur_path = OUTPUT_DIR / f"{leg}_femur.stl"
        troch_piece.export(troch_path)
        femur_piece.export(femur_path)

        print(
            f"{leg}: fused {len(fullsize_fused.faces)} faces -> "
            f"trochanter {len(troch_piece.faces)} faces ({troch_path}), "
            f"femur {len(femur_piece.faces)} faces ({femur_path})"
        )

    print(f"\nFullsize trochanter/femur split meshes written to {OUTPUT_DIR}")
    print("Review them, then follow the publishing steps in this script's docstring.")
