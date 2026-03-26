from dataclasses import dataclass, field

import trimesh
import numpy as np
import pandas as pd

from flygym import assets_dir


def simplify_mesh(
    input_mesh: trimesh.Trimesh,
    max_faces: int,
    mirror_by_xzplane: bool = False,
) -> trimesh.Trimesh:
    """Simplify a mesh by reducing its face count via quadric decimation.

    Args:
        input_mesh: The mesh to simplify. The original is never modified.
        max_faces: Target maximum number of faces in the output mesh.
            If the input mesh already has fewer or equal faces, it is
            returned unchanged.
        mirror_by_xzplane: If True, exploit bilateral symmetry across the
            XZ plane (y = 0) to improve simplification quality. The mesh is
            sliced at y = 0, the positive-Y half is simplified to
            ``max_faces // 2`` faces, then reflected back across the XZ
            plane and stitched into a single closed surface. This guarantees
            a perfectly symmetric output and prevents the decimation
            algorithm from introducing asymmetric artifacts or unevenly
            distributing faces across a nominally symmetric model.

    Returns:
        A new simplified ``trimesh.Trimesh``. The input mesh is not modified.
    """
    if not isinstance(input_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a trimesh.Trimesh, got {type(input_mesh)}")

    if len(input_mesh.faces) <= max_faces:
        return input_mesh.copy()
    elif mirror_by_xzplane:
        return _simplify_with_mirror(input_mesh, max_faces)
    else:
        return input_mesh.simplify_quadric_decimation(face_count=max_faces)


def _simplify_with_mirror(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    # Slice - keep the y >= 0 half, leave boundary open
    half = mesh.slice_plane(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0], cap=False)

    # Simplify the half
    half_simplified = half.simplify_quadric_decimation(face_count=max_faces // 2)

    # Snap any seam vertices that drifted slightly off y=0 back to the plane
    seam_mask = np.abs(half_simplified.vertices[:, 1]) < 1e-4
    half_simplified.vertices[seam_mask, 1] = 0.0

    # Mirror across XZ plane (negate Y), flip winding to match
    mirrored = half_simplified.copy()
    mirrored.vertices[:, 1] *= -1
    mirrored.faces[:, [1, 2]] = mirrored.faces[:, [2, 1]]  # reverse winding

    # Stitch: build a new mesh from concatenated raw arrays, let trimesh
    # merge coincident vertices and clean up in the constructor
    vertices = np.concatenate([half_simplified.vertices, mirrored.vertices])
    faces = np.concatenate(
        [
            half_simplified.faces,
            mirrored.faces + len(half_simplified.vertices),
        ]
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


@dataclass
class MeshSimilarityResult:
    mean_distance: float
    rms_distance: float
    hausdorff_distance: float
    n_samples: int
    scale: (
        float  # reference length (e.g. longest bounding box edge of the original mesh)
    )

    # Populated by __post_init__
    mean_distance_rel: float = field(init=False)
    rms_distance_rel: float = field(init=False)
    hausdorff_distance_rel: float = field(init=False)

    def __post_init__(self):
        self.mean_distance_rel = self.mean_distance / self.scale
        self.rms_distance_rel = self.rms_distance / self.scale
        self.hausdorff_distance_rel = self.hausdorff_distance / self.scale


def mesh_similarity(
    mesh_a: trimesh.Trimesh,
    mesh_b: trimesh.Trimesh,
    n_samples: int = 10_000,
) -> MeshSimilarityResult:
    """Quantify geometric similarity between two meshes via surface sampling.

    Samples points uniformly on both surfaces and computes point-to-surface
    distances in both directions (A->B and B->A), then aggregates symmetrically.
    Absolute distances are in the same units as the mesh coordinates; relative
    distances are normalised by the longest bounding box edge of ``mesh_a``.

    Args:
        mesh_a: First mesh (e.g. the original).
        mesh_b: Second mesh (e.g. the simplified version).
        n_samples: Number of surface points to sample per mesh. Higher values
            give more accurate estimates at the cost of compute time.
            10 000 is a good default for typical CAD/body meshes.

    Returns:
        A ``MeshSimilarityResult`` with absolute and relative variants of:
        - ``mean_distance``: symmetric mean surface distance - the primary
          summary metric. Zero means identical surfaces.
        - ``rms_distance``: root-mean-square distance, which penalises large
          local deviations more heavily than the mean.
        - ``hausdorff_distance``: the maximum of all sampled point-to-surface
          distances — a worst-case bound on surface deviation.
        - ``n_samples``: total number of distance samples used (2 * n_samples).
        - ``scale``: the reference length used to compute relative metrics
          (longest bounding box edge of ``mesh_a``).
    """
    points_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples)
    points_b, _ = trimesh.sample.sample_surface(mesh_b, n_samples)

    _, dist_a_to_b, _ = trimesh.proximity.closest_point(mesh_b, points_a)
    _, dist_b_to_a, _ = trimesh.proximity.closest_point(mesh_a, points_b)

    all_distances = np.concatenate([dist_a_to_b, dist_b_to_a])

    return MeshSimilarityResult(
        mean_distance=float(np.mean(all_distances)),
        rms_distance=float(np.sqrt(np.mean(all_distances**2))),
        hausdorff_distance=float(np.max(all_distances)),
        n_samples=len(all_distances),
        scale=float(mesh_a.scale),
    )


if __name__ == "__main__":
    MESH_DIR_FULLSIZE = assets_dir / "model/meshes/fullsize/"
    MAX_FACES = 2000
    MESH_DIR_REDUCED = assets_dir / f"model/meshes/simplified_max{MAX_FACES}faces/"

    MESH_DIR_REDUCED.mkdir(exist_ok=True, parents=True)

    # Load all meshes
    meshes_orig = {}
    rows = []
    for path in sorted(MESH_DIR_FULLSIZE.glob("*.stl")):
        name = path.stem
        mesh = trimesh.load_mesh(path)
        meshes_orig[name] = mesh
        n_faces = len(mesh.faces)
        file_size_kb = int(path.stat().st_size / 1e3)
        rows.append([name, n_faces, file_size_kb])
    df = pd.DataFrame(rows, columns=["name", "n_faces_orig", "filesize_kb_orig"])
    df = df.sort_values("n_faces_orig", ascending=False).reset_index(drop=True)
    df = df.set_index("name")

    total_faces_orig = sum(row[1] for row in rows)
    df["face_pct_of_total_orig"] = df["n_faces_orig"] / total_faces_orig * 100
    print(df)
    print()

    # Simplify all meshes
    df["n_faces_reduced"] = df["n_faces_orig"]
    df["file_size_kb_reduced"] = df["filesize_kb_orig"]
    df["pct_mean_dist"] = np.nan
    df["pct_mse"] = np.nan
    df["pct_hausdorff_dist"] = np.nan
    for name, mesh in meshes_orig.items():
        if len(mesh.faces) > MAX_FACES:
            print(f"Simplifying {name}")
            mirror_by_xzplane = name.startswith("c_")
            simplified_mesh = simplify_mesh(mesh, MAX_FACES, mirror_by_xzplane)
            output_path = MESH_DIR_REDUCED / f"{name}.stl"
            simplified_mesh.export(output_path)
            file_size_kb = int(output_path.stat().st_size / 1e3)
            similarity = mesh_similarity(mesh, simplified_mesh)
            df.loc[name, "file_size_kb_reduced"] = file_size_kb
            df.loc[name, "n_faces_reduced"] = len(simplified_mesh.faces)
            df.loc[name, "pct_mean_dist"] = similarity.mean_distance_rel * 100
            df.loc[name, "pct_mse"] = similarity.rms_distance_rel * 100
            df.loc[name, "pct_hausdorff_dist"] = similarity.hausdorff_distance_rel * 100
    total_faces_reduced = df["n_faces_reduced"].sum()
    df["n_faces_reduced"][df["pct_mean_dist"].isna()] = np.nan
    df["face_pct_of_total_reduced"] = df["n_faces_reduced"] / total_faces_reduced * 100
    print(df)
    print()

    print(f"Simplified meshes saved to {MESH_DIR_REDUCED}")
    print(f"Total faces before reduction: {total_faces_orig}")
    print(f"Total faces after reduction: {total_faces_reduced}")
    pct_reduced = (total_faces_orig - total_faces_reduced) / total_faces_orig * 100
    print(f"Overall pct reduced: {pct_reduced:.2f}%")

    metadata_path = MESH_DIR_REDUCED / f"simplification_metadata.csv"
    df.reset_index().to_csv(metadata_path, index=False)
