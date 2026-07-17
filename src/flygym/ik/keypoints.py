"""Keypoint target definitions for `flygym.ik`."""

from dataclasses import dataclass

import mujoco as mj
import numpy as np

from flygym.anatomy import BodySegment

__all__ = ["KeypointTarget", "KeypointSet", "estimate_terminal_offset"]


@dataclass(frozen=True)
class KeypointTarget:
    """A single keypoint rigidly attached to a body segment's local frame.

    Attributes:
        body: The body segment this keypoint is attached to.
        local_offset: Offset of the keypoint from the body's origin, in the
            body's local frame, shape `(3,)`.
        name: Human-readable identifier, used e.g. by `KeypointSet.set_weight`.
            Defaults to `body.name`.
    """

    body: BodySegment
    local_offset: np.ndarray
    name: str | None = None

    def __post_init__(self):
        if isinstance(self.body, str):
            object.__setattr__(self, "body", BodySegment(self.body))
        local_offset = np.asarray(self.local_offset, dtype=float)
        if local_offset.shape != (3,):
            raise ValueError(
                f"local_offset must have shape (3,), got {local_offset.shape}."
            )
        object.__setattr__(self, "local_offset", local_offset)
        if self.name is None:
            object.__setattr__(self, "name", self.body.name)

    @classmethod
    def from_anatomical_names(
        cls,
        leg: str,
        parent_link: str,
        child_link: str | None,
        *,
        local_offset: np.ndarray | None = None,
        mj_model: mj.MjModel | None = None,
        body_segment_cls: type[BodySegment] = BodySegment,
    ) -> "KeypointTarget":
        """Build a `KeypointTarget` from a `(leg, parent_link, child_link)` triple.

        This matches the keypoint naming convention used by
        `flygym_demo.spotlight_data.preprocessing.MotionSnippet` (and other
        bundled experimental recordings): each keypoint is identified by the
        leg it belongs to and the two links it sits between.

        If `child_link` is given, the keypoint is the origin of body
        `'{leg}_{child_link}'` -- i.e. the `parent_link`-`child_link` joint
        location, since body origins in this model are defined at their joint
        to the parent (see `flygym.compose.fly.BaseFly.add_joint_sites`).
        `local_offset` then defaults to `(0, 0, 0)`.

        If `child_link` is `None` (the convention used for a leg's
        terminal/claw keypoint, e.g. `('lf', 'tarsus5', None)`), the keypoint
        is attached to body `'{leg}_{parent_link}'` itself, and `local_offset`
        cannot be inferred exactly -- either pass it explicitly, or pass
        `mj_model` so a mesh-vertex-based estimate is used (see
        `estimate_terminal_offset`).

        Args:
            leg: Leg position identifier (e.g. `'lf'`).
            parent_link: Proximal link name (e.g. `'tibia'`).
            child_link: Distal link name (e.g. `'tarsus1'`), or `None` for a
                terminal keypoint at the end of `parent_link`.
            local_offset: Explicit local offset, shape `(3,)`. Required when
                `child_link` is `None` and `mj_model` is not given.
            mj_model: Compiled model to estimate a terminal offset from, when
                `child_link` is `None` and `local_offset` is not given.
            body_segment_cls: Body segment class to use (defaults to
                `flygym.anatomy.BodySegment`; pass e.g. `FlyBodyBodySegment`
                for the FlyBody model).

        Returns:
            A `KeypointTarget`.

        Raises:
            ValueError: If `child_link` is `None` and neither `local_offset`
                nor `mj_model` is given.
        """
        if child_link is not None:
            body = body_segment_cls(f"{leg}_{child_link}")
            offset = np.zeros(3) if local_offset is None else local_offset
            name = f"{leg}-{parent_link}-{child_link}"
            return cls(body=body, local_offset=offset, name=name)

        body = body_segment_cls(f"{leg}_{parent_link}")
        if local_offset is not None:
            offset = local_offset
        elif mj_model is not None:
            offset = estimate_terminal_offset(mj_model, body)
        else:
            raise ValueError(
                "Terminal keypoint (child_link=None) requires either an "
                "explicit local_offset or an mj_model to estimate one from."
            )
        return cls(body=body, local_offset=offset, name=f"{leg}-{parent_link}-tip")


def estimate_terminal_offset(mj_model: mj.MjModel, body: BodySegment) -> np.ndarray:
    """Estimate the local offset of a leaf body's distal tip.

    Approximates the tip as the mesh vertex farthest from the body's origin,
    reasonable for a rod-like segment (e.g. `tarsus5`) whose body origin sits
    at its proximal joint -- but the farthest vertex is not necessarily the
    true anatomical tip (e.g. a wider point elsewhere on the mesh could be
    farther by straight-line distance). Callers who need precision should
    pass an explicit `local_offset` instead.

    Mesh vertices (`mj_model.mesh_vert`) are given in the *geom's* local
    frame, which is not generally the same as the *body's* frame (a geom can
    have its own `pos`/`quat` relative to its body). All vertices are
    transformed into the body frame *before* ranking them by distance --
    ranking in the geom's own frame instead would measure distance from the
    wrong reference point and can pick a vertex near the proximal joint
    instead of the distal tip, whenever `geom_pos` is (as is typical for a
    leaf segment) comparable in magnitude to the segment's own length.

    Args:
        mj_model: Compiled MuJoCo model.
        body: Body segment to estimate the offset for (e.g. `lf_tarsus5`).

    Returns:
        Local offset in the body's frame, shape `(3,)`.

    Raises:
        ValueError: If the body is not found, does not have exactly one
            geom, or that geom is not a mesh.
    """
    body_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, body.name)
    if body_id < 0:
        raise ValueError(f"Body '{body.name}' not found in the model.")
    geom_count = mj_model.body_geomnum[body_id]
    if geom_count != 1:
        raise ValueError(
            f"Body '{body.name}' has {geom_count} geoms; "
            "estimate_terminal_offset requires exactly one."
        )
    geom_id = mj_model.body_geomadr[body_id]
    mesh_id = mj_model.geom_dataid[geom_id]
    if mesh_id < 0:
        raise ValueError(
            f"Body '{body.name}' has a non-mesh geom; "
            "estimate_terminal_offset requires a mesh geom."
        )

    vert_start = mj_model.mesh_vertadr[mesh_id]
    vert_count = mj_model.mesh_vertnum[mesh_id]
    verts_geom_frame = mj_model.mesh_vert[vert_start : vert_start + vert_count].astype(
        float
    )

    geom_rotmat = np.empty(9)
    mj.mju_quat2Mat(geom_rotmat, mj_model.geom_quat[geom_id])
    # Transform every vertex into the body frame *before* ranking by distance:
    # the body's own origin (0, 0, 0) is the proximal joint, so "farthest from
    # the body origin" is what picks out the distal tip. Ranking in the geom's
    # own local frame instead (as a previous version of this function did)
    # measures distance from the wrong reference point whenever geom_pos is
    # non-zero, which for a leaf segment like tarsus5 is typically comparable
    # in magnitude to the segment's own length -- it can trivially pick a
    # vertex near the *proximal* end instead of the tip.
    verts_body_frame = (
        verts_geom_frame @ geom_rotmat.reshape(3, 3).T + (mj_model.geom_pos[geom_id])
    )
    return verts_body_frame[np.argmax(np.linalg.norm(verts_body_frame, axis=1))]


@dataclass
class KeypointSet:
    """An ordered collection of `KeypointTarget`s with per-keypoint weights.

    Attributes:
        targets: The keypoint targets, in a fixed order.
        weights: Per-keypoint weight, shape `(len(targets),)`. Defaults to all
            ones. Use `set_weight` to down- or up-weight individual keypoints
            (e.g. to reduce the influence of a keypoint that is hard to track
            reliably).
    """

    targets: list[KeypointTarget]
    weights: np.ndarray | None = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = np.ones(len(self.targets))
        else:
            self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.shape != (len(self.targets),):
            raise ValueError(
                f"weights must have shape ({len(self.targets)},), "
                f"got {self.weights.shape}."
            )
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative.")

    @classmethod
    def from_keypoint_triples(
        cls,
        triples: list[tuple[str, str, str | None]],
        *,
        mj_model: mj.MjModel | None = None,
        body_segment_cls: type[BodySegment] = BodySegment,
    ) -> "KeypointSet":
        """Build a `KeypointSet` from `(leg, parent_link, child_link)` triples.

        This matches the format of `MotionSnippet.keypoints` and similar
        bundled experimental data (e.g. `snippet.keypoints`, used directly as
        `triples`). See `KeypointTarget.from_anatomical_names` for how each
        triple is resolved to a body and local offset; `mj_model` is required
        if any triple has `child_link=None` (a terminal/claw keypoint) and no
        further offset information is available.

        Args:
            triples: `(leg, parent_link, child_link)` tuples.
            mj_model: Compiled model, used to estimate terminal keypoint
                offsets (see `estimate_terminal_offset`).
            body_segment_cls: Body segment class to use for all targets.

        Returns:
            A `KeypointSet` with unit weights.
        """
        targets = [
            KeypointTarget.from_anatomical_names(
                leg,
                parent_link,
                child_link,
                mj_model=mj_model,
                body_segment_cls=body_segment_cls,
            )
            for leg, parent_link, child_link in triples
        ]
        return cls(targets=targets)

    def _index_of(self, name: str) -> int:
        for i, target in enumerate(self.targets):
            if target.name == name:
                return i
        raise ValueError(f"No keypoint named '{name}' in this KeypointSet.")

    def set_weight(self, name: str, weight: float) -> None:
        """Set the weight of the keypoint named `name` in place.

        Example:

            keypoints.set_weight("lf-thorax-coxa", 0.1)
        """
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        self.weights[self._index_of(name)] = weight

    def local_offsets(self) -> np.ndarray:
        """Stacked local offsets, shape `(len(targets), 3)`."""
        return np.stack([t.local_offset for t in self.targets])

    def resolve_body_ids(self, mj_model: mj.MjModel) -> np.ndarray:
        """Resolve each target's body to a MuJoCo body id in `mj_model`.

        Raises:
            ValueError: If a body name is not found in the model.
        """
        body_ids = np.empty(len(self.targets), dtype=np.int32)
        for i, target in enumerate(self.targets):
            body_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, target.body.name)
            if body_id < 0:
                raise ValueError(
                    f"Body '{target.body.name}' (keypoint '{target.name}') not "
                    "found in the model."
                )
            body_ids[i] = body_id
        return body_ids
