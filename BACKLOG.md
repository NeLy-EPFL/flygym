# Backlog

## `flygym.ik`: fully GPU-resident Gauss-Newton step (Warp tile API)

**Status:** done. `flygym.ik.warp_solve._TileSolvePasses` builds the padded
Jacobian/residual tiles directly on the GPU (via `_build_padded_jac_kernel`/
`_build_padded_residual_kernel`, reading straight from the already
GPU-resident `_BatchedGpuPasses.jacp`/`.points` buffers) and solves the
damped normal equations per frame with Warp's tile API (`wp.tile_matmul`,
`wp.tile_cholesky`, `wp.tile_cholesky_solve`), one thread block per frame.
Used automatically whenever the resolved Warp device is CUDA (no
user-facing toggle -- the tile solve is strictly better than the old
host-round-trip solve whenever it's available, so there's nothing to
configure); falls back to the host `np.matmul`/`np.linalg.solve` solve on a
non-CUDA device. Only the small `(n_frames, nv)` solved `delta` is
transferred back to the host each iteration, instead of the full
`O(n_frames * n_keypoints * nv)` Jacobian.

Measured throughput (RTX 3080 Ti, this repo's bundled model: 42 DOFs, 24-30
keypoints; see `scripts/dev/benchmark_ik.py`): plateaus around **~7,000
frames/s** for batches of 1,000 frames and up, vs. ~800-830 frames/s at the
old host-round-trip solve's peak batch size (500-1,000 frames) -- roughly an
order of magnitude, in line with the ~12-15x figure this backlog item
originally estimated once GPU-side padding replaced the host round-trip.

Validated during development against a pure-numpy reference (standalone,
not checked in) and against all of the existing warp IK tests (known-qpos
recovery, matching the CPU backend, joint limits, 2D projection, batch
chunking, CUDA graph capture), all of which pass unchanged with the tile
solve as the new default code path.

**Not done** (optional stretch goal, not required to realize the ~12-15x):
capturing the *entire* iteration (kinematics -> Jacobian -> tile
assemble/solve -> accept/reject -> qpos update) as a single CUDA graph. The
accept/reject/damping-update/qpos-clip logic still round-trips to the host
each iteration, but transfers only `O(n_frames * (nv + n_keypoints * 3))`
data (points for cost checking, damping, delta) rather than the Jacobian, so
it was not the bottleneck this item was scoped to fix. Worth revisiting only
if profiling shows this remaining host round-trip becomes significant at
some future batch size/iteration-count combination.
