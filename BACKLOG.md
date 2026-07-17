# Backlog

## `flygym.ik`: fully GPU-resident Gauss-Newton step (Warp tile API)

**Status:** prototyped and validated for correctness; not integrated.

The Warp-batched IK solver (`flygym.ik.warp_solve`) currently assembles the
batched normal equations (`J^T J`, `J^T r`) and solves them on the CPU via
`np.matmul` + `np.linalg.solve`, after fixing an earlier `np.einsum` bottleneck
(`np.matmul` dispatches to a batched BLAS GEMM and is ~5-6x faster than the
equivalent `einsum`). Even after that fix, profiling shows this host-side
step is still **~85-88% of total per-iteration time** at realistic batch
sizes (e.g. 26.5 ms of a ~30 ms iteration at 1,000 frames) -- the GPU forward
kinematics/Jacobian passes are comparatively cheap.

Warp 1.14 exposes a tile API (`wp.tile_load`, `wp.tile_matmul`,
`wp.tile_transpose`, `wp.tile_cholesky`, `wp.tile_cholesky_solve`,
`wp.launch_tiled`) that can run this same batched small-SPD-solve entirely on
the GPU, one thread-block per frame (a naive per-thread approach doesn't work
here -- a 42x42 matrix in per-thread registers would blow the register
budget; the tile API uses shared memory across a block instead).

Prototyped and validated (`/tmp/tile_full_kernel_test2.py` at the time of
writing, not checked in):
- Correctness: matches the `numpy` reference to ~1e-15 (`float64` tiles).
- With data already GPU-resident, the tile-based assemble+solve is **~12-15x
  faster** than the CPU `matmul`+`solve` (2.16 ms vs. 25.6 ms at 1,000
  frames; 20.4 ms vs. 309 ms at 10,000 frames).
- But building the padded tile layout via a host round-trip each iteration
  (`numpy` pad -> upload -> kernel -> download, matching how data currently
  reaches the solver) gives only **~1.7-1.8x realistic speedup** -- the
  padding/transfer overhead eats most of the theoretical win.

**To realize the full ~12-15x, the padded Jacobian/residual tiles need to be
built directly on the GPU** from the already-resident `_BatchedGpuPasses`
buffers (`points`, `jacp`), via new kernels, rather than round-tripping
through `numpy`. This also opens the door to capturing the *entire* iteration
(kinematics -> Jacobian -> assemble -> solve -> accept/reject -> qpos update)
as a single CUDA graph, replayed via `wp.capture_launch` with effectively
zero Python-side overhead per iteration -- a further, currently unquantified
win on top of the ~12-15x.

**Estimated scope:** a few new kernels (pad Jacobian into `(n_frames,
TILE_RES, TILE_NV)` tile layout with weighting and zero-padding; likewise for
residual; handle the `projection_axes` 2D case), plus moving the
accept/reject/damping-update/qpos-clip logic onto the GPU if going for full
iteration-level graph capture. One-time kernel compile cost observed during
prototyping: ~2-5.6 s (cached to disk afterward via Warp's kernel cache, so
this is a first-run cost only).

**Worth doing if:** the tens-of-thousands-of-frames use case becomes a
regular workload and this specific bottleneck (rather than e.g. I/O or
downstream processing) is the limiting factor.
