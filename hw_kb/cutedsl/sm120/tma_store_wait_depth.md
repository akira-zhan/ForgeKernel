# tma_store_wait_depth (the staging depth B of a TMA-store epilogue on sm120: `cp_async_bulk_wait_group(B - 1, read=True)`, its `CPWAIT` footprint, and where it carries no load)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified, KEEP-certified (a numerically invisible knob: the evidence is the dossier census of shipped
> KEEPs plus one same-job A/B probe, not a `max_diff` contract). Both directions were observed on RTX PRO 5000, CuTe DSL
> 4.8.0: B 2 -> 4 moved the commit axis `CPWAIT {LE 0x0 x1, LE 0x1 x8}` -> `{LE 0x0 x1, LE 0x3 x8}` (KEEP in job
> 1794023369, census job 1794023393), and B 4 -> 2 moved it back (KEEP in job 1794030395, census job 1794030419). Every
> B was bit-identical to every other B (probe `notes/m1_levers_probe.py`, `PROBE PASS max_diff=0.0  # d14_B2 vs base`,
> job 1794030350).
>
> **What is certified**: the knob -> footprint mapping, the correctness of the ring at B = 2 and B = 4, and the
> reading that **B carried no measurable load in the two epilogue structures below**. It does not show that B is never
> load-bearing (see "When needed").

## Purpose

An epilogue that writes the output through TMA (`stmatrix` into an smem staging buffer, then one bulk tensor S2G per
sub-tile) has to stop overwriting a staging buffer until the bulk store that is still reading it is done. B staging
buffers used round robin, plus a bulk read-wait that leaves at most B - 1 store groups in flight, are that throttle.
B is a lever: a larger B lets the stores of more sub-tiles overlap the next sub-tiles' register -> smem writes and
costs B x (sub-tile bytes) of smem.

## When needed (trigger)

- You write a TMA-store epilogue and have to choose B, or a lever list carries "staging depth" / "store wait depth".
- The dossier's commit row has `LDGDEPBAR x0` and a `CPWAIT {... LE 0x<N> xK}` term whose N you want to explain (it is
  B - 1, and K is the number of sub-tiles per tile).
- **Price it before ranking it high.** In both structures measured here (1 persistent CTA / SM, CTA tile 128 x 128,
  8 store sub-tiles per tile, one warp issuing each store, the mainloop holding the tensor pipe at ~86 %), B = 2 and
  B = 4 were the same within noise; and a same-structure check that removed the per-sub-tile read-wait entirely
  moved the geomean by +0.02 % (see Readings). Whether B carries load in some other structure (an
  epilogue that is a larger share of the tile time, a different store issue pattern) is untested.

## Minimal runnable example

The epilogue loop of the KEEP kernels, with the operator's math removed (CuTe DSL 4.8.0, `sm_120a`). Correctness at
B = 2 and B = 4 is the evidence above.

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as hh
from cutlass.tensor_utils import LayoutEnum

B = 2                      # staging buffers = bulk read-wait depth B - 1 (lever; 2 and 4 measured equal)
NSUB = 8                   # store sub-tiles per output tile, unrolled at compile time
assert NSUB % B == 0       # buffer index j % B restarts with every tile -- gotcha 2

# host / trace side: a B-deep staging ring and the S2G TMA atom built on one slot of it
sE_l = hh.make_smem_layout_epi(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (EPI_M, EPI_N), B)
tma_y, tY = cute.nvgpu.cpasync.make_tiled_tma_atom(
    cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), mY, cute.slice_(sE_l, (None, None, 0)), (EPI_M, EPI_N))

# kernel side, once per CTA
gY = cute.local_tile(tY, (EPI_M, EPI_N), (None, None))
tYs, tYg = cute.nvgpu.cpasync.tma_partition(tma_y, 0, cute.make_layout(1),
                                            cute.group_modes(sE, 0, 2), cute.group_modes(gY, 0, 2))

for tile in cutlass.range(cta_id, num_tiles, num_ctas):            # persistent tile loop
    # ... mainloop fills the register accumulator ...
    for j in cutlass.range_constexpr(NSUB):
        buf = j % B
        # registers -> output dtype, then stmatrix into staging slot `buf`
        cute.copy(stsm_tiled_copy, rE, tEsE[None, None, None, buf])
        cute.arch.fence_view_async_shared()                         # generic smem writes -> visible to the TMA
        cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)
        if warp_idx == STORE_WARP:                                  # one whole warp issues the store
            cute.copy(tma_y, tYs[None, buf], tYg[None, sub_m(tile, j), sub_n(tile, j)])
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(B - 1, read=True)    # SASS DEPBAR.LE SB0, B-1 -> CPWAIT {LE 0x<B-1> xNSUB}
        cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=EPI_THREADS)  # slot j+1 % B is free for everyone

if warp_idx == STORE_WARP:
    cute.arch.cp_async_bulk_wait_group(0, read=True)                # drain before exit -> CPWAIT {LE 0x0 x1}
```

`sub_m` / `sub_n` stand for the operator's sub-tile -> output coordinate map; `EPI_BAR` / `EPI_THREADS` for the
threads that write the staging buffer (a whole-CTA-consumer rendezvous or a warpgroup-local one: both were measured).

## Footprint (dossier `commit` axis, the evidence of the knob)

The census reads `DEPBAR.LE SB0, N` as `CPWAIT {LE N xK}` (see
[`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 22). For this epilogue, on the consumer role's
row (`LDGDEPBAR x0`: no cp.async in the loop, so every `DEPBAR` is a bulk-store wait):

| B | commit row (verbatim) |
|---|---|
| 2 | `mma-consumer: DEPBAR {} LDGDEPBAR x0 CPWAIT {LE 0x0 x1, LE 0x1 x8}` |
| 4 | `mma-consumer: DEPBAR {} LDGDEPBAR x0 CPWAIT {LE 0x0 x1, LE 0x3 x8}` |

`x8` = one read-wait per unrolled sub-tile at depth B - 1; `LE 0x0 x1` = the drain `wait_group(0)`, unchanged by B.
No other contract axis moved with B in either direction (cache_ops `UTMASTG.2D`, loops, mbarrier objects and edges
identical before and after). smem: +B x sub-tile bytes (64 x 32 bf16 sub-tiles: 82,944 B at B = 2 -> 91,136 B at
B = 4, under the 101,376 B per-block ceiling of [`launch_identity_stamp`](launch_identity_stamp.md)).

## Readings (RTX PRO 5000, two GEMM shapes, geomean)

| Structure | Change | Reading | Verdict |
|---|---|---|---|
| Epilogue shared by both consumer warpgroups: 64 x 32 bf16 sub-tiles, two 256-thread named-barrier rendezvous per sub-tile, warp 0 issues | B 2 -> 4 | paired ratio to the same-job baseline 1.0032 -> 1.0026 (per shape 0.9998 -> 1.0033 and 1.0067 -> 1.0019) | null |
| same, B = 4 | read-wait removed entirely (control) | +0.02 % gm (+0.03 % / +0.01 %) | the wait itself costs nothing |
| Warpgroup-local epilogue: each consumer warpgroup stores its own 64 x 16 half sub-tiles from its own store warp, 128-thread rendezvous | B 4 -> 2 | same-job A/B +0.18 % gm (+0.11 % / +0.25 %), A/A +0.02 %, 7 interleaved rounds; gate 0.3327 -> 0.3335 ms (+0.24 %, shipped together with one other lever change priced +0.42 % in the same A/B) | null (smallest B kept) |

**How to read the first row**: that KEEP's gate went 0.7085 -> 0.33565 ms (-52.6 %), but the device window was
redefined between the two evals -- the same-job baseline moved 0.7108 -> 0.3357 ms with it. A contract-axis KEEP whose
gate jumps by tens of percent is read on the paired ratio to its same-job baseline, never on the raw window.

## Gotchas

1. **Price B before spending a lever slot on it.** With the depth at 1, at 3, or the wait removed, the gate did not
   move in either structure (Readings). Take the smallest B inside the noise -- it is also the smallest smem -- unless
   a pricing run on your tree says otherwise.
2. **The compile-time ring index restarts every tile: NSUB % B must be 0.** `buf = j % B` with `j` unrolled per tile
   resets to 0 at the next tile, while the group count of `wait_group(B - 1)` does not. With NSUB % B != 0 (e.g. B = 3,
   NSUB = 8) the first sub-tile of the next tile writes the slot of a store that may still be in flight from the
   previous tile. Either keep NSUB a multiple of B (the assert in the example) or carry a runtime ring index across
   tiles. (The constraint follows from the group-count semantics and was guarded by the assert; no run exercised the
   violating case.)
3. **Reading the axis**: a `CPWAIT` term on a row with `LDGDEPBAR x0` is a bulk-store wait, not a cp.async ring depth
   (that one is [`cpasync_multistage_pipeline`](cpasync_multistage_pipeline.md)'s `LE 0x<S-2>`). A changed `LE 0x<N>`
   on such a row is a staging-depth change; record it, but it is not by itself a suspect for a gate move.

## Sources

- PTX ISA: `cp.async.bulk.tensor` (shared::cta -> global, `.bulk_group`), `cp.async.bulk.commit_group`,
  `cp.async.bulk.wait_group.read`, `fence.proxy.async.shared::cta`, `stmatrix`, `bar.sync`:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CuTe DSL API symbols: `cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp`, `cute.nvgpu.cpasync.make_tiled_tma_atom`,
  `cute.nvgpu.cpasync.tma_partition`, `cute.arch.cp_async_bulk_commit_group`, `cute.arch.cp_async_bulk_wait_group`
  (`read=True`), `cute.arch.fence_view_async_shared`, `cute.arch.barrier`, `cute.nvgpu.warp.StMatrix8x8x16bOp`,
  `cute.make_tiled_copy_C`, `cutlass.utils.hopper_helpers.make_smem_layout_epi`, `cutlass.range_constexpr`.
- KEEP certification, B 2 -> 4: KEEP in job 1794023369 (dossier census job 1794023393; watermark confirmation job
  1794023408), axis `commit` `CPWAIT {LE 0x0 x1, LE 0x1 x8}` -> `{LE 0x0 x1, LE 0x3 x8}`, gate 0.7085 -> 0.33565 ms
  (window redefined; paired ratio 1.0032 -> 1.0026).
- KEEP certification, B 4 -> 2: KEEP in job 1794030395 (dossier census job 1794030419), axis `commit`
  `CPWAIT {LE 0x0 x1, LE 0x3 x8}` -> `{LE 0x0 x1, LE 0x1 x8}`, gate 0.3327 -> 0.3335 ms.
- Same-job A/B: probe `notes/m1_levers_probe.py`, job 1794030350, `PROBE PASS max_diff=0.0  # d14_B2 vs base` (7/7
  variants PASS in one round), table line `d14_B2 0.1670 0.6799 0.3369 +0.18%`, A/A line `base2 ... +0.02%`.
- Read-wait removal control: check `nowait` of a store-mechanism probe (the probe is not included in this release),
  hypothesis "the per-sub-tile bulk read wait holds >= 1 % gm" refuted at `gm=+0.02%,square4k=+0.03%,wide8k=+0.01%`.
- Census reading of `DEPBAR.LE SB0` after bulk stores: [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md)
  gotcha 22.
