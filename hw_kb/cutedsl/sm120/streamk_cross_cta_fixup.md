# streamk_cross_cta_fixup (stream-K on sm120: a CTA's contiguous range of (tile, k-tile) units fed through one continuous TMA ring, plus a cross-CTA f32 fixup in last-arriver form with a self-resetting tile counter)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified. Correctness: probe `notes/streamk_fixup_probe.py`, 24/24 `PROBE PASS` after 1 round
> (job 1794113745), on four K-reduction shapes at real size. Shipped as a KEEP (eval job 1794113794, census job
> 1794113821): gate 0.0597 -> 0.0548 ms; contract axes `rings` `[]` -> `["stride=0x1,wrap=sel"]` and `commit`
> `CPWAIT {LE 0x1 x4}` -> `{LE 0x0 x1, LE 0x1 x4}`. Both axis moves are loop-structure follow-ons, not knobs of
> their own (see "Census footprint").
>
> **What is certified**: the decomposition, the continuous ring across piece boundaries, the fixup ordering below,
> the counter self-reset across back-to-back launches, and bit-identical results over repeated launches on
> exact-integer data. **Not certified**: which element of the ordering chain is individually necessary, run-to-run
> bit identity on general float data, concurrent launches sharing one workspace (gotchas 4-6).

## Purpose

When a GEMM-shaped kernel runs **1 CTA per SM** and the number of output tiles is just above the SM count (a second
wave of a few CTAs) or far below it (idle SMs), split the work at the granularity of **units** = (output tile,
k-tile) instead of tiles: every CTA gets an equal contiguous unit range, and a tile whose k-range is spread over
several CTAs is completed by a cross-CTA f32 reduction through a global workspace.

## When needed (trigger)

- 1 CTA / SM (here: 82944 B dynamic smem, 384 threads) and `tiles % #SM` is small but nonzero, or `tiles < #SM` with
  a long K: the grid either runs a near-empty tail wave or leaves SMs idle.
- The non-contract dossier readings that flag it: `grid_size` = number of output tiles, not the SM count; after the
  change `grid_size` = `min(#SM, units)`.
- **What it was worth here (one reading, not a universal rate)**: a weight-streaming skinny-M FP8 GEMM (M <= 64,
  CTA tile N = 128, BK = 128, TMA + mbarrier ring feeding `LDSM` -> `QMMA.16832`, 110 SMs, dominant kernel at
  `dram_pct` 68 %) moved the gate geomean −8.2 % (0.0597 -> 0.0548 ms). Per structure:

  | Tiles x k-tiles per tile, on 110 SMs | Before -> after (device ms) | Change |
  |---|---|---|
  | 112 x 32, M = 1 | 0.0577 -> 0.0535 | −7.3 % |
  | 112 x 32, M = 16 | 0.0578 -> 0.0535 | −7.4 % |
  | 112 x 32, M = 64 | 0.0656 -> 0.0567 | −13.6 % |
  | 32 x 112, M = 16 | 0.0581 -> 0.0555 | −4.5 % |

  On the dominant kernel (112 tiles) `dram_pct` rose 68.16 -> 77.13, `grid_size` 112 -> 110, `stall_barrier`
  0.42 -> 1.93 (the fixup rendezvous), `regs_per_thread` 44 -> 40, spills 0 / 0 both sides.
  A shape whose tile count is a multiple of the SM count and whose K is short has nothing to gain and pays the
  fixup; that case was not measured.

## Minimal runnable example

The skeleton of the certified kernel, operator math and epilogue removed. `U` units in tile-major order
(unit `u` = k-tile `u % KT` of tile `u // KT`), `G` CTAs, `S`-stage `PipelineTmaAsync` ring, one TMA producer warp,
`NC` consumer threads on named barrier `BAR`.

### 1) Host: grid, workspace, counters

```python
import torch
num_sm = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
U = n_tiles * KT
G = min(num_sm, U)                      # G <= U -> every CTA owns >= 1 unit
n_slots = 2 * G                         # (CTA, first / last piece of its range) -- gotcha 2
ws  = torch.empty(n_slots * NC * nacc, dtype=torch.float32, device=dev)   # f32 partial slots
cnt = torch.zeros(n_tiles, dtype=torch.int32, device=dev)                  # zeroed ONCE -- gotcha 1
# cache (compiled kernel, ws, cnt) per shape key; pass ws / cnt as two extra pointer arguments
# workspace view: thread-contiguous fragments, 128-bit per access
mWS = cute.make_tensor(pWS, cute.make_layout((4, nacc // 4, NC, n_slots),
                                             stride=(1, 4, nacc, nacc * NC)))
# launch grid = [1, G, 1]
```

### 2) Kernel: this CTA's unit range

```python
u_start = (bidy * U) // G
u_end   = ((bidy + 1) * U) // G
first_tile   = u_start // KT
n_pieces_cta = (u_end - 1) // KT - first_tile + 1      # range cut at tile boundaries
```

### 3) Producer warp: one flat loop, one ring state, no drain at piece boundaries

```python
tBsB, tBgB_all = cute.nvgpu.cpasync.tma_partition(
    tma_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 2),
    cute.group_modes(cute.local_tile(tB, (BN, BK), (None, None)), 0, 2))   # all tiles, all k-tiles
ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, S)
for u in cutlass.range(u_start, u_end, unroll=1):
    tile = u // KT
    j = u - tile * KT
    pipe.producer_acquire(ps)
    cute.copy(tma_a, tAgA_all[None, j], tAsA[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
    cute.copy(tma_b, tBgB_all[None, tile, j], tBsB[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
    pipe.producer_commit(ps)
    ps.advance()
pipe.producer_tail(ps)
```

### 4) Consumers: per piece, mainloop over the piece's k-tiles, then the fixup

```python
st_ws = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)
ld_ws = cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), cutlass.Float32, num_bits_per_copy=128,
                            load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL)       # L2 (.cg) load
acc_v = cute.make_tensor(acc.iterator, cute.make_layout((4, nacc // 4)))
part  = cute.make_rmem_tensor(cute.make_layout((4, nacc // 4)), cutlass.Float32)

cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)   # also continuous over pieces
for p in cutlass.range(n_pieces_cta, unroll=1):
    tile = first_tile + p
    t_lo = tile * KT
    nk = cutlass.min(u_end, t_lo + KT) - cutlass.max(u_start, t_lo)        # >= 1
    c_first = ((t_lo + 1) * G - 1) // U          # owner(u) = ((u + 1) * G - 1) // U
    c_last  = ((t_lo + KT) * G - 1) // U
    acc.fill(0.0)
    # ... prologue / for t in cutlass.range(nk - 1) / peeled last k-tile, each consumer_wait /
    #     consumer_release / cs.advance() exactly as in a one-tile mainloop ...

    is_fin = cutlass.Int32(1)
    if c_last > c_first:                         # this tile is split across CTAs
        slot = 2 * bidy + cutlass.min(p, 1)
        cute.copy(st_ws, acc_v, mWS[None, None, ctid, slot])          # (1) publish the partial
        cute.arch.fence_acq_rel_gpu()                                  # (2)
        cute.arch.barrier(barrier_id=BAR, number_of_threads=NC)        # (3) all partial stores done
        if ctid == 0:
            old = cute.arch.atomic_add((mCnt.iterator + tile).llvm_ptr, cutlass.Int32(1),
                                       sem="acq_rel", scope="gpu")     # (4) arrive on the tile counter
            last = cutlass.Int32(0)
            if old == c_last - c_first:
                last = cutlass.Int32(1)
            sFlag[0] = last                                            # broadcast through smem
        cute.arch.barrier(barrier_id=BAR, number_of_threads=NC)        # (5)
        is_fin = sFlag[0]
        if is_fin != 0:                                                # (6) the last arriver finishes
            cute.arch.fence_acq_rel_gpu()
            for cc in cutlass.range(c_first, c_last + 1, unroll=1):
                if cc != bidy:
                    cslot = 2 * cc + 1                                 # peer's last piece ...
                    if ((cc * U) // G) // KT == tile:
                        cslot = 2 * cc                                 # ... or its first piece
                    cute.copy(ld_ws, mWS[None, None, ctid, cslot], part)
                    for i in cutlass.range_constexpr(nacc):
                        acc_v[i] = acc_v[i] + part[i]
            if ctid == 0:
                mCnt[tile] = cutlass.Int32(0)                          # self-reset for the next launch
    if is_fin != 0:
        pass  # ... the unchanged epilogue for this tile ...
```

`sFlag` is one `Int32` in the shared-storage struct. `ctid` is the consumer thread index 0..NC-1. Every consumer
thread owns the same fragment shape in every CTA, so slot `[*, *, ctid, s]` lines up element for element.

## Gotchas

1. **The tile counter must come back to 0 by itself, or the second launch is wrong.** The counters are zeroed once
   at allocation and the finishing CTA writes `0` after it has read every peer slot; there is no per-call clear and
   no second kernel (1 launch per call). The probe calls the kernel twice back to back on the same cached workspace
   and counters and requires an exact match both times (`PROBE PASS max_diff=0.0  # exact-int ... call2`): a
   missing reset makes call 2 mis-pick its last arriver.
2. **Two slots per CTA are enough.** A contiguous range cut at tile boundaries can share a tile with other CTAs only
   at its first and its last piece; every middle piece covers a whole tile, has `c_first == c_last` and skips the
   fixup. Slot `2c` holds CTA c's first piece and `2c + 1` its last (a one-piece range uses `2c`). The finishing CTA
   finds a peer's slot by asking whether the peer's range *starts* inside this tile. `G = min(#SM, U)` keeps every
   range non-empty.
3. **Keep both ring states continuous across pieces.** The producer walks the CTA's whole unit range in one flat loop
   with one `PipelineState`; the consumers keep one `PipelineState` over all pieces and only re-run the per-piece
   prologue / loop / peeled last k-tile with `nk` = the piece's k-tiles (`nk = 1` runs the prologue and the peeled
   tile only). Nothing is drained or re-initialised at a piece boundary. Same rule as a persistent tile loop
   ([`../sm90/persistent_tile_loop_ring_continuity`](../sm90/persistent_tile_loop_ring_continuity.md), certified on
   sm90).
4. **The ordering chain is certified as a whole, not element by element.** The shipped form is: partial store ->
   `fence_acq_rel_gpu` -> rendezvous of the NC consumer threads (the producer warp is not in it and keeps streaming
   the next piece) -> one `atomic_add(..., sem="acq_rel", scope="gpu")` -> smem flag -> rendezvous -> the last
   arriver's `fence_acq_rel_gpu` -> peer loads through `CopyG2ROp` with `LoadCacheMode.GLOBAL` (an L2 load). 20
   repeated launches were bit-identical on all four shapes. No variant with an element removed (either fence, the
   `.cg` load, relaxed atomic semantics) was run, so none of them is known to be safe.
5. **Bit reproducibility on general data is not certified.** The last arriver adds its own partial first and the
   peers in CTA order, so which CTA arrives last decides the f32 summation order. The bit-identity check was run on
   exact-integer operands with power-of-two scales, where every order gives the same f32 sum; random / range-extreme
   / sparse data were checked against a tolerance (worst relative error 9.542e-04) and for exact zeros.
6. **The workspace and the counters are state of the cached kernel.** The certified use is serialized launches on
   one stream. Two launches in flight at once on the same cached entry would share slots and counters; that was
   not exercised.
7. **How the probe catches a broken fixup** (reusable for any cross-CTA reduction): exact-integer operands in a
   small range plus power-of-two scales make every partial sum exact in f32 in any order, so the result must equal
   the float64 reference rounded once, bit for bit; call twice (counter reset), then 20 more launches compared bit
   for bit (a race shows up as a mismatching launch); then random, range-extreme and sparse-row/column data against
   a tolerance, with exact zeros required on the zeroed rows and columns. One subprocess per shape, so a hang or a
   sticky error stays in its own rung ([`gpu_probe_ladder`](gpu_probe_ladder.md)).

## Census footprint (contract axes moved by this change; neither is a knob)

Readings of the SASS dossier before -> after, same kernel otherwise (same `PipelineTmaAsync` ring, same atoms, same
two bulk-store waits):

- **`rings`**: `[]` -> `["stride=0x1,wrap=sel"]`. The ring source did not change; the producer's loop went from a
  grid-stride tile loop with a nested k loop to one flat unit loop, and the consumer's outer loop to a piece loop. In
  the same census the count axes moved too: `loops` `{copy-producer: 1, mma-consumer: 4}` -> `{copy-producer: 5,
  mma-consumer: 5}`, `mbar_objs` 9 -> 10, `mbar_tiers` `{CONST: 8, THREAD: 1}` -> `{CONST: 9, RING: 1}`,
  `mbar_unresolved` 0 -> 3, `edges` gained `mma-consumer -> copy-producer` and lost `copy-producer -> spin`. Per
  [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotchas 12 / 25 / 26, a ring appearing together with
  moving count axes is codegen, not a ring change.
- **`commit`**: `mma-consumer: DEPBAR {} LDGDEPBAR x0 CPWAIT {LE 0x1 x4}` ->
  `mma-consumer: DEPBAR {} LDGDEPBAR x0 CPWAIT {LE 0x0 x1, LE 0x1 x4}`. Both trees call
  `cp_async_bulk_wait_group(1, read=True)` once per store sub-tile (4 per tile) and `cp_async_bulk_wait_group(0,
  read=True)` once after the consumer's outer loop; the new `LE 0x0 x1` is the same final drain now counted inside a
  consumer region ([`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 26). No wait depth changed.
- **Where the −8.2 % came from**: not from either axis. The non-contract readings that moved with it are
  `grid_size` 112 -> 110 (no tail wave) and `dram_pct` 68.16 -> 77.13.

## Sources

- API symbols (CuTe DSL): `cutlass.pipeline.PipelineTmaAsync`, `pipeline.make_pipeline_state`,
  `cute.nvgpu.cpasync.tma_partition`, `cute.arch.atomic_add(ptr, val, sem=, scope=)`,
  `cute.arch.fence_acq_rel_gpu`, `cute.arch.barrier(barrier_id=, number_of_threads=)`,
  `cute.nvgpu.CopyUniversalOp`, `cute.nvgpu.CopyG2ROp(load_cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL)`.
- PTX ISA: memory consistency model (release / acquire patterns, `fence.acq_rel.gpu`, `atom` with `.acq_rel.gpu`),
  `ld.global.cg`.
- This repo's GPU certification: probe `notes/streamk_fixup_probe.py`, job 1794113745, 24/24 lines
  `PROBE PASS` after 1 round (exact-int `max_diff=0.0` on call 1, call 2 and 20 repeats per shape; worst
  `PROBE PASS max_diff=9.542e-04  # random ... zeros_ok=True`).
- KEEP in job 1794113794 (census job 1794113821): axis `rings` `[]` -> `["stride=0x1,wrap=sel"]`, axis `commit`
  `CPWAIT {LE 0x1 x4}` -> `{LE 0x0 x1, LE 0x1 x4}`, gate 0.0597 -> 0.0548 ms.
- Detailed provenance archive: docs/kb-provenance/streamk_cross_cta_fixup.md (for human review; not in any agent read set).
