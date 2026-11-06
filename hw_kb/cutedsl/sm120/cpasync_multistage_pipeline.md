# cpasync_multistage_pipeline (the sm120 GEMM mainloop: cp.async into S smem stages, `wait_group(S-2)` + `__syncthreads`, ldmatrix + `mma.sync`; what S buys and what it does not)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified (GPU probe `probes/sm120/mechanisms/cpasync_multistage_check.py`, `PROBE PASS cpasync-multistage contracts=19`,
> RTX PRO 5000, CuTe DSL 4.8.0): a 4096^3 GEMM, bf16 and e4m3, 128 B and 64 B of K per smem row, 2 and 3 stages -- all eight
> bit-exact (`max_diff=0.0`); 128 B rows x 4 stages (128 KiB) refused at launch; SASS: the loop carries exactly one
> `DEPBAR.LE SB0, 0x<S-2>` and the page S `LDGDEPBAR`; the orchestration dossier reads one mma-consumer loop with
> commit axis `CPWAIT {LE 0x<S-2> x1}`; NCU: the shared-memory occupancy limit goes 1 -> 2 CTA / SM when the stage
> footprint halves, active warps 4.00 -> 7.81; four more tile / warp / register-cap shapes bit-exact, and forcing a third
> CTA per SM with `min_blocks_per_mp=3` caps registers at 168 and spills (local-load sectors 0 -> 16.8 M, instructions x1.20).
>
> **What is certified**: numerics, the wait-depth footprint and the occupancy readings. The TFLOPS table is INFO
> (median of 20 launches per configuration, clocks unlocked, same run), no significance test.

## Purpose

sm120 has no TMA-fed warpgroup MMA and no TMEM: the simplest high-throughput mainloop is the Ampere-shaped one -- each
thread `cp.async`s 16 B chunks of the next K tile into one of S shared-memory stages while the warps `ldmatrix` the
current stage into registers and issue `mma.sync`. This entry pins its construction on sm_120a and, from measurements,
**which knob actually moves its throughput on this card**.

## When needed (trigger)

- Writing any GEMM-shaped mainloop on sm120 (dense / grouped GEMM, the QK / PV loops of attention).
- A kernel's tensor pipe sits around 70 % with ~4 active warps per SM.
- Choosing the stage count, the K depth per stage (bytes of K per smem row) and the warp count of a CTA.

## Minimal runnable example

The full kernel (CTA tile 128 x 128, 4 warps 2 x 2) is the probe's `PipeGemm`; the parts that are this mechanism:

```python
# per stage: (BM + BN) x row_bytes bytes; sA/sB = make_smem_layout_a/b(LayoutEnum.from_tensor(m), (BM, BN, bk), dt, S)
gA = cute.local_tile(mA, (BM, bk), (bidx, None))          # (BM, bk, k_tiles); same for gB with bidy
tpr = row_bytes // 16                                      # threads per smem row, 16 B each
g2s = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
                          dt, num_bits_per_copy=128)
tcopy = cute.make_tiled_copy_tv(g2s, cute.make_layout((nthr // tpr, tpr), stride=(tpr, 1)),
                                cute.make_layout((1, 128 // dt.width)))
thr = tcopy.get_slice(tidx)
tAgA, tAsA = thr.partition_S(gA), thr.partition_D(sA)      # (CPY, M, K, k_tiles) / (CPY, M, K, S)
lA = cpA.get_slice(tidx).partition_S(sA)                   # ldmatrix view (CPY, M, KB, S); cpA = make_tiled_copy_A(ldsm, tiled_mma)

for s in cutlass.range_constexpr(S - 1):                   # prologue: S-1 stages in flight
    cute.copy(tcopy, tAgA[None, None, None, s], tAsA[None, None, None, s])   # (+ B)
    cute.arch.cp_async_commit_group()
rd, wr, nxt = cutlass.Int32(0), cutlass.Int32(S - 1), cutlass.Int32(S - 1)
for kt in cutlass.range(k_tiles):                          # runtime trip count: one SASS loop
    cute.arch.cp_async_wait_group(S - 2)                   # -> DEPBAR.LE SB0, S-2: stage `rd` has landed
    cute.arch.sync_threads()                               # every warp is done reading slot `wr` (read last iteration)
    if nxt < k_tiles:
        cute.copy(tcopy, tAgA[None, None, None, nxt], tAsA[None, None, None, wr])  # (+ B)
    cute.arch.cp_async_commit_group()                      # commit even when nothing was issued: keeps the group count
    for kb in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        cute.copy(cpA, lA[None, None, kb, rd], rA[None, None, kb])          # (+ B)
        cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
    rd = rd + 1
    if rd == S:
        rd = cutlass.Int32(0)
    wr = wr + 1
    if wr == S:
        wr = cutlass.Int32(0)
    nxt = nxt + 1
```

Footprint of the loop (`orch_tools` dossier): `loops = {mma-consumer: 1}`, `commit = [mma-consumer: DEPBAR {} LDGDEPBAR x1
CPWAIT {LE 0x<S-2> x1}]`, `cache_ops = [LDGSTS.E.BYPASS.LTC128B.128]`, one symmetric `BAR.SYNC` rendezvous.

## Readings (RTX PRO 5000, 4096^3, CTA 128 x 128, 4 warps)

| dtype | bytes of K per row | stages | smem / CTA | CTA / SM (smem limit) | TFLOPS | % of `mma.sync` peak |
|---|---|---|---|---|---|---|
| bf16 | 128 | 2 / 3 | 64 / 96 KiB | 1 | 206 / 205 | 70 % |
| bf16 | 64 | 2 / 3 | 32 / 48 KiB | 2 | 233.5 / 233.4 | 80 % |
| bf16 | 128 | 4 | 128 KiB | -- | launch refused | -- |
| e4m3 | 128 | 2 / 3 | 64 / 96 KiB | 1 | 405 / 405 | 69 % |
| e4m3 | 64 | 2 / 3 | 32 / 48 KiB | 2 | 455 / 454 | 77 % |

NCU on e4m3, 3 stages: 128 B rows -> 4.00 active warps / SM, tensor pipe 73.0 %; 64 B rows -> 7.78 warps, 84.0 %. Both are
also limited to 2 CTA / SM by registers. (Peaks from [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md): 293 / 587 TFLOPS.)

How to read it:
- **The stage count is not the lever here; CTAs per SM are.** 2 and 3 stages read the same at both row widths: the K tile
  of a 4096-deep problem sits in L2 (95 % hit rate measured) and two in-flight stages already cover it. What moved the
  throughput 13 % (bf16) / 12 % (e4m3) is halving the stage footprint so that **two** CTAs fit the 100 KiB per SM:
  one CTA of 4 warps is one warp per sub-partition, which caps `mma.sync` near the 92 % the device probe measured
  for that case, before the loop's other instructions take their share.
- **The budget arithmetic**: smem / CTA = S x (BM + BN) x row_bytes (+ 1 KiB reserved per block); two CTAs per SM need
  <= ~49 KiB each, one CTA may take up to 101376 B. Four 128 B-row stages of a 128 x 128 tile (128 KiB) do not launch.
- **What did not help** (same run family, not separate contracts): an explicit ldmatrix register double buffer (load
  k-block kb+1 before the MMAs of kb) -- identical times, ptxas already hoists the `LDSM` of the fully unrolled k-block
  loop; 8 warps per CTA (2 x 4 or 4 x 2 warp grid, smaller warp tiles) -- 5-12 % slower than the 4-warp 64 B-row
  configuration; 32 B of K per row -- 19 % (bf16) / 22 % (e4m3) slower (a `__syncthreads` per K tile amortised over half
  the MMAs).
- The next limit at this tile is registers (2 CTA / SM, 198 per thread): a 64 x 64 fp32 accumulator per warp is 128
  registers per thread before fragments. **Pulling that lever does not pay** -- see the next table.

### The register lever, measured (e4m3, 64 B rows, 2 stages: shared memory admits 3 CTA / SM)

| CTA tile / warps (warp tile) | registers / thread | CTA / SM (regs, smem) | active warps / SM | TFLOPS |
|---|---|---|---|---|
| 128 x 128 / 4 (64 x 64) | 198 | 2, 3 | 7.8 | **454** |
| 128 x 64 / 4 (64 x 32) | 128 | 4, 4 | -- | 433 |
| 128 x 128 / 8 (64 x 32) | 114 | 2, 3 | 15.3 (3-stage run) | 427 |
| 128 x 128 / 4, `min_blocks_per_mp=3` | 168 | 3, 3 | 11.2 | 417 (spills: 16.8 M local-load sectors, x1.20 instructions) |

- Every way of buying more warps cost more than it bought: a smaller warp tile issues more `LDSM` and address math per
  `mma.sync` (less fragment reuse), and capping registers at 64 x 64 makes ptxas spill the accumulators.
- `min_blocks_per_mp=N` on `launch(...)` is the CuTe DSL spelling of `__launch_bounds__(threads, N)` (it lowers to
  `nvvm.minctasm`); it does what it says -- registers <= 65536 / (N x threads) -- and the spills it causes are visible
  only in `spill_ld_sectors` / `spill_st_sectors`, not in the numerics.
- At this tile the mainloop is ~84 % tensor-pipe busy with 53 % of its executed instructions being `QMMA` (16.8 M of
  31.8 M, `smsp__inst_executed_pipe_tensor.sum`, an NCU reading taken beside the probe, not a contract); the rest is
  `LDSM`, `LDGSTS`, address arithmetic and the per-tile barrier. Going further needs a different structure (the copy
  and its address math off the MMA warps), not more warps of the same loop.

## Gotchas

1. **Wait depth is `S-2`, not `S-1`.** With S-1 groups committed in the prologue and one per iteration, the tile read in
   this iteration is the oldest of S-1 outstanding groups: `cp.async.wait_group(S - 2)` (SASS `DEPBAR.LE SB0, 0x<S-2>`)
   is the deepest wait that still guarantees it landed. A larger immediate would let the MMAs read a slot whose copy
   has not landed, with no error raised (not exercised by the probe: every certified configuration uses `S - 2`).
2. **Commit every iteration, even past the last tile.** `wait_group(N)` counts *groups*; skipping the commit when there is
   nothing left to load shifts the count, and the last S-2 tiles are consumed before they land.
3. **The `__syncthreads` after the wait is what makes slot reuse safe.** The slot written this iteration is the one every
   warp read in the previous iteration; the barrier after the wait is the only thing ordering those reads before the
   new `LDGSTS`. One `BAR.SYNC` per K tile is the cost of the whole scheme -- which is why very shallow K tiles lose.
4. **Loop-carried stage indices are Runtime values.** `rd`, `wr`, `nxt` must be bound (as `cutlass.Int32`) before the
   `cutlass.range` loop and only reassigned inside it; index the staged views with them directly
   (`tAsA[None, None, None, wr]`). See [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md) for the scope rule.
5. **`LoadCacheMode.GLOBAL` lands as `LDGSTS.E.BYPASS.LTC128B.128` inside this loop** (the single-tile kernel of
   [`warp_mma_fp8_ldsm`](warp_mma_fp8_ldsm.md) landed `LDGSTS.E.BYPASS.128`): the L2 prefetch width is chosen by the
   compiler; read it from `cache_ops`, do not assume it.
6. **The fill side is invisible to the default smem counters** (cp.async fills): judge it by
   `smem_wavefronts_ldgsts / ldgsts_inst` on sm_120 cards ([`smem_traffic_triage`](smem_traffic_triage.md)).
7. **A compact mainloop is ~350 SASS instructions**, under the 500-instruction floor the SASS tools used to demand of a
   tensor kernel; pages that issue an `mma.sync` family are analysed from 200 instructions up
   ([`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 21).

## Sources

- PTX ISA: `cp.async` (`.cg`, 16 B), `cp.async.commit_group`, `cp.async.wait_group`, `bar.sync`, `ldmatrix`, `mma.sync`:
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA C++ Programming Guide: asynchronous data copies (`cp.async` pipelines), shared memory per SM / per block on
  compute capability 12.0: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Nsight Compute metrics `launch__occupancy_limit_shared_mem`, `launch__occupancy_limit_registers`,
  `sm__warps_active.avg.per_cycle_active`, `sm__pipe_tensor_cycles_active`: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- CuTe DSL API symbols: `cute.local_tile`, `cute.make_tiled_copy_tv`, `cute.nvgpu.cpasync.CopyG2SOp`,
  `cute.arch.cp_async_commit_group`, `cute.arch.cp_async_wait_group`, `cute.arch.sync_threads`, `cutlass.range`,
  `cute.nvgpu.warp.MmaF16BF16Op`, `cute.nvgpu.warp.MmaFP8Op`, `cute.nvgpu.warp.LdMatrix8x8x16bOp`,
  `cutlass.utils.hopper_helpers.make_smem_layout_a` / `make_smem_layout_b`.
- GPU probe `probes/sm120/mechanisms/cpasync_multistage_check.py`: `PROBE PASS cpasync-multistage contracts=19`
  (`numerics-{bf16,fp8}-r{128,64}-s{2,3}`, `smem-ceiling`, `numerics-shape-*` x4, `sass-wait-depth-s{2,3}`,
  `orch-commit-s{2,3}`, `ncu-occupancy`, `ncu-register-cap-spills`).
- CuTe DSL launch option `min_blocks_per_mp` (`LaunchConfig`, lowered to `nvvm.minctasm`); PTX ISA performance-tuning
  directive `.minnctapersm`.
