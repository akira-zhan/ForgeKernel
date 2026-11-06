# tma_mbarrier_mainloop (the sm120 GEMM mainloop fed by TMA into an mbarrier ring: `make_tiled_tma_atom` + `PipelineTmaAsync`; in-loop producer vs a dedicated producer warp)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified (GPU probe `probes/sm120/mechanisms/tma_mbarrier_mainloop_check.py`, `PROBE PASS tma-mbarrier contracts=12`,
> RTX PRO 5000, CuTe DSL 4.8.0): a 4096^3 GEMM (C fp32 = A . B^T, K-major, CTA 128 x 128, 4 MMA warps) with the supply
> done by TMA into a 3-stage full / empty mbarrier ring, in two producer shapes, bf16 and e4m3, every consumer fencing
> (`fence_view_async_shared`) before it releases a stage -- all five configurations bit-exact, the in-loop e4m3 kernel
> exact on 30 of 30 launches; SASS: `UTMALDG.2D`, no `LDGSTS`, mbarrier try-waits, exactly one `BAR.SYNC` on the page,
> no stage release nominated by `orch_tools.xproxy_releases` (and the unfenced build nominated `bare`, the control); the
> orchestration dossier resolves 6 mbarrier objects (full + empty x 3 stages, all CONST) and, for the producer-warp
> shape, both edge directions; NCU: in-loop producer 2 CTA / SM (tensor pipe 85.7 %), producer warp 1 CTA / SM (78.2 %).
>
> **What is certified**: numerics, the footprint and the occupancy readings. The TFLOPS table is INFO (median of 20
> launches, clocks unlocked), no significance test.
>
> **Correction (this revision)**: the first version of this entry released stages without the proxy fence. That kernel
> passed every contract (and read 0 wrong in 1000 launches of each shape since, e4m3 at 4096^3) but the same release pattern races measurably
> in [`blockscaled_mma`](blockscaled_mma.md) -- see gotcha 1. The readings below are of the fenced kernels.

## Purpose

Replace the cp.async supply of [`cpasync_multistage_pipeline`](cpasync_multistage_pipeline.md) with TMA: one elected
thread issues a 2D bulk tensor copy per operand per K tile, the transaction count on a per-stage **full** mbarrier tells
the consumers when the tile landed, and a per-stage **empty** mbarrier tells the producer when every consumer warp is
done with the slot. No thread computes a per-element address, and there is no per-tile `__syncthreads`.

## When needed (trigger)

- An sm120 mainloop whose tensor pipe sits in the low 80s % with the cp.async supply already at 2 CTA / SM.
- Writing the supply side of any sm120 tensor kernel (GEMM, attention) that reads rectangular tiles.
- Choosing between a dedicated producer warp and issuing the copies from a consumer warp.

## Minimal runnable example

```python
import cutlass.pipeline as pipeline
# host side (@cute.jit): one atom per operand; the smem layout passed is ONE stage of the staged layout
sA_1 = cute.slice_(sA_l, (None, None, 0))            # sA_l = make_smem_layout_a(..., stages=S)
op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(op, mA, sA_1, (BM, bk), num_multicast=1)   # (+ B)
tx = cute.size_in_bytes(dt, sA_1) + cute.size_in_bytes(dt, sB_1)   # bytes landing per stage, both operands

@cute.struct
class Storage:
    bars: cute.struct.MemRange[cutlass.Int64, S * 2]                      # full[S] + empty[S]
    sA: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sA_l)], 1024]
    sB: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sB_l)], 1024]
# launch(..., block=[128, 1, 1], smem=Storage.size_in_bytes())

# kernel side
st = SmemAllocator().allocate(Storage)
sA = st.sA.get_tensor(sA_l.outer, swizzle=sA_l.inner)                     # (+ sB)
pipe = pipeline.PipelineTmaAsync.create(
    num_stages=S, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),   # one arrive per consumer WARP
    tx_count=tx, barrier_storage=st.bars.data_ptr(), cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))
gA = cute.local_tile(tA, (BM, bk), (bidx, None))                          # tA = the TMA coordinate tensor
tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(tma_a, 0, cute.make_layout(1),
                                              cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))

def issue(ps, kt):                                    # called by warp 0 only (the whole warp)
    pipe.producer_acquire(ps)                         # wait slot empty, arm full with tx bytes
    cute.copy(tma_a, tAgA[None, kt], tAsA[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
    cute.copy(tma_b, tBgB[None, kt], tBsB[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
    pipe.producer_commit(ps)                          # no-op for TMA: the copy completes the transaction
    ps.advance()

ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, S)
cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)
# in-loop producer: warp 0 fills all S stages, then refills the slot every consumer just released
# for kt in cutlass.range(k_tiles):
#     pipe.consumer_wait(cs, pipe.consumer_try_wait(cs))
#     ... ldmatrix + mma.sync over stage cs.index (as in warp_mma_fp8_ldsm) ...
#     cute.arch.fence_view_async_shared()   # REQUIRED (gotcha 1): ldmatrix reads before the TMA refill of the slot
#     pipe.consumer_release(cs); cs.advance()
#     if warp == 0 and next tile < k_tiles: issue(ps, next tile)
```

## Readings (RTX PRO 5000, 4096^3, CTA 128 x 128, 4 MMA warps)

| Supply | dtype | K bytes / row x stages | threads | CTA / SM | TFLOPS | tensor pipe |
|---|---|---|---|---|---|---|
| cp.async ([`cpasync_multistage_pipeline`](cpasync_multistage_pipeline.md)) | e4m3 | 64 x 2-3 | 128 | 2 | 455 | 83.7 % |
| TMA, warp 0 issues in-loop | e4m3 | 64 x 3 | 128 | 2 | **468** | **85.7 %** |
| TMA, warp 0 issues in-loop | bf16 | 64 x 3 | 128 | 2 | **249** (85 % of peak) | -- |
| TMA, dedicated producer warp | e4m3 | 64 x 3 | 160 | 1 | 428 | 78.2 % |
| TMA, dedicated producer warp | e4m3 | 128 x 2 | 160 | 1 | 444 | -- |
| TMA, dedicated producer warp | bf16 | 64 x 3 | 160 | 1 | 225 | -- |

(cp.async bf16 best: 233.5. Peaks from [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md): 293 / 587. The
unfenced first version read 463 / 248 / 426 / 442 / 225 and 85.1 % / 78.1 %: the fence costs nothing measurable.)

How to read it:
- **TMA in-loop is the best supply of the three at this tile: +3 % e4m3, +7 % bf16 over cp.async.** Not by executing
  fewer instructions (31.78 M vs 31.87 M, same tile) but by 20 fewer registers per thread (178 vs 198) and no CTA-wide
  barrier per K tile -- each warp waits only on the stage it reads.
- **A dedicated producer warp costs the second CTA per SM.** NCU reads a register limit of 1 CTA / SM for the 5-warp CTA
  at 178 registers even though 2 x 160 x 178 < 64 K; the explanation consistent with it (not separately probed):
  registers are allocated per warp on a sub-partition (16 K each), and a second 5-warp CTA needs three such warps on
  one sub-partition. Its smem (3 x 16 KiB + barriers = 50.2 KiB) also just misses
  the two-per-SM budget. With 4 MMA warps per SM it still beats cp.async at 1 CTA / SM (428-444 vs 405).
- The stage count moved nothing measurable here either (e4m3 64 B rows: 2 / 3 / 4 / 6 stages read 422-426 with the
  producer warp; 463.5 / 463.8 for 2 / 3 stages in-loop; this sweep was taken on the unfenced first version).

## Gotchas

1. **Fence before every `consumer_release`: `cute.arch.fence_view_async_shared()`.** The consumers read the stage with
   `ldmatrix` -- a generic-proxy access -- and the release lets the producer refill the slot with TMA, which writes
   through the async proxy. PTX orders accesses of different proxies only through a proxy fence (ISA 8.6, "A *proxy
   fence* is required to synchronize memory operations across different *proxies*"; `.async.bulk` accesses are listed
   there as non-generic). In SASS the unfenced release (`SYNCS.ARRIVE.TRANS64.A1T0`) sits right after the stage's last
   `LDSM`, waiting on no scoreboard, ahead of the MMAs that consume those registers: the refill can land before the
   read. The fence lowers to `MEMBAR.ALL.CTA` + `FENCE.VIEW.ASYNC.S`, and the release then waits on the fence's
   scoreboard (decoded on this probe's builds: inline writes SB3 and the release waits on SB3, producer-warp SB4 / SB4). Measured (block-scaled e4m3, the same pipeline): 782 of 1000 launches wrong without the fence, 0 of 1000
   with it; this entry's plain kernels read 0 of 1000 unfenced -- a narrow window, not a safe pattern
   ([`blockscaled_mma`](blockscaled_mma.md) gotcha 1 has the table, including a 1-in-1000 failure with no global
   loads in the loop). On Hopper the same `PipelineTmaAsync` usage needs no fence because its consumers are wgmma,
   which read smem through the async proxy (PTX ISA 9.7.17.4 "Async Proxy"; listed in the table of contents, body not
   re-read in this pass); copying that usage onto `mma.sync` consumers is the bug. Check a
   page with `orch_tools.xproxy_releases` (`bare` = nothing between read and release; `barrier-only` = a `BAR.SYNC`
   but no fence, the CUTLASS C++ sm120 mainloop shape).
2. **The TMA tensor is a coordinate tensor.** `make_tiled_tma_atom` returns `(atom, tensor)`; pass that tensor (not the
   original) to the kernel and `local_tile` it. Its `element_type` is not the data type -- `mA.element_type` inside the
   kernel raised `AttributeError: mlir_type` at `make_copy_atom`; carry the dtype as a Python-level constant.
3. **Pass one stage of the smem layout to `make_tiled_tma_atom`** (`cute.slice_(staged, (None, None, 0))`) and index
   the stage on the partitioned smem view (`tAsA[None, stage]`).
4. **`tx_count` is the bytes of everything that completes on that barrier**: with A and B on one barrier it is the sum
   of both stage sizes. (By the transaction-count semantics a short count lets the barrier complete before all bytes
   land and a long one never completes; neither case is exercised by the probe.)
5. **The consumer group counts warps, not threads**: `CooperativeGroup(Agent.Thread, 4)` for 4 consumer warps; one
   lane per warp performs the empty-barrier arrive in `consumer_release`.
6. **In-loop issue order: release, then acquire.** Warp 0 refills a slot only after its own `consumer_release`;
   `producer_acquire` then waits for the other three warps to release the same slot. Acquiring before its own release
   would make warp 0 wait for an arrive only it can perform (not exercised by the probe).
7. **A dedicated producer must drain before exit** (`producer_tail`), or the kernel ends with arrives in flight.
8. **Do not compute the register limit from the SM total.** `launch__occupancy_limit_registers` read 1 for a 5-warp
   CTA whose total register bill fits the SM twice (see Readings); read the limit from NCU, size warp-specialized CTAs
   in multiples of 4 warps, or budget for 1 CTA / SM.
9. **The mbarrier address in SASS is `ULEA Ux, SR_CgaCtaId-reg, offset-reg, 0x18`** (shared window base + offset), with
   the two source registers in either order depending on allocation; read the offset from the addend
   ([`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 23).

## Sources

- PTX ISA: `cp.async.bulk.tensor` (2D tile, `.mbarrier::complete_tx::bytes`), `mbarrier.init`,
  `mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity`, `prefetch.tensormap`, `elect.sync`; memory consistency
  model 8.6 "Proxies" (the proxy-fence sentence quoted in gotcha 1; "async-proxy and tensormap-proxy accesses by
  .async.bulk operations" in its list of non-generic accesses) and 8.9.5 (proxy-preserved base causality order):
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CuTe DSL 4.8.0 `cute.arch.fence_view_async_shared` docstring (`cutlass/cute/arch/nvvm_wrappers.py`): "The fence is
  required to synchronize the shared memory load/store and let the pipeline release or commit the buffer. This
  function is usually used for async execution unit (like TMA, UMMA) after the load/store operations."
- CUDA C++ Programming Guide: the Tensor Memory Accelerator, asynchronous barriers:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CuTe DSL API symbols: `cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp`, `cute.nvgpu.cpasync.make_tiled_tma_atom`,
  `cute.nvgpu.cpasync.tma_partition`, `cute.nvgpu.cpasync.prefetch_descriptor`, `cutlass.pipeline.PipelineTmaAsync`
  (`create`, `producer_acquire`, `producer_get_barrier`, `producer_commit`, `producer_tail`, `consumer_try_wait`,
  `consumer_wait`, `consumer_release`), `cutlass.pipeline.CooperativeGroup`, `cutlass.pipeline.make_pipeline_state`,
  `cute.struct.MemRange`, `cute.struct.Align`, `cute.arch.make_warp_uniform`, `cute.arch.warp_idx`.
- GPU probe `probes/sm120/mechanisms/tma_mbarrier_mainloop_check.py`: `PROBE PASS tma-mbarrier contracts=12`
  (`numerics-{inline,ws}-{fp8,bf16}-r64-s3`, `numerics-ws-fp8-r128-s2`, `stress-inline-fp8`, `sass-{inline,ws}`,
  `orch-{inline,ws}`, `ctl-unfenced-inline`, `N-occupancy`); fixtures `tests/fixtures/sm120/tma_mbarrier_fp8_{inline,ws}.sass`
  (fenced) and `tma_mbarrier_fp8_inline_unfenced.sass` (the control).
- Instruction totals (31.78 M / 31.87 M): `smsp__inst_executed.sum` from the NCU contracts of this probe and of
  `probes/sm120/mechanisms/cpasync_multistage_check.py`.
