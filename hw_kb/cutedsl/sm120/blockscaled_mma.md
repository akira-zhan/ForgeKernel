# blockscaled_mma (block-scaled `mma.sync` on sm_120a: MXF8 `QMMA.SF`, MXF4 / NVF4 `OMMA.SF`, fed by TMA into an mbarrier ring)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified (GPU probe `probes/sm120/mechanisms/blockscaled_mma_check.py`, `PROBE PASS blockscaled-mma contracts=15`,
> RTX PRO 5000, CuTe DSL 4.8.0): C (fp32) = (A x SFA) . (B x SFB)^T at 2048 x 2048 x 4096 for mxf8 (e4m3, ue8m0 scale
> per 32), mxf4 (e2m1, ue8m0 per 32) and nvf4 (e2m1, ue4m3 per 16) -- exact, and exact on 30 of 30 launches each; the K
> mapping exact under a selector pattern; three controls reproduce the three faults met while building it (no proxy
> fence before the stage release: 22 of 30 launches wrong; the element-unit swizzle on the e2m1 view: half the outputs
> wrong; `recast_tensor` dropping the swizzle); SASS `QMMA.SF.16832.F32.E4M3.E4M3.E8` /
> `OMMA.SF.16864.F32.E2M1.E2M1.E8` / `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, no stage release nominated by
> `orch_tools.xproxy_releases`.
>
> **What is certified**: numerics, the K mapping, the three controls, the SASS forms. The TFLOPS readings are INFO
> (median of 20 launches, clocks unlocked) and come from a scale-factor supply that is **not** the native recipe (see
> the next section): a correctness baseline, not a performance recipe.

## Purpose

Run the sm_120a block-scaled tensor-core instructions from CuTe DSL with an exact check: which layouts the operands
need (bytes for TMA, e2m1 elements for ldmatrix and the MMA), how the scale-factor fragments are built, and the one
pipeline rule that makes a TMA + ldmatrix mainloop correct.

## Native on sm_120a vs. borrowed from another architecture (read first)

| Piece | Native on sm_120a? | This probe |
|---|---|---|
| `mma.sync.aligned ... kind::mxf8f6f4.block_scale` (`QMMA.SF`), `kind::mxf4.block_scale` and `kind::mxf4nvf4.block_scale.scale_vec::4X` (`OMMA.SF`) | yes, sm120's own (not in sm90; sm100 scales through tcgen05 instead); needs the arch-specific target `sm_120a`; warp-level, accumulator in registers | `MmaMXF8Op` / `MmaMXF4Op` / `MmaMXF4NVF4Op` |
| TMA (`cp.async.bulk.tensor`) + mbarrier full / empty ring | yes | `make_tiled_tma_atom` + `PipelineTmaAsync`, warp 0 issues in-loop ([`tma_mbarrier_mainloop`](tma_mbarrier_mainloop.md)) |
| Consumers read smem with `ldmatrix` | yes -- there is no wgmma on sm120 | needs `fence_view_async_shared()` before `consumer_release` (gotcha 1). The Hopper usage of `PipelineTmaAsync` has wgmma consumers and no such fence (wgmma reads smem through the async proxy: PTX ISA 9.7.17.4 "Async Proxy", listed in the table of contents, body not re-read in this pass); copying it onto `ldmatrix` consumers is the race this entry measured |
| Scale-factor fragments `partition_fragment_SFA/SFB`, `thrfrg_SFA/SFB` (`cutlass.utils.blackwell_helpers`) | yes, sm120's own despite the module name: "Thread-fragment scale factor A tensor for SM120 block-scaled MMA", a port of `mma_traits_sm120.hpp` | used as is (the SFB fragment regrouped, gotcha 6) |
| Scales loaded gmem -> registers by each thread, once per K tile (`autovec_copy`) | **not** the CUTLASS recipe | CUTLASS C++ (`sm120_blockscaled_mma_tma.hpp`) TMA-loads SFA / SFB into smem (Sm1xx scale-factor layout) in the same stage as A / B and copies them smem -> registers per k-block. Not probed here yet |
| The e2m1 smem layout from `hopper_helpers.make_smem_layout_a` with an e2m1 type | no: that helper swizzles in element units (gotcha 4) | layout built on a 1-byte type; e2m1 view with element strides and the byte-unit swizzle (gotcha 2) |

## When needed (trigger)

- Writing an MXF8 / MXF4 / NVF4 GEMM or attention mainloop on sm120 in CuTe DSL.
- An sm120 TMA mainloop that is exact on most launches and wrong on some -- more often when the loop also issues
  global loads.
- An FP4 kernel exact on random data but wrong on structured data, an FP4 TMA copy dying with an illegal address, or
  a cp.async misalignment on an FP4 tile.

## Minimal runnable example

```python
# host side (@cute.jit) -- mxf4 shown; mxf8: MmaMXF8Op(Float8E4M3FN, Float32, Float8E8M0FNU), atom (16, 8, 32), V = 32;
# nvf4: MmaMXF4NVF4Op(Float4E2M1FN, Float32, Float8E4M3FN), atom (16, 8, 64), V = 16
BY = cutlass.Float8E4M3FN                               # 1-byte container: TMA and the smem layout work in bytes
op = cute.nvgpu.warp.MmaMXF4Op(cutlass.Float4E2M1FN, cutlass.Float32, cutlass.Float8E8M0FNU)
tiled_mma = cute.make_tiled_mma(op, cute.make_layout((2, 2, 1)), permutation_mnk=(32, 32, 64))
KB_ = K // 2                                             # bytes per row of packed e2m1 (element 2i = low nibble)
mA = cute.make_tensor(cute.recast_ptr(mA_raw.iterator, dtype=BY), cute.make_layout((M, KB_), stride=(KB_, 1)))
sA_l = hh.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (BM, BN, row_bytes), BY, S)   # byte swizzle: S<2,4,3> for 64-B rows
tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(), mA,
                                                   cute.slice_(sA_l, (None, None, 0)), (BM, row_bytes), num_multicast=1)
# scales: (rows, K / V) bytes viewed as a (rows, K) tensor that is constant over each V-block of K
sfa = cute.make_tensor(cute.recast_ptr(mSFA.iterator, dtype=cutlass.Float8E8M0FNU),
                       cute.make_layout((M, (V, K // V)), stride=(K // V, (0, 1))))

# kernel side
sA_b = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), sA_l.inner, dtype=BY), sA_l.outer)  # TMA destination
lA4 = cute.recast_layout(4, 8, sA_l)
sA = st.sA.get_tensor(lA4.outer, swizzle=sA_l.inner)    # e2m1 view: element strides, BYTE-unit swizzle (gotcha 2)
gSFA = cute.local_tile(sfa, (BM, bk), (bidx, None))     # bk = K elements per stage
tCrSFA = bwh.partition_fragment_SFA(gSFA[None, None, 0], thr_mma, tidx)
tCrSFB = bwh.partition_fragment_SFB(gSFB[None, None, 0], thr_mma, tidx)
tCrSFB = cute.make_tensor(tCrSFB.iterator, cute.group_modes(cute.flatten(tCrSFB.layout), 0, 2))   # (V, N, K) (gotcha 6)
frgA = bwh.thrfrg_SFA(gSFA[None, None, 0].layout, thr_mma)
vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
for kt in cutlass.range(k_tiles):
    vA = cute.make_tensor(gSFA[None, None, kt].iterator, frgA)                   # this thread's scales of tile kt
    cute.autovec_copy(cute.group_modes(cute.flatten(vA[(vmnk[0], (vmnk[1], vmnk[3])), (None, None)]), 0, 2), tCrSFA)
    # (+ the same for B with vmnk[2])
    pipe.consumer_wait(cs, pipe.consumer_try_wait(cs))
    for kb in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        cute.copy(cpA, lA[None, None, kb, cs.index], rA[None, None, kb])        # ldmatrix: generic-proxy reads
        cute.copy(cpB, lB[None, None, kb, cs.index], rB[None, None, kb])
        cute.gemm(tiled_mma, acc, [tCrA[None, None, kb], tCrSFA[None, None, kb]],
                  [tCrB[None, None, kb], tCrSFB[None, None, kb]], acc)
    cute.arch.fence_view_async_shared()                  # REQUIRED (gotcha 1)
    pipe.consumer_release(cs); cs.advance()
    # warp 0: producer_acquire + the two TMA copies of the next K tile into the slot just released
```

The full kernel (in-loop producer, 3 stages x 64-B rows, CTA 128 x 128, 4 MMA warps):
`probes/sm120/mechanisms/blockscaled_mma_check.py`.

## Readings (RTX PRO 5000, CTA 128 x 128, 4 MMA warps, 3 stages x 64 B rows)

| Kind | SASS MMA (64 per K tile) | PRMT | TFLOPS 4096^3 | of the kind's peak | TFLOPS 2048^2 x 4096 |
|---|---|---|---|---|---|
| mxf8 (e4m3, ue8m0 / 32) | `QMMA.SF.16832.F32.E4M3.E4M3.E8` | 24 | 275 | 47 % of 587 | 215 |
| mxf4 (e2m1, ue8m0 / 32) | `OMMA.SF.16864.F32.E2M1.E2M1.E8` | 12 | 518 | 44 % of 1173 | 399 |
| nvf4 (e2m1, ue4m3 / 16) | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | 0 | 537 | 46 % of 1173 | 368 |

How to read it:
- **A correctness baseline, not a recipe.** The plain e4m3 GEMM of [`tma_mbarrier_mainloop`](tma_mbarrier_mainloop.md)
  reaches 468 TFLOPS (80 % of peak) on the same pipeline at 4096^3; block-scaled e4m3 here reaches 275. Candidates
  for the gap, none probed yet: the per-thread scale loads issued every K tile (global loads in the consumer loop),
  the stage geometry (64-B rows), the scale-byte packing (`PRMT`, see gotcha 7).
- 2048 x 2048 is 256 CTAs on 110 SMs -- a partial second wave whatever the occupancy -- so its column reads lower than
  4096^3 for that reason too.

## Gotchas

1. **Fence before every `consumer_release`: `cute.arch.fence_view_async_shared()`.** ldmatrix reads the stage
   through the generic proxy; the TMA refill after the release writes it through the async proxy; PTX orders the two
   only through a proxy fence (ISA 8.6 "Proxies": "A *proxy fence* is required to synchronize memory operations
   across different *proxies*", with "async-proxy and tensormap-proxy accesses by .async.bulk operations" among the
   non-generic accesses). Measured on this kernel (mxf8 2048 x 2048 x 4096; launches of one compiled kernel):

   | Before `consumer_release` | Scales in the loop | Wrong launches |
   |---|---|---|
   | nothing | per-tile global loads | 782 / 1000 (probe control: 22 / 30) |
   | nothing, all scales 1 | per-tile global loads | 724 / 1000 |
   | nothing, all scales 1 | loaded once before the loop | **1 / 1000** |
   | `bar.sync 1, 128` of the 4 MMA warps (the choice of the CUTLASS C++ sm120 collectives) | per-tile global loads | 1 / 1120 (1 / 100, then 0 / 1000 and 0 / 20) |
   | `fence_view_async_shared()` | per-tile global loads | **0 / 1000** for each of mxf8, mxf4, nvf4 |
   | nothing -- the plain bf16 / e4m3 kernels of [`tma_mbarrier_mainloop`](tma_mbarrier_mainloop.md) | none | 0 / 1000 (inline), 0 / 1000 (producer warp) |

   (Rows other than the probe control: development harness on the probe's kernel, INFO.) SASS of the three shapes
   (mxf8, decoded scoreboard bits): unfenced, `SYNCS.ARRIVE.TRANS64.A1T0` follows the stage's last two `LDSM`
   (scoreboards 1, 2) with an empty wait mask, ahead of the `QMMA.SF` that consume them; with the barrier,
   `BAR.SYNC.DEFER_BLOCKING 0x1, 0x80` and the arrive both have empty wait masks -- the LDSMs can still be in flight;
   with the fence, `MEMBAR.ALL.CTA`, `FENCE.VIEW.ASYNC.S` (writes scoreboard 3), and the arrive waits on scoreboard 3.
   So: the release does not wait for the reads; global loads in the loop widen the window from 1 / 1000 to 724 / 1000
   launches (unit scales); a barrier narrows it but does not order the proxies; only the fence makes the arrive depend on the
   reads, at no measurable cost ([`tma_mbarrier_mainloop`](tma_mbarrier_mainloop.md): 468 vs 463 TFLOPS).
   References that release without a proxy fence: CUTLASS C++ `sm120_mma_tma.hpp` / `sm120_blockscaled_mma_tma.hpp`
   (`NamedBarrier::sync` of the MMA threads, then `consumer_release`) and the CuTe DSL example
   `blackwell_geforce/dense_gemm.py` (release at the last k-block with neither) -- not run here; the CUTLASS C++
   sm120 GEMM page in `tests/fixtures/sm120/` is nominated `barrier-only` by `orch_tools.xproxy_releases`, while the
   same page fences its tile-scheduler pipeline (`LDS`, `MEMBAR.ALL.CTA`, `FENCE.VIEW.ASYNC.S`, release).
2. **The e2m1 smem view keeps the BYTE-unit swizzle.** TMA writes bytes through the byte layout (`S<2,4,3>` for 64-B
   rows, `S<3,4,3>` for 128-B rows). The e2m1 view takes its strides from `recast_layout(4, 8, byte_layout).outer`
   (element units) but its swizzle from `byte_layout.inner` -- in DSL 4.8.0 the swizzle of a sub-byte smem pointer acts
   on byte addresses (observed: only this combination is exact under the selector). `recast_layout(4, 8, ...)` also
   rescales the swizzle to element units (`S<3,5,3>` for 128-B rows); using it read 8192 of 16384 outputs wrong under
   the selector (`ctl-f4-elem-swizzle`).
3. **`recast_tensor` drops the swizzle.** `recast_tensor(byte_view, Float4E2M1FN)` of
   `ptr<f8E4M3FN, smem, align<1024>, S<3,4,3>>` gives `ptr<f4E2M1FN, smem, align<1024>>` (`ctl-recast-tensor`): build
   the e2m1 view with `get_tensor(layout, swizzle=...)` or `recast_ptr(ptr, swizzle, dtype=...)` instead.
4. **TMA and the smem layout on a 1-byte type.** An e2m1-typed TMA copy died with an illegal address and the Hopper
   `make_smem_layout_a` called with `Float4E2M1FN` swizzles in element units (`S<3,4,3>` = 8-byte chunks), which broke
   the 16-byte alignment of a cp.async fill (both seen while building the probe, not contracts): give TMA and the
   layout helper a 1-byte type (`Float8E4M3FN` as a container) over the packed bytes.
5. **Exact on random data does not prove the operand mapping.** A K permutation applied the same way to A and B
   leaves A . B^T unchanged: the swizzle fault of gotcha 2 read exact on random integers. The `kmap-*` contracts use a
   selector: B[n, k] = 1 at k = n + off and 0 elsewhere, A encodes k (16 distinguishable values), unit scales, so C[m, n]
   must equal A[m, n + off] -- at off = 0 and 128.
6. **Regroup the SFB fragment.** `partition_fragment_SFB` returns (V, (N, K)); index it as (V, N, K) after
   `cute.group_modes(cute.flatten(layout), 0, 2)`, like the B operand fragment.
7. **Scale encodings and packing.** ue8m0 = biased exponent (2^e -> e + 127); nvf4's ue4m3 = the e4m3 byte of a positive
   scale (here 2^e). The MMA takes each scale as a 32-bit register plus byte / thread selectors; ptxas packs the scale
   bytes with `PRMT`: 24 / 12 / 0 on the kernel page (64 MMAs = one K tile per warp) for mxf8 / mxf4 / nvf4.
8. **The op constructors take three types**: `MmaMXF8Op(ab_dtype, acc_dtype, sf_dtype)`; the shape is fixed by the
   kind -- (16, 8, 32) for mxf8, (16, 8, 64) for the FP4 kinds -- and `permutation_mnk`'s K must be a multiple of it.

## Sources

- PTX ISA: `mma.sync` with `.kind::mxf8f6f4.block_scale`, `.kind::mxf4.block_scale`,
  `.kind::mxf4nvf4.block_scale.scale_vec::4X`; memory consistency model 8.6 "Proxies" (quoted in gotcha 1) and 8.9.5
  (proxy-preserved base causality order): https://docs.nvidia.com/cuda/parallel-thread-execution/
- CuTe DSL 4.8.0 `cute.arch.fence_view_async_shared` docstring (`cutlass/cute/arch/nvvm_wrappers.py`): "The fence is
  required to synchronize the shared memory load/store and let the pipeline release or commit the buffer. This
  function is usually used for async execution unit (like TMA, UMMA) after the load/store operations."
- CuTe DSL API symbols: `cute.nvgpu.warp.MmaMXF8Op`, `cute.nvgpu.warp.MmaMXF4Op`, `cute.nvgpu.warp.MmaMXF4NVF4Op`,
  `cutlass.utils.blackwell_helpers.partition_fragment_SFA` / `partition_fragment_SFB` / `thrfrg_SFA` / `thrfrg_SFB`,
  `cute.recast_layout`, `cute.recast_ptr`, `cute.recast_tensor`, `cute.arch.fence_view_async_shared`,
  `cutlass.pipeline.PipelineTmaAsync`, `cute.nvgpu.cpasync.make_tiled_tma_atom`.
- CUTLASS C++ v4.8.1: `include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` (scale factors through TMA into
  smem; `NamedBarrier::sync` then `consumer_release`), `include/cutlass/gemm/collective/sm120_mma_tma.hpp`;
  CuTe DSL example `examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py` (the copy bundled with FlashInfer 0.7.2).
- GPU probe `probes/sm120/mechanisms/blockscaled_mma_check.py`: `PROBE PASS blockscaled-mma contracts=15`
  (`numerics-{mxf8,mxf4,nvf4}`, `stress-{mxf8,mxf4,nvf4}`, `kmap-{mxf8,mxf4,nvf4}`, `ctl-f4-elem-swizzle`,
  `ctl-recast-tensor`, `ctl-unfenced-races`, `sass-{mxf8,mxf4,nvf4}`).
