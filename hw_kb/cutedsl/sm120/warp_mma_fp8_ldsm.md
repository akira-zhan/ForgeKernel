# warp_mma_fp8_ldsm (FP8 `mma.sync` on sm120: `MmaFP8Op` (16,8,32) = `QMMA.16832`, fed by `ldmatrix` from K-major shared memory)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified (GPU probe `probes/sm120/mechanisms/warp_mma_fp8_ldsm_check.py`, `PROBE PASS warp-mma-fp8-ldsm contracts=17`,
> RTX PRO 5000, CuTe DSL 4.8.0): one CTA computes C[128 x 128] = A[128 x 256] . B[128 x 256]^T (e4m3, fp32 accumulate) in three
> variants (swizzled smem, linear smem, linear smem + a conflicting cp.async fill); all three bit-exact (`max_diff=0.0`);
> SASS census per variant exactly 256 `QMMA.16832.F32.E4M3.E4M3`, 64 `LDSM.16.M88.4`, 32 `LDGSTS.E.BYPASS.128`, 0 scalar
> shared loads, PTX counts equal SASS counts per family; NCU: swizzled reads conflict-free, linear reads 8.0x wavefronts
> with the bank-conflict counter equal to the excess; the cp.async fill is visible only to SASS-patched counters.
>
> **What is certified**: numerics, the instruction footprint and the shared-memory counters of this construction. **No
> throughput claim**: a single 128 x 128 tile is not a mainloop; for rates see
> [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md).

## Purpose

On sm120 the only tensor-core path is warp-level `mma.sync`; the accumulator lives in registers and the A / B fragments
must be in registers before each issue. This entry is the minimal correct chain for FP8:
**gmem --cp.async--> smem --ldmatrix--> registers --QMMA.16832--> fp32 accumulators**, with the smem layout that keeps
`ldmatrix` conflict-free.

## When needed (trigger)

- Writing any FP8 (e4m3 / e5m2) GEMM-like mainloop on sm120 (dense GEMM, attention QK^T / PV with FP8 operands).
- SASS of an FP8 kernel shows scalar `LDS.U8` / `LDS.U16` feeding `QMMA`, or `HMMA` where `QMMA` was intended.
- NCU shows `bankconf_ld` far above zero in a tensor-core mainloop.
- Porting an sm90 / sm100 FP8 kernel: wgmma / tcgen05 read operands from smem descriptors; on sm120 every operand goes
  through registers, so the ldmatrix stage and its smem layout are new work.

## Minimal runnable example

```python
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as hh          # make_smem_layout_a / _b (swizzled K-major smem layouts)
from cutlass.memory import SmemAllocator            # cutlass.utils.SmemAllocator is deprecated in DSL 4.8.0
from cutlass.tensor_utils import LayoutEnum         # cutlass.utils.LayoutEnum is deprecated in DSL 4.8.0
F8 = cutlass.Float8E4M3FN
M, N, K = 128, 128, 256                             # one CTA tile; A (M,K) and B (N,K) both K-major

class Fp8Tile:
    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        mma_op = cute.nvgpu.warp.MmaFP8Op(F8, cutlass.Float32, (16, 8, 32))           # QMMA.16832.F32.E4M3.E4M3
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((2, 2, 1)),        # 4 warps, 2 x 2
                                        permutation_mnk=(32, 32, 32))
        sA_l = hh.make_smem_layout_a(LayoutEnum.from_tensor(mA), (M, N, K), F8, 1)  # swizzled, 1 stage
        sB_l = hh.make_smem_layout_b(LayoutEnum.from_tensor(mB), (M, N, K), F8, 1)
        self.kernel(mA, mB, mC, tiled_mma, sA_l, sB_l).launch(grid=[1, 1, 1], block=[128, 1, 1])

    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, tiled_mma: cute.TiledMma,
               sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout):
        tidx, _, _ = cute.arch.thread_idx()
        smem = SmemAllocator()
        sA = smem.allocate_tensor(F8, sA_l, byte_alignment=1024)[None, None, 0]
        sB = smem.allocate_tensor(F8, sB_l, byte_alignment=1024)[None, None, 0]
        # gmem -> smem: cp.async 16 B per thread (LDGSTS.E.BYPASS.128)
        g2s = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
                                  F8, num_bits_per_copy=128)
        tcopy = cute.make_tiled_copy_tv(g2s, cute.make_layout((8, 16), stride=(16, 1)), cute.make_layout((1, 16)))
        thr = tcopy.get_slice(tidx)
        cute.copy(tcopy, thr.partition_S(mA), thr.partition_D(sA))
        cute.copy(tcopy, thr.partition_S(mB), thr.partition_D(sB))
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # smem -> registers with ldmatrix (LDSM.16.M88.4), then QMMA per k-block of 32
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA, tCsB = thr_mma.partition_A(sA), thr_mma.partition_B(sB)
        tCrA, tCrB = tiled_mma.make_fragment_A(tCsA), tiled_mma.make_fragment_B(tCsB)
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((M, N)), cutlass.Float32)
        acc.fill(0.0)
        ldsm = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), F8)
        cpA, cpB = cute.make_tiled_copy_A(ldsm, tiled_mma), cute.make_tiled_copy_B(ldsm, tiled_mma)
        tAsA, tBsB = cpA.get_slice(tidx).partition_S(sA), cpB.get_slice(tidx).partition_S(sB)
        tArA, tBrB = cpA.retile(tCrA), cpB.retile(tCrB)
        for kb in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
            cute.copy(cpA, tAsA[None, None, kb], tArA[None, None, kb])
            cute.copy(cpB, tBsB[None, None, kb], tBrB[None, None, kb])
            cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
        cute.autovec_copy(acc, thr_mma.partition_C(mC))

# host: from_dlpack(t, assumed_align=16) for A, B (torch.float8_e4m3fn) and C (fp32), cute.compile(Fp8Tile(), ...)
```

Expected footprint of this tile (the census the probe asserts; 128 threads, 4 warps of 64 x 64):

| Opcode (executed form) | Count | Why |
|---|---|---|
| `QMMA.16832.F32.E4M3.E4M3` | 256 | per warp 4 (M) x 8 (N) atoms x 8 k-blocks of 32 |
| `LDSM.16.M88.4` | 64 | per k-block 4 for A + 4 for B (each x4 = 4 matrices of 8 rows x 16 B) |
| `LDGSTS.E.BYPASS.128` | 32 | per thread 256 B of A + 256 B of B, 16 B each |
| `LDS` / `LDS.U8` / `LDS.U16` (scalar) | 0 | the failure form: per-element operand loads |

PTX side, one to one: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` = 256,
`ldmatrix.sync.aligned.m8n8.x4.shared.b16` = 64, `cp.async.cg.shared.global` = 32.

## Gotchas

1. **The FP8 `ldmatrix` is the 16-bit atom.** For K-major e4m3 operands use `LdMatrix8x8x16bOp(transpose=False,
   num_matrices=4)` instantiated with the fp8 dtype: each 8 x 8 "b16" matrix is 8 rows x 16 B = 8 rows x 16 e4m3 values,
   which is exactly the register layout `m16n8k32` expects (SASS `LDSM.16.M88.4`). The 8-bit atoms
   (`LdMatrix16x16x8bOp`, `LdMatrix16x8x8bOp`) exist for transposed / MN-major operands and were not used here.
2. **A K-major tile with 256 B rows needs the swizzle.** With a plain row-major smem layout (`stride=(K, 1)`, K = 256
   e4m3 = 256 B), the 8 rows of each ldmatrix matrix start on the same bank group: NCU reads **7.99x** the load
   wavefronts of the swizzled tile and `bankconf_ld` equal to the excess (7171 of 8195); the swizzled layout from
   `make_smem_layout_a/b` reads 1026 wavefronts for 256 warp-level `LDSM.x4` (4 each = conflict-free) with 2 conflicts.
   Numerics are identical either way: **only NCU or a wavefront-per-instruction count reveals the missing swizzle.**
3. **The cp.async fill is invisible to the four smem counters the loop reads.** `smem_wavefronts_st` / `bankconf_st`
   read 0 for an `LDGSTS` fill, and so do the hardware counters `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts` /
   `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts` on GB202. Only the SASS-patched
   `sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts` sees it: 512 for the two 32 KiB fills (4 per warp-level
   16 B-per-lane `LDGSTS`). A fill pattern that conflicts (one row per thread, lanes 256 B apart) reads **4096 (8x) at the
   same `LDGSTS` count** -- and the SASS-patched **bank-conflict** counter still reads 0. Judge a fill by
   wavefronts per `LDGSTS` (`sm__sass_..._op_ldgsts` / `sm__sass_inst_executed_op_ldgsts`, 4 is conflict-free for 16 B
   per lane), never by a conflict counter.
4. **cp.async needs a 16 B-aligned source the compiler can see.** `from_dlpack(t)` assumes element alignment (1 B for
   fp8); the 128-bit `CopyG2SOp` then fails IR verification with `'cute.copy' op src ptr alignment (8 bits) does not
   meet requirement (128 bits)`. Pass `from_dlpack(t, assumed_align=16)` (and make it true).
5. **Count executed forms only.** The SASS of this kernel carries three `@!PT LDS RZ, [RZ]` before the `LDGSTS` run:
   predicate constant-false, never executed. A census that counts them reports scalar shared loads that do not exist
   (see [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md)).
6. **`permutation_mnk` was (32, 32, 32) for a (2, 2, 1) warp layout** (N doubled relative to 2 x 8), matching what
   `make_tiled_copy_B` needs for the x4 B load; other permutations were not tested here.
7. **Exact numerics need exact inputs.** Integer-valued e4m3 in [-2, 2] make every product and partial sum exact in
   fp32, so the contract is `max_diff == 0.0` regardless of accumulation order; random fp8 data only supports a
   tolerance.
8. **The accumulator choice does not change the rate on this card** (f16 and f32 accumulate both ~2030 dense flop / clk / SM),
   so f32 accumulation costs nothing in issue rate; it costs registers (4 per thread per atom vs 2).

## Sources

- PTX ISA: `mma.sync.aligned.m16n8k32` with `.e4m3` / `.e5m2` (warp-level matrix multiply-accumulate), `ldmatrix`,
  `cp.async`: https://docs.nvidia.com/cuda/parallel-thread-execution/
- Nsight Compute metrics (`l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st,ldgsts}`, `l1tex__data_bank_conflicts_*`,
  `sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts`, `sm__sass_inst_executed_op_ldgsts`):
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- CuTe DSL API symbols: `cute.nvgpu.warp.MmaFP8Op`, `cute.nvgpu.warp.LdMatrix8x8x16bOp`, `cute.make_tiled_mma`,
  `cute.make_tiled_copy_A`, `cute.make_tiled_copy_B`, `cute.make_tiled_copy_tv`, `cute.nvgpu.cpasync.CopyG2SOp`,
  `cute.nvgpu.cpasync.LoadCacheMode`, `cutlass.utils.hopper_helpers.make_smem_layout_a` / `make_smem_layout_b`,
  `cutlass.tensor_utils.LayoutEnum`, `cutlass.memory.SmemAllocator`, `cute.arch.cp_async_commit_group`,
  `cute.arch.cp_async_wait_group`, `cute.runtime.from_dlpack(assumed_align=...)`.
- GPU probe `probes/sm120/mechanisms/warp_mma_fp8_ldsm_check.py`: `PROBE PASS warp-mma-fp8-ldsm contracts=17`
  (`numerics-{swz,lin,linfill}`, `census-qmma-*`, `census-ldsm-*`, `census-ptx-sass-*`, `N-swz-clean`, `N-lin-conflict`,
  `N-ldgsts-blind-hw-counters`, `N-linfill-fill-wavefronts`, `N-ldgsts-conflict-counter-blind`).
