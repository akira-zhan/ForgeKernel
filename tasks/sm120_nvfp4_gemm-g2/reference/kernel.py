"""sm120_nvfp4_gemm — the student's kernel.

Value-returning: run(<inputs, in order>) returns the outputs (a tensor, or a tuple in the
order below), in the declared dtypes.

Shapes (one kernel serves all of them):
    square4k: {"m": 4096, "n": 4096, "k": 4096}
    wide8k: {"m": 8192, "n": 8192, "k": 4096}
Inputs:
    a: float4_e2m1fn_x2 [m, 2048]
    sa: float8_e4m3fn [m, 256]
    b: float4_e2m1fn_x2 [n, 2048]
    sb: float8_e4m3fn [n, 256]
Outputs:
    y: bfloat16 [m, n]

Judged against the task's judge (eval/leaderboard/problem.json, "reference") on every input
group of every shape, and timed on all of them (eval/leaderboard.py).

Design (one persistent warp-specialized kernel, sm_120a):
  * block = 384 threads: warps 0-7 = two consumer warpgroups (8 MMA warps, 4 (M) x 2 (N), warp tile 32 x 64),
    warps 8-11 = producer warpgroup (warp 8 issues every TMA load, warps 9-11 idle).
  * CTA tile 128 x 128, k-tile 256 e2m1 (128 B rows), 2-stage ring of A / B / SFA / SFB fed by TMA into full / empty
    mbarriers (full: 1 arrive + 36864 tx bytes, empty: one arrive per consumer warp, fenced release).
  * OMMA.SF.16864 (MmaMXF4NVF4Op), fragments via ldmatrix (A, B) and 32-bit / vector LDS of the blocked scales.
  * static persistent schedule: grid = min(#tiles, #SMs), CTA b takes tiles b, b + grid, ...
  * epilogue: 8 sub-tiles of 64 x 32 bf16 -> stmatrix into 2 staging buffers -> TMA store, <= 1 store in flight.
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as bwh
import cutlass.utils.hopper_helpers as hh
from cutlass.cute.runtime import make_ptr
from cutlass.memory import SmemAllocator
from cutlass.tensor_utils import LayoutEnum
from cuda.bindings import driver as cuda

BM, BN = 128, 128          # CTA output tile
BK = 256                   # e2m1 elements per k-tile
ROWB = BK // 2             # bytes per operand row per k-tile
STAGES = 2
EPI_M, EPI_N = 64, 32      # epilogue sub-tile (one staging buffer)
EPI_BUFS = 2
SF_V = 16                  # e2m1 elements per scale
SF_STAGE = BM * BK // SF_V  # 2048 bytes of scales per operand per stage
NUM_CONSUMER_THREADS = 256
PRODUCER_WARP = 8
EPI_BAR_ID = 1
PRODUCER_REGS = 40
CONSUMER_REGS = 232


class Nvf4Gemm:
    def __init__(self, M, N, K, num_sms, use_tma=True, do_math=True, do_epi=True, setmaxreg=True):
        assert M % BM == 0 and N % BN == 0 and K % BK == 0
        self.M, self.N, self.K = M, N, K
        self.tiles_m, self.tiles_n = M // BM, N // BN
        self.num_tiles = self.tiles_m * self.tiles_n
        self.grid = min(self.num_tiles, num_sms)
        self.k_tiles = K // BK
        self.use_tma, self.do_math, self.do_epi, self.setmaxreg = use_tma, do_math, do_epi, setmaxreg

    @cute.jit
    def __call__(self, pA: cute.Pointer, pSFA: cute.Pointer, pB: cute.Pointer, pSFB: cute.Pointer,
                 pY: cute.Pointer, stream: cuda.CUstream):
        M, N, K = self.M, self.N, self.K
        BY = cutlass.Float8E4M3FN                     # 1-byte container for TMA / smem layouts of packed e2m1
        KB_ = K // 2
        # ---- operands: packed e2m1 bytes, K-major
        mA = cute.make_tensor(cute.recast_ptr(pA, dtype=BY), cute.make_layout((M, KB_), stride=(KB_, 1)))
        mB = cute.make_tensor(cute.recast_ptr(pB, dtype=BY), cute.make_layout((N, KB_), stride=(KB_, 1)))
        sA_l = hh.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (BM, BN, ROWB), BY, STAGES)   # byte S<3,4,3>
        sB_l = hh.make_smem_layout_b(LayoutEnum.ROW_MAJOR, (BM, BN, ROWB), BY, STAGES)
        g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(g2s, mA, cute.slice_(sA_l, (None, None, 0)), (BM, ROWB))
        tma_b, tB = cute.nvgpu.cpasync.make_tiled_tma_atom(g2s, mB, cute.slice_(sB_l, (None, None, 0)), (BN, ROWB))
        # ---- scales: 128x4 blocked layout consumed as is; one k-tile of one 128-row block = 2048 contiguous bytes
        # (4 chunks of 512 B), viewed as (256, 8, k_tile, row_block) so the TMA box dims stay <= 256
        blk_stride = (K // 64) * 512
        sf_g_l_a = cute.make_layout((256, 8, K // BK, M // BM), stride=(1, 256, SF_STAGE, blk_stride))
        sf_g_l_b = cute.make_layout((256, 8, K // BK, N // BN), stride=(1, 256, SF_STAGE, blk_stride))
        mSFA = cute.make_tensor(cute.recast_ptr(pSFA, dtype=BY), sf_g_l_a)
        mSFB = cute.make_tensor(cute.recast_ptr(pSFB, dtype=BY), sf_g_l_b)
        sf_s_l = cute.make_layout((256, 8, STAGES), stride=(1, 256, SF_STAGE))
        tma_sfa, tSFA = cute.nvgpu.cpasync.make_tiled_tma_atom(g2s, mSFA, cute.slice_(sf_s_l, (None, None, 0)),
                                                               (256, 8))
        tma_sfb, tSFB = cute.nvgpu.cpasync.make_tiled_tma_atom(g2s, mSFB, cute.slice_(sf_s_l, (None, None, 0)),
                                                               (256, 8))
        # ---- output: bf16 row-major, staging 64 x 32 with 64-B rows, byte S<2,4,3> == bf16 Swizzle<2,3,3>
        mY = cute.make_tensor(pY, cute.make_layout((M, N), stride=(N, 1)))
        epi_l = hh.make_smem_layout_epi(cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (EPI_M, EPI_N), EPI_BUFS)
        s2g = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_y, tY = cute.nvgpu.cpasync.make_tiled_tma_atom(s2g, mY, cute.slice_(epi_l, (None, None, 0)),
                                                           (EPI_M, EPI_N))
        # ---- MMA: 8 warps 4 (M) x 2 (N); N permutation: warp wn owns columns 32 g + 16 wn + 8 h
        op = cute.nvgpu.warp.MmaMXF4NVF4Op(cutlass.Float4E2M1FN, cutlass.Float32, cutlass.Float8E4M3FN)
        perm_n = cute.make_layout((8, 2, 2), stride=(1, 16, 8))
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((4, 2, 1)), permutation_mnk=(BM, perm_n, 64))
        # epilogue view of the same warps over one 64 x 32 sub-tile
        tiled_mma_epi = cute.make_tiled_mma(op, cute.make_layout((4, 2, 1)), permutation_mnk=(EPI_M, perm_n, 64))

        tx = 2 * cute.size_in_bytes(BY, cute.slice_(sA_l, (None, None, 0))) + 2 * SF_STAGE
        if cutlass.const_expr(not self.use_tma):
            tx = 0

        @cute.struct
        class Storage:
            bars: cute.struct.MemRange[cutlass.Int64, STAGES * 2]
            sEpi: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, EPI_M * EPI_N * EPI_BUFS], 1024]
            sA: cute.struct.Align[cute.struct.MemRange[cutlass.Float4E2M1FN, cute.cosize(sA_l) * 2], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[cutlass.Float4E2M1FN, cute.cosize(sB_l) * 2], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, SF_STAGE * STAGES], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, SF_STAGE * STAGES], 128]

        self.storage_t = Storage
        self.kernel(tma_a, tA, tma_b, tB, tma_sfa, tSFA, tma_sfb, tSFB, tma_y, tY,
                    tiled_mma, tiled_mma_epi, sA_l, sB_l, sf_s_l, epi_l, tx).launch(
            grid=[self.grid, 1, 1], block=[384, 1, 1], smem=Storage.size_in_bytes(), stream=stream,
            min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, tma_a: cute.CopyAtom, tA: cute.Tensor, tma_b: cute.CopyAtom, tB: cute.Tensor,
               tma_sfa: cute.CopyAtom, tSFA: cute.Tensor, tma_sfb: cute.CopyAtom, tSFB: cute.Tensor,
               tma_y: cute.CopyAtom, tY: cute.Tensor,
               tiled_mma: cute.TiledMma, tiled_mma_epi: cute.TiledMma,
               sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout, sf_s_l: cute.Layout,
               epi_l: cute.ComposedLayout, tx: cutlass.Constexpr):
        BY = cutlass.Float8E4M3FN
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bid, _, _ = cute.arch.block_idx()
        num_tiles = self.num_tiles
        tiles_n = self.tiles_n
        grid = self.grid
        k_tiles = self.k_tiles

        st = SmemAllocator().allocate(self.storage_t)
        # TMA destinations: byte views carrying the byte swizzle
        sA_b = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), sA_l.inner, dtype=BY), sA_l.outer)
        sB_b = cute.make_tensor(cute.recast_ptr(st.sB.data_ptr(), sB_l.inner, dtype=BY), sB_l.outer)
        sSFA_b = cute.make_tensor(st.sSFA.data_ptr(), sf_s_l)
        sSFB_b = cute.make_tensor(st.sSFB.data_ptr(), sf_s_l)
        # e2m1 views for ldmatrix: element strides, BYTE-unit swizzle (KB blockscaled_mma gotcha 2)
        lA4 = cute.recast_layout(4, 8, sA_l)
        lB4 = cute.recast_layout(4, 8, sB_l)
        sA = st.sA.get_tensor(lA4.outer, swizzle=sA_l.inner)
        sB = st.sB.get_tensor(lB4.outer, swizzle=sB_l.inner)
        # staging buffers
        sEpi = st.sEpi.get_tensor(epi_l.outer, swizzle=epi_l.inner)

        if warp == PRODUCER_WARP:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_sfa)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_sfb)
        if warp == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_y)

        # full[s] (1 arrive + tx bytes) / empty[s] (one arrive per consumer warp)
        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=STAGES, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8), tx_count=tx,
            barrier_storage=st.bars.data_ptr(), cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))

        if warp >= PRODUCER_WARP:
            # ------------------------------------------------------------------ producer warpgroup
            if cutlass.const_expr(self.setmaxreg):
                cute.arch.setmaxregister_decrease(PRODUCER_REGS)
            if warp == PRODUCER_WARP:
                gA = cute.local_tile(tA, (BM, ROWB), (None, None))         # (BM, ROWB, tiles_m, k_tiles)
                gB = cute.local_tile(tB, (BN, ROWB), (None, None))
                tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(tma_a, 0, cute.make_layout(1),
                                                              cute.group_modes(sA_b, 0, 2), cute.group_modes(gA, 0, 2))
                tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(tma_b, 0, cute.make_layout(1),
                                                              cute.group_modes(sB_b, 0, 2), cute.group_modes(gB, 0, 2))
                tFsA, tFgA = cute.nvgpu.cpasync.tma_partition(tma_sfa, 0, cute.make_layout(1),
                                                              cute.group_modes(sSFA_b, 0, 2),
                                                              cute.group_modes(tSFA, 0, 2))
                tFsB, tFgB = cute.nvgpu.cpasync.tma_partition(tma_sfb, 0, cute.make_layout(1),
                                                              cute.group_modes(sSFB_b, 0, 2),
                                                              cute.group_modes(tSFB, 0, 2))
                ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
                for tile in cutlass.range(bid, num_tiles, grid):
                    tm = tile // tiles_n
                    tn = tile % tiles_n
                    for kt in cutlass.range(k_tiles):
                        pipe.producer_acquire(ps)
                        if cutlass.const_expr(self.use_tma):
                            bar = pipe.producer_get_barrier(ps)
                            cute.copy(tma_a, tAgA[None, tm, kt], tAsA[None, ps.index], tma_bar_ptr=bar)
                            cute.copy(tma_b, tBgB[None, tn, kt], tBsB[None, ps.index], tma_bar_ptr=bar)
                            cute.copy(tma_sfa, tFgA[None, kt, tm], tFsA[None, ps.index], tma_bar_ptr=bar)
                            cute.copy(tma_sfb, tFgB[None, kt, tn], tFsB[None, ps.index], tma_bar_ptr=bar)
                        pipe.producer_commit(ps)
                        ps.advance()
                pipe.producer_tail(ps)
        else:
            # ------------------------------------------------------------------ consumer warpgroups
            if cutlass.const_expr(self.setmaxreg):
                cute.arch.setmaxregister_increase(CONSUMER_REGS)
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA[None, None, 0])
            tCsB = thr_mma.partition_B(sB[None, None, 0])
            tCrA = tiled_mma.make_fragment_A(tCsA)                          # (V, 2, 4): k-steps of one k-tile
            tCrB = tiled_mma.make_fragment_B(tCsB)                          # (V, 8, 4)
            ldsm = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                                       cutlass.Float4E2M1FN)
            cpA = cute.make_tiled_copy_A(ldsm, tiled_mma)
            cpB = cute.make_tiled_copy_B(ldsm, tiled_mma)
            lA = cpA.get_slice(tidx).partition_S(sA)                       # (CPY, 2, 4, STAGES)
            lB = cpB.get_slice(tidx).partition_S(sB)
            rA = cpA.retile(tCrA)
            rB = cpB.retile(tCrB)
            # scale views: logical (row, k element) of one stage over the blocked smem layout:
            # byte = (k // 64) * 512 + (r % 32) * 16 + (r // 32) * 4 + (k // 16) % 4
            sf_view = cute.make_layout(((32, 4), (SF_V, 4, BK // 64)), stride=((16, 4), (0, 1, 512)))
            sf_view_staged = cute.make_layout(((32, 4), (SF_V, 4, BK // 64), STAGES),
                                              stride=((16, 4), (0, 1, 512), SF_STAGE))
            sSFA0 = cute.make_tensor(st.sSFA.data_ptr(), sf_view_staged)
            sSFB0 = cute.make_tensor(st.sSFB.data_ptr(), sf_view_staged)
            tCrSFA = bwh.partition_fragment_SFA(sSFA0[None, None, 0], thr_mma, tidx)
            tCrSFB = bwh.partition_fragment_SFB(sSFB0[None, None, 0], thr_mma, tidx)
            # regroup to (V, N, K) like the B fragment; N = (h, g) with column slab index h + 2 g
            tCrSFB = cute.make_tensor(tCrSFB.iterator,
                                      cute.group_modes(cute.group_modes(cute.flatten(tCrSFB.layout), 0, 2), 1, 3))
            frgA = bwh.thrfrg_SFA(sf_view, thr_mma)
            frgB = bwh.thrfrg_SFB(sf_view, thr_mma)
            vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)

            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((BM, BN)), cutlass.Float32)

            # epilogue: stmatrix of one 64 x 32 sub-tile into a staging buffer
            stsm = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2),
                                       cutlass.BFloat16)
            cpE = cute.make_tiled_copy_C(stsm, tiled_mma_epi)
            thr_e = cpE.get_slice(tidx)
            tEsE = thr_e.partition_D(sEpi)                                  # (CPY, 1, 1, EPI_BUFS)
            frag_epi_shape = tiled_mma_epi.get_slice(tidx).partition_shape_C((EPI_M, EPI_N))   # ((2,2),1,2)
            gY = cute.local_tile(tY, (EPI_M, EPI_N), (None, None))          # (64, 32, M/64, N/32)
            tYsY, tYgY = cute.nvgpu.cpasync.tma_partition(tma_y, 0, cute.make_layout(1),
                                                          cute.group_modes(sEpi, 0, 2), cute.group_modes(gY, 0, 2))

            cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
            for tile in cutlass.range(bid, num_tiles, grid):
                tm = tile // tiles_n
                tn = tile % tiles_n
                acc.fill(0.0)
                # pre-warm: wait k-tile 0, load its k-step-0 fragments
                pipe.consumer_wait(cs)
                if cutlass.const_expr(self.do_math):
                    self.load_kstep(0, cs.index, cpA, cpB, lA, lB, rA, rB, sSFA0, sSFB0, frgA, frgB, vmnk,
                                    tCrSFA, tCrSFB)
                for kt in cutlass.range(k_tiles - 1):
                    s = cs.index
                    if cutlass.const_expr(self.do_math):
                        for ks in cutlass.range_constexpr(BK // 64 - 1):
                            self.load_kstep(ks + 1, s, cpA, cpB, lA, lB, rA, rB, sSFA0, sSFB0, frgA, frgB, vmnk,
                                            tCrSFA, tCrSFB)
                            if ks < BK // 64 - 2:
                                self.mma_kstep(ks, tiled_mma, acc, tCrA, tCrB, tCrSFA, tCrSFB)
                        self.mma_kstep(BK // 64 - 2, tiled_mma, acc, tCrA, tCrB, tCrSFA, tCrSFB)
                    # last smem read of stage s issued: fence (generic -> async proxy), release
                    cute.arch.fence_view_async_shared()
                    pipe.consumer_release(cs)
                    cs.advance()
                    # wait for k-tile kt + 1, load its k-step 0 under k-step 3's MMAs
                    pipe.consumer_wait(cs)
                    if cutlass.const_expr(self.do_math):
                        self.load_kstep(0, cs.index, cpA, cpB, lA, lB, rA, rB, sSFA0, sSFB0, frgA, frgB, vmnk,
                                        tCrSFA, tCrSFB)
                        self.mma_kstep(BK // 64 - 1, tiled_mma, acc, tCrA, tCrB, tCrSFA, tCrSFB)
                # peeled last k-tile: release, no wait on a next k-tile
                s = cs.index
                if cutlass.const_expr(self.do_math):
                    for ks in cutlass.range_constexpr(BK // 64 - 1):
                        self.load_kstep(ks + 1, s, cpA, cpB, lA, lB, rA, rB, sSFA0, sSFB0, frgA, frgB, vmnk,
                                        tCrSFA, tCrSFB)
                        if ks < BK // 64 - 2:
                            self.mma_kstep(ks, tiled_mma, acc, tCrA, tCrB, tCrSFA, tCrSFB)
                    self.mma_kstep(BK // 64 - 2, tiled_mma, acc, tCrA, tCrB, tCrSFA, tCrSFB)
                cute.arch.fence_view_async_shared()
                pipe.consumer_release(cs)
                cs.advance()
                if cutlass.const_expr(self.do_math):
                    self.mma_kstep(BK // 64 - 1, tiled_mma, acc, tCrA, tCrB, tCrSFA, tCrSFB)

                if cutlass.const_expr(self.do_epi):
                    # ---------------------------------------------------- epilogue: 8 sub-tiles (mi, g)
                    for j in cutlass.range_constexpr(8):
                        mi = j // 4
                        g = j % 4
                        buf = j % EPI_BUFS
                        sub = cute.make_rmem_tensor(frag_epi_shape, cutlass.Float32)
                        for h in cutlass.range_constexpr(2):
                            for v in cutlass.range_constexpr(4):
                                sub[v, 0, h] = acc[v, mi, 2 * g + h]
                        rE = cute.make_rmem_tensor(frag_epi_shape, cutlass.BFloat16)
                        rE.store(sub.load().to(cutlass.BFloat16))
                        cute.copy(cpE, cpE.retile(rE), tEsE[None, None, None, buf])
                        cute.arch.fence_view_async_shared()
                        cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=NUM_CONSUMER_THREADS)
                        if warp == 0:
                            cute.copy(tma_y, tYsY[None, buf], tYgY[None, 2 * tm + mi, 4 * tn + g])
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(EPI_BUFS - 1, read=True)
                        cute.arch.barrier(barrier_id=EPI_BAR_ID, number_of_threads=NUM_CONSUMER_THREADS)
            if warp == 0:
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    def load_kstep(self, ks, s, cpA, cpB, lA, lB, rA, rB, sSFA0, sSFB0, frgA, frgB, vmnk, tCrSFA, tCrSFB):
        """Fragments of k-step ks of stage s: ldmatrix A / B, scale words from the blocked smem layout."""
        cute.copy(cpA, lA[None, None, ks, s], rA[None, None, ks])
        cute.copy(cpB, lB[None, None, ks, s], rB[None, None, ks])
        vA = cute.make_tensor(sSFA0[None, None, s].iterator, frgA)
        vB = cute.make_tensor(sSFB0[None, None, s].iterator, frgB)
        pA = cute.group_modes(cute.flatten(vA[(vmnk[0], (vmnk[1], vmnk[3])), (None, None)]), 0, 2)
        # (V, (h, g), k-step): the 4 g-slabs of one h are 16 contiguous bytes (rows n, n+32, n+64, n+96)
        pB = cute.group_modes(cute.group_modes(cute.flatten(vB[(vmnk[0], (vmnk[2], vmnk[3])), (None, None)]), 0, 2),
                              1, 3)
        cute.autovec_copy(pA[None, None, ks], tCrSFA[None, None, ks])
        cute.autovec_copy(pB[None, None, ks], tCrSFB[None, None, ks])

    def mma_kstep(self, ks, tiled_mma, acc, tCrA, tCrB, tCrSFA, tCrSFB):
        cute.gemm(tiled_mma, acc, [tCrA[None, None, ks], tCrSFA[None, None, ks]],
                  [tCrB[None, None, ks], tCrSFB[None, None, ks]], acc)


_CACHE = {}


def _num_sms():
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def get_compiled(M, N, K, **kw):
    key = (M, N, K, tuple(sorted(kw.items())))
    f = _CACHE.get(key)
    if f is None:
        gemm = Nvf4Gemm(M, N, K, _num_sms(), **kw)
        gmem = cute.AddressSpace.gmem
        args = [make_ptr(cutlass.Uint8, 0, gmem, assumed_align=128),
                make_ptr(cutlass.Uint8, 0, gmem, assumed_align=128),
                make_ptr(cutlass.Uint8, 0, gmem, assumed_align=128),
                make_ptr(cutlass.Uint8, 0, gmem, assumed_align=128),
                make_ptr(cutlass.BFloat16, 0, gmem, assumed_align=128),
                cuda.CUstream(0)]
        f = cute.compile(gemm, *args)
        _CACHE[key] = f
    return f


def launch(f, a, sa, b, sb, y):
    gmem = cute.AddressSpace.gmem
    f(make_ptr(cutlass.Uint8, a.data_ptr(), gmem, assumed_align=128),
      make_ptr(cutlass.Uint8, sa.data_ptr(), gmem, assumed_align=128),
      make_ptr(cutlass.Uint8, b.data_ptr(), gmem, assumed_align=128),
      make_ptr(cutlass.Uint8, sb.data_ptr(), gmem, assumed_align=128),
      make_ptr(cutlass.BFloat16, y.data_ptr(), gmem, assumed_align=128),
      cuda.CUstream(torch.cuda.current_stream().cuda_stream))


def run(a, sa, b, sb):
    M, KB_ = a.shape
    N = b.shape[0]
    K = KB_ * 2
    y = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    launch(get_compiled(M, N, K), a, sa, b, sb, y)
    return y
