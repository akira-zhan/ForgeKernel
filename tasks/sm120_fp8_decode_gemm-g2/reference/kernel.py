"""sm120_fp8_decode_gemm — the student's kernel.

Value-returning: run(<inputs, in order>) returns the outputs (a tensor, or a tuple in the
order below), in the declared dtypes.

Shapes (one kernel serves all of them):
    m1_up: {"m": 1, "n": 14336, "k": 4096}
    m16_up: {"m": 16, "n": 14336, "k": 4096}
    m64_up: {"m": 64, "n": 14336, "k": 4096}
    m16_down: {"m": 16, "n": 4096, "k": 14336}
Inputs:
    a: float8_e4m3fn [m, k]
    b: float8_e4m3fn [n, k]
    scale_a: float32 [m, 1]
    scale_b: float32 [n, 1]
Outputs:
    y: bfloat16 [m, n]

Judged against the task's judge (eval/leaderboard/problem.json, "reference") on every input
group of every shape, and timed on all of them (eval/leaderboard.py).

Machine (milestone M0):
  - one launch per call, grid (1, ceil(M/bM) * N/128, 1), 384 threads, one bM x 128 tile per CTA,
    bM = smallest of {16, 32, 64, 128} >= M (compile-time per shape key)
  - warp 0: TMA producer (bM-row A slab then B slab per 128-deep k-tile) into an S_max(bM)-stage full/empty ring
  - warps 4..11: 8 consumer warps (WM = min(bM/16, 4) along M x 8/WM along N), LDSM x4 + QMMA.16832 e4m3,
    fp32 accumulators in registers
  - epilogue: acc * scale_b * scale_a -> bf16 -> STSM into 2 x 4 KiB staging (min(bM,64) x 32 subtiles) -> TMA store
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils.hopper_helpers as hh
from cutlass.cute.runtime import make_ptr
from cutlass.memory import SmemAllocator
from cutlass.tensor_utils import LayoutEnum

F8 = cutlass.Float8E4M3FN
F32 = cutlass.Float32
BF16 = cutlass.BFloat16
I32 = cutlass.Int32

BN, BK = 128, 128
S_MAX = {16: 4, 32: 4, 64: 3, 128: 2}   # ring depth per bM (lever, starting value S_max(bM))
KB = BK // 32                 # k32 steps per k-tile
EPI_N = 32                    # store subtile columns
EPI_B = 2                     # D staging buffers (lever D1.3, starting value)
D_BUF_ELEMS = 64 * 32         # each D buffer is 4,096 B (64 x 32 bf16) for every bM
THREADS = 384
NUM_CONSUMER_WARPS = 8
CONSUMER_THREADS = 256
CONSUMER_WARP0 = 4            # warps 4..11 = warpgroups 1, 2
PRODUCER_WARP = 0
STORE_WARP = CONSUMER_WARP0   # issues the TMA stores
EPI_BAR = 1                   # named barrier id of the 256 consumer threads

# Build-stage switch (trace-time): 1 skeleton, 2 + TMA ring, 3 + MMA mainloop, 4 + epilogue (full)
BUILD_STAGE = 4


class Fp8DecodeGemm:
    def __init__(self, M, N, K, build_stage=BUILD_STAGE):
        assert N % BN == 0 and K % BK == 0 and 1 <= M <= 128
        self.M, self.N, self.K = M, N, K
        self.k_tiles = K // BK
        bM = next(b for b in (16, 32, 64, 128) if b >= M)
        self.bM = bM
        self.stages = S_MAX[bM]
        self.WM = min(bM // 16, 4)              # consumer warps along M
        self.WN = 8 // self.WM                  # consumer warps along N
        self.MR = bM // (16 * self.WM)          # m16 atoms per warp (2 only at bM = 128)
        self.NR = BN // (8 * self.WN)           # n8 atoms per warp
        self.epi_m = min(bM, 64)
        self.WE = min(self.WN, 4)               # warps along N inside one 32-column store subtile
        self.NRE = EPI_N // (8 * self.WE)       # n8 atoms per warp inside one store subtile
        self.n_pre = 10 if bM == 128 else 0     # step-3 chains issued before release + next-tile probe
        self.tx_bytes = (bM + BN) * BK          # bytes landing per stage (A + B, OOB rows of A counted)
        self.num_tiles = ((M + bM - 1) // bM) * (N // BN)
        self.stage = build_stage

    @cute.jit
    def __call__(self, pA: cute.Pointer, pB: cute.Pointer, pSA: cute.Pointer, pSB: cute.Pointer,
                 pY: cute.Pointer, mM: cutlass.Int32, stream):
        M, N, K = self.M, self.N, self.K
        BM, STAGES, EPI_M = self.bM, self.stages, self.epi_m
        # A / Y take M as a runtime extent: a static M == 1 mode would be normalised to stride 0 and the
        # TMA coordinate tensor would collapse to rank 1 against the 2-D box basis
        mA = cute.make_tensor(pA, cute.make_layout((mM, K), stride=(K, 1)))
        mB = cute.make_tensor(pB, cute.make_layout((N, K), stride=(K, 1)))
        mSA = cute.make_tensor(pSA, cute.make_layout(M))
        mSB = cute.make_tensor(pSB, cute.make_layout(N))
        mY = cute.make_tensor(pY, cute.make_layout((mM, N), stride=(N, 1)))

        mma_op = cute.nvgpu.warp.MmaFP8Op(F8, F32, (16, 8, 32))
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((self.WM, self.WN, 1)),
                                        permutation_mnk=(BM, 16 * self.WN, 32))
        tiled_mma_epi = cute.make_tiled_mma(mma_op, cute.make_layout((self.WM, self.WE, 1)),
                                            permutation_mnk=(EPI_M, EPI_N, 32))

        sA_l = hh.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (BM, BN, BK), F8, STAGES)
        sB_l = hh.make_smem_layout_b(LayoutEnum.ROW_MAJOR, (BM, BN, BK), F8, STAGES)
        sD_l = hh.make_smem_layout_epi(BF16, LayoutEnum.ROW_MAJOR, (EPI_M, EPI_N), EPI_B)

        g2s = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(g2s, mA, cute.slice_(sA_l, (None, None, 0)), (BM, BK))
        tma_b, tB = cute.nvgpu.cpasync.make_tiled_tma_atom(g2s, mB, cute.slice_(sB_l, (None, None, 0)), (BN, BK))
        tma_y, tY = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), mY, cute.slice_(sD_l, (None, None, 0)), (EPI_M, EPI_N))

        @cute.struct
        class Storage:
            bars: cute.struct.MemRange[cutlass.Int64, STAGES * 2]       # full[S] + empty[S]
            clc_bars: cute.struct.MemRange[cutlass.Int64, 2]            # reserved for D1.2
            clc_resp: cute.struct.MemRange[cutlass.Int32, 4]            # reserved for D1.2
            sSB: cute.struct.Align[cute.struct.MemRange[F32, BN], 16]
            sD: cute.struct.Align[cute.struct.MemRange[BF16, EPI_B * D_BUF_ELEMS], 1024]
            sA: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(sA_l)], 1024]
            sB: cute.struct.Align[cute.struct.MemRange[F8, cute.cosize(sB_l)], 1024]

        self.kernel(tma_a, tA, tma_b, tB, tma_y, tY, mSA, mSB, tiled_mma, tiled_mma_epi,
                    sA_l, sB_l, sD_l, Storage).launch(
            grid=[1, self.num_tiles, 1], block=[THREADS, 1, 1],
            smem=Storage.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, tma_a: cute.CopyAtom, tA: cute.Tensor, tma_b: cute.CopyAtom, tB: cute.Tensor,
               tma_y: cute.CopyAtom, tY: cute.Tensor, mSA: cute.Tensor, mSB: cute.Tensor,
               tiled_mma: cute.TiledMma, tiled_mma_epi: cute.TiledMma,
               sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout, sD_l: cute.ComposedLayout,
               Storage: cutlass.Constexpr):
        M = self.M
        k_tiles = self.k_tiles
        BM, STAGES, EPI_M = self.bM, self.stages, self.epi_m
        WM, WN, WE, MR, NR, NRE = self.WM, self.WN, self.WE, self.MR, self.NR, self.NRE
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        _, bid_y, _ = cute.arch.block_idx()
        _, grid_y, _ = cute.arch.grid_dim()

        smem = SmemAllocator()
        st = smem.allocate(Storage)
        sA = st.sA.get_tensor(sA_l.outer, swizzle=sA_l.inner)
        sB = st.sB.get_tensor(sB_l.outer, swizzle=sB_l.inner)
        sD = st.sD.get_tensor(sD_l.outer, swizzle=sD_l.inner)
        sSB = st.sSB.get_tensor(cute.make_layout(BN))

        if warp == PRODUCER_WARP:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_y)

        # full: 1 producer arrive + tx bytes; empty: one arrive per consumer warp
        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, NUM_CONSUMER_WARPS),
            tx_count=self.tx_bytes,
            barrier_storage=st.bars.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))   # includes the prologue CTA rendezvous

        if warp == PRODUCER_WARP:
            # ---------------- TMA producer ----------------
            if cutlass.const_expr(self.stage >= 2):
                ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, STAGES)
                for tile in cutlass.range(bid_y, self.num_tiles, grid_y):
                    tile_n = tile
                    gA = cute.local_tile(tA, (BM, BK), (0, None))
                    gB = cute.local_tile(tB, (BN, BK), (tile_n, None))
                    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                        tma_a, 0, cute.make_layout(1), cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
                    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                        tma_b, 0, cute.make_layout(1), cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))
                    for kt in cutlass.range(k_tiles):
                        pipe.producer_acquire(ps)
                        cute.copy(tma_a, tAgA[None, kt], tAsA[None, ps.index],
                                  tma_bar_ptr=pipe.producer_get_barrier(ps))
                        cute.copy(tma_b, tBgB[None, kt], tBsB[None, ps.index],
                                  tma_bar_ptr=pipe.producer_get_barrier(ps))
                        pipe.producer_commit(ps)
                        ps.advance()
                pipe.producer_tail(ps)

        if warp >= CONSUMER_WARP0:
            # ---------------- consumers ----------------
            ctid = tidx - CONSUMER_WARP0 * 32
            lane = ctid % 32
            cwarp = ctid // 32
            w_m = cwarp % WM
            w_n = cwarp // WM
            row_base = w_m * 16 + lane // 4             # rows row_base + 64 mi + {0, 8}
            col_base = w_n * 8 + (lane % 4) * 2         # cols col_base + 8 WN ni + {0, 1}
            etid = (w_m + WM * (w_n % WE)) * 32 + lane  # thread index in the store-subtile tiled MMA

            thr_mma = tiled_mma.get_slice(ctid)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((BM, BN)), F32)

            ldsm = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), F8)
            cpA = cute.make_tiled_copy_A(ldsm, tiled_mma)
            cpB = cute.make_tiled_copy_B(ldsm, tiled_mma)
            tAsA_c = cpA.get_slice(ctid).partition_S(sA)
            tBsB_c = cpB.get_slice(ctid).partition_S(sB)
            tArA = cpA.retile(tCrA)
            tBrB = cpB.retile(tCrB)

            # epilogue partitions
            stsm = cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=2), BF16)
            r2s = cute.make_tiled_copy_C(stsm, tiled_mma_epi)
            thr_r2s = r2s.get_slice(etid)
            tRS_sD = thr_r2s.partition_D(sD)
            thr_mma_epi = tiled_mma_epi.get_slice(etid)
            rDf = cute.make_rmem_tensor(thr_mma_epi.partition_shape_C((EPI_M, EPI_N)), F32)
            rD = cute.make_rmem_tensor(thr_mma_epi.partition_shape_C((EPI_M, EPI_N)), BF16)
            tRS_rD = r2s.retile(rD)
            gY = cute.local_tile(tY, (EPI_M, EPI_N), (None, None))
            tYs, tYg = cute.nvgpu.cpasync.tma_partition(
                tma_y, 0, cute.make_layout(1), cute.group_modes(sD, 0, 2), cute.group_modes(gY, 0, 2))

            cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, STAGES)
            sa = cute.make_rmem_tensor(cute.make_layout(2 * MR), F32)

            for tile in cutlass.range(bid_y, self.num_tiles, grid_y):
                tile_n = tile
                acc.fill(0.0)
                if cutlass.const_expr(self.stage >= 2):
                    # prologue: pre-warm k-tile 0 step 0
                    pipe.consumer_wait(cs, pipe.consumer_try_wait(cs))
                    if cutlass.const_expr(self.stage >= 3):
                        cute.copy(cpA, tAsA_c[None, None, 0, cs.index], tArA[None, None, 0])
                        cute.copy(cpB, tBsB_c[None, None, 0, cs.index], tBrB[None, None, 0])
                    # trips 0 .. k_tiles-2: release k-tile t, wait k-tile t+1 inside step 3
                    for t in cutlass.range(k_tiles - 1):
                        s = cs.index
                        for kb in cutlass.range_constexpr(KB):
                            if cutlass.const_expr(self.stage >= 3):
                                if cutlass.const_expr(kb + 1 < KB):
                                    cute.copy(cpA, tAsA_c[None, None, kb + 1, s], tArA[None, None, kb + 1])
                                    cute.copy(cpB, tBsB_c[None, None, kb + 1, s], tBrB[None, None, kb + 1])
                            if cutlass.const_expr(kb + 1 < KB):
                                if cutlass.const_expr(self.stage >= 3):
                                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                            else:
                                # step 3: chains [0, n_pre), release + probe, remaining chains, wait, next-tile LDSM
                                if cutlass.const_expr(self.stage >= 3):
                                    for c in cutlass.range_constexpr(self.n_pre):
                                        mi = c // NR
                                        ni = c % NR
                                        acc_c = _chain(acc[None, mi, ni], 2)
                                        cute.gemm(tiled_mma, acc_c, _chain(tCrA[None, mi, kb], 1),
                                                  _chain(tCrB[None, ni, kb], 1), acc_c)
                                cute.arch.fence_view_async_shared()
                                pipe.consumer_release(cs)
                                cs.advance()
                                tok = pipe.consumer_try_wait(cs)
                                if cutlass.const_expr(self.stage >= 3):
                                    for c in cutlass.range_constexpr(self.n_pre, MR * NR):
                                        mi = c // NR
                                        ni = c % NR
                                        acc_c = _chain(acc[None, mi, ni], 2)
                                        cute.gemm(tiled_mma, acc_c, _chain(tCrA[None, mi, kb], 1),
                                                  _chain(tCrB[None, ni, kb], 1), acc_c)
                                pipe.consumer_wait(cs, tok)
                                if cutlass.const_expr(self.stage >= 3):
                                    cute.copy(cpA, tAsA_c[None, None, 0, cs.index], tArA[None, None, 0])
                                    cute.copy(cpB, tBsB_c[None, None, 0, cs.index], tBrB[None, None, 0])
                    # peeled last k-tile: scale loads overlap its MMA tail
                    s = cs.index
                    for kb in cutlass.range_constexpr(KB):
                        if cutlass.const_expr(self.stage >= 3):
                            if cutlass.const_expr(kb + 1 < KB):
                                cute.copy(cpA, tAsA_c[None, None, kb + 1, s], tArA[None, None, kb + 1])
                                cute.copy(cpB, tBsB_c[None, None, kb + 1, s], tBrB[None, None, kb + 1])
                        if cutlass.const_expr(kb == KB - 1):
                            cute.arch.fence_view_async_shared()
                            pipe.consumer_release(cs)
                            cs.advance()
                            if cutlass.const_expr(self.stage >= 4):
                                for i in cutlass.range_constexpr(2 * MR):
                                    row = row_base + (i // 2) * 64 + (i % 2) * 8
                                    sa[i] = F32(0.0)
                                    if row < M:
                                        sa[i] = mSA[row]
                                if ctid < BN:
                                    sSB[ctid] = mSB[tile_n * BN + ctid]
                        if cutlass.const_expr(self.stage >= 3):
                            cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)

                if cutlass.const_expr(self.stage >= 4):
                    # ---------------- epilogue ----------------
                    cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=CONSUMER_THREADS)
                    for h in cutlass.range_constexpr(MR):
                        for j in cutlass.range_constexpr(BN // EPI_N):
                            sub = h * (BN // EPI_N) + j
                            buf = sub % EPI_B
                            # owners of subtile j: every warp when WN <= 4; warp-column group j % 2 when WN == 8
                            owner = True
                            if cutlass.const_expr(WN == 8):
                                owner = (w_n // 4) == (j % 2)
                            if owner:
                                for nn in cutlass.range_constexpr(NRE):
                                    ni = j // 2 if cutlass.const_expr(WN == 8) else j * (4 // WN) + nn
                                    col = col_base + ni * (8 * WN)
                                    sb0 = sSB[col]
                                    sb1 = sSB[col + 1]
                                    for v in cutlass.range_constexpr(4):
                                        sbv = sb0 if v % 2 == 0 else sb1
                                        rDf[v, 0, nn] = (acc[v, h, ni] * sbv) * sa[h * 2 + v // 2]
                                rD.store(rDf.load().to(BF16))
                                cute.copy(r2s, tRS_rD, tRS_sD[None, None, None, buf])
                            cute.arch.fence_view_async_shared()
                            cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=CONSUMER_THREADS)
                            if warp == STORE_WARP:
                                cute.copy(tma_y, tYs[None, buf], tYg[None, h, tile_n * (BN // EPI_N) + j])
                                cute.arch.cp_async_bulk_commit_group()
                                cute.arch.cp_async_bulk_wait_group(EPI_B - 1, read=True)
                            cute.arch.barrier(barrier_id=EPI_BAR, number_of_threads=CONSUMER_THREADS)
            if cutlass.const_expr(self.stage >= 4):
                if warp == STORE_WARP:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.jit
def _chain(t: cute.Tensor, extra: cutlass.Constexpr):
    # one (m, n) chain as a rank-(1 + extra) view with unit M/N modes, as cute.gemm requires
    lay = t.layout
    for _ in cutlass.range_constexpr(extra):
        lay = cute.append(lay, cute.make_layout(1))
    return cute.make_tensor(t.iterator, lay)


_cache = {}


def _ptr(t, dtype):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def compile_for(M, N, K, build_stage=BUILD_STAGE):
    key = (M, N, K, build_stage)
    fn = _cache.get(key)
    if fn is None:
        op = Fp8DecodeGemm(M, N, K, build_stage)
        fake = lambda dt: make_ptr(dt, 16, cute.AddressSpace.gmem, assumed_align=16)
        stream = (cutlass_torch.current_stream() if torch.cuda.is_available()
                  else cute.runtime.make_fake_stream())        # no-GPU hosts: trace/compile check only
        fn = cute.compile(op, fake(F8), fake(F8), fake(F32), fake(F32), fake(BF16), I32(M), stream)
        _cache[key] = fn
    return fn


def run(a, b, scale_a, scale_b):
    M, K = a.shape
    N = b.shape[0]
    y = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    fn = compile_for(M, N, K)
    fn(_ptr(a, F8), _ptr(b, F8), _ptr(scale_a, F32), _ptr(scale_b, F32), _ptr(y, BF16),
       I32(M), cutlass_torch.current_stream())
    return y
