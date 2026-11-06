#!/usr/bin/env python3
"""Block-scaled mma.sync on sm_120a (CuTe DSL): MXF8 (QMMA.SF), MXF4 / NVF4 (OMMA.SF) in a TMA + mbarrier mainloop.

    python3 -u blockscaled_mma_check.py      (GPU; cuobjdump)

C (fp32) = (A * SFA) . (B * SFB)^T, A / B K-major, CTA 128 x 128, 4 MMA warps (2 x 2); warp 0 issues the TMA loads of
A and B (raw bytes: e4m3, or packed e2m1 two per byte) into a 3-stage full / empty mbarrier ring; each consumer warp
reads its scale factors for the K tile from gmem into registers (``partition_fragment_SFA/SFB``), ldmatrix-es the
operands and issues ``cute.gemm(atom, acc, [A, SFA], [B, SFB], acc)``; before releasing a stage it executes
``fence_view_async_shared()``. Integer operands in [-2, 2] and scales 2^[-3, 3] make every product a multiple of
2^-6 and keep the sum of their magnitudes at most 2^16.1 per output at K = 4096 (the max over this probe's draws),
under the 2^18 that 24 significand bits hold at that resolution: every partial sum is exact in fp32 whatever the
summation order. Shapes are multiples of the tiles (no
predication).

Contracts:
- ``numerics-<kind>``      : C == (A*SA) @ (B*SB)^T exactly at 2048 x 2048 x 4096 (multi-wave), random scales;
- ``stress-<kind>``        : the same compiled kernel, 30 launches, every one exact;
- ``kmap-<kind>``          : with B a selector (B[n, k] = 1 at k = n + off) C reproduces A[:, off:off+N] exactly -- the
  K mapping of A relative to B is the identity (a random-data test cannot see a K permutation shared by A and B);
- ``ctl-unfenced-races``   : (control) MXF8 without the fence before ``consumer_release``: at least one of 30 launches
  is wrong -- the generic-proxy (ldmatrix) reads race the async-proxy (TMA) refill;
- ``ctl-f4-elem-swizzle``  : (control) MXF4 with the e2m1 smem view carrying the element-unit swizzle produced by
  ``recast_layout(4, 8, byte_layout)``: the selector shows a K permutation -- the swizzle of a sub-byte smem pointer
  acts on byte addresses;
- ``ctl-recast-tensor``    : (control, trace time) ``recast_tensor`` of the swizzled byte view yields a pointer with no
  swizzle at all;
- ``sass-<kind>``          : the expected block-scaled opcode, and ``orch_tools.xproxy_releases`` finds no stage release
  (``SYNCS.ARRIVE...A1T0``) after an ``LDSM`` without a ``FENCE.VIEW.ASYNC.S`` between.
Facts: TFLOPS per kind (median of 20 launches, clocks unlocked), SASS histogram, the PRMT count (scale packing).
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "methodology"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from _ladder import n, run, v  # noqa: E402

KINDS = ("mxf8", "mxf4", "nvf4")
BIG = (2048, 2048, 4096)


def build():
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils.blackwell_helpers as bwh
    import cutlass.utils.hopper_helpers as hh
    from cutlass.tensor_utils import LayoutEnum
    from cutlass.memory import SmemAllocator

    BM, BN = 128, 128

    class BsGemm:
        def __init__(self, kind, M, N, K, stages=3, row_bytes=64, fence=True, f4_swizzle="byte", probe=None):
            self.kind, self.M, self.N, self.K = kind, M, N, K
            self.stages, self.row_bytes, self.fence, self.f4_swizzle = stages, row_bytes, fence, f4_swizzle
            self.fp4 = kind in ("mxf4", "nvf4")
            self.vec = 16 if kind == "nvf4" else 32
            self.probe = probe or {}

        def dt(self):
            return cutlass.Float4E2M1FN if self.fp4 else cutlass.Float8E4M3FN

        def sft(self):
            return cutlass.Float8E4M3FN if self.kind == "nvf4" else cutlass.Float8E8M0FNU

        @cute.jit
        def __call__(self, mA_raw: cute.Tensor, mB_raw: cute.Tensor, mSFA: cute.Tensor, mSFB: cute.Tensor,
                     mC: cute.Tensor):
            M, N, K, V = self.M, self.N, self.K, self.vec
            dt = self.dt()
            if cutlass.const_expr(self.kind == "mxf8"):
                op = cute.nvgpu.warp.MmaMXF8Op(dt, cutlass.Float32, cutlass.Float8E8M0FNU)
            elif cutlass.const_expr(self.kind == "mxf4"):
                op = cute.nvgpu.warp.MmaMXF4Op(dt, cutlass.Float32, cutlass.Float8E8M0FNU)
            else:
                op = cute.nvgpu.warp.MmaMXF4NVF4Op(dt, cutlass.Float32, cutlass.Float8E4M3FN)
            kat = 64 if cutlass.const_expr(self.fp4) else 32
            tiled_mma = cute.make_tiled_mma(op, cute.make_layout((2, 2, 1)), permutation_mnk=(32, 32, kat))
            bk = self.row_bytes * 8 // dt.width                      # K elements per stage row
            KB_ = K * dt.width // 8                                  # K bytes per gmem row
            BY = cutlass.Float8E4M3FN                                # 1-byte container for TMA / smem
            mA = cute.make_tensor(cute.recast_ptr(mA_raw.iterator, dtype=BY), cute.make_layout((M, KB_), stride=(KB_, 1)))
            mB = cute.make_tensor(cute.recast_ptr(mB_raw.iterator, dtype=BY), cute.make_layout((N, KB_), stride=(KB_, 1)))
            sA_l = hh.make_smem_layout_a(LayoutEnum.ROW_MAJOR, (BM, BN, self.row_bytes), BY, self.stages)
            sB_l = hh.make_smem_layout_b(LayoutEnum.ROW_MAJOR, (BM, BN, self.row_bytes), BY, self.stages)
            sA_1 = cute.slice_(sA_l, (None, None, 0))
            sB_1 = cute.slice_(sB_l, (None, None, 0))
            tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(tma_op, mA, sA_1, (BM, self.row_bytes), num_multicast=1)
            tma_b, tB = cute.nvgpu.cpasync.make_tiled_tma_atom(tma_op, mB, sB_1, (BN, self.row_bytes), num_multicast=1)
            tx = cute.size_in_bytes(BY, sA_1) + cute.size_in_bytes(BY, sB_1)
            # (rows, K/V) scale bytes viewed as (rows, K) elements, constant over each V-block of K
            sfa = cute.make_tensor(cute.recast_ptr(mSFA.iterator, dtype=self.sft()),
                                   cute.make_layout((M, (V, K // V)), stride=(K // V, (0, 1))))
            sfb = cute.make_tensor(cute.recast_ptr(mSFB.iterator, dtype=self.sft()),
                                   cute.make_layout((N, (V, K // V)), stride=(K // V, (0, 1))))

            @cute.struct
            class Storage:
                bars: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
                sA: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sA_l) * 8 // dt.width], 1024]
                sB: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sB_l) * 8 // dt.width], 1024]

            self.storage_t = Storage
            self.kernel(tma_a, tA, tma_b, tB, sfa, sfb, mC, tiled_mma, sA_l, sB_l, bk, tx).launch(
                grid=[M // BM, N // BN, 1], block=[128, 1, 1], smem=Storage.size_in_bytes())

        @cute.kernel
        def kernel(self, tma_a: cute.CopyAtom, mA: cute.Tensor, tma_b: cute.CopyAtom, mB: cute.Tensor,
                   sfa: cute.Tensor, sfb: cute.Tensor, mC: cute.Tensor, tiled_mma: cute.TiledMma,
                   sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout, bk: cutlass.Constexpr, tx: cutlass.Constexpr):
            S = self.stages
            dt = self.dt()
            tidx, _, _ = cute.arch.thread_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            bidx, bidy, _ = cute.arch.block_idx()
            st = SmemAllocator().allocate(self.storage_t)
            # TMA writes bytes through the byte-typed swizzled view
            sA_b = cute.make_tensor(cute.recast_ptr(st.sA.data_ptr(), sA_l.inner, dtype=cutlass.Float8E4M3FN), sA_l.outer)
            sB_b = cute.make_tensor(cute.recast_ptr(st.sB.data_ptr(), sB_l.inner, dtype=cutlass.Float8E4M3FN), sB_l.outer)
            if cutlass.const_expr(self.fp4):
                # e2m1 view: element-unit strides; the pointer swizzle of a sub-byte type acts on BYTE addresses, so it
                # keeps the byte-unit swizzle (the element-unit one from recast_layout is the control that misreads)
                lA4 = cute.recast_layout(4, 8, sA_l)
                lB4 = cute.recast_layout(4, 8, sB_l)
                swA = sA_l.inner if cutlass.const_expr(self.f4_swizzle == "byte") else lA4.inner
                swB = sB_l.inner if cutlass.const_expr(self.f4_swizzle == "byte") else lB4.inner
                sA = st.sA.get_tensor(lA4.outer, swizzle=swA)
                sB = st.sB.get_tensor(lB4.outer, swizzle=swB)
                if cutlass.const_expr(self.probe.get("print_recast") is True):
                    print("TRACE byte view ptr:", sA_b.iterator)
                    print("TRACE recast_tensor ptr:", cute.recast_tensor(sA_b, dt).iterator)
            else:
                sA, sB = sA_b, sB_b
            if warp == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_b)
            pipe = pipeline.PipelineTmaAsync.create(
                num_stages=S, producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4), tx_count=tx,
                barrier_storage=st.bars.data_ptr(), cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)))
            gA = cute.local_tile(mA, (BM, self.row_bytes), (bidx, None))
            gB = cute.local_tile(mB, (BN, self.row_bytes), (bidy, None))
            k_tiles = cute.size(gA, mode=[2])
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(tma_a, 0, cute.make_layout(1),
                                                          cute.group_modes(sA_b, 0, 2), cute.group_modes(gA, 0, 2))
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(tma_b, 0, cute.make_layout(1),
                                                          cute.group_modes(sB_b, 0, 2), cute.group_modes(gB, 0, 2))
            gSFA = cute.local_tile(sfa, (BM, bk), (bidx, None))
            gSFB = cute.local_tile(sfb, (BN, bk), (bidy, None))
            gC = cute.local_tile(mC, (BM, BN), (bidx, bidy))
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA[None, None, 0])
            tCsB = thr_mma.partition_B(sB[None, None, 0])
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
            g0A = gSFA[None, None, 0]
            g0B = gSFB[None, None, 0]
            tCrSFA = bwh.partition_fragment_SFA(g0A, thr_mma, tidx)
            tCrSFB = bwh.partition_fragment_SFB(g0B, thr_mma, tidx)
            # partition_fragment_SFB groups (N, K) into one mode; regroup to (V, N, K) like the B fragment
            tCrSFB = cute.make_tensor(tCrSFB.iterator, cute.group_modes(cute.flatten(tCrSFB.layout), 0, 2))
            frgA = bwh.thrfrg_SFA(g0A.layout, thr_mma)
            frgB = bwh.thrfrg_SFB(g0B.layout, thr_mma)
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((BM, BN)), cutlass.Float32)
            acc.fill(0.0)
            ldsm = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), dt)
            cpA = cute.make_tiled_copy_A(ldsm, tiled_mma)
            cpB = cute.make_tiled_copy_B(ldsm, tiled_mma)
            lA = cpA.get_slice(tidx).partition_S(sA)
            lB = cpB.get_slice(tidx).partition_S(sB)
            rA, rB = cpA.retile(tCrA), cpB.retile(tCrB)
            ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, S)
            cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)
            nxt = cutlass.Int32(0)
            if warp == 0:
                for s0 in cutlass.range_constexpr(S):
                    if nxt < k_tiles:
                        pipe.producer_acquire(ps)
                        cute.copy(tma_a, tAgA[None, nxt], tAsA[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                        cute.copy(tma_b, tBgB[None, nxt], tBsB[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                        pipe.producer_commit(ps)
                        ps.advance()
                    nxt = nxt + 1
            for kt in cutlass.range(k_tiles):
                # scale factors of this K tile: gmem -> registers, issued before the stage wait
                vA = cute.make_tensor(gSFA[None, None, kt].iterator, frgA)
                pA = cute.group_modes(cute.flatten(vA[(vmnk[0], (vmnk[1], vmnk[3])), (None, None)]), 0, 2)
                vB = cute.make_tensor(gSFB[None, None, kt].iterator, frgB)
                pB = cute.group_modes(cute.flatten(vB[(vmnk[0], (vmnk[2], vmnk[3])), (None, None)]), 0, 2)
                cute.autovec_copy(pA, tCrSFA)
                cute.autovec_copy(pB, tCrSFB)
                pipe.consumer_wait(cs, pipe.consumer_try_wait(cs))
                for kb in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                    cute.copy(cpA, lA[None, None, kb, cs.index], rA[None, None, kb])
                    cute.copy(cpB, lB[None, None, kb, cs.index], rB[None, None, kb])
                    cute.gemm(tiled_mma, acc, [tCrA[None, None, kb], tCrSFA[None, None, kb]],
                              [tCrB[None, None, kb], tCrSFB[None, None, kb]], acc)
                if cutlass.const_expr(self.fence):
                    # ldmatrix reads are generic-proxy; the refill is an async-proxy (TMA) write
                    cute.arch.fence_view_async_shared()
                pipe.consumer_release(cs)
                cs.advance()
                if warp == 0:
                    if nxt < k_tiles:
                        pipe.producer_acquire(ps)
                        cute.copy(tma_a, tAgA[None, nxt], tAsA[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                        cute.copy(tma_b, tBgB[None, nxt], tBsB[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                        pipe.producer_commit(ps)
                        ps.advance()
                    nxt = nxt + 1
            cute.autovec_copy(acc, thr_mma.partition_C(gC))

    return BsGemm


# --------------------------------------------------------------------------------- host helpers
E2M1 = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5, 4.0: 6, 6.0: 7}


def pack_e2m1(x):
    import torch
    code = torch.zeros_like(x, dtype=torch.int32)
    for val, c in E2M1.items():
        code[x.abs() == val] = c
    code = code | ((x < 0).to(torch.int32) << 3)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)       # element 2i in the low nibble


def operands(kind, a32, b32, ea, eb):
    import torch
    vec = 16 if kind == "nvf4" else 32
    if kind == "nvf4":
        sfa = torch.pow(2.0, ea.float()).to(torch.float8_e4m3fn).view(torch.uint8)
        sfb = torch.pow(2.0, eb.float()).to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        sfa, sfb = (ea + 127).to(torch.uint8), (eb + 127).to(torch.uint8)
    if kind in ("mxf4", "nvf4"):
        a_in, b_in = pack_e2m1(a32), pack_e2m1(b32)
    else:
        a_in, b_in = a32.to(torch.float8_e4m3fn).view(torch.uint8), b32.to(torch.float8_e4m3fn).view(torch.uint8)
    ad = a32 * torch.pow(2.0, ea.float()).repeat_interleave(vec, dim=1)
    bd = b32 * torch.pow(2.0, eb.float()).repeat_interleave(vec, dim=1)
    return [a_in, b_in, sfa.contiguous(), sfb.contiguous()], ad @ bd.T


def random_case(kind, M, N, K, seed=0):
    import torch
    torch.manual_seed(seed)
    vec = 16 if kind == "nvf4" else 32
    a32 = torch.randint(-2, 3, (M, K), device="cuda").float()
    b32 = torch.randint(-2, 3, (N, K), device="cuda").float()
    ea = torch.randint(-3, 4, (M, K // vec), device="cuda")
    eb = torch.randint(-3, 4, (N, K // vec), device="cuda")
    return operands(kind, a32, b32, ea, eb)


def compile_case(kind, M, N, K, ins, **kw):
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    BsGemm = build()
    c = torch.zeros(M, N, device="cuda")
    ts = [from_dlpack(t, assumed_align=16) for t in ins + [c]]
    f = cute.compile(BsGemm(kind, M, N, K, **kw), *ts)
    return f, ts, c


def launches_wrong(f, ts, c, ref, reps):
    import torch
    bad = 0
    for _ in range(reps):
        c.zero_(); f(*ts); torch.cuda.synchronize()
        bad += float((c - ref).abs().max()) != 0.0
    return bad


# --------------------------------------------------------------------------------- stages
def median_ms(f, ts, reps=20):
    import torch
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        e0.record(); f(*ts); e1.record(); torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    return sorted(times)[reps // 2]


def stage_numerics():
    import torch
    M, N, K = BIG
    for kind in KINDS:
        ins, ref = random_case(kind, M, N, K)
        f, ts, c = compile_case(kind, M, N, K, ins)
        f(*ts); torch.cuda.synchronize()
        md = float((c - ref).abs().max())
        ms = median_ms(f, ts)
        n(f"{kind} {M}x{N}x{K}: {ms:.3f} ms, {2.0 * M * N * K / ms / 1e9:.1f} TFLOPS")
        v(f"numerics-{kind}", md == 0.0, f"max_diff={md}")
        bad = launches_wrong(f, ts, c, ref, 30)
        v(f"stress-{kind}", bad == 0, f"{bad}/30 launches wrong")
    for kind in KINDS:          # rate at the size of the other mainloop entries (4096^3); INFO, no contract
        ins, ref = random_case(kind, 4096, 4096, 4096)
        f, ts, c = compile_case(kind, 4096, 4096, 4096, ins)
        f(*ts); torch.cuda.synchronize()
        md = float((c - ref).abs().max())
        ms = median_ms(f, ts)
        n(f"{kind} 4096^3: {ms:.3f} ms, {2.0 * 4096 ** 3 / ms / 1e9:.1f} TFLOPS (max_diff={md})")


def selector_case(kind, off):
    """A: per-k code pattern (16 distinguishable e2m1 / e4m3 values), B: selector, unit scales."""
    import torch
    M, N, K = 128, 128, 256
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    kk = torch.arange(K, device="cuda")
    a32 = (vals[kk % 8] * torch.where((kk // 8) % 2 == 1, -1.0, 1.0)).repeat(M, 1)
    b32 = torch.zeros(N, K, device="cuda")
    rows = torch.arange(N, device="cuda")
    b32[rows, (rows + off) % K] = 1.0
    vec = 16 if kind == "nvf4" else 32
    zeros_a = torch.zeros(M, K // vec, dtype=torch.long, device="cuda")
    zeros_b = torch.zeros(N, K // vec, dtype=torch.long, device="cuda")
    ins, _ = operands(kind, a32, b32, zeros_a, zeros_b)
    return ins, a32[:, [(r + off) % K for r in range(N)]]


def stage_kmap():
    import torch
    for kind in KINDS:
        ok, why = True, []
        for off in (0, 128):
            ins, expect = selector_case(kind, off)
            f, ts, c = compile_case(kind, 128, 128, 256, ins, stages=2, row_bytes=128)
            f(*ts); torch.cuda.synchronize()
            good = bool(torch.equal(c, expect))
            ok &= good
            why.append(f"off={off}:{'exact' if good else 'PERMUTED'}")
        v(f"kmap-{kind}", ok, " ".join(why))


def stage_ctl_elem_swizzle():
    import torch
    ins, expect = selector_case("mxf4", 0)
    f, ts, c = compile_case("mxf4", 128, 128, 256, ins, stages=2, row_bytes=128, f4_swizzle="elem")
    f(*ts); torch.cuda.synchronize()
    wrong = int((c != expect).sum())
    n(f"mxf4 with the element-unit swizzle on the e2m1 view: {wrong} of {c.numel()} outputs wrong under the selector")
    v("ctl-f4-elem-swizzle", wrong > 0, f"{wrong} outputs wrong")


def stage_ctl_recast_tensor():
    import io
    import contextlib
    ins, _ = selector_case("mxf4", 0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        compile_case("mxf4", 128, 128, 256, ins, stages=2, row_bytes=128, probe={"print_recast": True})
    lines = [ln for ln in buf.getvalue().splitlines() if ln.startswith("TRACE")]
    for ln in lines:
        n(ln)
    byte_sw = any("byte view" in ln and "S<" in ln for ln in lines)
    rec_sw = any("recast_tensor" in ln and "S<" in ln for ln in lines)
    v("ctl-recast-tensor", byte_sw and not rec_sw, f"byte view swizzled={byte_sw}, recast_tensor swizzled={rec_sw}")


def stage_ctl_unfenced():
    M, N, K = BIG
    ins, ref = random_case("mxf8", M, N, K)
    f, ts, c = compile_case("mxf8", M, N, K, ins, fence=False)
    bad = launches_wrong(f, ts, c, ref, 30)
    n(f"mxf8 without fence_view_async_shared before consumer_release: {bad}/30 launches wrong")
    v("ctl-unfenced-races", bad > 0, f"{bad}/30 launches wrong")


EXPECT_OP = {"mxf8": "QMMA.SF.16832.F32.E4M3.E4M3.E8", "mxf4": "OMMA.SF.16864.F32.E2M1.E2M1.E8",
             "nvf4": "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"}


def stage_sass():
    kind = os.environ["FK_KIND"]
    ins, _ = random_case(kind, 256, 256, 512)
    compile_case(kind, 256, 256, 512, ins)
    cubin = sorted(glob.glob(os.path.join(os.environ["CUTE_DSL_DUMP_DIR"], "*.cubin")), key=os.path.getmtime)[-1]
    sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
    hist = {}
    for m in re.finditer(r"/\*[0-9a-f]{4}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", sass):
        hist[m.group(1)] = hist.get(m.group(1), 0) + 1
    mma = {k: c for k, c in hist.items() if re.match(r"(HMMA|QMMA|OMMA|IMMA)", k)}
    from forgekernel.skills import orch_tools as OT
    xp = OT.summarize(sass)["xproxy_release"]
    n(f"{kind}: MMA {mma}; LDSM {sum(c for k, c in hist.items() if k.startswith('LDSM'))}; "
      f"PRMT {hist.get('PRMT', 0)}; MEMBAR {sum(c for k, c in hist.items() if k.startswith('MEMBAR'))}; "
      f"FENCE.VIEW.ASYNC.S {hist.get('FENCE.VIEW.ASYNC.S', 0)}; cross-proxy releases {xp}")
    v(f"sass-{kind}", list(mma) == [EXPECT_OP[kind]] and xp == {"bare": 0, "barrier-only": 0},
      f"mma={mma} xproxy_release={xp}")


STAGES = {"numerics": stage_numerics, "kmap": stage_kmap, "ctl_elem": stage_ctl_elem_swizzle,
          "ctl_recast": stage_ctl_recast_tensor, "ctl_unfenced": stage_ctl_unfenced, "sass": stage_sass}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
        sys.exit(0)
    root = tempfile.mkdtemp(prefix="fk_bs_")
    plan = [("numerics", [f"{p}-{k}" for k in KINDS for p in ("numerics", "stress")], None),
            ("kmap", [f"kmap-{k}" for k in KINDS], None),
            ("ctl_elem", ["ctl-f4-elem-swizzle"], None),
            ("ctl_recast", ["ctl-recast-tensor"], None),
            ("ctl_unfenced", ["ctl-unfenced-races"], None)]
    for kind in KINDS:
        d = os.path.join(root, kind)
        os.makedirs(d)
        plan.append(("sass", [f"sass-{kind}"], {"FK_KIND": kind, "CUTE_DSL_KEEP": "cubin", "CUTE_DSL_DUMP_DIR": d,
                                                "CUTE_DSL_NO_CACHE": "1"}))
    sys.exit(run(HERE, plan, "blockscaled-mma", cap_s=1800))
