#!/usr/bin/env python3
"""setmaxnreg producer / consumer register split on sm_120a (CuTe DSL): the quota a thread starts with, and who can
give registers back.

    python3 -u setmaxnreg_split_check.py      (GPU; ncu for the launch facts; one subprocess per rung, ~8 min:
                                               a rung that never returns is a verdict and costs RUNG_CAP_S)

The GEMM of tma_mbarrier_mainloop_check.py's ``ws`` shape (C fp32 = A . B^T, e4m3, 64 B rows, 3-stage TMA + mbarrier
ring, consumer warps = ldmatrix + mma.sync, one warp issues every TMA load), launched with ``min_blocks_per_mp=1``,
in four CTA shapes ``c<consumer warps>p<producer-role warps>``:
- ``c4p1`` 160 threads, ``c8p1`` 288 threads: the producer role is one warp, alone in its warpgroup;
- ``c4p4`` 256 threads, ``c8p4`` 384 threads: the producer role is a whole warpgroup (its first warp issues TMA,
  the other three execute the ``.dec`` and exit).
The producer role executes ``setmaxregister_decrease(P)``, the consumer warps ``setmaxregister_increase(C)``; a rung
is ``<shape> P/C`` with 0 = that side executes no setmaxnreg.

What the rungs separate: the per-thread quota at launch is the uniform share ``65536 / (128 x warpgroups of the
CTA)`` -- 256 for one or two warpgroups, 170 (168 usable) for three -- whatever the kernel compiled to (118 / 178
registers here). An ``.inc`` up to the share completes with no ``.dec`` anywhere; above it, it waits for registers a
``.dec`` gave back, and only a whole warpgroup's ``.dec`` gives any.

Contracts (``done`` = returns and C == A32 @ B32^T exactly; ``hang`` = compiled, then no return in RUNG_CAP_S):
- ``share-2wg-inc-232-no-dec``   : ``c4p4 0/232`` done -- 232 is under the two-warpgroup share, nobody has to release;
- ``share-3wg-inc-168``          : ``c8p1 24/168`` done -- 168 is under the three-warpgroup share;
- ``above-share-lone-dec-hangs`` : ``c8p1 24/176`` hang -- above the share, and the lone warp's ``.dec`` pays nothing;
- ``above-share-no-dec-hangs``   : ``c8p4 0/176`` hang -- above the share with nobody releasing (the instruction is
  live: an ``.inc`` nobody pays for waits);
- ``wg-dec-pays-232``            : ``c8p4 40/232`` done -- a whole producer warpgroup at 40 pays two consumer
  warpgroups at 232 (2 x 232 + 40 = 504 <= 512);
- ``wg-dec-pays-232-2wg``        : ``c4p4 40/232`` done;
- ``lone-under-share``           : ``c4p1 24/208`` done -- a lone producer warp is harmless while the consumers stay
  under the share.
Facts: registers / thread the kernel compiled to, per shape; every rung's outcome.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.abspath(__file__)

RUNG_CAP_S = 90           # JIT compile ~20 s + a sub-second launch; past this the launch is not coming back
STAGES_S, ROW_B = 3, 64


def build():
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils.hopper_helpers as hh
    from cutlass.memory import SmemAllocator
    from cutlass.tensor_utils import LayoutEnum

    class RegSplitGemm:
        """warps 0-3 consume (ldmatrix + mma.sync); warps 4 .. 3 + prod_warps are the producer role, warp 4 issues TMA."""

        def __init__(self, cons_warps, prod_warps, dec, inc, tile=(128, 128)):
            self.cons_warps, self.prod_warps, self.dec, self.inc = cons_warps, prod_warps, dec, inc
            self.BM, self.BN = tile

        @cute.jit
        def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
            dt = cutlass.Float8E4M3FN
            mma_op = cute.nvgpu.warp.MmaFP8Op(dt, cutlass.Float32, (16, 8, 32))
            bk = ROW_B
            wm = self.cons_warps // 2                     # consumer warps: wm (M) x 2 (N)
            tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((wm, 2, 1)), permutation_mnk=(16 * wm, 32, 32))
            sA_l = hh.make_smem_layout_a(LayoutEnum.from_tensor(mA), (self.BM, self.BN, bk), dt, STAGES_S)
            sB_l = hh.make_smem_layout_b(LayoutEnum.from_tensor(mB), (self.BM, self.BN, bk), dt, STAGES_S)
            sA_1 = cute.slice_(sA_l, (None, None, 0))
            sB_1 = cute.slice_(sB_l, (None, None, 0))
            op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(op, mA, sA_1, (self.BM, bk), num_multicast=1)
            tma_b, tB = cute.nvgpu.cpasync.make_tiled_tma_atom(op, mB, sB_1, (self.BN, bk), num_multicast=1)
            tx = cute.size_in_bytes(dt, sA_1) + cute.size_in_bytes(dt, sB_1)

            @cute.struct
            class Storage:
                bars: cute.struct.MemRange[cutlass.Int64, STAGES_S * 2]
                sA: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sA_l)], 1024]
                sB: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sB_l)], 1024]

            self.storage_t = Storage
            grid = [cute.size(mA, mode=[0]) // self.BM, cute.size(mB, mode=[0]) // self.BN, 1]
            self.kernel(tma_a, tA, tma_b, tB, mC, tiled_mma, sA_l, sB_l, bk, tx).launch(
                grid=grid, block=[32 * (self.cons_warps + self.prod_warps), 1, 1], smem=Storage.size_in_bytes(),
                min_blocks_per_mp=1)

        @cute.kernel
        def kernel(self, tma_a: cute.CopyAtom, mA: cute.Tensor, tma_b: cute.CopyAtom, mB: cute.Tensor, mC: cute.Tensor,
                   tiled_mma: cute.TiledMma, sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout,
                   bk: cutlass.Constexpr, tx: cutlass.Constexpr):
            S = STAGES_S
            dt = cutlass.Float8E4M3FN
            tidx, _, _ = cute.arch.thread_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            bidx, bidy, _ = cute.arch.block_idx()
            st = SmemAllocator().allocate(self.storage_t)
            sA = st.sA.get_tensor(sA_l.outer, swizzle=sA_l.inner)
            sB = st.sB.get_tensor(sB_l.outer, swizzle=sB_l.inner)
            if warp == self.cons_warps:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_b)
            pipe = pipeline.PipelineTmaAsync.create(
                num_stages=S,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.cons_warps),
                tx_count=tx,
                barrier_storage=st.bars.data_ptr(),
                cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            )
            gA = cute.local_tile(mA, (self.BM, bk), (bidx, None))
            gB = cute.local_tile(mB, (self.BN, bk), (bidy, None))
            k_tiles = cute.size(gA, mode=[2])
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(tma_a, 0, cute.make_layout(1),
                                                          cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(tma_b, 0, cute.make_layout(1),
                                                          cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))
            if warp >= self.cons_warps:
                # ---------------- producer role: every warp of it gives its quota back, its first warp issues TMA
                if cutlass.const_expr(self.dec > 0):
                    cute.arch.setmaxregister_decrease(self.dec)
                if warp == self.cons_warps:
                    ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, S)
                    for kt in cutlass.range(k_tiles):
                        pipe.producer_acquire(ps)
                        cute.copy(tma_a, tAgA[None, kt], tAsA[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                        cute.copy(tma_b, tBgB[None, kt], tBsB[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                        pipe.producer_commit(ps)
                        ps.advance()
                    pipe.producer_tail(ps)
            else:
                # ---------------- consumer warps: ldmatrix + mma.sync
                if cutlass.const_expr(self.inc > 0):
                    cute.arch.setmaxregister_increase(self.inc)
                gC = cute.local_tile(mC, (self.BM, self.BN), (bidx, bidy))
                thr_mma = tiled_mma.get_slice(tidx)
                tCsA = thr_mma.partition_A(sA[None, None, 0])
                tCsB = thr_mma.partition_B(sB[None, None, 0])
                tCrA = tiled_mma.make_fragment_A(tCsA)
                tCrB = tiled_mma.make_fragment_B(tCsB)
                acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((self.BM, self.BN)), cutlass.Float32)
                acc.fill(0.0)
                ldsm = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), dt)
                cpA = cute.make_tiled_copy_A(ldsm, tiled_mma)
                cpB = cute.make_tiled_copy_B(ldsm, tiled_mma)
                lA = cpA.get_slice(tidx).partition_S(sA)
                lB = cpB.get_slice(tidx).partition_S(sB)
                rA, rB = cpA.retile(tCrA), cpB.retile(tCrB)
                cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)
                for kt in cutlass.range(k_tiles):
                    pipe.consumer_wait(cs, pipe.consumer_try_wait(cs))
                    for kb in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                        cute.copy(cpA, lA[None, None, kb, cs.index], rA[None, None, kb])
                        cute.copy(cpB, lB[None, None, kb, cs.index], rB[None, None, kb])
                        cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                    # ldmatrix reads (generic proxy) ordered before the TMA refill (async proxy) of this slot
                    cute.arch.fence_view_async_shared()
                    pipe.consumer_release(cs)
                    cs.advance()
                cute.autovec_copy(acc, thr_mma.partition_C(gC))

    return RegSplitGemm


def stage_rung():
    """One rung: FK_RUNG = <prod_warps>_<dec>_<inc>. Prints DONE lines; the parent reads them (or the cap)."""
    import time
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    cons_warps, prod_warps, dec, inc = (int(x) for x in os.environ["FK_RUNG"].split("_"))
    M = N = K = 2048
    torch.manual_seed(0)
    a32 = torch.randint(-2, 3, (M, K), device="cuda").float()
    b32 = torch.randint(-2, 3, (N, K), device="cuda").float()
    c = torch.zeros(M, N, device="cuda")
    ts = [from_dlpack(t, assumed_align=16) for t in (a32.to(torch.float8_e4m3fn), b32.to(torch.float8_e4m3fn), c)]
    f = cute.compile(build()(cons_warps, prod_warps, dec, inc), *ts)
    print("COMPILED", flush=True)
    t0 = time.time()
    f(*ts)
    torch.cuda.synchronize()
    md = float((c - a32 @ b32.T).abs().max())
    print(f"DONE max_diff={md} s={time.time() - t0:.2f}", flush=True)


def rung(cons_warps, prod_warps, dec, inc):
    """(state, detail): state in {"done", "hang", "error"}."""
    env = dict(os.environ, FK_RUNG=f"{cons_warps}_{prod_warps}_{dec}_{inc}", CUTE_DSL_ARCH="sm_120a",
               CUTE_DSL_CACHE_DIR=tempfile.mkdtemp(prefix="fk_regsplit_"))
    try:
        p = subprocess.run([sys.executable, "-u", HERE, "rung"], env=env, capture_output=True, text=True,
                           timeout=RUNG_CAP_S)
        out = p.stdout
    except subprocess.TimeoutExpired as te:
        out = te.stdout.decode("utf-8", "replace") if isinstance(te.stdout, bytes) else (te.stdout or "")
        return ("hang" if "COMPILED" in out else "error"), f"no return in {RUNG_CAP_S} s (compiled={'COMPILED' in out})"
    done = [ln for ln in out.splitlines() if ln.startswith("DONE")]
    if done:
        return "done", done[-1][5:]
    tail = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()][-1:] or ["no output"]
    return "error", f"rc={p.returncode} {tail[0][:160]}"


def launch_regs():
    """registers / thread each shape compiled to (ncu launch metric), None where ncu gave nothing."""
    got = {}
    for cw, pw in ((4, 1), (4, 4), (8, 1), (8, 4)):
        env = dict(os.environ, FK_RUNG=f"{cw}_{pw}_0_0", CUTE_DSL_ARCH="sm_120a",
                   CUTE_DSL_CACHE_DIR=tempfile.mkdtemp(prefix="fk_regsplit_ncu_"))
        try:
            p = subprocess.run(["ncu", "--csv", "--kernel-name", "regex:RegSplitGemm", "--launch-count", "1",
                                "--metrics", "launch__registers_per_thread", sys.executable, "-u", HERE, "rung"],
                               env=env, capture_output=True, text=True, timeout=600)
            vals = [ln.rsplit('","', 1)[-1].strip('"') for ln in p.stdout.splitlines()
                    if "launch__registers_per_thread" in ln]
            got[f"c{cw}p{pw}"] = int(float(vals[-1])) if vals else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            got[f"c{cw}p{pw}"] = None
    return got


def share(cons_warps, prod_warps):
    """Per-thread quota at launch: 65536 registers over the CTA's warpgroups (threads rounded up to 128)."""
    wgs = -(-(cons_warps + prod_warps) // 4)
    return min(256, 65536 // (128 * wgs))


PLAN = [  # tag, consumer warps, producer-role warps, dec, inc, expected state
    ("share-2wg-inc-232-no-dec", 4, 4, 0, 232, "done"),
    ("share-3wg-inc-168", 8, 1, 24, 168, "done"),
    ("above-share-lone-dec-hangs", 8, 1, 24, 176, "hang"),
    ("above-share-no-dec-hangs", 8, 4, 0, 176, "hang"),
    ("wg-dec-pays-232", 8, 4, 40, 232, "done"),
    ("wg-dec-pays-232-2wg", 4, 4, 40, 232, "done"),
    ("lone-under-share", 4, 1, 24, 208, "done"),
]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rung":
        stage_rung()
        sys.exit(0)
    print(f"FACT registers / thread compiled, per shape: {launch_regs()}", flush=True)
    verdicts = []
    for tag, cw, pw, dec, inc, want in PLAN:
        state, detail = rung(cw, pw, dec, inc)
        exact = state == "done" and detail.startswith("max_diff=0.0 ")
        print(f"FACT rung c{cw}p{pw} {dec}/{inc}: {state} ({detail}); share {share(cw, pw)} registers / thread",
              flush=True)
        verdicts.append((tag, exact if want == "done" else state == "hang", f"c{cw}p{pw} {dec}/{inc} {state}: {detail}"))
    for tag, ok, md in verdicts:
        print(f"PROBE {'PASS' if ok else 'FAIL'} {tag} ({md})")
    all_ok = all(ok for _, ok, _ in verdicts)
    print(f"PROBE {'PASS' if all_ok else 'FAIL'} setmaxnreg-split contracts={len(verdicts)}")
    sys.exit(0 if all_ok else 1)
