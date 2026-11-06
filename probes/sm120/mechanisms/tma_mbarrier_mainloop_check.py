#!/usr/bin/env python3
"""TMA + mbarrier mainloop on sm_120a (CuTe DSL): TMA tiles into an S-stage mbarrier ring, ldmatrix + mma.sync.

    python3 -u tma_mbarrier_mainloop_check.py      (GPU; cuobjdump; ncu with counter access for the N-* contracts)

Same GEMM as cpasync_multistage_check.py (C fp32 = A . B^T, 4096^3, K-major, CTA 128 x 128, 4 MMA warps 2 x 2), with
the supply replaced by ``make_tiled_tma_atom(CopyBulkTensorTileG2SOp())`` + ``PipelineTmaAsync`` (full / empty
mbarrier per stage, tx count = A + B stage bytes). Two shapes of the producer:
- ``inline``: no producer warp; warp 0 fills all S stages up front, then after each ``consumer_release`` re-acquires
  the slot just released and issues the next tile (128 threads);
- ``ws``: warp 4 is a dedicated producer (TMA only, ``producer_tail`` before exit), warps 0-3 consume (160 threads).
Every consumer executes ``fence_view_async_shared()`` before ``consumer_release``: its ldmatrix reads of the stage are
generic-proxy, the TMA refill of the released slot writes through the async proxy, and PTX orders the two only through
a proxy fence (see blockscaled_mma_check.py for the measured race without it).

Contracts:
- ``numerics-<shape>-<dt>-r<row>-s<S>``: C == A32 @ B32^T exactly;
- ``stress-inline-fp8`` : the same compiled inline e4m3 kernel, 30 launches, every one exact;
- ``sass-<shape>``     : UTMALDG present, no LDGSTS, mbarrier try-wait present, exactly one BAR.SYNC on the page
  (the pipeline's init rendezvous) -- no per-tile CTA barrier -- and ``orch_tools.xproxy_releases`` finds no stage
  release after an LDSM without a FENCE.VIEW.ASYNC.S between;
- ``ctl-unfenced-inline``: (control) the inline build without the fence: the tool nominates its one release ``bare``;
- ``orch-<shape>``     : the dossier resolves 2 x S mbarrier objects, all CONST (full + empty per stage); for ``ws`` its
  edges carry both directions copy-producer -> mma-consumer and mma-consumer -> copy-producer;
- ``N-occupancy``      : (ncu) ``inline`` (64 B rows, 3 stages) runs 2 CTA / SM, ``ws`` 1 CTA / SM, and ``inline`` keeps
  the tensor pipe busier.
Facts: TFLOPS per configuration (median of 20, clocks unlocked), NCU readings.
"""
import glob
import json
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


def build():
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils.hopper_helpers as hh
    from cutlass.memory import SmemAllocator
    from cutlass.tensor_utils import LayoutEnum

    class TmaWsGemm:
        """C (fp32) = A . B^T; 1 TMA producer warp + 4 ldmatrix/mma.sync consumer warps, mbarrier ring of S stages."""

        def __init__(self, dtype_name, stages, row_bytes=64, tile=(128, 128), regs=None, ws=True, fence=True):
            self.ws = ws                         # True: warp 4 is a dedicated producer; False: warp 0 issues TMA in-loop
            self.fence = fence                   # proxy fence before each consumer_release (False = the control)
            self.dtype_name, self.stages, self.row_bytes = dtype_name, stages, row_bytes
            self.BM, self.BN = tile
            self.regs = regs                     # (producer, consumer) setmaxnreg targets, or None

        @cute.jit
        def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
            if cutlass.const_expr(self.dtype_name == "fp8"):
                dt = cutlass.Float8E4M3FN
                mma_op = cute.nvgpu.warp.MmaFP8Op(dt, cutlass.Float32, (16, 8, 32))
                bk = self.row_bytes
                perm = (32, 32, 32)
            else:
                dt = cutlass.BFloat16
                mma_op = cute.nvgpu.warp.MmaF16BF16Op(dt, cutlass.Float32, (16, 8, 16))
                bk = self.row_bytes // 2
                perm = (32, 32, 16)
            tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((2, 2, 1)), permutation_mnk=perm)
            sA_l = hh.make_smem_layout_a(LayoutEnum.from_tensor(mA), (self.BM, self.BN, bk), dt, self.stages)
            sB_l = hh.make_smem_layout_b(LayoutEnum.from_tensor(mB), (self.BM, self.BN, bk), dt, self.stages)
            sA_1 = cute.slice_(sA_l, (None, None, 0))
            sB_1 = cute.slice_(sB_l, (None, None, 0))
            op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            tma_a, tA = cute.nvgpu.cpasync.make_tiled_tma_atom(op, mA, sA_1, (self.BM, bk), num_multicast=1)
            tma_b, tB = cute.nvgpu.cpasync.make_tiled_tma_atom(op, mB, sB_1, (self.BN, bk), num_multicast=1)
            tx = cute.size_in_bytes(dt, sA_1) + cute.size_in_bytes(dt, sB_1)

            @cute.struct
            class Storage:
                bars: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
                sA: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sA_l)], 1024]
                sB: cute.struct.Align[cute.struct.MemRange[dt, cute.cosize(sB_l)], 1024]

            self.storage_t = Storage
            grid = [cute.size(mA, mode=[0]) // self.BM, cute.size(mB, mode=[0]) // self.BN, 1]
            self.kernel(tma_a, tA, tma_b, tB, mC, tiled_mma, sA_l, sB_l, bk, tx).launch(
                grid=grid, block=[160 if self.ws else 128, 1, 1], smem=Storage.size_in_bytes())

        @cute.kernel
        def kernel(self, tma_a: cute.CopyAtom, mA: cute.Tensor, tma_b: cute.CopyAtom, mB: cute.Tensor, mC: cute.Tensor,
                   tiled_mma: cute.TiledMma, sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout,
                   bk: cutlass.Constexpr, tx: cutlass.Constexpr):
            S = self.stages
            dt = cutlass.Float8E4M3FN if cutlass.const_expr(self.dtype_name == "fp8") else cutlass.BFloat16
            tidx, _, _ = cute.arch.thread_idx()
            warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            bidx, bidy, _ = cute.arch.block_idx()
            st = SmemAllocator().allocate(self.storage_t)
            sA = st.sA.get_tensor(sA_l.outer, swizzle=sA_l.inner)
            sB = st.sB.get_tensor(sB_l.outer, swizzle=sB_l.inner)
            if warp == (4 if self.ws else 0):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_a)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_b)
            pipe = pipeline.PipelineTmaAsync.create(
                num_stages=S,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
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
            if cutlass.const_expr(not self.ws):
                # ---------------- no producer warp: warp 0 issues TMA from inside the consumer loop
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
                ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, S)
                cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)
                nxt = cutlass.Int32(0)
                if warp == 0:
                    for s0 in cutlass.range_constexpr(S):         # fill every stage up front
                        if nxt < k_tiles:
                            pipe.producer_acquire(ps)
                            cute.copy(tma_a, tAgA[None, nxt], tAsA[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                            cute.copy(tma_b, tBgB[None, nxt], tBsB[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                            pipe.producer_commit(ps)
                            ps.advance()
                        nxt = nxt + 1
                for kt in cutlass.range(k_tiles):
                    pipe.consumer_wait(cs, pipe.consumer_try_wait(cs))
                    for kb in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                        cute.copy(cpA, lA[None, None, kb, cs.index], rA[None, None, kb])
                        cute.copy(cpB, lB[None, None, kb, cs.index], rB[None, None, kb])
                        cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                    if cutlass.const_expr(self.fence):
                        # ldmatrix reads (generic proxy) ordered before the TMA refill (async proxy) of this slot
                        cute.arch.fence_view_async_shared()
                    pipe.consumer_release(cs)
                    cs.advance()
                    if warp == 0:
                        if nxt < k_tiles:                          # refill the slot just released (waits for all 4 warps)
                            pipe.producer_acquire(ps)
                            cute.copy(tma_a, tAgA[None, nxt], tAsA[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                            cute.copy(tma_b, tBgB[None, nxt], tBsB[None, ps.index], tma_bar_ptr=pipe.producer_get_barrier(ps))
                            pipe.producer_commit(ps)
                            ps.advance()
                        nxt = nxt + 1
                cute.autovec_copy(acc, thr_mma.partition_C(gC))
            else:
                if warp == 4:
                    # ---------------- producer warp: TMA only
                    if cutlass.const_expr(self.regs is not None):
                        cute.arch.setmaxregister_decrease(self.regs[0])
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
                    if cutlass.const_expr(self.regs is not None):
                        cute.arch.setmaxregister_increase(self.regs[1])
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
                        if cutlass.const_expr(self.fence):
                            cute.arch.fence_view_async_shared()
                        pipe.consumer_release(cs)
                        cs.advance()
                    cute.autovec_copy(acc, thr_mma.partition_C(gC))

    return TmaWsGemm


def run_cfg(dt, stages, row_bytes, ws, reps=20, check=True, M=4096, N=4096, K=4096, fence=True, stress=0):
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    TmaWsGemm = build()
    torch.manual_seed(0)
    a32 = torch.randint(-2, 3, (M, K), device="cuda").float()
    b32 = torch.randint(-2, 3, (N, K), device="cuda").float()
    tdt = torch.float8_e4m3fn if dt == "fp8" else torch.bfloat16
    c = torch.zeros(M, N, device="cuda")
    ts = [from_dlpack(t, assumed_align=16) for t in (a32.to(tdt), b32.to(tdt), c)]
    f = cute.compile(TmaWsGemm(dt, stages, row_bytes, ws=ws, fence=fence), *ts)
    f(*ts)
    torch.cuda.synchronize()
    ref = a32 @ b32.T
    md = float((c - ref).abs().max()) if check else float("nan")
    if stress:
        bad = 0
        for _ in range(stress):
            c.zero_(); f(*ts); torch.cuda.synchronize()
            bad += float((c - ref).abs().max()) != 0.0
        return bad
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        e0.record(); f(*ts); e1.record(); torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    ms = sorted(times)[len(times) // 2]
    return md, ms, 2.0 * M * N * K / ms / 1e9


CFGS = [("inline", "fp8", 64, 3), ("inline", "bf16", 64, 3), ("ws", "fp8", 64, 3), ("ws", "bf16", 64, 3),
        ("ws", "fp8", 128, 2)]


def stage_numerics():
    for shape, dt, row, s in CFGS:
        md, ms, tf = run_cfg(dt, s, row, shape == "ws")
        n(f"{shape} {dt} rows={row}B stages={s}: {ms:.3f} ms, {tf:.1f} TFLOPS")
        v(f"numerics-{shape}-{dt}-r{row}-s{s}", md == 0.0, f"max_diff={md}")


def stage_stress():
    bad = run_cfg("fp8", 3, 64, False, stress=30)
    v("stress-inline-fp8", bad == 0, f"{bad}/30 launches wrong")


def stage_sass():
    shape = os.environ["FK_SHAPE"]
    fence = os.environ.get("FK_FENCE", "1") == "1"
    S = 3
    run_cfg("fp8", S, 64, shape == "ws", reps=1, check=False, fence=fence)
    cubin = sorted(glob.glob(os.path.join(os.environ["CUTE_DSL_DUMP_DIR"], "*.cubin")), key=os.path.getmtime)[-1]
    sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
    cnt = lambda pat: len(re.findall(pat, sass))
    c = {"UTMALDG": cnt(r"\bUTMALDG"), "LDGSTS": cnt(r"\bLDGSTS"), "TRYWAIT": cnt(r"SYNCS\.PHASECHK\.TRANS64\.TRYWAIT"),
         "BAR.SYNC": cnt(r"\bBAR\.SYNC"), "QMMA": cnt(r"\bQMMA"), "LDSM": cnt(r"\bLDSM"),
         "MEMBAR": cnt(r"\bMEMBAR"), "FENCE.VIEW.ASYNC.S": cnt(r"\bFENCE\.VIEW\.ASYNC\.S")}
    from forgekernel.skills import orch_tools as OT
    d = OT.summarize(sass)
    xp = d["xproxy_release"]
    if os.environ.get("FK_FIXTURE"):
        open(os.environ["FK_FIXTURE"], "w").write(
            "\n".join(re.sub(r"\s*/\* 0x[0-9a-f]{16} \*/", "", ln) for ln in sass.splitlines() if ln.strip()) + "\n")
    if not fence:
        n(f"{shape} without the fence: SASS {c}; cross-proxy releases {xp}")
        v(f"ctl-unfenced-{shape}", xp == {"bare": 1, "barrier-only": 0}, f"xproxy_release={xp}")
        return
    n(f"{shape}: SASS {c}; cross-proxy releases {xp}")
    v(f"sass-{shape}", c["UTMALDG"] > 0 and c["LDGSTS"] == 0 and c["TRYWAIT"] > 0 and c["BAR.SYNC"] == 1
      and xp == {"bare": 0, "barrier-only": 0}, f"{c} xproxy_release={xp}")
    got = {k: d.get(k) for k in ("loops", "mbar_objs", "mbar_tiers", "rings", "edges", "cache_ops")}
    n(f"{shape}: dossier {json.dumps(got)}")
    ok = d["mbar_objs"] == 2 * S and d["mbar_tiers"] == {"CONST": 2 * S}
    if shape == "ws":
        e = [tuple(x) for x in d["edges"]]
        ok = ok and ("copy-producer", "mma-consumer") in e and ("mma-consumer", "copy-producer") in e
    v(f"orch-{shape}", ok, f"mbar_objs={d['mbar_objs']} tiers={d['mbar_tiers']} edges={d['edges']}")


def stage_target():
    run_cfg("fp8", 3, 64, os.environ["FK_SHAPE"] == "ws", reps=1, check=False)


def ncu_occupancy():
    mets = ["launch__occupancy_limit_registers", "launch__occupancy_limit_shared_mem", "launch__registers_per_thread",
            "sm__warps_active.avg.per_cycle_active", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "smsp__inst_executed.sum"]
    got = {}
    for shape in ("inline", "ws"):
        env = dict(os.environ, FK_SHAPE=shape, CUTE_DSL_ARCH="sm_120a",
                   CUTE_DSL_CACHE_DIR=tempfile.mkdtemp(prefix="fk_tma_ncu_"))
        p = subprocess.run(["ncu", "--csv", "--kernel-name", "regex:TmaWsGemm", "--launch-skip", "1", "--launch-count", "1",
                            "--metrics", ",".join(mets), sys.executable, "-u", HERE, "target"],
                           env=env, capture_output=True, text=True, timeout=900)
        vals = {}
        for line in p.stdout.splitlines():
            m = re.match(r'"[^"]*",.*"([a-z0-9_]+__[a-z0-9_.]+)","[^"]*","([^"]*)"$', line)
            if m:
                vals[m.group(1)] = float(m.group(2).replace(",", ""))
        got[shape] = vals
        print(f"FACT ncu fp8 rows=64B stages=3 {shape}: {json.dumps(vals)}", flush=True)
    try:
        i, w = got["inline"], got["ws"]
        lim = lambda d: min(d["launch__occupancy_limit_registers"], d["launch__occupancy_limit_shared_mem"])
        t = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
        ok = lim(i) == 2 and lim(w) == 1 and i[t] > w[t]
        md = f"inline {lim(i):.0f} CTA/SM tensor {i[t]:.1f} %, ws {lim(w):.0f} CTA/SM tensor {w[t]:.1f} %"
    except KeyError as e:
        ok, md = False, f"missing {e}"
    return [("N-occupancy", ok, md)]


STAGES = {"numerics": stage_numerics, "stress": stage_stress, "sass": stage_sass, "target": stage_target}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
        sys.exit(0)
    root = tempfile.mkdtemp(prefix="fk_tma_")
    plan = [("numerics", [f"numerics-{s}-{d}-r{r}-s{st}" for s, d, r, st in CFGS], None),
            ("stress", ["stress-inline-fp8"], None)]
    for shape, fence in (("inline", "1"), ("ws", "1"), ("inline", "0")):
        dd = os.path.join(root, f"{shape}{'' if fence == '1' else '_unfenced'}")
        os.makedirs(dd)
        env = {"FK_SHAPE": shape, "FK_FENCE": fence, "CUTE_DSL_KEEP": "cubin", "CUTE_DSL_DUMP_DIR": dd,
               "CUTE_DSL_NO_CACHE": "1"}
        if os.environ.get("FK_FIXTURE_DIR"):
            env["FK_FIXTURE"] = os.path.join(os.environ["FK_FIXTURE_DIR"],
                                             f"tma_mbarrier_fp8_{shape}{'' if fence == '1' else '_unfenced'}.sass")
        plan.append(("sass", [f"sass-{shape}", f"orch-{shape}"] if fence == "1" else [f"ctl-unfenced-{shape}"], env))
    rc = run(HERE, plan, "tma-mbarrier-core", cap_s=900)
    occ = ncu_occupancy()
    for tag, ok, md in occ:
        print(f"PROBE {'PASS' if ok else 'FAIL'} {tag} ({md})")
    all_ok = rc == 0 and all(ok for _, ok, _ in occ)
    print(f"PROBE {'PASS' if all_ok else 'FAIL'} tma-mbarrier contracts={sum(len(t) for _, t, _ in plan) + len(occ)}")
    sys.exit(0 if all_ok else 1)
