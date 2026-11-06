#!/usr/bin/env python3
"""cp.async multistage mainloop on sm_120a (CuTe DSL): gmem --LDGSTS--> S smem stages --ldmatrix--> mma.sync.

    python3 -u cpasync_multistage_check.py      (GPU; cuobjdump; ncu with counter access for the occupancy contracts)

C (fp32, M = N = K = 4096) = A . B^T with A, B K-major, bf16 (``MmaF16BF16Op`` 16x8x16 = HMMA.16816) or e4m3
(``MmaFP8Op`` 16x8x32 = QMMA.16832); CTA tile 128 x 128, 4 warps (2 x 2); each stage holds ``row_bytes`` of K per
row (128 B or 64 B) of A and B in ``make_smem_layout_a/b`` swizzled layouts. Loop (runtime trip count = K tiles):
``cp.async.wait_group(S-2)`` -> ``__syncthreads`` -> issue the load of tile k+S-1 into the slot read last iteration
-> ``cp.async.commit_group`` -> ldmatrix + MMA over the current slot. Integer-valued inputs make the fp32 result
exact (|partial sums| <= 4 * 4096).

Contracts:
- ``numerics-<dt>-r<row>-s<S>``: C == A32 @ B32^T exactly, for bf16 / fp8 x 128 / 64 B rows x 2 / 3 stages;
- ``smem-ceiling``           : bf16, 128 B rows, 4 stages (128 KiB) fails at launch with cudaErrorInvalidValue;
- ``sass-wait-depth-s<S>``   : the loop carries exactly one ``DEPBAR.LE SB0, 0x<S-2>`` and the page S ``LDGDEPBAR``
  (S-1 in the prologue, 1 in the loop);
- ``orch-commit-s<S>``       : the orchestration dossier reads one mma-consumer loop and its commit axis
  ``CPWAIT {LE 0x<S-2> x1}`` (the 2- and 3-stage pages differ there and nowhere else on that axis);
- ``ncu-occupancy``          : (ncu) 128 B rows x 3 stages are limited to 1 CTA / SM by shared memory, 64 B rows x 3
  stages to 2 (``launch__occupancy_limit_shared_mem``), and the 64 B variant achieves ~2x the active warps.
- ``numerics-shape-*``       : e4m3, 64 B rows, 2 stages, bit-exact for CTA 128 x 128 / 4 warps, 128 x 64 / 4 warps,
  128 x 128 / 8 warps and 128 x 128 / 4 warps with ``min_blocks_per_mp=3``;
- ``ncu-register-cap-spills`` : (ncu) at 64 B rows x 2 stages shared memory admits 3 CTA / SM but registers (198 / thread)
  only 2; ``min_blocks_per_mp=3`` brings registers to <= 170 and the register limit to 3, and the kernel spills
  (local-load sectors 0 -> > 0) and executes more instructions.
Facts: TFLOPS per configuration and per shape (median of 20 launches, unlocked clocks), NCU readings.
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
from _ladder import err_text, n, run, v  # noqa: E402


def build():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.hopper_helpers as hh
    from cutlass.memory import SmemAllocator
    from cutlass.tensor_utils import LayoutEnum

    BM, BN = 128, 128


    class PipeGemm:
        """C (fp32) = A . B^T, A (M,K) / B (N,K) K-major; cp.async multistage -> ldmatrix -> mma.sync."""

        def __init__(self, dtype_name, stages, row_bytes=128, tile=(128, 128), warps=(2, 2), min_blocks=0):
            self.dtype_name, self.stages, self.warps = dtype_name, stages, warps
            self.BM, self.BN = tile
            self.min_blocks = min_blocks
            self.row_bytes = row_bytes                             # bytes of K per smem row per stage

        @cute.jit
        def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
            if cutlass.const_expr(self.dtype_name == "fp8"):
                dt = cutlass.Float8E4M3FN
                mma_op = cute.nvgpu.warp.MmaFP8Op(dt, cutlass.Float32, (16, 8, 32))
                bk = self.row_bytes
                perm = (self.warps[0] * 16, self.warps[1] * 16, 32)
            else:
                dt = cutlass.BFloat16
                mma_op = cute.nvgpu.warp.MmaF16BF16Op(dt, cutlass.Float32, (16, 8, 16))
                bk = self.row_bytes // 2
                perm = (self.warps[0] * 16, self.warps[1] * 16, 16)
            tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((self.warps[0], self.warps[1], 1)),
                                            permutation_mnk=perm)
            sA_l = hh.make_smem_layout_a(LayoutEnum.from_tensor(mA), (self.BM, self.BN, bk), dt, self.stages)
            sB_l = hh.make_smem_layout_b(LayoutEnum.from_tensor(mB), (self.BM, self.BN, bk), dt, self.stages)
            nthr = self.warps[0] * self.warps[1] * 32
            grid = [cute.size(mA, mode=[0]) // self.BM, cute.size(mB, mode=[0]) // self.BN, 1]
            self.kernel(mA, mB, mC, tiled_mma, sA_l, sB_l, bk).launch(grid=grid, block=[nthr, 1, 1],
                                                                      min_blocks_per_mp=self.min_blocks)

        @cute.kernel
        def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, tiled_mma: cute.TiledMma,
                   sA_l: cute.ComposedLayout, sB_l: cute.ComposedLayout, bk: cutlass.Constexpr):
            S = self.stages
            nthr = self.warps[0] * self.warps[1] * 32
            dt = mA.element_type
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()
            smem = SmemAllocator()
            sA = smem.allocate_tensor(dt, sA_l, byte_alignment=1024)
            sB = smem.allocate_tensor(dt, sB_l, byte_alignment=1024)
            gA = cute.local_tile(mA, (self.BM, bk), (bidx, None))      # (BM, bk, k_tiles)
            gB = cute.local_tile(mB, (self.BN, bk), (bidy, None))
            gC = cute.local_tile(mC, (self.BM, self.BN), (bidx, bidy))
            k_tiles = cute.size(gA, mode=[2])
            # gmem -> smem: 16 B per thread per copy, 8 threads per 128 B row
            g2s = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
                                      dt, num_bits_per_copy=128)
            epc = 128 // dt.width                                  # elements per 16 B
            tpr = self.row_bytes // 16                             # threads per smem row (16 B each)
            tcopy = cute.make_tiled_copy_tv(g2s, cute.make_layout((nthr // tpr, tpr), stride=(tpr, 1)),
                                            cute.make_layout((1, epc)))
            thr = tcopy.get_slice(tidx)
            tAgA, tAsA = thr.partition_S(gA), thr.partition_D(sA)
            tBgB, tBsB = thr.partition_S(gB), thr.partition_D(sB)
            # mma + ldmatrix
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
            lA = cpA.get_slice(tidx).partition_S(sA)               # (CPY, M, KB, S)
            lB = cpB.get_slice(tidx).partition_S(sB)
            rA, rB = cpA.retile(tCrA), cpB.retile(tCrB)
            # prologue: S-1 stages in flight
            for s in cutlass.range_constexpr(S - 1):
                cute.copy(tcopy, tAgA[None, None, None, s], tAsA[None, None, None, s])
                cute.copy(tcopy, tBgB[None, None, None, s], tBsB[None, None, None, s])
                cute.arch.cp_async_commit_group()
            rd = cutlass.Int32(0)
            wr = cutlass.Int32(S - 1)
            nxt = cutlass.Int32(S - 1)
            for kt in cutlass.range(k_tiles):
                cute.arch.cp_async_wait_group(S - 2)
                cute.arch.sync_threads()
                if nxt < k_tiles:
                    cute.copy(tcopy, tAgA[None, None, None, nxt], tAsA[None, None, None, wr])
                    cute.copy(tcopy, tBgB[None, None, None, nxt], tBsB[None, None, None, wr])
                cute.arch.cp_async_commit_group()
                for kb in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                    cute.copy(cpA, lA[None, None, kb, rd], rA[None, None, kb])
                    cute.copy(cpB, lB[None, None, kb, rd], rB[None, None, kb])
                    cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
                rd = rd + 1
                if rd == S:
                    rd = cutlass.Int32(0)
                wr = wr + 1
                if wr == S:
                    wr = cutlass.Int32(0)
                nxt = nxt + 1
            cute.autovec_copy(acc, thr_mma.partition_C(gC))

    return PipeGemm


def run_cfg(dt, stages, row_bytes, reps=20, check=True, M=4096, N=4096, K=4096, tile=(128, 128), warps=(2, 2),
            min_blocks=0):
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    PipeGemm = build()
    torch.manual_seed(0)
    a32 = torch.randint(-2, 3, (M, K), device="cuda").float()
    b32 = torch.randint(-2, 3, (N, K), device="cuda").float()
    tdt = torch.float8_e4m3fn if dt == "fp8" else torch.bfloat16
    c = torch.zeros(M, N, device="cuda")
    ts = [from_dlpack(t, assumed_align=16) for t in (a32.to(tdt), b32.to(tdt), c)]
    f = cute.compile(PipeGemm(dt, stages, row_bytes, tile, warps, min_blocks), *ts)
    f(*ts)
    torch.cuda.synchronize()
    md = float((c - a32 @ b32.T).abs().max()) if check else float("nan")
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(reps):
        e0.record(); f(*ts); e1.record(); torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    ms = sorted(times)[len(times) // 2]
    return md, ms, 2.0 * M * N * K / ms / 1e9


CFGS = [(dt, row, s) for dt in ("bf16", "fp8") for row in (128, 64) for s in (2, 3)]


def stage_numerics():
    for dt, row, s in CFGS:
        md, ms, tf = run_cfg(dt, s, row)
        n(f"{dt} rows={row}B stages={s}: {ms:.3f} ms, {tf:.1f} TFLOPS")
        v(f"numerics-{dt}-r{row}-s{s}", md == 0.0, f"max_diff={md}")


def stage_ceiling():
    try:
        run_cfg("bf16", 4, 128, reps=1, check=False)
        v("smem-ceiling", False, "4 stages x 32 KiB launched")
    except Exception as e:  # noqa: BLE001
        t = err_text(e)
        n(f"bf16 128 B rows x 4 stages (128 KiB): {t}")
        v("smem-ceiling", "cudaErrorInvalidValue" in t, t)


def stage_sass():
    s = int(os.environ["FK_STAGES"])
    run_cfg("fp8", s, 64, reps=1, check=False)
    cubin = sorted(glob.glob(os.path.join(os.environ["CUTE_DSL_DUMP_DIR"], "*.cubin")), key=os.path.getmtime)[-1]
    sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
    waits = re.findall(r"DEPBAR\.LE SB0, (0x[0-9a-f]+)", sass)
    commits = len(re.findall(r"\bLDGDEPBAR\b", sass))
    n_instr = len(re.findall(r"^\s+/\*[0-9a-f]{4}\*/", sass, re.M))
    n(f"stages={s}: DEPBAR.LE SB0 immediates {waits}, LDGDEPBAR x{commits}, {n_instr} instructions")
    v(f"sass-wait-depth-s{s}", waits == [hex(s - 2)] and commits == s, f"waits={waits} LDGDEPBAR={commits}")
    from forgekernel.skills import orch_tools as OT
    d = OT.summarize(sass)
    got = None if d is None else (d["loops"], d["commit"])
    n(f"stages={s}: dossier loops/commit = {got}")
    v(f"orch-commit-s{s}", d is not None and d["loops"] == {"mma-consumer": 1}
      and d["commit"] == [f"mma-consumer: DEPBAR {{}} LDGDEPBAR x1 CPWAIT {{LE {hex(s - 2)} x1}}"], str(got))


def stage_target():
    run_cfg(os.environ["FK_DT"], int(os.environ["FK_STAGES"]), int(os.environ["FK_ROW"]), reps=1, check=False,
            tile=tuple(int(x) for x in os.environ.get("FK_TILE", "128,128").split(",")),
            warps=tuple(int(x) for x in os.environ.get("FK_WARPS", "2,2").split(",")),
            min_blocks=int(os.environ.get("FK_MINBLK", "0")))


SHAPES = [((128, 128), (2, 2), 0), ((128, 64), (2, 2), 0), ((128, 128), (2, 4), 0), ((128, 128), (2, 2), 3)]


def stage_shapes():
    """e4m3, 64 B rows, 2 stages: warp-tile / warp-count / register-cap variants (numerics + timing facts)."""
    for tile, warps, mb in SHAPES:
        md, ms, tf = run_cfg("fp8", 2, 64, tile=tile, warps=warps, min_blocks=mb)
        n(f"fp8 rows=64B stages=2 tile={tile} warps={warps} min_blocks={mb}: {ms:.3f} ms, {tf:.1f} TFLOPS")
        v(f"numerics-shape-{tile[0]}x{tile[1]}-w{warps[0]}x{warps[1]}-mb{mb}", md == 0.0, f"max_diff={md}")


def _ncu(env_extra, mets):
    env = dict(os.environ, CUTE_DSL_ARCH="sm_120a", CUTE_DSL_CACHE_DIR=tempfile.mkdtemp(prefix="fk_pipe_ncu_"),
               **env_extra)
    p = subprocess.run(["ncu", "--csv", "--kernel-name", "regex:PipeGemm", "--launch-skip", "1", "--launch-count", "1",
                        "--metrics", ",".join(mets), sys.executable, "-u", HERE, "target"],
                       env=env, capture_output=True, text=True, timeout=900)
    vals = {}
    for line in p.stdout.splitlines():
        m = re.match(r'"[^"]*",.*"([a-z0-9_]+__[a-z0-9_.]+)","[^"]*","([^"]*)"$', line)
        if m:
            vals[m.group(1)] = float(m.group(2).replace(",", ""))
    return vals


OCC = ["launch__occupancy_limit_shared_mem", "launch__occupancy_limit_registers", "launch__registers_per_thread",
       "sm__warps_active.avg.per_cycle_active", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
       "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
       "smsp__inst_executed.sum"]


def ncu_occupancy():
    out = []
    got = {}
    for row in (128, 64):
        got[row] = _ncu({"FK_DT": "fp8", "FK_STAGES": "3", "FK_ROW": str(row)}, OCC)
        print(f"FACT ncu fp8 rows={row}B stages=3: {json.dumps(got[row])}", flush=True)
    try:
        ok = (got[128]["launch__occupancy_limit_shared_mem"] == 1 and got[64]["launch__occupancy_limit_shared_mem"] == 2
              and got[64]["sm__warps_active.avg.per_cycle_active"] > 1.8 * got[128]["sm__warps_active.avg.per_cycle_active"])
        md = (f"smem limit {got[128]['launch__occupancy_limit_shared_mem']:.0f} -> "
              f"{got[64]['launch__occupancy_limit_shared_mem']:.0f} CTA/SM, active warps "
              f"{got[128]['sm__warps_active.avg.per_cycle_active']:.2f} -> {got[64]['sm__warps_active.avg.per_cycle_active']:.2f}")
    except KeyError as e:
        ok, md = False, f"missing {e}"
    out.append(("ncu-occupancy", ok, md))
    # the register cap: 2 stages x 64 B rows fit 3 CTAs by smem; min_blocks_per_mp=3 forces <= 170 registers
    rc = {mb: _ncu({"FK_DT": "fp8", "FK_STAGES": "2", "FK_ROW": "64", "FK_MINBLK": str(mb)}, OCC) for mb in (0, 3)}
    for mb in (0, 3):
        print(f"FACT ncu fp8 rows=64B stages=2 min_blocks={mb}: {json.dumps(rc[mb])}", flush=True)
    try:
        a, b = rc[0], rc[3]
        ok = (a["launch__occupancy_limit_registers"] == 2 and a["launch__occupancy_limit_shared_mem"] == 3
              and a["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"] == 0
              and b["launch__registers_per_thread"] <= 170 and b["launch__occupancy_limit_registers"] >= 3
              and b["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"] > 0
              and b["smsp__inst_executed.sum"] > a["smsp__inst_executed.sum"])
        md = (f"registers {a['launch__registers_per_thread']:.0f} -> {b['launch__registers_per_thread']:.0f}, CTA/SM by "
              f"registers {a['launch__occupancy_limit_registers']:.0f} -> {b['launch__occupancy_limit_registers']:.0f}, "
              f"local ld sectors 0 -> {b['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum']:.0f}, instructions "
              f"x{b['smsp__inst_executed.sum'] / a['smsp__inst_executed.sum']:.2f}")
    except KeyError as e:
        ok, md = False, f"missing {e}"
    out.append(("ncu-register-cap-spills", ok, md))
    return out


STAGES = {"numerics": stage_numerics, "ceiling": stage_ceiling, "sass": stage_sass, "target": stage_target,
          "shapes": stage_shapes}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
        sys.exit(0)
    root = tempfile.mkdtemp(prefix="fk_pipe_")
    plan = [("numerics", [f"numerics-{dt}-r{row}-s{s}" for dt, row, s in CFGS], None),
            ("ceiling", ["smem-ceiling"], None),
            ("shapes", [f"numerics-shape-{t[0]}x{t[1]}-w{w[0]}x{w[1]}-mb{mb}" for t, w, mb in SHAPES], None)]
    for s in (2, 3):
        d = os.path.join(root, f"s{s}")
        os.makedirs(d)
        plan.append(("sass", [f"sass-wait-depth-s{s}", f"orch-commit-s{s}"],
                     {"FK_STAGES": str(s), "CUTE_DSL_KEEP": "cubin", "CUTE_DSL_DUMP_DIR": d, "CUTE_DSL_NO_CACHE": "1"}))
    rc = run(HERE, plan, "cpasync-multistage-core", cap_s=900)
    occ = ncu_occupancy()
    for tag, ok, md in occ:
        print(f"PROBE {'PASS' if ok else 'FAIL'} {tag} ({md})")
    n_core = len(CFGS) + 1 + len(SHAPES) + 4
    all_ok = rc == 0 and all(ok for _, ok, _ in occ)
    print(f"PROBE {'PASS' if all_ok else 'FAIL'} cpasync-multistage contracts={n_core + len(occ)}")
    sys.exit(0 if all_ok else 1)
