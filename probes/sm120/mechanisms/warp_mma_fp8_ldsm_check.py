#!/usr/bin/env python3
"""FP8 ``mma.sync`` (QMMA.16832) fed by ``ldmatrix`` from K-major shared memory, on sm_120a (CuTe DSL).

    python3 -u warp_mma_fp8_ldsm_check.py          (needs cuobjdump; ncu with counter access for the N-* contracts)

One CTA, 4 warps (2 x 2), computes C[128 x 128] (fp32) = A[128 x 256] . B[128 x 256]^T with A, B e4m3 K-major:
cp.async 16 B gmem -> smem, ``LdMatrix8x8x16bOp(num_matrices=4)`` smem -> registers, ``MmaFP8Op(e4m3, f32, (16,8,32))``
via ``make_tiled_mma(..., permutation_mnk=(32, 32, 32))``. Two smem layouts: ``swz`` (``make_smem_layout_a/b``) and
``lin`` (plain K-major rows of 256 B), and ``linfill`` (``lin`` with a one-row-per-thread cp.async fill). Integer-valued fp8 inputs make every partial sum exact in fp32.

Contracts (per variant unless noted):
- ``numerics-<v>``     : C == A32 @ B32^T exactly;
- ``census-qmma-<v>``  : executed-form SASS has exactly 256 ``QMMA.16832.F32.E4M3.E4M3`` and no other MMA family;
- ``census-ldsm-<v>``  : exactly 64 ``LDSM.16.M88.4`` and 0 scalar shared loads (``LDS``/``LDS.U8``/``LDS.U16``)
  once never-executed ``@!PT`` instructions are dropped;
- ``census-ptx-sass-<v>``: PTX counts equal SASS counts per family (mma.sync == QMMA, ldmatrix == LDSM,
  cp.async.cg == LDGSTS);
- ``N-swz-clean``      : (ncu) the swizzled tile reads with bank conflicts < 1 % of its load wavefronts;
- ``N-lin-conflict``   : (ncu) the linear tile lands in the conflict quadrant of ``smem_traffic_triage`` against ``swz``;
- ``N-ldgsts-blind-hw-counters``: (ncu) the cp.async smem fill shows 0 in ``l1tex__..._op_st`` and in
  ``l1tex__..._op_ldgsts``; only the SASS-patched ``sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts``
  sees it (512 = 64 KiB / 128 B for the conflict-free fills);
- ``N-linfill-fill-wavefronts``: (ncu) ``linfill`` (same smem layout as ``lin``, one row per thread in the fill) raises
  the SASS-patched fill wavefronts at the same LDGSTS count, the ld side unchanged;
- ``N-ldgsts-conflict-counter-blind``: (ncu) that fill-side excess shows 0 in both ldgsts bank-conflict counters
  (``sm__sass_..._bank_conflicts_..._op_ldgsts`` and ``l1tex__..._op_ldgsts``): a fill conflict is visible only as
  wavefronts per LDGSTS.
Facts: SASS histogram, the ``@!PT`` filler count, NCU counters.
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

M, N, K = 128, 128, 256
EXPECT = {"QMMA": 256, "LDSM": 64, "LDGSTS": 32}


def build():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.hopper_helpers as hh
    from cutlass.memory import SmemAllocator
    from cutlass.tensor_utils import LayoutEnum
    F8 = cutlass.Float8E4M3FN

    class Fp8Tile:
        def __init__(self, swizzle, row_fill=False):
            self.swizzle, self.row_fill = swizzle, row_fill

        @cute.jit
        def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
            mma_op = cute.nvgpu.warp.MmaFP8Op(F8, cutlass.Float32, (16, 8, 32))
            tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((2, 2, 1)), permutation_mnk=(32, 32, 32))
            if cutlass.const_expr(self.swizzle):
                sA_l = hh.make_smem_layout_a(LayoutEnum.from_tensor(mA), (M, N, K), F8, 1)
                sB_l = hh.make_smem_layout_b(LayoutEnum.from_tensor(mB), (M, N, K), F8, 1)
            else:
                sA_l = cute.make_layout((M, K, 1), stride=(K, 1, M * K))
                sB_l = cute.make_layout((N, K, 1), stride=(K, 1, N * K))
            self.kernel(mA, mB, mC, tiled_mma, sA_l, sB_l).launch(grid=[1, 1, 1], block=[128, 1, 1])

        @cute.kernel
        def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, tiled_mma: cute.TiledMma,
                   sA_l: cute.ComposedLayout | cute.Layout, sB_l: cute.ComposedLayout | cute.Layout):
            tidx, _, _ = cute.arch.thread_idx()
            smem = SmemAllocator()
            sA = smem.allocate_tensor(F8, sA_l, byte_alignment=1024)[None, None, 0]
            sB = smem.allocate_tensor(F8, sB_l, byte_alignment=1024)[None, None, 0]
            g2s = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL), F8,
                num_bits_per_copy=128)
            if cutlass.const_expr(self.row_fill):
                # one row per thread: the 32 lanes of a warp write 16 B each, 256 B apart -> same banks
                tcopy = cute.make_tiled_copy_tv(g2s, cute.make_layout((128, 1), stride=(1, 0)), cute.make_layout((1, 16)))
            else:
                tcopy = cute.make_tiled_copy_tv(g2s, cute.make_layout((8, 16), stride=(16, 1)), cute.make_layout((1, 16)))
            thr = tcopy.get_slice(tidx)
            cute.copy(tcopy, thr.partition_S(mA), thr.partition_D(sA))
            cute.copy(tcopy, thr.partition_S(mB), thr.partition_D(sB))
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((M, N)), cutlass.Float32)
            acc.fill(0.0)
            ldsm = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), F8)
            cpA = cute.make_tiled_copy_A(ldsm, tiled_mma)
            cpB = cute.make_tiled_copy_B(ldsm, tiled_mma)
            tAsA = cpA.get_slice(tidx).partition_S(sA)
            tBsB = cpB.get_slice(tidx).partition_S(sB)
            tArA = cpA.retile(tCrA)
            tBrB = cpB.retile(tCrB)
            for kb in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
                cute.copy(cpA, tAsA[None, None, kb], tArA[None, None, kb])
                cute.copy(cpB, tBsB[None, None, kb], tBrB[None, None, kb])
                cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrB[None, None, kb], acc)
            cute.autovec_copy(acc, thr_mma.partition_C(mC))

    return Fp8Tile


def _run(variant):
    import torch
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    Fp8Tile = build()
    torch.manual_seed(0)
    a32 = torch.randint(-2, 3, (M, K), device="cuda").float()
    b32 = torch.randint(-2, 3, (N, K), device="cuda").float()
    c = torch.zeros(M, N, device="cuda")
    ts = [from_dlpack(t, assumed_align=16) for t in (a32.to(torch.float8_e4m3fn), b32.to(torch.float8_e4m3fn), c)]
    f = cute.compile(Fp8Tile(variant == "swz", row_fill=(variant == "linfill")), *ts)
    f(*ts)
    torch.cuda.synchronize()
    return float((c - a32 @ b32.T).abs().max())


def sass_census(sass):
    """{opcode-with-qualifiers: count} over executed-form instructions, and the @!PT filler count."""
    hist, never = {}, 0
    for line in sass.splitlines():
        m = re.match(r"\s+/\*[0-9a-f]{4}\*/\s+(@!?U?P[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if not m:
            continue
        if m.group(1) and m.group(1).strip() == "@!PT":   # predicate constant-false: never executes
            never += 1
            continue
        hist[m.group(2)] = hist.get(m.group(2), 0) + 1
    return hist, never


def fam(hist, family):
    return {k: c for k, c in hist.items() if k.split(".", 1)[0] == family}


def stage_run():
    variant = os.environ["FK_VARIANT"]
    dump = os.environ["CUTE_DSL_DUMP_DIR"]
    md = _run(variant)
    v(f"numerics-{variant}", md == 0.0, f"max_diff={md}")
    cubin = sorted(glob.glob(os.path.join(dump, "*.cubin")), key=os.path.getmtime)[-1]
    ptx = open(sorted(glob.glob(os.path.join(dump, "*.ptx")), key=os.path.getmtime)[-1]).read()
    sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
    hist, never = sass_census(sass)
    top = sorted(hist.items(), key=lambda kv: -kv[1])[:10]
    n(f"{variant}: SASS histogram (executed forms) {top}; never-executed @!PT instructions={never}")
    mma = {k: c for k, c in hist.items() if re.match(r"(HMMA|QMMA|OMMA|IMMA|HGMMA)", k)}
    v(f"census-qmma-{variant}", mma == {"QMMA.16832.F32.E4M3.E4M3": EXPECT["QMMA"]}, str(mma))
    ldsm, lds = fam(hist, "LDSM"), fam(hist, "LDS")
    v(f"census-ldsm-{variant}", ldsm == {"LDSM.16.M88.4": EXPECT["LDSM"]} and not lds, f"LDSM={ldsm} scalar LDS={lds}")
    p = {"mma": len(re.findall(r"mma\.sync\.aligned\.m16n8k32\.row\.col\.f32\.e4m3\.e4m3\.f32", ptx)),
         "ldmatrix": len(re.findall(r"ldmatrix\.sync\.aligned\.m8n8\.x4\.shared\.b16", ptx)),
         "cp.async": len(re.findall(r"cp\.async\.cg\.shared\.global", ptx))}
    s = {"mma": sum(mma.values()), "ldmatrix": sum(ldsm.values()), "cp.async": sum(fam(hist, "LDGSTS").values())}
    n(f"{variant}: PTX {p} vs SASS {s}")
    v(f"census-ptx-sass-{variant}", p == s and s["cp.async"] == EXPECT["LDGSTS"], f"ptx={p} sass={s}")


def stage_target():
    _run(os.environ["FK_VARIANT"])


def ncu_contracts():
    """Parent side: profile the two variants and read the counters through ForgeKernel's parser."""
    from forgekernel.inner.ncu import parse_ncu_csv, profiler_from_ncu
    mets = ["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum", "gpu__time_active.sum",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum",
            "sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum",
            "sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum",
            "sm__sass_inst_executed_op_ldgsts.sum"]
    m = {}
    for variant in ("swz", "lin", "linfill"):
        env = dict(os.environ, FK_VARIANT=variant, CUTE_DSL_ARCH="sm_120a",
                   CUTE_DSL_CACHE_DIR=tempfile.mkdtemp(prefix=f"fk_ncu_{variant}_"))
        p = subprocess.run(["ncu", "--csv", "--kernel-name", "regex:Fp8Tile", "--metrics", ",".join(mets),
                            sys.executable, "-u", HERE, "target"], env=env, capture_output=True, text=True, timeout=600)
        m[variant] = profiler_from_ncu(p.stdout) if "Fp8Tile" in p.stdout else {}
        raw = parse_ncu_csv(p.stdout)
        ent = next((e for k, e in raw.items() if "Fp8Tile" in k), {})
        for key, met in (("ldgsts_wavefronts", mets[5]), ("ldgsts_bankconf", mets[6]), ("sass_ldgsts_wavefronts", mets[7]),
                         ("sass_ldgsts_bankconf", mets[8]), ("ldgsts_inst", mets[9])):
            val = ent.get(met)
            m[variant][key] = float(val.replace(",", "")) if val not in (None, "", "n/a") else None
        print(f"FACT ncu {variant}: " + " ".join(f"{k}={m[variant].get(k)}" for k in
              ("smem_wavefronts_ld", "bankconf_ld", "smem_wavefronts_st", "bankconf_st", "ldgsts_wavefronts",
               "ldgsts_bankconf", "sass_ldgsts_wavefronts", "sass_ldgsts_bankconf", "ldgsts_inst", "time_ns")), flush=True)
    ok = []
    s, l = m["swz"], m["lin"]
    try:
        clean = s["bankconf_ld"] < 0.01 * s["smem_wavefronts_ld"]
        wf = l["smem_wavefronts_ld"] / s["smem_wavefronts_ld"]
        bc = l["bankconf_ld"] / max(s["bankconf_ld"], 1)
        ok.append(("N-swz-clean", clean, f"bankconf_ld={s['bankconf_ld']} wavefronts_ld={s['smem_wavefronts_ld']}"))
        ok.append(("N-lin-conflict", wf >= 1.3 and bc >= 3.0, f"ld wavefronts {wf:.2f}x, bankconf {bc:.1f}x vs swz"))
        # the cp.async fill (2 x 32 KiB per CTA = 512 x 128 B wavefronts when conflict-free)
        ok.append(("N-ldgsts-blind-hw-counters",
                   all(m[x]["smem_wavefronts_st"] == 0 and not m[x]["ldgsts_wavefronts"] for x in m)
                   and all(m[x]["sass_ldgsts_wavefronts"] == 512 for x in ("swz", "lin")),
                   " ".join(f"{x}: op_st={m[x]['smem_wavefronts_st']} l1tex_ldgsts={m[x]['ldgsts_wavefronts']} "
                            f"sass_ldgsts={m[x]['sass_ldgsts_wavefronts']}" for x in m)))
        f, base = m["linfill"], m["lin"]
        excess = f["sass_ldgsts_wavefronts"] - base["sass_ldgsts_wavefronts"]
        ok.append(("N-linfill-fill-wavefronts",
                   f["sass_ldgsts_wavefronts"] >= 1.3 * base["sass_ldgsts_wavefronts"] and excess > 0
                   and f["ldgsts_inst"] == base["ldgsts_inst"] and f["smem_wavefronts_ld"] == base["smem_wavefronts_ld"],
                   f"fill wavefronts {f['sass_ldgsts_wavefronts']} vs {base['sass_ldgsts_wavefronts']} at the same LDGSTS "
                   f"count {f['ldgsts_inst']}; ld side unchanged {f['smem_wavefronts_ld']}"))
        ok.append(("N-ldgsts-conflict-counter-blind",
                   excess > 0 and f["sass_ldgsts_bankconf"] == 0 and not f["ldgsts_bankconf"],
                   f"fill wavefront excess {excess} with sass_ldgsts_bankconf={f['sass_ldgsts_bankconf']} "
                   f"l1tex_ldgsts_bankconf={f['ldgsts_bankconf']}"))
    except (KeyError, TypeError, ZeroDivisionError) as e:
        ok += [(t, False, f"no counters ({e})") for t in ("N-swz-clean", "N-lin-conflict", "N-ldgsts-blind-hw-counters",
                                                          "N-linfill-fill-wavefronts", "N-ldgsts-conflict-counter-blind")]
    return ok


STAGES = {"run": stage_run, "target": stage_target}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
        sys.exit(0)
    root = tempfile.mkdtemp(prefix="fk_fp8_ldsm_")
    plan = []
    for variant in ("swz", "lin", "linfill"):
        d = os.path.join(root, variant)
        os.makedirs(d)
        plan.append(("run", [f"numerics-{variant}", f"census-qmma-{variant}", f"census-ldsm-{variant}",
                             f"census-ptx-sass-{variant}"],
                     {"FK_VARIANT": variant, "CUTE_DSL_KEEP": "cubin,ptx", "CUTE_DSL_DUMP_DIR": d,
                      "CUTE_DSL_NO_CACHE": "1"}))
    rc = run(HERE, plan, "warp-mma-fp8-ldsm-census")
    nc = ncu_contracts()
    for tag, ok, md in nc:
        print(f"PROBE {'PASS' if ok else 'FAIL'} {tag} ({md})")
    all_ok = rc == 0 and all(ok for _, ok, _ in nc)
    print(f"PROBE {'PASS' if all_ok else 'FAIL'} warp-mma-fp8-ldsm contracts={12 + len(nc)}")
    sys.exit(0 if all_ok else 1)
