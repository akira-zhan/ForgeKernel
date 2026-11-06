#!/usr/bin/env python3
"""Re-earn ``ptx_escape`` (``llvm.inline_asm`` inside ``@dsl_user_op``) on CuTe DSL 4.8.0 / sm_120a.

    python3 -u ptx_escape_check.py      (needs cuobjdump)

One kernel uses the entry's three forms together: form 2 (``cp.async.cg.shared.global.L2::256B``, address
operands) copies 16 B per thread gmem -> smem, form 1 (``cp.async.commit_group`` / ``wait_group``, no
operands) completes it, the smem tile is copied back out and checked; form 3 (``cvt.rn.f16x2.f32``,
a return value) packs two fp32 per thread. The cubin is kept (``CUTE_DSL_KEEP=cubin`` into a dump directory
that exists, with ``CUTE_DSL_NO_CACHE=1``) and its SASS is counted.

Contracts:
- ``PE-A-cache-mode-knob``  : rung A -- ``CopyG2SOp`` has a ``cache_mode`` parameter (signature + enum printed as facts);
- ``PE-form2-numerics``     : the smem round trip of the inline cp.async is bit-exact;
- ``PE-form2-footprint``    : SASS carries ``LDGSTS`` with ``BYPASS`` and ``LTC256B`` (the two modifiers are separate bits);
- ``PE-form1-footprint``    : SASS carries ``LDGDEPBAR`` (commit) and a ``DEPBAR`` wait;
- ``PE-form3-numerics``     : ``cvt.rn.f16x2.f32 $0, $2, $1`` with operands (a, b) puts f16(a) in the low half, bit-exact;
- ``PE-G2-percent-placeholder``: writing ``%0`` instead of ``$0`` fails to compile;
- ``PE-G4-no-side-effects-dce``: the no-output commit asm without ``has_side_effects=True`` leaves no ``LDGDEPBAR``;
- ``PE-dump-needs-existing-dir``: with a dump directory that does not exist, no cubin is written and nothing fails.
"""
import glob
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(HERE))
from _ladder import err_text, n, run, v  # noqa: E402

DUMP = os.environ.get("FK_DUMP", "")


def _ops(side_effects=True, placeholder="$"):
    import cutlass
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    @dsl_user_op
    def cp_async_commit(*, loc=None, ip=None):
        llvm.inline_asm(None, [], "cp.async.commit_group;", "",
                        has_side_effects=side_effects, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    @dsl_user_op
    def cp_async_wait_group(nn, *, loc=None, ip=None):
        llvm.inline_asm(None, [], "cp.async.wait_group %d;" % int(nn), "",
                        has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    @dsl_user_op
    def cp_async_16_l2_256(dst_smem_addr, src_gmem_addr, *, loc=None, ip=None):
        p = placeholder
        llvm.inline_asm(None, [dst_smem_addr.ir_value(loc=loc, ip=ip), src_gmem_addr.ir_value(loc=loc, ip=ip)],
                        f"cp.async.cg.shared.global.L2::256B [{p}0], [{p}1], 16;", "r,l",
                        has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

    @dsl_user_op
    def cvt_f16x2_f32(a, b, *, loc=None, ip=None) -> cutlass.Int32:
        return cutlass.Int32(llvm.inline_asm(
            T.i32(), [cutlass.Float32(a).ir_value(loc=loc, ip=ip), cutlass.Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $2, $1;", "=r,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))

    return cp_async_commit, cp_async_wait_group, cp_async_16_l2_256, cvt_f16x2_f32


def _build_and_run(side_effects=True, placeholder="$"):
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.memory import SmemAllocator
    from cutlass.cute.runtime import from_dlpack
    commit, wait_group, cp16, cvt = _ops(side_effects, placeholder)

    @cute.kernel
    def k(gs: cute.Tensor, gd: cute.Tensor, gi: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        smem = SmemAllocator()
        s = smem.allocate_tensor(cutlass.Float32, cute.make_layout(512), byte_alignment=128)
        cp16((s.iterator + tidx * 4).toint(), (gs.iterator + tidx * 4).toint())   # form 2
        commit()                                                                     # form 1
        wait_group(0)
        cute.arch.sync_threads()
        for j in cutlass.range_constexpr(4):
            gd[tidx * 4 + j] = s[(tidx * 4 + j + 4) % 512]                           # a neighbour's slice
        gi[tidx] = cvt(gs[tidx * 4], gs[tidx * 4 + 1])                               # form 3

    @cute.jit
    def host(gs: cute.Tensor, gd: cute.Tensor, gi: cute.Tensor):
        k(gs, gd, gi).launch(grid=[1, 1, 1], block=[128, 1, 1])

    src = torch.randn(512, device="cuda")
    dst = torch.zeros(512, device="cuda")
    gi = torch.zeros(128, device="cuda", dtype=torch.int32)
    ts = [from_dlpack(t) for t in (src, dst, gi)]
    f = cute.compile(host, *ts)
    f(*ts)
    torch.cuda.synchronize()
    return src, dst, gi


def _sass():
    cubins = sorted(glob.glob(os.path.join(DUMP, "*.cubin")), key=os.path.getmtime)
    if not cubins:
        return None
    return subprocess.run(["cuobjdump", "-sass", cubins[-1]], capture_output=True, text=True).stdout


def stage_rung_a():
    import inspect
    from cutlass.cute.nvgpu import cpasync
    sig = str(inspect.signature(cpasync.CopyG2SOp.__init__))
    members = [m for m in dir(cpasync.LoadCacheMode) if not m.startswith("_")]
    n(f"CopyG2SOp.__init__{sig}")
    n(f"LoadCacheMode members: {members}")
    v("PE-A-cache-mode-knob", "cache_mode" in sig, sig)


def stage_forms():
    import torch
    src, dst, gi = _build_and_run()
    exp = torch.roll(src, -4)
    v("PE-form2-numerics", bool(torch.equal(dst, exp)), f"max_diff={float((dst - exp).abs().max())}")
    a = src.view(-1, 4)[:, 0].half().view(torch.int16).to(torch.int32) & 0xFFFF
    b = src.view(-1, 4)[:, 1].half().view(torch.int16).to(torch.int32) & 0xFFFF
    v("PE-form3-numerics", bool(torch.equal(gi & 0xFFFF, a)) and bool(torch.equal((gi >> 16) & 0xFFFF, b)),
      "low half = f16(a), high half = f16(b)")
    sass = _sass()
    if sass is None:
        v("PE-form2-footprint", False, "no cubin dumped"); v("PE-form1-footprint", False, "no cubin dumped")
        return
    ldgsts = re.findall(r"\bLDGSTS[A-Z0-9_.]*", sass)
    n(f"form 2/1 SASS: LDGSTS spellings {sorted(set(ldgsts))}; LDGDEPBAR={len(re.findall(r'LDGDEPBAR', sass))} "
      f"DEPBAR={len(re.findall(r'(?<!LDG)DEPBAR', sass))} F2FP={sorted(set(re.findall(r'F2FP[A-Z0-9_.]*', sass)))}")
    v("PE-form2-footprint", any("BYPASS" in x and "LTC256B" in x for x in ldgsts), str(sorted(set(ldgsts))))
    v("PE-form1-footprint", "LDGDEPBAR" in sass and bool(re.search(r"(?<!LDG)DEPBAR", sass)), "")


def stage_no_side_effects():
    _build_and_run(side_effects=False)
    sass = _sass()
    cnt = len(re.findall(r"LDGDEPBAR", sass or ""))
    n(f"commit asm without has_side_effects: LDGDEPBAR count={cnt} (cubin dumped={sass is not None})")
    v("PE-G4-no-side-effects-dce", sass is not None and cnt == 0, f"LDGDEPBAR={cnt}")


def stage_percent():
    try:
        _build_and_run(placeholder="%")
        v("PE-G2-percent-placeholder", False, "compiled and ran")
    except Exception as e:  # noqa: BLE001
        n(f"%0 placeholder: {err_text(e)}")
        v("PE-G2-percent-placeholder", True, err_text(e))


def stage_dump_missing():
    _build_and_run()
    written = glob.glob(os.path.join(DUMP, "*.cubin"))
    n(f"dump dir {DUMP!r} existed={os.path.isdir(DUMP)} after compile: cubins={len(written)}")
    v("PE-dump-needs-existing-dir", not written, f"cubins={len(written)}")


def _env(dump, make=True):
    if make:
        os.makedirs(dump, exist_ok=True)
    return {"CUTE_DSL_KEEP": "cubin", "CUTE_DSL_DUMP_DIR": dump, "FK_DUMP": dump, "CUTE_DSL_NO_CACHE": "1"}


STAGES = {k[len("stage_"):]: f for k, f in dict(globals()).items() if k.startswith("stage_")}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
    else:
        root = tempfile.mkdtemp(prefix="fk_ptx_escape_")
        PLAN = [
            ("rung_a", ["PE-A-cache-mode-knob"], None),
            ("forms", ["PE-form2-numerics", "PE-form3-numerics", "PE-form2-footprint", "PE-form1-footprint"],
             _env(os.path.join(root, "forms"))),
            ("no_side_effects", ["PE-G4-no-side-effects-dce"], _env(os.path.join(root, "nse"))),
            ("percent", ["PE-G2-percent-placeholder"], None),
            ("dump_missing", ["PE-dump-needs-existing-dir"], _env(os.path.join(root, "absent"), make=False)),
        ]
        sys.exit(run(HERE, PLAN, "ptx-escape"))
