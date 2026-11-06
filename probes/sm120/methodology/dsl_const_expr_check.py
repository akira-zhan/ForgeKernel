#!/usr/bin/env python3
"""Re-earn ``cute_dsl_const_expr_if`` on CuTe DSL 4.8.0 / sm_120a.

    python3 -u dsl_const_expr_check.py

The kernel ``k_ce`` is the entry's minimal example verbatim; ``_GEAR`` is a module global read at
trace time, set per process from ``FK_GEAR`` (each stage has its own process and its own
``CUTE_DSL_CACHE_DIR``). ``k_plain_b`` / ``k_plain_c`` are the same kernel with the
``cutlass.const_expr(`` wrapper removed at site (b) / (c) only.

Contracts:
- ``C1-gear0``  : gear 0 output bit-equal to the host formula (6x - 3, halved for tidx < 32) and the folded
  branch's auxiliary-plane write never landed (aux all 0);
- ``C2-gear1``  : gear 1 output bit-equal (23x, halved for tidx < 32), aux all 1, and the two gears' references differ;
- ``C3-plain-b-nameerror``: site (b) without the wrapper fails at trace time naming ``w``;
- ``C6-plain-c-nameerror``: site (c) (kernel top level) without the wrapper fails at trace time naming ``b``;
- ``G8-dynamic-index``    : a Python tuple subscripted by a ``cutlass.range`` induction variable fails at trace time;
  the same loop as ``range_constexpr`` runs correctly inside a Runtime-trip ``cutlass.range`` body.
Facts: ``G4`` -- changing the module global in the same process and compiling the same host object again;
two processes (gear 0, then gear 1) sharing one ``CUTE_DSL_CACHE_DIR``.
"""
import os
import sys

HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(HERE))
from _ladder import err_text, n, run, v  # noqa: E402

_GEAR = int(os.environ.get("FK_GEAR", "1"))
_SHARED = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"fk_const_expr_shared_{os.getpid()}")


def _kernels():
    import cutlass
    import cutlass.cute as cute

    @cute.kernel
    def k_ce(mIn: cute.Tensor, mOut: cute.Tensor, mAux: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        x = mIn[i]
        for u in cutlass.range_constexpr(2):
            z = x * cutlass.Float32(float(u))
        y = z
        for t in cutlass.range(4):
            if cutlass.const_expr(_GEAR >= 1):
                w = x * cutlass.Float32(4.0)
                mAux[i] = cutlass.Float32(1.0)
            mOut[i] = mOut[i] + x
            if cutlass.const_expr(_GEAR >= 1):
                mOut[i] = mOut[i] + w
        if cutlass.const_expr(_GEAR >= 1):
            b = x * cutlass.Float32(2.0)
        else:
            b = x - cutlass.Float32(3.0)
        mOut[i] = mOut[i] + b + y
        if tidx < 32:
            mOut[i] = mOut[i] * cutlass.Float32(0.5)

    @cute.kernel
    def k_plain_b(mIn: cute.Tensor, mOut: cute.Tensor, mAux: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        x = mIn[i]
        for t in cutlass.range(4):
            if _GEAR >= 1:
                w = x * cutlass.Float32(4.0)
                mAux[i] = cutlass.Float32(1.0)
            mOut[i] = mOut[i] + x
            if _GEAR >= 1:
                mOut[i] = mOut[i] + w

    @cute.kernel
    def k_plain_c(mIn: cute.Tensor, mOut: cute.Tensor, mAux: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 128 + tidx
        x = mIn[i]
        if _GEAR >= 1:
            b = x * cutlass.Float32(2.0)
        else:
            b = x - cutlass.Float32(3.0)
        mOut[i] = mOut[i] + b

    return cute, {"ce": k_ce, "plain_b": k_plain_b, "plain_c": k_plain_c}


def _run(which):
    import torch
    from cutlass.cute.runtime import from_dlpack
    cute, ks = _kernels()
    k = ks[which]

    @cute.jit
    def host(a: cute.Tensor, o: cute.Tensor, x: cute.Tensor):
        k(a, o, x).launch(grid=[2, 1, 1], block=[128, 1, 1])

    xin = (torch.arange(256, device="cuda", dtype=torch.float32) % 17) - 8.0   # small ints: exact in fp32
    out = torch.zeros(256, device="cuda")
    aux = torch.zeros(256, device="cuda")
    ts = [from_dlpack(t) for t in (xin, out, aux)]
    cute.compile(host, *ts)(*ts)
    torch.cuda.synchronize()
    return xin, out, aux


def _ref(xin, gear):
    import torch
    r = 23.0 * xin if gear >= 1 else 6.0 * xin - 3.0
    half = (torch.arange(256, device=xin.device) % 128) < 32
    return torch.where(half, r * 0.5, r)


def stage_gear():
    xin, out, aux = _run("ce")
    ref, other = _ref(xin, _GEAR), _ref(xin, 1 - _GEAR)
    dev = float((out - ref).abs().max())
    aux_exp = 1.0 if _GEAR >= 1 else 0.0
    aux_dev = float((aux - aux_exp).abs().max())
    sep = float((ref - other).abs().max())
    tag = "C2-gear1" if _GEAR >= 1 else "C1-gear0"
    v(tag, max(dev, aux_dev) == 0.0 and sep > 0, f"max_diff={max(dev, aux_dev)} ref_sep={sep}")


def _expect_nameerror(tag, which, name):
    try:
        _run(which)
        v(tag, False, "compiled and ran")
    except Exception as e:  # noqa: BLE001
        t = err_text(e)
        n(f"{which}: {t}")
        v(tag, f"'{name}'" in t or f"`{name}`" in t or f" {name} " in t, t)


def stage_plain_b():
    _expect_nameerror("C3-plain-b-nameerror", "plain_b", "w")


def stage_plain_c():
    _expect_nameerror("C6-plain-c-nameerror", "plain_c", "b")


def stage_g8():
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    RUNS = (0, 2, 1, 3)                              # a static run table (Python tuple)

    def make(constexpr_inner):
        @cute.kernel
        def k(g: cute.Tensor, nrun: cutlass.Int32):
            tidx, _, _ = cute.arch.thread_idx()
            acc = cutlass.Float32(0.0)
            for r in cutlass.range(nrun):            # Runtime trip count
                s = [None] * 4                       # Python staging list built inside the body
                if cutlass.const_expr(constexpr_inner):
                    for q in cutlass.range_constexpr(4):
                        s[RUNS[q]] = cutlass.Float32(float(q + 1))
                else:
                    for q in cutlass.range(4):
                        s[RUNS[q]] = cutlass.Float32(1.0)
                acc = acc + s[0] + s[3]
            g[tidx] = acc

        @cute.jit
        def host(g: cute.Tensor, nrun: cutlass.Int32):
            k(g, nrun).launch(grid=[1, 1, 1], block=[32, 1, 1])
        return host

    a = torch.zeros(32, device="cuda")
    t = from_dlpack(a)
    try:
        cute.compile(make(False), t, cutlass.Int32(2))(t, cutlass.Int32(2))
        dyn_err = None
    except Exception as e:  # noqa: BLE001
        dyn_err = err_text(e)
    n(f"tuple subscripted by a cutlass.range induction variable: {dyn_err}")
    cute.compile(make(True), t, cutlass.Int32(2))(t, cutlass.Int32(2))
    torch.cuda.synchronize()
    # s[RUNS[q]] = q + 1 -> s[0] = 1 (q=0), s[3] = 4 (q=3); two trips -> 10
    val = float(a[0])
    n(f"range_constexpr inside the Runtime-trip body: value={val}")
    v("G8-dynamic-index", dyn_err is not None and val == 10.0, f"{dyn_err} | constexpr value={val}")


def stage_g4():
    """Same process, SAME host object: compile at gear 0, flip the module global, compile it again."""
    global _GEAR
    import torch
    from cutlass.cute.runtime import from_dlpack
    cute, ks = _kernels()
    k = ks["ce"]

    @cute.jit
    def host(a: cute.Tensor, o: cute.Tensor, x: cute.Tensor):
        k(a, o, x).launch(grid=[2, 1, 1], block=[128, 1, 1])

    xin = (torch.arange(256, device="cuda", dtype=torch.float32) % 17) - 8.0
    res = {}
    for gear in (0, 1):
        _GEAR = gear
        out, aux = torch.zeros(256, device="cuda"), torch.zeros(256, device="cuda")
        ts = [from_dlpack(t) for t in (xin, out, aux)]
        cute.compile(host, *ts)(*ts)
        torch.cuda.synchronize()
        res[gear] = {g: float((out - _ref(xin, g)).abs().max()) == 0.0 for g in (0, 1)}
    n(f"G4 same host object: compile at gear 0 -> matches gear0={res[0][0]}; global flipped to 1 and compiled "
      f"again -> matches gear1={res[1][1]} gear0={res[1][0]}")


def stage_gear_shared():
    """Fact only: one stage of a pair that shares one CUTE_DSL_CACHE_DIR (gear 0 first, then gear 1)."""
    xin, out, _ = _run("ce")
    n(f"shared on-disk cache, process with gear {_GEAR}: matches gear {_GEAR}="
      f"{float((out - _ref(xin, _GEAR)).abs().max()) == 0.0}, matches gear {1 - _GEAR}="
      f"{float((out - _ref(xin, 1 - _GEAR)).abs().max()) == 0.0}")


STAGES = {k[len("stage_"):]: f for k, f in dict(globals()).items() if k.startswith("stage_")}
PLAN = [
    ("gear", ["C1-gear0"], {"FK_GEAR": "0"}),
    ("gear", ["C2-gear1"], {"FK_GEAR": "1"}),
    ("plain_b", ["C3-plain-b-nameerror"], None),
    ("plain_c", ["C6-plain-c-nameerror"], None),
    ("g8", ["G8-dynamic-index"], None),
    ("g4", [], None),
    ("gear_shared", [], {"FK_GEAR": "0", "CUTE_DSL_CACHE_DIR": _SHARED}),
    ("gear_shared", [], {"FK_GEAR": "1", "CUTE_DSL_CACHE_DIR": _SHARED}),
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
    else:
        sys.exit(run(HERE, PLAN, "dsl-const-expr"))
