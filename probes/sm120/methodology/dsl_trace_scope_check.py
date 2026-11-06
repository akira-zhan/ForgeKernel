#!/usr/bin/env python3
"""Re-earn ``cute_dsl_trace_scope`` on CuTe DSL 4.8.0 / sm_120a: one stage (process) per gotcha.

    python3 -u dsl_trace_scope_check.py

Contracts:
- ``TS1-undecorated-helper-rejects``  : a plain helper comparing a Runtime value in ``if`` fails at trace time;
- ``TS1-decorated-helper-ok``         : the same helper with ``@cute.jit`` (decorator syntax, even on a nested def)
  compiles, runs, and its writes into the caller's register fragment reach the caller (values checked per thread);
- ``TS1-call-form-not-rewritten``     : wrapping the plain helper as ``cute.jit(helper)`` (call, not decorator syntax)
  does NOT get its control flow rewritten: it fails exactly like the undecorated helper;
- ``TS2-closure-capture-rejected``    : a nested function closure-capturing Runtime values, called inside a staged
  ``if``, fails at trace time;
- ``TS2-explicit-args-ok``            : the same work with every Runtime value passed as an argument runs correctly;
- ``TS3-constexpr-shipped-ok``        : ``cute.compile(entry, t, 4)`` with ``n: cutlass.Constexpr`` runs correctly;
- ``TS3-transcribed-wrapper-differs`` : a wrapper that re-spells ``n`` without the annotation does not behave like
  the shipped entry (fails to compile, or bakes a different value) -- the transcription trap;
- ``TS4-make-rmem-tensor``            : ``cute.make_rmem_tensor`` exists (fact: whether ``cute.make_fragment`` does);
- ``TS5-range-constexpr-needs-decorator``: in an undecorated helper ``const_expr`` folds but ``range_constexpr`` fails;
  the same helper with a plain ``range`` runs;
- ``TS6-attr-subscript-store-rejected``: ``obj.face[i] = v`` inside a staged ``for`` fails at trace time;
- ``TS6-local-binding-ok``            : ``face = obj.face`` before the loop, then ``face[i] = v``, runs correctly.
"""
import os
import sys

HERE = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(HERE))
from _ladder import err_text, n, run, v  # noqa: E402


def _rt():
    import torch
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    return torch, cutlass, cute, from_dlpack


def _launch1(cute, kernel, threads=128):
    @cute.jit
    def host(g: cute.Tensor):
        kernel(g).launch(grid=[1, 1, 1], block=[threads, 1, 1])
    return host


# ------------------------------------------------------------------ TS1
def _ts1(form):
    torch, cutlass, cute, from_dlpack = _rt()

    def helper_body(frag, m, s):
        if m >= s:                                   # m is Runtime (from tidx)
            frag[0] = cutlass.Float32(1.0)
        else:
            frag[0] = cutlass.Float32(2.0)

    @cute.jit
    def helper_deco(frag, m, s):                     # same body, decorator syntax
        if m >= s:
            frag[0] = cutlass.Float32(1.0)
        else:
            frag[0] = cutlass.Float32(2.0)

    helper = {"plain": helper_body, "deco": helper_deco, "call": cute.jit(helper_body)}[form]

    @cute.kernel
    def k(g: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        frag = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
        frag[0] = cutlass.Float32(0.0)
        helper(frag, tidx, cutlass.Int32(64))
        g[tidx] = frag[0]

    a = torch.zeros(128, device="cuda")
    t = from_dlpack(a)
    cute.compile(_launch1(cute, k), t)(t)
    torch.cuda.synchronize()
    exp = torch.tensor([2.0] * 64 + [1.0] * 64, device="cuda")
    return float((a - exp).abs().max())


def stage_ts1_plain():
    try:
        md = _ts1("plain")
        v("TS1-undecorated-helper-rejects", False, f"compiled and ran, max_diff={md}")
    except Exception as e:  # noqa: BLE001
        n(f"undecorated helper: {err_text(e)}")
        v("TS1-undecorated-helper-rejects", True, err_text(e))


def stage_ts1_jit():
    md = _ts1("deco")
    v("TS1-decorated-helper-ok", md == 0.0, f"max_diff={md} (writes into the caller's fragment)")


def stage_ts1_call():
    try:
        md = _ts1("call")
        v("TS1-call-form-not-rewritten", False, f"compiled and ran, max_diff={md}")
    except Exception as e:  # noqa: BLE001
        n(f"cute.jit(f) call form: {err_text(e)}")
        v("TS1-call-form-not-rewritten", True, err_text(e))


# ------------------------------------------------------------------ TS2
def _add_one(g, i, x):  # module-level helper, everything explicit
    g[i] = x + 1.0


def _ts2(capture):
    torch, cutlass, cute, from_dlpack = _rt()

    @cute.kernel
    def k(g: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        x = g[tidx]
        if cutlass.const_expr(capture):
            def add_cap():
                g[tidx] = x + 1.0                    # closure-captures g, tidx, x
            if tidx < 64:
                add_cap()
        else:
            if tidx < 64:
                _add_one(g, tidx, x)

    a = torch.arange(128, device="cuda", dtype=torch.float32)
    t = from_dlpack(a)
    cute.compile(_launch1(cute, k), t)(t)
    torch.cuda.synchronize()
    exp = torch.arange(128, device="cuda", dtype=torch.float32)
    exp[:64] += 1
    return float((a - exp).abs().max())


def stage_ts2_capture():
    try:
        md = _ts2(True)
        v("TS2-closure-capture-rejected", False, f"compiled and ran, max_diff={md}")
    except Exception as e:  # noqa: BLE001
        n(f"closure capture: {err_text(e)}")
        v("TS2-closure-capture-rejected", True, err_text(e))


def stage_ts2_args():
    md = _ts2(False)
    v("TS2-explicit-args-ok", md == 0.0, f"max_diff={md}")


# ------------------------------------------------------------------ TS3
def _ts3_kernels():
    torch, cutlass, cute, from_dlpack = _rt()

    @cute.kernel
    def k(g: cute.Tensor, n: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        acc = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(n):         # needs n at trace time
            acc = acc + cutlass.Float32(1.0)
        g[tidx] = acc

    @cute.jit
    def entry(g: cute.Tensor, n: cutlass.Constexpr):
        k(g, n).launch(grid=[1, 1, 1], block=[32, 1, 1])

    @cute.jit
    def wrapper(g: cute.Tensor, n):                  # the transcription: annotation dropped
        entry(g, n)

    return torch, cute, from_dlpack, entry, wrapper


def stage_ts3_shipped():
    torch, cute, from_dlpack, entry, _ = _ts3_kernels()
    a = torch.zeros(32, device="cuda")
    t = from_dlpack(a)
    cute.compile(entry, t, 4)(t)
    torch.cuda.synchronize()
    v("TS3-constexpr-shipped-ok", float(a.max()) == 4.0 and float(a.min()) == 4.0, f"value={float(a[0])}")


def stage_ts3_wrapper():
    torch, cute, from_dlpack, _, wrapper = _ts3_kernels()
    a = torch.zeros(32, device="cuda")
    t = from_dlpack(a)
    try:
        f = cute.compile(wrapper, t, 4)
        n("transcribed wrapper compiled")
        try:
            f(t, 4)
        except TypeError:
            f(t)
        torch.cuda.synchronize()
        n(f"transcribed wrapper ran: value={float(a[0])}")
        v("TS3-transcribed-wrapper-differs", float(a[0]) != 4.0, f"value={float(a[0])}")
    except Exception as e:  # noqa: BLE001
        n(f"transcribed wrapper: {err_text(e)}")
        v("TS3-transcribed-wrapper-differs", True, err_text(e))


# ------------------------------------------------------------------ TS4
def stage_ts4():
    _, _, cute, _ = _rt()
    n(f"hasattr(cute, 'make_fragment')={hasattr(cute, 'make_fragment')} "
      f"hasattr(cute, 'make_fragment_like')={hasattr(cute, 'make_fragment_like')} "
      f"hasattr(cute, 'make_rmem_tensor')={hasattr(cute, 'make_rmem_tensor')} "
      f"hasattr(cute, 'make_rmem_tensor_like')={hasattr(cute, 'make_rmem_tensor_like')}")
    v("TS4-make-rmem-tensor", hasattr(cute, "make_rmem_tensor"), "")


# ------------------------------------------------------------------ TS5
def _ts5(loop):
    torch, cutlass, cute, from_dlpack = _rt()

    def helper(frag):                                # undecorated
        if cutlass.const_expr(True):
            frag[0] = cutlass.Float32(1.0)
        if cutlass.const_expr(loop == "range_constexpr"):
            for i in cutlass.range_constexpr(2):
                frag[0] = frag[0] + cutlass.Float32(1.0)
        else:
            for i in range(2):                       # plain range under a static bound
                frag[0] = frag[0] + cutlass.Float32(1.0)

    @cute.kernel
    def k(g: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        frag = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
        helper(frag)
        g[tidx] = frag[0]

    a = torch.zeros(32, device="cuda")
    t = from_dlpack(a)
    cute.compile(_launch1(cute, k, 32), t)(t)
    torch.cuda.synchronize()
    return float(a[0])


def stage_ts5():
    try:
        val = _ts5("range_constexpr")
        rc_fails, why = False, f"range_constexpr ran, value={val}"
    except Exception as e:  # noqa: BLE001
        rc_fails, why = True, err_text(e)
    n(f"undecorated helper with range_constexpr: {why}")
    val = _ts5("range")
    n(f"undecorated helper with plain range: value={val}")
    v("TS5-range-constexpr-needs-decorator", rc_fails and val == 3.0, f"{why} | plain range value={val}")


# ------------------------------------------------------------------ TS6
class _Holder:
    pass


def _ts6(local):
    torch, cutlass, cute, from_dlpack = _rt()

    @cute.kernel
    def k(g: cute.Tensor, nrun: cutlass.Int32):
        tidx, _, _ = cute.arch.thread_idx()
        h = _Holder()
        h.face = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
        if cutlass.const_expr(local):
            face = h.face
            for i in cutlass.range(nrun):            # staged loop (Runtime trip count)
                face[i] = cutlass.Float32(1.0) + i.to(cutlass.Float32)
        else:
            for i in cutlass.range(nrun):
                h.face[i] = cutlass.Float32(1.0) + i.to(cutlass.Float32)
        g[tidx] = h.face[0] + h.face[3]

    @cute.jit
    def host(g: cute.Tensor, nrun: cutlass.Int32):
        k(g, nrun).launch(grid=[1, 1, 1], block=[32, 1, 1])

    a = torch.zeros(32, device="cuda")
    t = from_dlpack(a)
    cute.compile(host, t, cutlass.Int32(4))(t, cutlass.Int32(4))
    torch.cuda.synchronize()
    return float(a[0])


def stage_ts6_attr():
    try:
        val = _ts6(False)
        v("TS6-attr-subscript-store-rejected", False, f"compiled and ran, value={val}")
    except Exception as e:  # noqa: BLE001
        n(f"attribute subscript store in a staged for: {err_text(e)}")
        v("TS6-attr-subscript-store-rejected", True, err_text(e))


def stage_ts6_local():
    val = _ts6(True)
    v("TS6-local-binding-ok", val == 5.0, f"value={val}")


STAGES = {k[len("stage_"):]: f for k, f in dict(globals()).items() if k.startswith("stage_")}
PLAN = [
    ("ts1_plain", ["TS1-undecorated-helper-rejects"], None),
    ("ts1_jit", ["TS1-decorated-helper-ok"], None),
    ("ts1_call", ["TS1-call-form-not-rewritten"], None),
    ("ts2_capture", ["TS2-closure-capture-rejected"], None),
    ("ts2_args", ["TS2-explicit-args-ok"], None),
    ("ts3_shipped", ["TS3-constexpr-shipped-ok"], None),
    ("ts3_wrapper", ["TS3-transcribed-wrapper-differs"], None),
    ("ts4", ["TS4-make-rmem-tensor"], None),
    ("ts5", ["TS5-range-constexpr-needs-decorator"], None),
    ("ts6_attr", ["TS6-attr-subscript-store-rejected"], None),
    ("ts6_local", ["TS6-local-binding-ok"], None),
]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
    else:
        sys.exit(run(HERE, PLAN, "dsl-trace-scope"))
