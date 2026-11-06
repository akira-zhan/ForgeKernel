#!/usr/bin/env python3
"""Re-earn the instrument rules of ``gpu_probe_ladder`` on an sm_120a device (CuTe DSL).

    python3 -u probe_ladder_check.py            # parent: runs every stage in its own process
    python3 -u probe_ladder_check.py <stage>    # child: one stage (used by the parent)

Contracts (one ``PROBE PASS|FAIL <tag>`` line each, printed in the final DIGEST block):
- ``L0-compile-only-lowers``    : the faulting kernel below lowers for sm_120a without a launch;
- ``L1-launch-numerics``        : a trivial kernel launches and computes the right sum;
- ``G7-fault-has-no-source-line``: launching the faulting kernel raises an error that names no
  line of this file (compile-only and launch contracts carry different localisation);
- ``G1-sticky-in-process``      : after that fault, a correct kernel in the SAME process fails too;
- ``G1-fresh-process-recovers`` : the same correct kernel in a NEW process passes;
- ``G1-hang-unbuffered-keeps-readings``: a child that prints a reading and then hangs on the GPU
  is killed by the stage timeout, and with ``-u`` the reading reaches the parent;
- ``G13-dsl-error-attributes``  : trace-time user and IR-build errors are ``DSLBaseError`` with a
  ``message`` attribute;
- ``G13-build-error-cause-carries-the-verifier-text``: for an IR-verifier rejection ``message`` is
  generic and the verifier's text lives in ``cause``;
- ``G13-shape-error-is-not-a-dsl-error``: a layout shape/stride mismatch raises a plain Python
  exception (no DSL attributes to read).
Facts (``[[N]]`` lines, never verdicts): the error texts, the buffered-child control, the
first 150 characters of the rendered DSL error.
"""
import os
import re
import subprocess
import sys
import time

HERE = os.path.abspath(__file__)
STAGE_CAP_S = 120
HANG_CAP_S = 25


# ----------------------------------------------------------------------------- child side
def _v(tag, ok, md=""):
    print(f"[[V]]{tag}|{int(bool(ok))}|{md}", flush=True)


def _n(text):
    print(f"[[N]]{text}", flush=True)


def _kernels():
    import cutlass
    import cutlass.cute as cute

    @cute.kernel
    def k_add(gA: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        gA[tidx] = gA[tidx] + cutlass.Float32(1.0)

    @cute.kernel
    def k_oob(gA: cute.Tensor):
        # a view whose stride walks ~8.5 GB past a 512 B allocation: lowers fine, faults at run time
        far = cute.make_tensor(gA.iterator, cute.make_layout(128, stride=1 << 24))
        tidx, _, _ = cute.arch.thread_idx()
        far[tidx] = cutlass.Float32(7.0)

    @cute.kernel
    def k_spin(gA: cute.Tensor):
        # gA[0] stays 0: spins forever
        while gA[0] == cutlass.Float32(0.0):
            gA[1] = gA[1] + cutlass.Float32(1.0)

    @cute.jit
    def run_add(mA: cute.Tensor):
        k_add(mA).launch(grid=[1, 1, 1], block=[128, 1, 1])

    @cute.jit
    def run_oob(mA: cute.Tensor):
        k_oob(mA).launch(grid=[1, 1, 1], block=[128, 1, 1])

    @cute.jit
    def run_spin(mA: cute.Tensor):
        k_spin(mA).launch(grid=[1, 1, 1], block=[1, 1, 1])

    return run_add, run_oob, run_spin


def _tensor():
    import torch
    from cutlass.cute.runtime import from_dlpack
    a = torch.zeros(128, device="cuda", dtype=torch.float32)
    return a, from_dlpack(a)


def _add_ok(run_add):
    import torch
    import cutlass.cute as cute
    a, ta = _tensor()
    cute.compile(run_add, ta)(ta)
    torch.cuda.synchronize()
    return a.sum().item()


def stage_compile():
    import cutlass.cute as cute
    _, run_oob, _ = _kernels()
    _, ta = _tensor()
    cute.compile(run_oob, ta)  # no launch
    _v("L0-compile-only-lowers", True, "n/a")


def stage_launch():
    run_add, _, _ = _kernels()
    s = _add_ok(run_add)
    _v("L1-launch-numerics", s == 128.0, f"sum={s}")


def stage_sticky():
    import torch
    import cutlass.cute as cute
    run_add, run_oob, _ = _kernels()
    a, ta = _tensor()
    f = cute.compile(run_oob, ta)
    try:
        f(ta)
        torch.cuda.synchronize()
        err_a = None
    except Exception as e:  # noqa: BLE001 - the fault itself is the reading
        err_a = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    _n(f"fault launch -> {err_a}")
    no_line = err_a is not None and os.path.basename(HERE) not in err_a and not re.search(r"line \d+", err_a)
    _v("G7-fault-has-no-source-line", no_line, err_a or "no error raised")
    try:
        s = _add_ok(run_add)
        err_b = None
        _n(f"correct kernel after the fault, same process -> sum={s}")
    except Exception as e:  # noqa: BLE001
        err_b = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        _n(f"correct kernel after the fault, same process -> {err_b}")
    _v("G1-sticky-in-process", err_b is not None, err_b or "later rung survived")


def stage_after_sticky():
    run_add, _, _ = _kernels()
    s = _add_ok(run_add)
    _v("G1-fresh-process-recovers", s == 128.0, f"sum={s}")


def stage_hang():
    import torch
    import cutlass.cute as cute
    _, _, run_spin = _kernels()
    a, ta = _tensor()
    f = cute.compile(run_spin, ta)
    print("[[N]]hang-stage reading before the hang: 42")  # deliberately NOT flushed
    f(ta)
    torch.cuda.synchronize()  # never returns
    print("[[N]]hang-stage returned (unexpected)")


def stage_dslerr():
    import cutlass
    import cutlass.cute as cute
    from cutlass.base_dsl.common import DSLBaseError
    from cutlass.cute.runtime import from_dlpack
    import torch
    ta = from_dlpack(torch.zeros(128, device="cuda"))

    def user_error():  # a Runtime value where a code-read-time constant is required
        @cute.kernel
        def k(gA: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            if cutlass.const_expr(tidx > 3):
                gA[0] = cutlass.Float32(1.0)

        @cute.jit
        def h(m: cute.Tensor):
            k(m).launch(grid=[1, 1, 1], block=[32, 1, 1])
        cute.compile(h, ta)

    def build_error():  # cute.copy between fragments of different size: rejected by the IR verifier
        @cute.kernel
        def k(gA: cute.Tensor):
            r = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
            d = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
            cute.autovec_copy(r, d)

        @cute.jit
        def h(m: cute.Tensor):
            k(m).launch(grid=[1, 1, 1], block=[32, 1, 1])
        cute.compile(h, ta)

    def shape_error():  # shape / stride rank mismatch: a plain Python exception, not a DSL one
        @cute.jit
        def h():
            cute.make_layout((4, 4), stride=(1,))
        h()

    got = {}
    for name, fn in (("user", user_error), ("build", build_error), ("shape", shape_error)):
        try:
            fn()
            got[name] = None
            _n(f"{name}: no error raised")
        except Exception as e:  # noqa: BLE001 - the error object is the reading
            got[name] = e
            _n(f"{name}: type={type(e).__name__} is_DSLBaseError={isinstance(e, DSLBaseError)}")
            for attr in ("message", "error_code", "cause", "suggestion", "filename", "line"):
                v = getattr(e, attr, None)
                if v:
                    _n(f"{name}: .{attr} = {str(v).strip().splitlines()[0][:160]}")
            raw = str(e)
            _n(f"{name}: str() has ANSI escapes={chr(27) in raw}; str()[:150] = "
               + " / ".join(raw[:150].replace(chr(27), "ESC").splitlines()))
    u, b = got["user"], got["build"]
    _v("G13-dsl-error-attributes",
       all(isinstance(e, DSLBaseError) and getattr(e, "message", None) for e in (u, b)),
       f"user={type(u).__name__} build={type(b).__name__}")
    cause = getattr(b, "cause", None)
    _v("G13-build-error-cause-carries-the-verifier-text",
       cause is not None and "same size" in str(cause) and "same size" not in str(getattr(b, "message", "")),
       f"message={str(getattr(b, 'message', ''))[:40]!r}")
    _v("G13-shape-error-is-not-a-dsl-error", got["shape"] is not None and not isinstance(got["shape"], DSLBaseError),
       type(got["shape"]).__name__)


STAGES = {"compile": stage_compile, "launch": stage_launch, "sticky": stage_sticky,
          "after_sticky": stage_after_sticky, "hang": stage_hang, "dslerr": stage_dslerr}


# ----------------------------------------------------------------------------- parent side
PLAN = [  # (stage, cap seconds, python flags, verdict tags the stage owns)
    ("compile", STAGE_CAP_S, ["-u"], ["L0-compile-only-lowers"]),
    ("launch", STAGE_CAP_S, ["-u"], ["L1-launch-numerics"]),
    ("sticky", STAGE_CAP_S, ["-u"], ["G7-fault-has-no-source-line", "G1-sticky-in-process"]),
    ("after_sticky", STAGE_CAP_S, ["-u"], ["G1-fresh-process-recovers"]),
    ("dslerr", STAGE_CAP_S, ["-u"], ["G13-dsl-error-attributes", "G13-build-error-cause-carries-the-verifier-text",
                                     "G13-shape-error-is-not-a-dsl-error"]),
]


def _run(stage, cap, flags, env):
    cmd = [sys.executable] + flags + [HERE, stage]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=cap)
        return p.returncode, p.stdout.decode("utf-8", "replace"), time.time() - t0
    except subprocess.TimeoutExpired as te:
        return "timeout", (te.stdout or b"").decode("utf-8", "replace"), time.time() - t0


def _postmortem(stage, rc, dt, body, keep=8):
    lines = body.splitlines()
    out = [f"stage {stage!r} exit={rc} after {dt:.0f}s, {len(lines)} lines"]
    if not lines:
        out.append("  child produced NO OUTPUT AT ALL")
        return out
    hits = [ln for ln in lines if re.search(r"Error|error:|Traceback|CUDA|Exception|Killed", ln)]
    out += ["  | " + ln[:200] for ln in (hits[-keep:] if hits else lines[-keep:])]
    return out


def main():
    env = dict(os.environ)
    env.setdefault("CUTE_DSL_ARCH", "sm_120a")
    env["CUTE_DSL_SHOW_STACKTRACE"] = "1"
    verdicts, facts, why = {}, [], []
    for stage, cap, flags, tags in PLAN:
        rc, body, dt = _run(stage, cap, flags, env)
        seen = set()
        for ln in body.splitlines():
            if ln.startswith("[[V]]"):
                tag, ok, md = ln[5:].split("|", 2)
                verdicts[tag] = (ok == "1", md); seen.add(tag)
            elif ln.startswith("[[N]]"):
                facts.append(f"{stage}: {ln[5:]}")
        missing = [t for t in tags if t not in seen]  # no verdict = FAIL, whatever rc says
        for t in missing:
            verdicts[t] = (False, f"no verdict emitted (stage rc={rc})")
        if missing:
            why += _postmortem(stage, rc, dt, body)
        print(f"[parent] stage {stage} rc={rc} {dt:.0f}s verdicts={sorted(seen)}", flush=True)

    # hang: the same child twice, unbuffered (-u) and block-buffered (stdout is a pipe)
    rc_u, body_u, dt_u = _run("hang", HANG_CAP_S, ["-u"], env)
    env_b = dict(env); env_b.pop("PYTHONUNBUFFERED", None)
    rc_b, body_b, dt_b = _run("hang", HANG_CAP_S, [], env_b)
    kept_u = "reading before the hang: 42" in body_u
    kept_b = "reading before the hang: 42" in body_b
    facts.append(f"hang: -u child rc={rc_u} after {dt_u:.0f}s, reading kept={kept_u}")
    facts.append(f"hang: buffered child rc={rc_b} after {dt_b:.0f}s, reading kept={kept_b}")
    verdicts["G1-hang-unbuffered-keeps-readings"] = (rc_u == "timeout" and kept_u,
                                                     f"rc={rc_u} kept={kept_u}")
    # the device must still be usable after the killed hang
    rc, body, dt = _run("launch", STAGE_CAP_S, ["-u"], env)
    ok = "[[V]]L1-launch-numerics|1|" in body
    facts.append(f"after the killed hang, a new process launches correctly: {ok}")

    print("=== DIGEST ===")
    for f in facts:
        print(f"FACT {f}")
    if why:
        print("=== WHY ===")
        for ln in why:
            print(ln)
    fails = []
    for tag, (ok, md) in verdicts.items():
        print(f"PROBE {'PASS' if ok else 'FAIL'} {tag} ({md})")
        if not ok:
            fails.append(tag)
    print(f"PROBE {'PASS' if not fails else 'FAIL'} probe-ladder contracts={len(verdicts)}"
          + (f" failed={','.join(fails)}" if fails else ""))
    return 0 if not fails else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        STAGES[sys.argv[1]]()
    else:
        sys.exit(main())
