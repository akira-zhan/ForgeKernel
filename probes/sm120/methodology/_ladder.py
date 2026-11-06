"""Minimal process-per-stage runner shared by the CuTe DSL semantics probes (gpu_probe_ladder §1-§2).

Child side: ``v(tag, ok, md)`` prints a verdict, ``n(text)`` a fact.
Parent side: ``run(script, plan)`` with plan = [(stage, [tags], extra_env)], one subprocess per stage
(``python -u``), missing verdicts are FAIL, the digest (facts, causes of death, verdicts) is printed last.
"""
import os
import re
import subprocess
import sys
import tempfile


def v(tag, ok, md=""):
    print(f"[[V]]{tag}|{int(bool(ok))}|{str(md).replace(chr(10), ' ')[:300]}", flush=True)


def n(text):
    print(f"[[N]]{str(text).replace(chr(10), ' ')[:400]}", flush=True)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def err_text(e):
    """type + diagnostic tag + the first line of message (or str) of an exception, ANSI-free.

    The tag (``PHASE_DYNAMIC_TO_STATIC_BOOL`` ...) is taken from ``error_code`` when set, else from the
    ``error[TAG]`` head of the rendered text (DSL 4.8.0 often leaves ``error_code`` empty)."""
    raw = _ANSI.sub("", str(e))
    code = getattr(e, "error_code", None)
    if not code:
        m = re.search(r"error\[([A-Z_]+)\]", raw)
        code = m.group(1) if m else None
    msg = _ANSI.sub("", str(getattr(e, "message", None) or raw))
    first = next((ln.strip() for ln in msg.splitlines() if ln.strip()), "")
    return f"{type(e).__name__}" + (f"[{code}]" if code else "") + f": {first[:200]}"


def run(script, plan, name, cap_s=300):
    base_env = dict(os.environ)
    base_env.setdefault("CUTE_DSL_ARCH", "sm_120a")
    base_env["CUTE_DSL_SHOW_STACKTRACE"] = "1"
    verdicts, facts, why = {}, [], []
    for stage, tags, extra in plan:
        env = dict(base_env, **(extra or {}))
        env.setdefault("CUTE_DSL_CACHE_DIR", tempfile.mkdtemp(prefix=f"fk_{stage}_"))  # no cache shared across stages
        try:
            p = subprocess.run([sys.executable, "-u", script, stage], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=cap_s)
            rc, body = p.returncode, p.stdout.decode("utf-8", "replace")
        except subprocess.TimeoutExpired as te:
            rc, body = "timeout", (te.stdout or b"").decode("utf-8", "replace")
        seen = set()
        for ln in body.splitlines():
            if ln.startswith("[[V]]"):
                tag, ok, md = ln[5:].split("|", 2)
                verdicts[tag] = (ok == "1", md)
                seen.add(tag)
            elif ln.startswith("[[N]]"):
                facts.append(f"{stage}: {ln[5:]}")
        missing = [t for t in tags if t not in seen]
        for t in missing:
            verdicts[t] = (False, f"no verdict emitted (stage rc={rc})")
        if missing:
            why += [f"stage {stage} rc={rc}"] + ["  | " + ln[:220] for ln in body.splitlines()
                                                 if "Error" in ln or "error" in ln][-10:]
        print(f"[parent] stage {stage} rc={rc} verdicts={sorted(seen)}", flush=True)
    print("=== DIGEST ===")
    for f in facts:
        print(f"FACT {f}")
    for ln in why:
        print(ln)
    fails = [t for t, (ok, _) in verdicts.items() if not ok]
    for t, (ok, md) in verdicts.items():
        print(f"PROBE {'PASS' if ok else 'FAIL'} {t} ({md})")
    print(f"PROBE {'PASS' if not fails else 'FAIL'} {name} contracts={len(verdicts)}"
          + (f" failed={','.join(fails)}" if fails else ""))
    return 0 if not fails else 1
