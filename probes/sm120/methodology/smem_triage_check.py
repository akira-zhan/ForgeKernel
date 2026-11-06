#!/usr/bin/env python3
"""Re-earn ``smem_traffic_triage``'s reading matrix on an sm_120a device.

    python3 -u smem_triage_check.py [--out DIR]     (needs nvcc, cuobjdump, ncu with counter access)

``smem_triage.cu`` moves the same useful bytes through shared memory three ways:
``k_smem<1, 1>`` (base: conflict-free 128-bit), ``k_smem<8, 1>`` (conflict: 128 B lane stride,
same instruction count) and ``k_smem<1, 2>`` (volume: conflict-free, one extra smem round trip).
The four counters are read through ForgeKernel's own NCU parser
(``forgekernel.inner.ncu.profiler_from_ncu``), i.e. under the keys a loop digest uses.

Contracts:
- ``M-metrics-exist``       : smem_wavefronts_st/ld and bankconf_st/ld are numbers (not n/a) for all three;
- ``S-executed-shape``      : base and conflict EXECUTE the same number of shared st / ld instructions,
  volume twice that (dynamic counts from NCU; static SASS counts differ with unrolling);
- ``T-conflict-quadrant``   : the conflict kernel reads "wavefronts over the line + bankconf over the line";
- ``T-volume-quadrant``     : the volume kernel reads "wavefronts over the line + bankconf clean",
  with the st and ld sides inflated by a similar ratio (the extra-transit signature).
Facts: every counter, the ratios, and how much of the conflict kernel's wavefront excess the
bank-conflict counter accounts for.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from forgekernel.inner.ncu import parse_ncu_csv, profiler_from_ncu  # noqa: E402

METRICS = [
    "gpu__time_active.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
]
KERNELS = {"base": "k_smem<1, 1>", "conflict": "k_smem<8, 1>", "volume": "k_smem<1, 2>"}
OVER = 1.3       # wavefronts "over the line": >= 1.3x the anchor (ForgeKernel's elevated threshold)
CONFLICTED = 3.0  # bankconf "over the line": >= 3x the anchor (floor 1 so an anchor of 0 still compares)


def sh(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p.returncode, p.stdout + p.stderr


def quadrant(m, a):
    wf = (m["smem_wavefronts_st"] + m["smem_wavefronts_ld"]) / (a["smem_wavefronts_st"] + a["smem_wavefronts_ld"])
    bc = m["bankconf_st"] + m["bankconf_ld"]
    bc_a = max(a["bankconf_st"] + a["bankconf_ld"], 1)
    over, conf = wf >= OVER, bc >= CONFLICTED * bc_a
    kind = {(True, True): "conflict", (True, False): "volume",
            (False, True): "conflict-mild", (False, False): "healthy"}[(over, conf)]
    return kind, wf, bc / bc_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".sm120_scratch" / "smem_triage"))
    out = Path(ap.parse_args().out); out.mkdir(parents=True, exist_ok=True)
    verdicts, facts = [], []

    def verdict(tag, ok, md):
        verdicts.append((tag, ok)); print(f"PROBE {'PASS' if ok else 'FAIL'} {tag} ({md})", flush=True)

    binary = out / "smem_triage"
    rc, log = sh(["nvcc", "-gencode", "arch=compute_120a,code=sm_120a", "-O3", "-lineinfo",
                  "-o", str(binary), str(HERE / "smem_triage.cu")])
    if rc:
        print(log); verdict("build", False, "nvcc failed"); return 1

    rc, csv = sh(["ncu", "--csv", "--metrics", ",".join(METRICS), str(binary)], timeout=600)
    (out / "ncu.csv").write_text(csv)
    parsed = parse_ncu_csv(csv)
    names = {k: next((n for n in parsed if pat in n), None) for k, pat in KERNELS.items()}
    keys = ("smem_wavefronts_st", "smem_wavefronts_ld", "bankconf_st", "bankconf_ld")
    m = {k: (profiler_from_ncu(csv, kernel=n) if n else {}) for k, n in names.items()}
    for k in KERNELS:
        facts.append(f"{k}: " + " ".join(f"{x}={m[k].get(x)}" for x in keys + ("time_ns",)))
    ex = {k: tuple(float(parsed[names[k]].get(f"smsp__sass_inst_executed_op_shared_{o}.sum", "nan").replace(",", ""))
                   for o in ("st", "ld")) if names[k] else (None, None) for k in KERNELS}
    facts.append(f"executed shared (st, ld) instructions {ex}")
    b, c, v = ex["base"], ex["conflict"], ex["volume"]
    verdict("S-executed-shape", None not in b and b == c and v == (2 * b[0], 2 * b[1]), str(ex))
    ok_metrics = all(isinstance(m[k].get(x), (int, float)) for k in KERNELS for x in keys)
    verdict("M-metrics-exist", ok_metrics, f"kernels={list(names.values())}")
    if not ok_metrics:
        print(csv[-2000:])
        return 1

    a = m["base"]
    kc, wfc, bcc = quadrant(m["conflict"], a)
    kv, wfv, bcv = quadrant(m["volume"], a)
    st_r = m["volume"]["smem_wavefronts_st"] / a["smem_wavefronts_st"]
    ld_r = m["volume"]["smem_wavefronts_ld"] / a["smem_wavefronts_ld"]
    excess = (m["conflict"]["smem_wavefronts_st"] + m["conflict"]["smem_wavefronts_ld"]
              - a["smem_wavefronts_st"] - a["smem_wavefronts_ld"])
    explained = (m["conflict"]["bankconf_st"] + m["conflict"]["bankconf_ld"]) / excess if excess else float("nan")
    facts.append(f"conflict: wavefront ratio {wfc:.2f}x, bankconf ratio {bcc:.1f}x -> {kc}")
    facts.append(f"volume: wavefront ratio {wfv:.2f}x (st {st_r:.2f}x, ld {ld_r:.2f}x), bankconf ratio {bcv:.2f}x -> {kv}")
    facts.append(f"conflict: bank-conflict counter / wavefront excess = {explained:.3f}")
    verdict("T-conflict-quadrant", kc == "conflict", f"wf {wfc:.2f}x bankconf {bcc:.1f}x")
    verdict("T-volume-quadrant", kv == "volume" and abs(st_r - ld_r) < 0.2 * max(st_r, ld_r),
            f"wf {wfv:.2f}x st {st_r:.2f}x ld {ld_r:.2f}x bankconf {bcv:.2f}x")

    print("=== DIGEST ===")
    for f in facts:
        print(f"FACT {f}")
    fails = [t for t, ok in verdicts if not ok]
    print(f"PROBE {'PASS' if not fails else 'FAIL'} smem-triage contracts={len(verdicts)}"
          + (f" failed={','.join(fails)}" if fails else ""))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
