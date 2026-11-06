#!/usr/bin/env python3
"""SM120 device-facts probe driver.

    python3 run_facts.py --gpu <nvidia-smi index> [--out DIR] [--quick]

Builds ``facts.cu`` for the arch-specific target (sm_120a), then runs one measurement
per process (a hang or a sticky CUDA error costs that rung only, never the readings
already printed by the others), each under its own timeout.

Contracts (each ends in one ``PROBE PASS <name>`` / ``PROBE FAIL <name> ...`` line):
- ``build``: facts.cu compiles for compute_120a/sm_120a;
- ``sass-<kind>``: the mma kernel of that kind issues the expected SASS opcode;
- ``numerics-<kind>``: one accumulator chain with all-ones operands (scales = 1.0)
  ends at exactly iters * K — the throughput reading of a kind whose numerics fail is
  not reported;
- ``stage-<name>``: the stage ran to completion within its timeout.

Readings are facts, not contracts: median of 5-7 launches, SM clock UNLOCKED and
measured inside each timed kernel (clock64 / globaltimer). The GPU's utilisation and
other compute processes are sampled before and after; any sign of a co-tenant marks the
whole digest ``CONTENDED`` (the readings are then not facts about the device).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KINDS = {  # kind -> (template index in facts.cu, expected SASS opcode)
    "bf16_f32acc": (0, "HMMA.16816.F32.BF16"),
    "f16_f16acc": (1, "HMMA.16816.F16"),
    "e4m3_f32acc": (2, "QMMA.16832.F32.E4M3.E4M3"),
    "e4m3_f16acc": (3, "QMMA.16832.F16.E4M3.E4M3"),
    "e2m1_f8f6f4_f32acc": (4, "QMMA.16832.F32.E2M1.E2M1"),
    "s8_s32acc": (5, "IMMA.16832.S8.S8"),
    "mxf8_1x_f32acc": (6, "QMMA.SF.16832.F32.E4M3.E4M3.E8"),
    "mxf4_2x_f32acc": (7, "OMMA.SF.16864.F32.E2M1.E2M1.E8"),
    "nvf4_4x_f32acc": (8, "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X"),
}
LAT_KIB = [16, 256, 4096, 16384, 32768, 49152, 65536, 81920, 98304, 131072, 1048576]
L2_MIB = [8, 16, 32, 48, 64, 80, 96, 128, 256, 1024]
QUICK_LAT_KIB = [256, 16384, 262144]
QUICK_L2_MIB = [16, 64, 256]


def sh(cmd, timeout=None, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    return p.returncode, p.stdout + p.stderr


def gpu_state(idx):
    """utilisation / clocks / power / memory of one GPU plus the PIDs computing on it."""
    q = "index,uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,clocks_event_reasons.active"
    _, out = sh(["nvidia-smi", f"--id={idx}", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    vals = [v.strip() for v in out.strip().split(",")]
    st = dict(zip(q.split(","), vals))
    _, apps = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    st["foreign_pids"] = [l.split(",")[1].strip() for l in apps.splitlines()
                          if l.strip() and l.split(",")[0].strip() == st.get("uuid")]
    return st


def contended(st, own_pid=None):
    util = int(re.sub(r"\D", "", st.get("utilization.gpu", "0")) or 0)
    others = [p for p in st["foreign_pids"] if p != str(own_pid)]
    return util > 5 or bool(others), f"util={util}% other_pids={others}"


def parse_facts(text):
    facts, checks = [], []
    for line in text.splitlines():
        if line.startswith("FACT "):
            facts.append(dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line[5:])))
        elif line.startswith("CHECK "):
            checks.append(dict(re.findall(r"(\w+)=(\S+)", line[6:])))
    return facts, checks


def sass_kinds(binary):
    """{template index: set of MMA opcodes} from cuobjdump of the k_mma<KIND> instances."""
    _, text = sh(["cuobjdump", "-sass", str(binary)])
    out, cur = {}, None
    for line in text.splitlines():
        m = re.search(r"Function : _Z5k_mmaILi(\d+)EE", line)
        if m:
            cur = int(m.group(1)); out[cur] = set(); continue
        if "Function :" in line:
            cur = None; continue
        if cur is not None:
            m = re.search(r"\b((?:HMMA|QMMA|OMMA|IMMA)[A-Z0-9._]*)", line)
            if m:
                out[cur].add(m.group(1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, help="nvidia-smi index of the GPU to measure")
    ap.add_argument("--out", default=str(HERE.parents[2] / ".sm120_scratch" / "device_facts"))
    ap.add_argument("--quick", action="store_true", help="fewer sizes (functional check)")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    verdicts, digest = [], {"gpu_index": args.gpu, "stages": {}, "checks": {}}

    def verdict(name, ok, why=""):
        line = f"PROBE {'PASS' if ok else 'FAIL'} {name}" + (f" {why}" if why else "")
        verdicts.append((name, ok)); print(line, flush=True)

    binary = out / "facts"
    rc, log = sh(["nvcc", "-gencode", "arch=compute_120a,code=sm_120a", "-O3", "-lineinfo",
                  "-o", str(binary), str(HERE / "facts.cu")])
    verdict("build", rc == 0, "" if rc == 0 else log.strip().splitlines()[-1][:200])
    if rc:
        return 1
    _, nvcc_v = sh(["nvcc", "--version"])
    digest["nvcc"] = nvcc_v.strip().splitlines()[-1]
    _, drv = sh(["nvidia-smi", "--query-gpu=driver_version", f"--id={args.gpu}", "--format=csv,noheader"])
    digest["driver"] = drv.strip()

    found = sass_kinds(binary)
    for kind, (ti, op) in KINDS.items():
        ok = op in found.get(ti, set())
        verdict(f"sass-{kind}", ok, "" if ok else f"expected {op}, saw {sorted(found.get(ti, []))}")

    before = gpu_state(args.gpu)
    digest["gpu_before"] = before
    busy, why = contended(before)
    if busy:
        print(f"WARN gpu {args.gpu} is in use before the run ({why}): readings will be marked CONTENDED", flush=True)

    lat = QUICK_LAT_KIB if args.quick else LAT_KIB
    l2 = QUICK_L2_MIB if args.quick else L2_MIB
    stages = [("attrs", ["attrs"], 60), ("launch", ["launch"], 120), ("alu", ["alu"], 120),
              ("smem", ["smem"], 120), ("lat", ["lat"] + [str(k) for k in lat], 900),
              ("l2", ["l2"] + [str(m) for m in l2], 900), ("dram", ["dram", "1024" if args.quick else "2048"], 300)]
    stages += [(f"mma-{k}", ["mma", k], 300) for k in KINDS]
    sampled = []
    for name, cmd, tmo in stages:
        t0 = time.time()
        try:
            rc, text = sh([str(binary)] + cmd, timeout=tmo, env=env)
        except subprocess.TimeoutExpired as te:
            rc, text = "timeout", (te.stdout or b"").decode("utf-8", "replace") if isinstance(te.stdout, bytes) else (te.stdout or "")
        facts, checks = parse_facts(text)
        digest["stages"][name] = {"rc": rc, "seconds": round(time.time() - t0, 1), "facts": facts}
        for line in text.splitlines():
            if line.startswith(("FACT ", "CHECK ", "ERROR")):
                print(f"  [{name}] {line}", flush=True)
        verdict(f"stage-{name}", rc == 0 and f"DONE mode={cmd[0]}" in text, "" if rc == 0 else f"rc={rc}")
        for c in checks:
            ok = c.get("ok") == "1"
            digest["checks"][c["mma"]] = c
            verdict(f"numerics-{c['mma']}", ok, "" if ok else f"expect={c.get('expect')} err={c.get('max_abs_err')}")
        sampled.append(gpu_state(args.gpu))

    after = gpu_state(args.gpu)
    digest["gpu_after"] = after
    busy_after, why_after = contended(after)
    digest["contended"] = busy or busy_after
    digest["contention_note"] = f"before: {why}; after: {why_after}"
    digest["clock_event_reasons_seen"] = sorted({s.get("clocks_event_reasons.active", "") for s in sampled})
    (out / "digest.json").write_text(json.dumps(digest, indent=1))

    fails = [n for n, ok in verdicts if not ok]
    print(f"DIGEST {out / 'digest.json'}")
    if digest["contended"]:
        print(f"FACTS-CONTENDED {digest['contention_note']}")
    print(f"PROBE {'PASS' if not fails else 'FAIL'} device-facts contracts={len(verdicts)}"
          + (f" failed={','.join(fails)}" if fails else ""))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
