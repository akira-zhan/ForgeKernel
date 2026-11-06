# gpu_probe_ladder (instrument engineering for multi-contract GPU probes: process isolation / self-contained digest / fact-verdict split / candidate sweep / double-edge bijection / timing readings)

> **Certification source: methodology transcribed from the CuTe DSL mode knowledge base (`hw_kb/cutedsl/sm90`, `sm100`, `sm120`) · re-checked on sm120**
> for the language-independent CUDA-level behaviour of gotcha 1 and §5's fingerprint (probe
> `probes/sm120/methodology/probe_ladder_check.py`, `PROBE PASS probe-ladder contracts=9`) and for §6 (CUDA C++ probe
> `probes/sm120/device_facts/run_facts.py`, `PROBE PASS device-facts contracts=35`). What is transcribed is the part
> independent of the kernel language: the probe's instrument structure (process isolation, digest, verdict/fact split,
> double-edge ramp) is carried by the Python driver script, and CUDA-mode probes are Python driver scripts too —
> they compile and load a `notes/<name>.cu` with the operator tree's builder (`load_cuda_entry` in `eval/leaderboard.py`).
> The DSL-specific parts (how to read trace-time exceptions, sweep examples of view spellings) are not transcribed.
>
> One file per hardware interface. Operator-agnostic: mechanism only. This entry is a **reading recipe**: it teaches no mechanism,
> only how to build the probe that carries the verdicts so that one GPU round brings back **more than one** fact.

## Purpose

GPU rounds are the most expensive resource in this pipeline, and one round can ask only once. The probe's **structure** decides how much information
the round brings back: the same submission can bring back "11 FAILs, 1 fact" or
"5 verdicts + a measured table of the candidate space". This entry collects the constructions that turn the former into the latter.

## When needed (trigger)

When writing any multi-contract / multi-rung GPU certification probe. And after these symptoms appear:

- a round comes back **all FAIL carrying only one fact** (every rung died at the step they share);
- some rung has "no verdict", and the output holding the reason has already scrolled out of the retained log tail;
- verdicts go round in circles on questions like "which spelling matches", **burning one GPU round per guess**;
- the kernel hangs / never returns after launch, and one bare run burns the whole job's clock.

## Minimal runnable example

### 1) One rung per process

One sub-interpreter per stage, each timed separately, the parent printing as it goes. Start the child with `-u`, so the readings that a hanging stage has already printed
survive (a fully buffered child that is SIGKILLed lets its readings rot in the buffer).

```python
# (stage, budget seconds, sanitizer allowed?, verdict tags this stage owns)
PLAN = [
    ("S",     130, True,  ["S1-row-ramp", "S2-col-ramp"]),
    ("L0",     50, True,  ["L0-path-builds"]),
    ("E",      90, True,  ["E1-row-ramp", "E2-col-ramp"]),
    ("whole", 210, False, ["C1-smoke", "C2-numeric", "C3-footprint"]),
]

def _run(stage, timeout, sanitize=False):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), stage]   # re-exec itself
    if sanitize:
        cmd = ["compute-sanitizer", "--tool", "memcheck"] + cmd
    try:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as te:                           # hang = timeout
        return "timeout", (te.stdout or b"").decode("utf-8", "replace")
```

The child talks to the parent only through prefixed lines, which the parent routes by prefix:

```
[[BEGIN]]<tag>     # printed before touching CUDA, separates "never started" from "started, then died silently"
[[V]]tag|ok|md     # verdict (the parent produces PROBE PASS/FAIL from it)
[[E]]tag|reason    # the verdict's reason
[[N]]...           # fact (produces no verdict)
```

In CUDA mode the first step of a stage body is usually to build and load the `.cu` under test (the same source reuses its build within one job):

```python
sys.path.insert(0, "eval")
from leaderboard import load_cuda_entry
fn = load_cuda_entry("notes/<name>.cu", "<entry name>")   # a compile failure raises: nvcc output carries .cu line numbers
```

### 2) Self-contained DIGEST: fold the cause of death into the block printed last

```python
def _postmortem(stage, rc, dt, body, keep=14):
    lines = body.splitlines()
    head = ["stage %r exit=%s after %.0fs, %d bytes / %d lines"
            % (stage, rc, dt, len(body), len(lines))]
    if not lines:
        head.append("  child produced NO OUTPUT AT ALL "
                    "(process never reached its first print)")
        return head
    if not any(ln.startswith("[[BEGIN]]") for ln in lines):
        head.append("  no [[BEGIN]] marker: died before the stage body "
                    "(import / build / interpreter level)")
    interesting = [ln for ln in lines
                   if ln.startswith(("[[E]]", "[[parent]]", "note:"))
                   or re.search(r"Error|error:|Traceback|assert|CUDA|"
                                r"Exception|Killed|Aborted", ln)]
    for ln in (interesting[-keep:] if interesting else lines[-keep:]):
        head.append("  | " + ln[:200])
    return head
```

The parent calls it for every stage that "has tags but no verdict" and **folds the returned lines into the digest**; the digest is
the last thing printed in the whole run, so the cause of death survives log truncation by construction.

### 3) A candidate sweep instead of guessing round by round (compress N rounds into 1)

Some questions are not "does the mechanism work" but "which spelling matches" (which swizzle tier, which lane mapping, which descriptor
field value) — they have no answer you can reason out, only one you can measure. Do not guess one per round: build one variant per candidate (in the same
`.cu` via a template parameter or a `switch`, or one `.cu` each), run each in its own child process against the same reference, **all on the fact channel**
(`[[N]]cand <name> row=<..> col=<..>`); only the shipped candidate takes a verdict line. Then "learning which spelling is right"
never reads as `PROBE FAIL`, and one sweep round brings back the measured table for the whole candidate space.

### 4) Double-edge ramp pins the bijection

One marginal cannot pin a bijection: a row ramp is blind to column permutations, a column ramp is blind to row permutations, and
**a mapping that preserves both marginals at once is the identity**.

```python
# batch 0 = row ramp (value = packed row index); batch 1 = column ramp (value = column index)
# both ramps are integers < 256 -> exactly representable in bf16 -> the criterion is a strict 0.0, not a tolerance
ref[0, r, c] = float(r)
ref[1, r, c] = float(c)
verdict("bijection", max_diff_row == 0.0 and max_diff_col == 0.0)
```

### 5) Pay for the sanitizer only on faults without a source line

```python
NO_LINE_FINGERPRINT = re.compile(             # CUDA-level causes of death: naturally carry no source line
    r"illegal memory access|unspecified launch failure|misaligned address"
    r"|operation not supported on global/shared address space",   # sm120's text for a wild global store
    re.IGNORECASE)
py_level = "Traceback (most recent call last)" in body
line_less = (rc != "timeout") and not py_level and bool(
    NO_LINE_FINGERPRINT.search(body)          # illegal memory access / launch failure
    or (isinstance(rc, int) and rc != 0)
    or missing)                               # tags but no verdict = silent death
if sanitizable and line_less and any_failed:
    retry.append((stage, tags))               # rerun only the **lowest** failing rung
```

For the sanitizer to report `.cu` line numbers, add `// fk-cuda-cflags: -lineinfo` as the first line of the source under test (within the operator tree's whitelist).

### 6) Timing readings: make the timed region contain what you think it contains

A rate reading is a contract on the **SASS of the timed loop**, not on the source. Before trusting a number, read the
loop in `cuobjdump -sass` and check the instruction you meant to time is in it, once per iteration; and check the
reading against a physical bound (a lane-count x clock ceiling, a bus-width x data-rate ceiling). Each item below
produced a plausible-looking, wrong number on sm120 before it was caught (the fixed readings are in
[`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md)):

| Trap | Wrong reading it produced | Fix |
|---|---|---|
| inputs identical across threads (`v = x + j`) | the FFMA chain went to the uniform datapath (`UFFMA`): 65 instead of 123 FFMA/clk/SM | seed per-thread values (`+ threadIdx.x * eps`), check SASS for vector `FFMA` |
| a shared-memory read loop over a few fixed addresses | ptxas hoisted the `LDS` out of the loop **even inside `asm volatile("ld.shared...")`**: 26 PB/s | `ld.volatile.shared` (one `LDS` per iteration survives), or make the address depend on the previous value |
| a grid larger than one resident wave sweeping a "footprint" | each wave re-read only its own slice, which fit in cache: a 256 MiB read reported 5.5 TB/s on a 1.23 TB/s bus | size the grid with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` x SMs (one resident wave) |
| default `ld.global` in an "L2" sweep | small footprints were served from L1 | `ld.global.cg` (`__ldcg`) for an L2 reading |
| a pointer chase with fewer steps than the footprint has lines | a 1 GiB "DRAM" chase touched 25 MB and read L2 latency | walk the whole permutation cycle once to warm, once timed |

The clock is part of the reading: with clocks unlocked, record the SM clock **of the timed kernel** (`clock64` delta
over `globaltimer` delta on one thread) and quote per-clock rates; a kernel too short for the `globaltimer`
resolution reports an impossible clock (above the card's maximum), which is itself a sign the timed region is empty.

## Gotchas

1. **A per-rung `try/except` is not isolation; it only looks like it.** A sticky CUDA context
   error, an illegal memory access or a hang are **not Python exceptions** — they take down every later
   rung in the same process. Observed shape: all five rungs report `no verdict emitted (stage rc=0)`,
   while there is only one real cause of death. **The granularity of isolation must be the process** (the parent re-execs itself + `argv[1]`
   selects the stage), not a `try` block.
   Measured on sm120: after a kernel's wild global store, a correct kernel launched in the **same** process fails with
   the same CUDA error; the same kernel in a **new** process passes. After a hung child is killed at its stage
   timeout, the next process launches normally; a child started with `python -u` keeps the readings it printed
   before the hang, a block-buffered one (stdout is a pipe) loses them.
2. **Infer verdicts from "was there output", not from the exit code.** A rung can have `rc=0` and still emit not a single verdict.
   The parent must go by the tag list: `missing = [t for t in tags if t not in seen]`;
   every missing one is recorded as FAIL + reason and triggers the postmortem.
3. **The log tail gets truncated, and by default the reason is exactly in the half that was cut.** The parent's echo of the child's output
   is printed **before** the digest, so it has scrolled away exactly when you need it. The fix is §2: **fold** the cause of death
   into the digest. Also distinguish three ways of dying — "no output at all" (the process never reached its first print),
   "no `[[BEGIN]]` marker" (died at the import / build / interpreter level), "marker but no verdict"
   (died inside the stage body). The next action for each is entirely different.
4. **A ladder cannot locate a fault in the step all its rungs share.** If every rung and the shipped file
   write that shared step the same way, the N FAILs of a round carry only **one** fact — the ladder's
   whole resolution is wasted. Two fixes, take one as the case requires: have the shared step **directly reuse the same code
   from the shipped kernel** (the same `__device__` function, not a transcribed copy); or, if the shared step itself is the unknown,
   **promote it to a swept axis** (§3).
5. **Do not let "learning" read as FAIL.** In a sweep, non-shipped candidates, layout prints, API audits and similar observations all go on the fact channel;
   only contracts take `PROBE` verdict lines. Otherwise the structure of 10 PASS and 1 FAIL in a round loses its meaning — the log reader
   cannot tell "the kernel is broken" from "I just learned that 3 of 4 candidates do not match".
6. **Split "does it compile" and "does it launch" into two contracts.** A compile failure is reported by nvcc with **`.cu` line numbers**
   (brought back via a Python exception); a launch failure has **no line number**. Split into two contracts, which one the FAIL lands on is itself a piece of
   locating information; merged into one, the two faults are indistinguishable. A side benefit of the compile contract: it **cannot hang**, so it can safely go anywhere.
7. **One marginal cannot pin a bijection.** A row ramp cannot see column permutations, a column ramp cannot see row permutations.
   Run both, and choose the values as **small integers exactly representable in bf16** (< 256), so the criterion is a strict
   `max_diff == 0.0` — not a tolerance that needs explaining.
8. **The sanitizer is only worth it for faults without a source line, and only affordable on small rungs.** A Python traceback and an nvcc error
   already name the line; **hangs are not memcheck's domain** (a wrong mbarrier arrive count or phase waits forever rather than
   faulting; that can only be located by the process split of §1). The 10–100× instrumentation slowdown is affordable only on single-CTA small rungs;
   **never sanitize a whole-machine stage**; within budget rerun only the **lowest** failing rung (higher rungs are most likely its downstream).
9. **The parent needs a global deadline, not just per-stage timeouts.** When the budget runs out, record the remaining tags as
   "skipped: out of job budget" and print the digest as usual — a probe cut off by the job system on timeout
   brings back zero facts, the most expensive kind of failure. In CUDA mode the first build of each new `.cu` variant (CUTLASS headers)
   takes about half a minute; count build time into the budget too.
10. **To pin the bijection of a matrix atom, use two patterns, "fine + coarse", not one.** The row / column ramps of gotcha 7 pin the
    element-wise path; a matrix atom (`ldmatrix` / `stmatrix`) can go wrong at two granularities — permutations **inside** an 8 × 8 / 16 × 16 matrix
    (including picking the wrong `.trans` member) and swaps between **whole** matrices — and one pattern sees only one of them.
    The fine pattern is injective within each 16 × 16 tile and repeats across tiles (`((i % 16) * 16 + (j % 16)) − 128`); the coarse pattern is
    constant within a tile and injective across tiles (`((i // 16) * 4 + (j // 16)) * 17 − 128`); both are integers in [−128, 127],
    exact in bf16, criterion a strict `0.0`. Three companions: **sentinel prefill** (fill the destination face first with a value neither pattern can take and that is bf16-exact,
    e.g. −1024 — an element no path touched is then a FAIL rather than a coincidence); **a scalar path as control** (the same fragment,
    the same conversion, written element by element — when the control itself does not equal the host pattern, the readback geometry is wrong, not the atom, and the two FAILs are separable);
    **look the pattern up in a host table by coordinate** (the probe does no "index → float" conversion, so the reference the host compares against is bitwise identical by construction).
11. **On the lowest failing rung, attach controls in the same round: candidate fixes + isolation controls, each with numerics — diagnosis and fix come back in one round.**
    Once the parent knows the lowest failing rung L, it starts additional child processes on L: one branch per candidate fix, one isolation control that zeroes the suspect variable,
    one alternative reading (a fact line). **Controls compare numerics too**; a fix that compiles but computes wrong is not a fix.
    All on the fact channel (gotcha 5). Next round: (a) swap the passing candidate in as the shipped spelling; (b) **start the ladder from L,
    not from 1** — the low rungs already brought back readings last round, rerunning them only eats budget; (c) **deduplicate and print again** the fact group the next round needs
    in the tail block — a fact block at the front is "disposable", only the tail block is the part of the log that survives (gotcha 3).

12. **A timing probe's verdict is the SASS of its loop, not its source.** Every trap in §6 compiled cleanly, ran
    without error and printed a number in a believable unit. The contract that catches them is cheap and
    mechanical: count the timed opcode inside the loop body in `cuobjdump -sass` (expected opcode, expected count
    per unrolled iteration), and compare the reading with its physical ceiling; only a reading that passes both is a
    fact.
13. **The no-line fingerprint list is per architecture.** A wild global store on sm120 dies with `operation not
    supported on global/shared address space`, not `illegal memory access`; a fingerprint regex that lists only the
    latter misclassifies it (§5). The `rc != 0` / missing-verdict fallbacks still catch it; keep them.

## Sources

- This repo's GPU certification on sm120 (gotcha 1, §5 fingerprint, gotcha 13): `probes/sm120/methodology/probe_ladder_check.py`
  -- `PROBE PASS probe-ladder contracts=9` (`G1-sticky-in-process`, `G1-fresh-process-recovers`,
  `G1-hang-unbuffered-keeps-readings`, `G7-fault-has-no-source-line` among them).
- This repo's GPU certification on sm120 (§6, gotcha 12): `probes/sm120/device_facts/run_facts.py` --
  `PROBE PASS device-facts contracts=35`.
- CUDA Runtime API `cudaOccupancyMaxActiveBlocksPerMultiprocessor`: https://docs.nvidia.com/cuda/cuda-runtime-api/ ;
  PTX ISA (`ld.volatile`, `ld.global.cg`, `%clock64`, `%globaltimer`): https://docs.nvidia.com/cuda/parallel-thread-execution/
- GPU-certified in this repo: the certification probes and verdict lines listed in the CuTe DSL mode knowledge base's `hw_kb/cutedsl/sm90/gpu_probe_ladder.md` and `hw_kb/cutedsl/sm100/gpu_probe_ladder.md`
  (ladder probes with multiple contracts all passing in the same round, `max_diff=0.0` verdicts for the double-edge ramp and the fine / coarse pattern pair).
  This entry transcribes only the instrument constructions in them that are independent of the kernel language.
