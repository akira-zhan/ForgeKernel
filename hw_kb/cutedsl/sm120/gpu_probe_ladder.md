# gpu_probe_ladder (instrument engineering for multi-contract GPU probes: process isolation / self-contained digest / fact-verdict split / candidate sweep / two-marginal bijection / timing readings)

> **Certification source: the sm90 tree, partially re-checked on sm100 · re-checked on sm120** for §1, §2, §5's
> fingerprint, §6 and gotchas 1, 2, 3, 7, 13 (probe `probes/sm120/methodology/probe_ladder_check.py`,
> `PROBE PASS probe-ladder contracts=9`, CuTe DSL 4.8.0, RTX PRO 5000) and for §7 (probe
> `probes/sm120/device_facts/run_facts.py`, `PROBE PASS device-facts contracts=35`). Gotchas 4-6, 8-12, 14, 15 are
> transcribed method; their readings were taken on sm90 / sm100.
>
> One entry per hardware interface. Operator-agnostic; mechanism only. This entry is an **interpretation recipe**: it does not teach how to use any mechanism;
> it only teaches how to build the probe that carries the verdicts so that one GPU round brings back **more than one** fact.
> Names of the form `sm90/xxx` / `sm100/xxx` are entries of other trees and **are not in the read set of an sm120 run**: the loop hands
> a run only the one tree `hw_kb/cutedsl/<kb_arch>/` (`loop.toml [loop] kb_arch`). The names are written out so that you
> know they exist; every reading in them was taken on other hardware and **is not a conclusion for this architecture**; if you need one, re-earn it here.

## Purpose

GPU rounds are the most expensive resource in this pipeline, and each round can only ask once. The probe's **structure** decides how much information
the round brings back: the same submission can bring back "11 FAILs, 1 fact", or
"5 verdicts + a measured table of the candidate space". This entry records the constructions that turn the former into the latter, all verified on the GPU
in the form "one cut, and the next round's verdict flips".

## When needed (trigger)

When writing any multi-contract / multi-rung GPU certification probe. And after these symptoms appear:

- A round comes back **all FAIL, carrying only one fact** (every rung died at the step they share);
- Some rung has "no verdict", and the stretch of output holding the reason has already scrolled out of the retained log tail;
- Verdicts go round in circles on questions like "which spelling fits", **burning one GPU round per guess**.

## Minimal runnable example

### 1) One rung per process

> **Ready-made implementation**: a probe-guard helper (not included in this release).
> `run_stages({"base": base, "ring": ring}, cap_s=120)` -- one sub-interpreter per stage,
> each timed on its own, with the parent printing as it goes. The children are started with `-u`, so **readings already printed by a stage that then hangs are preserved**
> (a fully buffered child killed by SIGKILL lets its readings rot in the buffer; this was hit in practice: in one probe the first two variants
> had produced numbers, the third hung, the whole process was killed, and all three sets of readings were lost together). The code below is its shape;
> you can write it yourself, but there is no reason to rewrite it.

```python
# (stage, budget seconds, sanitizer allowed?, verdict tags this stage is responsible for)
PLAN = [
    ("S",     130, True,  ["S1-row-ramp", "S2-col-ramp"]),
    ("L0",     50, True,  ["L0-path-lowers"]),
    ("E",      90, True,  ["E1-row-ramp", "E2-col-ramp"]),
    ("whole", 210, False, ["C1-smoke", "C2-numeric", "C3-footprint"]),
]

def _run(stage, timeout, sanitize=False):
    cmd = [sys.executable, os.path.abspath(__file__), stage]      # re-exec itself
    if sanitize:
        cmd = ["compute-sanitizer", "--tool", "memcheck"] + cmd
    try:
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as te:                        # hang = timeout
        return "timeout", (te.stdout or b"").decode("utf-8", "replace")
```

The child communicates with the parent only through three kinds of prefixed lines, and the parent routes by prefix:

```
[[BEGIN]]<tag>     # printed before touching CUDA, to tell "never started" from "started then died silently"
[[V]]tag|ok|md     # verdict (the parent produces PROBE PASS/FAIL from it)
[[E]]tag|reason    # the reason for a verdict
[[N]]...           # fact (produces no verdict at all)
```

### 2) A self-contained DIGEST: fold the cause of death into the last printed block

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
                    "(import / interpreter / launcher level)")
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

### 3) A candidate sweep instead of guessing round by round (compressing N rounds into 1)

Some questions are not "does the mechanism work" but "which acceptable view spelling fits" -- there is no answer to reason out,
only an answer to measure. Do not guess one per round:

```python
_CANDS = ("shipped", "spelling-A", "spelling-B", "reinterpret", "flat-dst")

# (a) first print the layout of every candidate as a fact (including the ones that do not lower)
def attempt(f):
    try:
        t = f()
        return "shape=%s layout=%s" % (t.shape, t.layout)
    except Exception as e:                    # not lowering is also a fact
        return "<%s: %s>" % (type(e).__name__, str(e).splitlines()[0][:90])

nrec("shape  spelling-A  = %s" % attempt(lambda: tiled.retile(rSrc)))
nrec("shape  partition_D = %s" % attempt(lambda: thr.partition_D(sDst)))

# (b) start one subprocess per candidate, running the same round-trip reference
for cand in _CANDS:
    for mode in ("row", "col"):
        md[mode] = one(cand, mode)          # exception text is recorded as a fact too
    nrec("cand %-22s row=%-42s col=%s" % (cand, md["row"], md["col"]))
    if cand == "shipped":
        vrec("bijection", md["row"] == 0.0 and md["col"] == 0.0, ...)
        #  ^^ only the shipped candidate is a verdict
```

**The key is the last two lines**: every non-shipped candidate goes through the fact channel. So "learning which spelling is right"
itself never reads as `PROBE FAIL`, and one round of sweeping brings back the measured table of the whole candidate space.

### 4) Two-marginal ramps pin a bijection

One marginal cannot pin a bijection: a row ramp is blind to column permutations, a column ramp is blind to row permutations, and
**a mapping that preserves both marginals at once is the identity**.

```python
# batch 0 = row ramp (value = packed row number); batch 1 = column ramp (value = column number)
# both ramps are integers < 256 -> exactly representable in bf16 -> the criterion is a strict 0.0, not a tolerance
ref[0, r, c] = float(r)
ref[1, r, c] = float(c)
verdict("bijection", max_diff_row == 0.0 and max_diff_col == 0.0)
```

### 5) Pay for the sanitizer only for faults "without a source line"

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

### 6) Read the DSL exception's **attributes**, not its rendered text; the cause of death gets its own channel, printed last

```python
def exc_facts(stage, exc):
    """`DSLBaseError` keeps message / error_code / cause / suggestion / filename / line as attributes."""
    wrec(stage, "type", type(exc).__name__)
    for attr in ("message", "error_code", "cause", "context", "suggestion"):
        v = getattr(exc, attr, None)
        if v:
            wrec(stage, attr, v)                       # [[W]]stage|key|<=500 chars, one line per field
    fn, ln = getattr(exc, "filename", None), getattr(exc, "line", None)
    if fn:
        wrec(stage, "at", "%s:%s" % (os.path.basename(str(fn)), ln))
    tag = re.search(r"error\[([A-Z_]+)\]", re.sub(r"\x1b\[[0-9;]*m", "", str(exc)))
    if tag:                                            # DSL 4.8.0: the diagnostic tag lives only in the rendering
        wrec(stage, "tag", tag.group(1))
    if not getattr(exc, "message", None):              # not a DSL exception: the rendered text is everything
        wrec(stage, "str", _one_line(exc))
    frames = [l.strip() for l in traceback.format_exc().splitlines() if l.strip().startswith("File ")]
    wrec(stage, "tb", " << ".join(frames[-7:]))        # keep the last few frames verbatim; do not filter by file name

env["CUTE_DSL_SHOW_STACKTRACE"] = "1"                  # child: make the DSL spit out the technical details
env["CUTE_DSL_LINEINFO"] = "1"
# parent: print a `=== WHY step L ===` block only for the lowest failing rung, after LADDER and before the verdict lines
```

What the three error families of CuTe DSL 4.8.0 carry (measured on sm120, see gotcha 13):

| Raised for | Type | `.message` | Where the useful text is | `.filename` / `.line` |
|---|---|---|---|---|
| user-code rule at trace time (Runtime value in `const_expr`, a variable changing type across a dynamic branch) | `DSLUserCodeError` | the cause; the diagnostic tag (`PHASE_REQUIRES_CONSTANT`, `TYPE_UNSTABLE_JOIN`, `SCOPE_CLOSURE_CAPTURE` ...) is **not** in `error_code` (left empty) but in the `error[TAG]` head of the rendered text | `.message` + `.suggestion` (a list) | **your** file and line |
| IR verifier rejection (e.g. `cute.copy` between fragments of different size) | `DSLOperationBuildError` | generic: `Operation verification failed` | **`.cause`** (`MLIRError`): `'cute.copy' op expects same size in rank 1 ...` | a DSL-internal file, not yours |
| layout algebra misuse (shape / stride not congruent) | plain `ValueError` | -- (no DSL attributes) | `str(exc)` | traceback only |

### 7) Timing readings: make the timed region contain what you think it contains

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
   error, an illegal memory access, a hang are **not Python exceptions** -- they take with them every later
   rung in the same process. The observed shape: all five rungs report `no verdict emitted (stage rc=0)`,
   while there is only one real cause of death. **The granularity of isolation must be the process** (the parent re-execs itself + `argv[1]`
   selects the stage), not a `try` block.
   Measured on sm120: after a kernel's wild global store, a correct kernel launched in the **same** process fails with
   the same `AcceleratorError`; the same kernel in a **new** process passes. After a hung child is killed at its stage
   timeout, the next process launches normally.
2. **Infer verdicts from "is there output", not from the exit code.** A rung may have `rc=0` yet emit not a single
   verdict. The parent must go by the tag list: `missing = [t for t in tags if t not in seen]`;
   every missing one is recorded as FAIL + reason, and triggers the postmortem.
3. **The log tail gets truncated, and by default the reason is in exactly the half that gets cut.** The parent's echo of the child output
   is printed **before** the digest, so when you need it, it has just scrolled away. The fix is §2: **fold** the cause of death
   **into** the digest. Also distinguish three ways of dying -- "no output at all" (the process never reached its first print),
   "no `[[BEGIN]]` marker" (died at the import / interpreter / launcher level), "marker but no verdict"
   (died inside the stage body). The next action is completely different for each.
4. **A ladder cannot localise a fault in the step all its rungs share.** If every rung and the shipped file
   write that shared step with the same spelling, the N FAILs a round brings back carry **one** fact -- the ladder's
   whole resolution is wasted. Two fixes, pick one by situation:
   - the shared step **calls the shipped file's own constructor** instead of transcribing it (same as
     [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 3);
   - or, if the shared step itself is the unknown, **promote it to a swept axis** (§3).
5. **A shape/spelling rejection is not a "does the mechanism work" problem; do not debug it as one.** For example
   `Expected source and destination tensors to have the same size in mode-1,
   but got 1 and 8` asks "which acceptable source view fits this destination" -- going by guesses it
   **burns one GPU round per round**, and each round rules out only one candidate. Switch to the sweep of §3: run them all in one round.
6. **Do not let "learning" read as FAIL.** Non-shipped candidates in a sweep, layout prints, api-audit
   and observations like these always go through the fact channel; only contracts occupy `PROBE` verdict lines. Otherwise a round structure of 10 PASS
   and 1 FAIL loses its meaning -- whoever reads the log cannot tell "the kernel is broken" from "I just learned that
   3 of 4 candidates do not lower". The same discipline as
   [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 4 for api-audit.
7. **Split compile-only contracts and launch contracts into two.** A failure of "can it lower" throws a Python exception with a
   **line number**; a failure of "can it launch" has **no line number**. With two contracts,
   which one the FAIL lands on is itself localisation information; merged into one, the two kinds of fault are indistinguishable.
   A side benefit of a compile-only contract: it **cannot hang**, so it can safely be placed anywhere.
8. **One marginal cannot pin a bijection.** A row ramp cannot see column permutations, a column ramp cannot see row permutations.
   Run both, and choose the values as **small integers exactly representable in bf16** (< 256), so the criterion is a strict
   `max_diff == 0.0` -- not a tolerance that needs explaining.
9. **The sanitizer is only valuable for faults without a source line, and only affordable for small rungs.** A Python
   traceback already names the line; **a hang is not memcheck's domain** (a wrong mbarrier arrive-count
   or phase waits forever rather than faulting; that can only be localised by the process split of §1).
   The 10–100× instrumentation slowdown is affordable only on small single-CTA rungs, **whole-machine stages are never sanitized**;
   within budget, rerun only the **lowest** failing rung (the higher rungs are most likely its downstream).
10. **The parent needs a global deadline, not just per-stage timeouts.** When the budget runs out, record the remaining tags as
    "skipped: out of job budget" and print the digest as usual -- a probe cut off directly by the job system for timing out
    brings back zero facts, the most expensive kind of failure.
11. **To pin the bijection of a matrix atom, use two patterns "fine + coarse", not one.** The row / column ramps of gotcha 8 pin
    the element-wise path; a matrix atom (`ldmatrix` / `stmatrix`) can go wrong at two granularities -- a permutation **inside** an 8 × 8 / 16 × 16 matrix
    (including picking the wrong `transpose` member) and a swap of **whole** matrices -- and one pattern can see only one of the levels.
    The fine pattern is injective within each 16 × 16 tile and repeats across tiles (`((i % 16) * 16 + (j % 16)) − 128`); the coarse pattern is
    constant within a tile and injective across tiles (`((i // 16) * 4 + (j // 16)) * 17 − 128`); both are integers in [−128, 127],
    exact in bf16, with a strict `0.0` criterion. Three companions: **sentinel prefill** (fill the destination face first with a value neither pattern can take that is exact in
    bf16, such as −1024 -- an element touched by no path is then a FAIL rather than a coincidence); **a scalar path as control** (for the same slice and
    the same conversion, also write a copy element by element, `face[i, j] = v` -- when the control itself differs from the host pattern, what is wrong is the readback geometry, not the
    atom, so the two kinds of FAIL separate); **look up the pattern by coordinate in a host table** (`v = BF16(gPat[i, j])`; the probe does no
    "index → float" conversion, so the reference compared on the host is bit-identical by construction).
12. **When a trace-time error text does not contain your line number, the first step is to bisect the kernel body, not to change the most suspicious line.** A
    `'tuple' object is not callable`-type TypeError can come from a DSL-internal call on one of your arguments, with the traceback
    stopping inside the DSL; every change to "the most suspicious-looking" construct in the kernel body (Constexpr-ification, where the tiled copy is built, whether there is a `retile`)
    is one GPU round, while the real culprit line may stay untouched for six rounds (measured: `allocate_tensor` received a shape tuple,
    see `sm100/smem_exchange_face_matrix_atoms_ldsm_mt_stsm_sw128`
    gotcha 13). Practice: comment out half of the kernel body from the middle, run one round of the compile-only contract (gotcha 7) to see whether it lives or dies, and locate the line within
    two or three rounds; cheaper still is **not building a toy geometry** -- write the probe directly on the geometry of the shipped entry
    ([`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) gotcha 3); every line a toy brings along is a new way to die.

13. **Read a DSL error by attribute, and by the right attribute for its family; never by a truncated rendering.**
    On CuTe DSL 4.8.0 the rendered text of a `DSLBaseError` starts with an error tag (`error[PHASE_REQUIRES_CONSTANT]:`,
    `[Error] MLIRError: Verification failed:`) wrapped in **ANSI colour escapes even when stdout is a pipe**, so a log
    grep for `error:` or a fixed-width slice of `str(exc)` reads escape bytes and cuts the cause. The families differ
    (table in §6): for an IR-verifier rejection `.message` is only `Operation verification failed` and the verifier's
    sentence lives in `.cause`; its `.filename` / `.line` point into the DSL, not at your line, so the rung is
    located by the ladder (gotcha 14) or by bisection (gotcha 12), not by the error. A layout shape/stride mismatch is
    a plain `ValueError` with no attributes at all. The diagnostic tag of a user-code error (`PHASE_DYNAMIC_TO_STATIC_BOOL`,
    `SCOPE_CLOSURE_CAPTURE` ...) is not in `error_code` (empty on 4.8.0): take it from the `error[TAG]` head of the
    ANSI-stripped rendering. Send `type`, the tag, `message`, `cause`, `suggestion` field
    by field through `[[W]]`, keep the tail frames of the traceback verbatim, turn on `CUTE_DSL_SHOW_STACKTRACE=1`
    in the child. (On the DSL builds used for the sm90 / sm100 trees the first 150 characters of an internal error
    were a fixed "[Internal Error] ... This is a bug in the DSL" envelope; the rule -- attributes, not text -- is the same.)
14. **"The lowest failing rung = the defect" holds only when each rung adds exactly one construct; AST constructs count as constructs.**
    A rung that adds `make_tensor(displaced iterator)` / `partition_A/B` / `make_fragment_A/B` /
    `partition_S` / `retile` / `cute.copy` / `cute.gemm` all at once -- seven things -- says only "one of the seven" when it dies;
    split into seven rungs, the same round names the `cute.copy` line directly. Two companion disciplines:
    - **Python syntax the traced body sees for the first time is also a construct**: a conditional expression like `x = A if gear >= 4 else 1`,
      a nested `def` in the kernel body, a store subscripted through an object attribute -- each is a new AST shape;
      hoist them out of the kernel body and compute them as Python ints in the closure, and only then is a rung's delta "one mechanism".
    - **A compile-only rung must also bring back a number**: every rung writes a tap (e.g. `sA[(0,0)]` and the coordinate it claims),
      compared on the host; when the tap was not written it reports `NOT WRITTEN (n/N)`, and must not be read as a clean `0.0`.
      So rungs 4..7, of the "return right after building the partition" kind, bring back "the view lands at the right coordinates",
      not just "it compiled".
15. **Hang controls on the lowest failing rung in the same round: candidate fixes + an isolation control, each with numerics -- diagnosis and fix come back in one round.**
    Once the parent knows the lowest failing rung L, it starts further subprocesses on L: one per candidate fix
    (e.g. `.align(16)` / `local_tile`), one isolation control zeroing the suspect variable (zero displacement: is the construct itself
    broken), one alternative reading (fact line). **Controls also compare numerics** (slice map / partial sums); a fix that compiles but computes wrongly
    does not count as a fix. All go through the fact channel (gotcha 6). Next round: (a) replace the shipped spelling with the candidate that passed;
    (b) **start the ladder from L, not from 1** -- the low rungs already brought back readings last round, rerunning them only eats budget, and the high rungs
    (full K, exchange, reduction) are the slow ones; (c) the fact groups the next round needs (alignment audit, API signatures) are printed **again, de-duplicated,** in the tail block
    -- a fact block at the front is "droppable", only the tail block is the part the log keeps (gotcha 3).
    Measured: all three controls passed in the same round, slice maps each 0.0, and the next round PASSed in one go.

16. **A timing probe's verdict is the SASS of its loop, not its source.** Every trap in §7 compiled cleanly, ran
    without error and printed a number in a believable unit. The contract that catches them is cheap and
    mechanical: count the timed opcode inside the loop body in `cuobjdump -sass` (expected opcode, expected count
    per unrolled iteration), and compare the reading with its physical ceiling; only a reading that passes both is a
    fact.
17. **The no-line fingerprint list is per architecture.** A wild global store on sm120 dies with `operation not
    supported on global/shared address space`, not `illegal memory access`; a fingerprint regex that lists only the
    latter misclassifies it (§5). The `rc != 0` / missing-verdict fallbacks still catch it; keep them.

## Sources

- This repo's GPU certification on sm120 (§1, §2, §5 fingerprint, §6 table, gotchas 1, 2, 3, 7, 13, 17):
  `probes/sm120/methodology/probe_ladder_check.py` -- `PROBE PASS probe-ladder contracts=9`, all in the same round:
  `L0-compile-only-lowers`, `L1-launch-numerics`, `G7-fault-has-no-source-line`, `G1-sticky-in-process`,
  `G1-fresh-process-recovers`, `G1-hang-unbuffered-keeps-readings` (the block-buffered control lost its reading),
  `G13-dsl-error-attributes`, `G13-build-error-cause-carries-the-verifier-text`, `G13-shape-error-is-not-a-dsl-error`.
- This repo's GPU certification on sm120 (§7, gotcha 16): `probes/sm120/device_facts/run_facts.py` --
  `PROBE PASS device-facts contracts=35`.
- CuTe DSL API symbols: `cutlass.base_dsl.common.DSLBaseError` (`message`, `error_code`, `cause`, `suggestion`,
  `filename`, `line`), `cute.compile`, `cutlass.const_expr`; CUDA Runtime API
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`: https://docs.nvidia.com/cuda/cuda-runtime-api/
- PTX ISA (`ld.volatile`, `ld.global.cg`, `%clock64`, `%globaltimer`): https://docs.nvidia.com/cuda/parallel-thread-execution/
- Transcribed certifications of the sm90 / sm100 trees (readings taken on that hardware):

- This repo's GPU certification (gotchas 1, 2, 3, 4, 5, 6, 8, 9):
  certified by a GPU probe of a ping-pong epilogue (the probe is not included in this release) -- eleven contracts pass in the same round,
  including `PROBE PASS max_diff=0  # S1-r2s-row-ramp` / `PROBE PASS max_diff=0  # S2-r2s-col-ramp`
  (the two-marginal pair of §4, the shipped candidate found by the sweep), `PROBE PASS max_diff=n/a  # L0-store-path-lowers`
  (the compile-only contract of gotcha 7), `PROBE PASS max_diff=n/a  # C1-whole-machine-smoke`.
  The failure shape of gotchas 1/3 (all five rungs `no verdict emitted (stage rc=0)` in the same round,
  the reason scrolled out of the log tail) and that of gotchas 4/5 (every rung hitting the same mode-1 shape rejection)
  are recorded in that probe's per-version docstrings.
- This repo's GPU certification (gotchas 5, 6, 8): an epilogue probe --
  a four-candidate sweep settled it in one round, after which `PROBE PASS max_diff=0.0  (bijection ...)` serves as the regression contract,
  with `PROBE PASS max_diff=n/a  (smoke: ... lowers + launch completes)` and
  `PROBE PASS max_diff=0.009644746780395508  (whole kernel vs fp32 math SDPA)` in the same round.
- This repo's GPU certification (gotchas 11, 12): an stmatrix-pattern probe -- after 7 rounds
  `PROBE PASS max_diff=0.0`, all four contracts pass in the same round: `X0-atom` n/a, `X1-threads` n/a,
  `X2-fine` 0.0, `X3-coarse` 0.0 (gotcha 11's fine / coarse pattern pair + sentinel + scalar control come from this probe); the first six rounds
  gave no verdict, the changes in the six probe versions all fell on the toy geometry's `tile_n` / tiled copy position / `retile`, while the culprit line
  (`allocate_tensor` receiving a shape tuple) stayed untouched for six versions; the seventh version's docstring is on record -- gotcha 12's reading.
- How to write the verdict lines themselves (PTX/SASS count contracts) without wronging the kernel:
  `sm90/ptx_sass_census_criteria`.
- The specific spelling settled by the sweep (the source view of the epilogue r2s):
  `sm90/tma_store_epilogue` gotcha 14.
- This repo's GPU certification (§6, gotchas 13, 14, 15): a QK `ldmatrix` probe -- after 6 rounds three
  `PROBE PASS max_diff=0.0` in the same round (`Q1-s-tile0` / `Q2-s-tile1` / `Q3-rowmax`). Per-version docstrings on record:
  in round 3 the ladder located the rung but truncated the `DSLRuntimeError` to its envelope and the traceback filter dropped the DSL frame (gotcha 13's reading);
  round 4 switched to reading by attribute + one construct per rung (the seven-in-one rung split into rungs 3..9) + a tap per rung;
  round 5 `LOWEST FAILING STEP: 8 (cute.copy)` with the three controls `c2 / c3 / c4` all passing in the same round,
  `A-frag vs standard m16n8k16 map=0.0`; round 6 started the ladder from rung 8, PASS. For the mechanism-side conclusion see
  `sm100/warp_mma_ldsm_operand_displaced_view`.
