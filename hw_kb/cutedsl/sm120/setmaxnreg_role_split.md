# setmaxnreg_role_split (run-time register split between a producer warp and consumer warps on sm120: `setmaxregister_decrease / _increase`, the hang above the launch value, and what it costs when nothing needs it)

> One entry per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified on RTX PRO 5000, CuTe DSL 4.8.0, `sm_120a`, in one CTA shape: a 288-thread CTA of 8 `mma.sync`
> consumer warps (warpgroups 0 and 1) plus **one** TMA producer warp (warp 8, alone in warpgroup 2), 1 CTA / SM,
> 168 registers / thread at launch. Evidence: a hang-bisection probe (`notes/d21_regsplit_probe.py`, job 1794048427),
> a smoke + numerics + same-job A/B probe (`notes/d21_smoke_probe.py`, `PROBE PASS` 6/6, job 1794048954), and two
> KEEPs whose own NCU census moved the `rings` axis: inserting the split (KEEP in job 1794048999) and removing it again
> (KEEP in job 1794049160).
>
> **What is certified**: `setmaxnreg` executes on sm120 (an `.inc` nobody pays for blocks forever); in the CTA shape
> above, any consumer `.inc` above the launch value hangs, and the producer's `.dec` alone, with no consumer gain,
> cost about 10 % with consumer-side spills appearing. **Not certified**: which register split, if any, pays on sm120;
> whether a CTA made of whole warpgroups avoids the hang (see Gotchas 1, 6).

## Purpose

`setmaxnreg.{dec,inc}.sync.aligned.u32` moves per-thread register quota between roles of one CTA at run time: a
producer role that needs few registers hands its quota back (`.dec`), a consumer role that needs more takes it
(`.inc`). The mechanism and its sm90 rules are in [`../sm90/warp_specialization.md`](../sm90/warp_specialization.md);
this entry is what was measured on sm120, where the tensor core is driven by `mma.sync` and the accumulator lives in
registers.

## When needed (trigger)

- **Only when the consumer is short of registers**: `spill_ld/st_sectors > 0` on the consumer's code, or
  `regs_per_thread` at the per-thread cap `floor(65536 / blockDim)` rounded down to a multiple of 8. When the consumer
  already fits at its launch value with 0 spills, the split has nothing to buy, and in the shape measured here it was
  a pure loss (Readings).
- A lever list carries "register split between roles" / "setmaxnreg" on an sm120 warp-specialized kernel: read
  Gotcha 1 before choosing the consumer value; the sm90 pattern (producer is a whole warpgroup, consumers go above
  the launch value) is not what was certified here.
- The dossier's `rings` axis changes value on a KEEP that added or removed `setmaxregister_*` (Footprint).

## Minimal runnable example

The role branch of the KEEP kernel with the operator's math removed (CuTe DSL 4.8.0, `sm_120a`). The values are the
only ones measured to launch with a producer `.dec` in this CTA shape: producer 24, consumer `.inc` to exactly the
launch value (a no-op for the consumer). Correctness of this form: `notes/d21_smoke_probe.py`, `PROBE PASS
max_diff=0.003407` (randn / shifted inputs, two shapes, job 1794048954).

```python
import cutlass
import cutlass.cute as cute

NCW = 8                    # consumer warps (warpgroups 0, 1)
NTHR = (NCW + 1) * 32      # + 1 producer warp = 288 threads; warp 8 is alone in warpgroup 2
REG_PRODUCER = 24          # multiple of 8 in [24, 256]
REG_CONSUMER = 168         # = the launch value; every value above it hung in this shape (Gotcha 1)

# host side: the launch needs a min-CTAs-per-SM attribute (.minnctapersm) next to the block size (.reqntid)
kernel_obj.launch(grid=grid, block=[NTHR, 1, 1], smem=smem_bytes,
                  min_blocks_per_mp=1)   # carried from sm90 warp_specialization gotcha 1; not re-checked on sm120

@cute.kernel
def kernel(...):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())   # warp-uniform predicate: setmaxnreg is .sync.aligned
    # shared prologue BEFORE the role branch: smem carve-up, mbarrier init + fence, sync_threads, tma_partition
    # (sm90 warp_specialization gotchas 5, 11: no CTA-wide barrier after the branch point, prologue under the entry quota)
    ...
    cute.arch.sync_threads()
    if warp == NCW:
        cute.arch.setmaxregister_decrease(REG_PRODUCER)          # first statement of the producer branch
        # ... TMA producer loop: wait empty -> arrive_and_expect_tx -> cute.copy(tma, ..., tma_bar_ptr=full) ...
    else:
        cute.arch.setmaxregister_increase(REG_CONSUMER)          # first statement of the consumer branch
        # ... consumer loop: wait full -> ldmatrix + mma.sync -> fence_view_async_shared -> release empty ...
```

## Readings (RTX PRO 5000; the CTA shape of the Status line)

Hang bisection, one rung per subprocess, 150 s timeout = hang (`notes/d21_regsplit_probe.py`, job 1794048427;
`min_blocks_per_mp=1` in every rung, `P_C` = producer `.dec` value / consumer `.inc` value, 0 = instruction absent):

| `P_C` | Result | ms / launch (5 launches, one shape) |
|---|---|---|
| `0_0` | `PROBE PASS` (completes) | 2.439 |
| `24_0` | `PROBE PASS` (completes) | 2.729 |
| `24_168` | `PROBE PASS` (completes) | 2.722 |
| `0_176` | `PROBE FAIL ... rc=timeout` | -- (compiled, never completed) |
| `24_176` | `PROBE FAIL ... rc=timeout` | -- |
| `24_184` | `PROBE FAIL ... rc=timeout` | -- |

Same-job A/B of `24_168` against the tree without `setmaxregister_*` (`notes/d21_smoke_probe.py`, job 1794048954):
`ab-4k head 2.6339 ms new 2.9146 ms new/head 1.1066`, `ab-16k head 9.9581 ms new 10.9947 ms new/head 1.1041`.

Shipped KEEP pair (driver NCU profile, geomean of two workloads):

| Metric | without split | with `24_168` (KEEP in job 1794048999) | split removed (KEEP in job 1794049160) |
|---|---|---|---|
| gate (device window) | 5.2769 ms | 5.7694 ms (+9.33 %) | 5.2881 ms (-8.34 %) |
| `regs_per_thread` | 168 | 168 | 168 |
| `spill_ld_sectors` / `spill_st_sectors` | 0 / 0 | 106,962,496 / 96,600,064 | 0 / 0 |
| `tensor_pct` | 84.84 | 74.8 | 84.86 |
| `bankconf_ld` | 1,527,758 | 14,888,785 | 1,527,545 |

**Load-bearing condition**: worth **-9.3 %** (a loss) in a kernel whose consumer already fit at 168 registers with
0 spills, at 1 CTA / SM, where the hang held the consumer at its launch value, so only the producer's `.dec` was
live. That is one reading under these conditions, not a price of `setmaxnreg` in general.

## Footprint (dossier `rings` axis)

Both directions, verbatim (census of the KEEP in job 1794048999, then of the KEEP in job 1794049160):

| Tree | `rings` |
|---|---|
| without split | `["stride=0x180", "stride=0x4,wrap=mask", "stride=0x5,wrap=mask"]` |
| with `24_168` | `["stride=0x18024,wrap=mask", "stride=0x4,wrap=mask", "stride=0x5,wrap=mask"]` |
| split removed | `["stride=0x180", "stride=0x4,wrap=mask", "stride=0x5,wrap=mask"]` |

In the same KEEPs `loops` (copy-producer 3 -> 4 -> 3), `mbar_objs` (19 -> 17 -> 19), `mbar_tiers` (`THREAD 1` gone and
back), `mbar_per_role` and `edges` (`copy-producer -> spin` gone and back) moved together with it, and `cache_ops`
(`UTMALDG.2D`, `UTMASTG.2D`) and `commit` did not. The source of the mbarrier ring was not touched, so this is
register allocation moving the address chain ([`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotchas 12,
25), not a ring change. Read it as "a `setmaxnreg` was inserted or removed", never as a ring-structure suspect.

## Gotchas

1. **In a CTA whose producer is a lone warp, any consumer `.inc` above the launch value hung, even when the pool
   arithmetic fits.** `24_184` asks 16 x 256 = 4,096 registers for the consumers against 144 x 32 = 4,608 released by
   the producer, and still never completed; `24_176` (2,048 asked) hung too. `0_176` (nobody releases) hanging is the
   expected `.inc`-waits-for-the-pool behaviour (sm90 warp_specialization gotcha 4) and shows the instruction is live
   on sm120. A hang shows as a timeout with no error line, after a clean compile -- probe it one rung per subprocess
   with a timeout ([`gpu_probe_ladder`](gpu_probe_ladder.md) §1). The probe did not isolate the cause. A candidate,
   **[pending certification]**: `setmaxnreg` is `.sync.aligned` over a warpgroup, and warp 8 here is 1 of 4 warps in
   its warpgroup, so its `.dec` may not return quota the way a whole-warpgroup `.dec` does. Untested: the same split
   with a producer that fills a whole warpgroup (a 384-thread CTA).
2. **The producer's `.dec` alone cost about 10 %, and the spills it brought were large.** `24_0` (no consumer
   instruction) was as slow as `24_168`, and `0_0` (attribute present, no `setmaxnreg`) was the fastest rung, so
   within the probe the cost follows the `.dec`, not the `min_blocks_per_mp=1` attribute (the attribute alone was not
   paired against its absence in one job). In the KEEP profile `regs_per_thread`
   stayed 168 while ~10^8 local sectors each way appeared and `tensor_pct` fell 10 points. Which role's code spilled
   was not attributed; the sm90 causes to check first are a live range straddling the branch (gotcha 6) and the
   prologue falling under the reduced quota (gotcha 11) of
   [`../sm90/warp_specialization.md`](../sm90/warp_specialization.md). Before keeping a split, read `spill_*` in the
   profile: a non-zero reading cancels whatever the split was meant to buy.
3. **Price it against the trigger, not against the sm90 habit.** The split exists to give a register-starved
   consumer more quota. With the consumer at 0 spills below the cap there is no gain available, and in Gotcha 1's
   CTA shape no gain is reachable anyway: the only launchable form is a pure producer `.dec`.
4. **`regs_per_thread` does not show the split.** NCU reports the launch value (168 before, with and after), so a
   profile cannot tell whether a split took effect. Use the time and `spill_*` readings, a deliberate `.inc` with no
   matching `.dec` as a liveness control (it must hang, rung `0_176`), or the `rings` footprint above.
5. **The `rings` value change is register allocation moving with the split** (Footprint). Removing the split put back
   the exact census of the tree without it, which makes it a reliable marker of the split's presence and an
   unreliable marker of anything else.
6. **Carried from sm90 without an sm120 check**: `min_blocks_per_mp=1` (`nvvm.minctasm` -> `.minnctapersm`) being
   required for ptxas to keep `setmaxnreg` (sm90 warp_specialization gotcha 1); every rung above had it, so its
   absence was not measured on sm120. Keep it: the rung that had it and no `setmaxnreg` was the fastest (Gotcha 2).

## Sources

- PTX ISA, `setmaxnreg` (`.inc` / `.dec`, `.sync.aligned`, value a multiple of 8 in [24, 256]):
  https://docs.nvidia.com/cuda/parallel-thread-execution/
- CuTe DSL API symbols: `cute.arch.setmaxregister_decrease`, `cute.arch.setmaxregister_increase`,
  `cute.arch.make_warp_uniform`, `cute.arch.warp_idx`, the `launch(..., min_blocks_per_mp=...)` option
  (`LaunchConfig`).
- Hang bisection: probe `notes/d21_regsplit_probe.py`, job 1794048427, 1 round: `PROBE PASS max_diff=n/a  # P_C=0_0
  rc=0`, `# P_C=24_0 rc=0`, `# P_C=24_168 rc=0`; `PROBE FAIL max_diff=n/a  # P_C=0_176 rc=timeout`, `# P_C=24_176
  rc=timeout`, `# P_C=24_184 rc=timeout` (the FAIL lines are the hang readings of Gotcha 1); timing lines
  `FACT 0_0 completes 2.439 ms/launch`, `FACT 24_0 completes 2.729 ms/launch`, `FACT 24_168 completes 2.722 ms/launch`.
- Smoke + numerics + same-job A/B of the `24_168` form: probe `notes/d21_smoke_probe.py`, PROBE PASS after 2 rounds
  (round 1, at `24_184`: `PROBE FAIL max_diff=n/a  # smoke-s5 no verdict (rc=timeout)`), winning job 1794048954:
  `PROBE PASS max_diff=0.003407  # num-small`, `PROBE PASS max_diff=0.002709  # num-16k`, A/B lines quoted in Readings.
- KEEP certification, split inserted: KEEP in job 1794048999 (NCU census job 1794049092), axis `rings`
  `["stride=0x180", "stride=0x4,wrap=mask", "stride=0x5,wrap=mask"]` -> `["stride=0x18024,wrap=mask",
  "stride=0x4,wrap=mask", "stride=0x5,wrap=mask"]`, gate 5.2769 -> 5.7694 ms.
- KEEP certification, split removed: KEEP in job 1794049160 (NCU census job 1794049253), axis `rings` back to
  `["stride=0x180", "stride=0x4,wrap=mask", "stride=0x5,wrap=mask"]`, gate 5.7694 -> 5.2881 ms.
- Detailed provenance archive: docs/kb-provenance/setmaxnreg_role_split.md (for human review; not in any agent read set).
