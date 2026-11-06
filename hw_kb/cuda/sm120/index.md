# SM120 (RTX Blackwell, `sm_120a`) interface index — CUDA mode

The sm120 tree of CUDA mode (single-file CUDA C++ student). It is fully separate from `hw_kb/cutedsl/sm120/`: entries there teach the `cute.*` spelling and are not copied over as they are. Demand-driven, exactly as `sm100/`: a recipe is
written when a student is blocked on that interface, follows `../TEMPLATE.md` (examples are CUDA C++ / CUTLASS (CuTe C++) / inline PTX), and is marked
✅ **only** on one of the two evidence classes below. This file is not a plan; it is the
ledger of what has been earned on this architecture.

Two classes earn a ✅, and nothing else does. **A GPU probe returning `PROBE PASS`** on an
sm_120 device is the default. The second is for **numerically invisible** knobs (cache-policy
bits, wait depth, issue order): **a shipped KEEP whose own NCU census shows the contract axis
moved**, cited with the before/after axis readings and the gate window. An entry carrying
neither does not belong in this file.

## What this architecture is (read before borrowing from another tree)

SM120 is **not a smaller SM100**. Its tensor cores are driven by **warp-level `mma.sync`**
only: no `wgmma` (SM90) and no `tcgen05` / TMEM / UMMA / `cta_group::2` (SM100), so the
accumulator lives in registers and is waited on through the scoreboard. In SASS the MMA
families are `HMMA` (f16 / bf16 / tf32), `QMMA` (`kind::f8f6f4`; `QMMA.SF` = the
`mxf8f6f4` block-scaled form) and `OMMA` (`OMMA.SF` = the `mxf4` / `mxf4nvf4` block-scaled
form). Readings and conclusions in `sm100/` (and in the CuTe DSL trees) were taken on other hardware:
**they are hypotheses here until a probe on this architecture re-earns them**, and an
entry transcribed from another tree carries a `Certification source` line saying where its
readings were taken.

## Find by symptom first (this section is the entry point, not a table of contents)

| What you see / what you are about to do | Look first at |
|---|---|
| **Pricing a plan on the RTX PRO 5000** -- a roofline or ridge point, a "% of peak" reading of an NCU profile, choosing bf16 vs FP8 vs block-scaled FP4 by rate, warps per SM to hide `mma.sync` latency, whether a working set stays in the 96 MiB L2 | [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md): measured per-kind `mma.sync` rate and latency, smem / L2 / DRAM bandwidth and latency, launch cost |
| **The FP4 kernel runs no faster than FP8** -- e2m1 operands fed through `kind::f8f6f4` (`QMMA.16832...E2M1`) | [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md) tensor-core table: only the block-scaled `OMMA.SF.16864` forms (`kind::mxf4` / `kind::mxf4nvf4`) run at 2x the FP8 rate |
| **A tensor-core mainloop stalls on shared-memory reads, not on the MMA** -- fragments re-loaded from smem for every `mma.sync` | [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md) "Shared memory cannot feed `mma.sync` without register reuse": 768 B per mma vs 2048 B per 16 clk per SM |

## Interfaces

| Interface | Purpose (one line) | Status |
|------|---------------|------|
| [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md) | Measured budget of the RTX PRO 5000 (GB202, 110 SMs): `mma.sync` rate/latency per kind, FFMA/MUFU, smem/L2/DRAM bandwidth and latency, launch cost | ✅ |

Status: ⬜ TODO / 🟡 draft / ✅ Verified (the example runs on sm_120 and its source is cited).
