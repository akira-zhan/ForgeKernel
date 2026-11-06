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
| **About to write a GPU probe** -- how to set the contracts, isolate rungs, survive a hang or a sticky CUDA error, read a DSL error, or trust a timing reading | [`gpu_probe_ladder`](gpu_probe_ladder.md); for a rate / latency reading, its §6 (the SASS of the timed loop is the contract) |
| **Kernel died with `operation not supported on global/shared address space`, `illegal memory access` or `unspecified launch failure`, and no source line** | [`gpu_probe_ladder`](gpu_probe_ladder.md) §1 (one rung per process: the error is sticky for the rest of the process) and §5 (sanitizer only on the lowest failing rung) |
| **Suspicious shared memory traffic** / bank conflicts / `smem_wavefronts` over the reference | [`smem_traffic_triage`](smem_traffic_triage.md): conflict-type vs volume-type; on sm120 the bank-conflict counter equals the wavefront excess of a conflict |
| **Want to split a register reduction chain into several accumulators / a tree**, or a kernel ported from sm100 relies on `.f32x2` (`FFMA2`) throughput, or bf16 widening (`SHF`/`LOP3`) outnumbers FP in SASS | [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md): 2 chains cover latency at 1 warp/SMSP, fewest instructions win above; `.f32x2` is two scalar issues; `fma.rn.f32.bf16` |
| **A probe's or kernel's SASS shows `UFFMA` / `ULOP3` / `USHF` where vector ops were expected** | [`gpu_probe_ladder`](gpu_probe_ladder.md) timing traps; [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md) gotcha 5 (inputs provably uniform across the warp) |
| **Not sure which compiled instance actually ran** (grid / cluster / stage count / dynamic smem), a dispatch cache may hand back the wrong executable, or a port from sm100 asks for more than 99 KiB of smem and dies at launch with `cudaErrorInvalidValue` / `invalid argument` | [`launch_identity_stamp`](launch_identity_stamp.md): the kernel stamps its own identity; 101376 B is the per-block ceiling |

## Interfaces

| Interface | Purpose (one line) | Status |
|------|---------------|------|
| [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md) | Measured budget of the RTX PRO 5000 (GB202, 110 SMs): `mma.sync` rate/latency per kind, FFMA/MUFU, smem/L2/DRAM bandwidth and latency, launch cost | ✅ |
| [`gpu_probe_ladder`](gpu_probe_ladder.md) | **Interpretation recipe**: instrument engineering of multi-contract GPU probes -- one rung per process, self-contained digest, fact/verdict split, candidate sweep, two-marginal bijection, timing readings; sm120 re-checks of the CUDA-level and DSL-error rules | ✅ |
| [`smem_traffic_triage`](smem_traffic_triage.md) | **Interpretation recipe**: smem traffic over the line -> conflict-type (bankconf over too) vs volume-type (bankconf clean); counters verified on GB202 | ✅ |
| [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md) | Register reduction chain on sm120: latency-bound at 1 warp/SMSP (2 chains suffice), issue-bound above; `.f32x2` = two scalar ops; full tree = N/2-way in SASS; `fma.rn.f32.bf16` cuts the widening | ✅ |
| [`launch_identity_stamp`](launch_identity_stamp.md) | **Instrument**: the kernel writes trace-time constants + launch special registers (`%nctaid`, `%cluster_nctaid` vs `%nclusterid`, `%dynamic_smem_size`) into a stamp table checked cell by cell; sm120 smem ceiling behaviour | ✅ |

Status: ⬜ TODO / 🟡 draft / ✅ Verified (the example runs on sm_120 and its source is cited).
