# SM120 (RTX Blackwell, `sm_120a`) interface index — CuTe DSL mode

The sm120 tree of CuTe DSL mode. Demand-driven, exactly as `sm90/` and `sm100/`: a recipe is
written when a student is blocked on that interface, follows `../TEMPLATE.md`, and is marked
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
form). Readings and conclusions in `sm90/` and `sm100/` were taken on other hardware:
**they are hypotheses here until a probe on this architecture re-earns them**, and an
entry transcribed from another tree carries a `Certification source` line saying where its
readings were taken.

## Find by symptom first (this section is the entry point, not a table of contents)

| What you see / what you are about to do | Look first at |
|---|---|
| **Pricing a plan on the RTX PRO 5000** -- a roofline or ridge point, a "% of peak" reading of an NCU profile, choosing bf16 vs FP8 vs block-scaled FP4 by rate, warps per SM to hide `mma.sync` latency, whether a working set stays in the 96 MiB L2 | [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md): measured per-kind `mma.sync` rate and latency, smem / L2 / DRAM bandwidth and latency, launch cost |
| **The FP4 kernel runs no faster than FP8** -- e2m1 operands fed through `kind::f8f6f4` (`QMMA.16832...E2M1`) | [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md) tensor-core table: only the block-scaled `OMMA.SF.16864` forms (`kind::mxf4` / `kind::mxf4nvf4`) run at 2x the FP8 rate |
| **A tensor-core mainloop stalls on shared-memory reads, not on the MMA** -- fragments re-loaded from smem for every `mma.sync` | [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md) "Shared memory cannot feed `mma.sync` without register reuse": 768 B per mma vs 2048 B per 16 clk per SM |
| **About to write a GPU probe** -- how to set the contracts, isolate rungs, survive a hang or a sticky CUDA error, read a DSL error, or trust a timing reading | [`gpu_probe_ladder`](gpu_probe_ladder.md); for a rate / latency reading, its §7 (the SASS of the timed loop is the contract) |
| **Kernel died with `operation not supported on global/shared address space`, `illegal memory access` or `unspecified launch failure`, and no source line** | [`gpu_probe_ladder`](gpu_probe_ladder.md) §1 (one rung per process: the error is sticky for the rest of the process) and §5 (sanitizer only on the lowest failing rung) |
| **Suspicious shared memory traffic** / bank conflicts / `smem_wavefronts` over the reference | [`smem_traffic_triage`](smem_traffic_triage.md): conflict-type vs volume-type; on sm120 the bank-conflict counter equals the wavefront excess of a conflict |
| **Want to split a register reduction chain into several accumulators / a tree**, or a kernel ported from sm100 relies on `.f32x2` (`FFMA2`) throughput, or bf16 widening (`SHF`/`LOP3`) outnumbers FP in SASS | [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md): 2 chains cover latency at 1 warp/SMSP, fewest instructions win above; `.f32x2` is two scalar issues; `fma.rn.f32.bf16` |
| **A probe's or kernel's SASS shows `UFFMA` / `ULOP3` / `USHF` where vector ops were expected** | [`gpu_probe_ladder`](gpu_probe_ladder.md) timing traps; [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md) gotcha 5 (inputs provably uniform across the warp) |
| **Not sure which compiled instance actually ran** (grid / cluster / stage count / dynamic smem), a dispatch cache may hand back the wrong executable, or a port from sm100 asks for more than 99 KiB of smem and dies at launch with `cudaErrorInvalidValue` / `invalid argument` | [`launch_identity_stamp`](launch_identity_stamp.md): the kernel stamps its own identity; 101376 B is the per-block ceiling |
| **Trace-time error in a helper** (`PHASE_DYNAMIC_TO_STATIC_BOOL`, `SCOPE_CLOSURE_CAPTURE`, `range_constexpr should be preprocessed`, `TYPE_DYNAMIC_EXPR_UNSUPPORTED`), or a `Constexpr` that works in the shipped entry but not in a probe wrapper, or `module 'cutlass.cute' has no attribute ...` | [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) (decorator *syntax* is required: `cute.jit(f)` as a call does not rewrite `f`) |
| **`NameError: name 'x' is not defined` from a name bound inside an `if`**, a compile-time switch whose both branches show up in SASS, `break`/`return` rejected in a region, or `PHASE_DYNAMIC_INDEX` on a Python list/tuple | [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md) |
| **A PTX instruction or modifier the DSL op does not expose** (`.L2::256B`, `cp.async` commit/wait, an unexported `cvt`), or a modifier that "should be there" but is absent from SASS, or no cubin appears in the dump directory | [`ptx_escape`](ptx_escape.md) (rung A before asm; `CUTE_DSL_KEEP=cubin` + an existing dump dir + `CUTE_DSL_NO_CACHE=1`) |
| **Writing an FP8 (e4m3 / e5m2) mainloop**, or an FP8 kernel's SASS feeds `QMMA` from scalar `LDS.U8`/`LDS.U16`, or `bankconf_ld` is high in a tensor-core loop | [`warp_mma_fp8_ldsm`](warp_mma_fp8_ldsm.md): `MmaFP8Op` (16,8,32) + `LdMatrix8x8x16bOp` x4 from a swizzled K-major tile; expected footprint table |
| **A PTX/SASS count contract fails while the numerics pass**, or a census shows scalar loads / zero counts you cannot explain | [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md): histogram first, family + qualifier, failure form apart, PTX vs SASS, drop `@!PT`, executed vs static counts |
| **`smem_wavefronts_st` / `bankconf_st` read 0 in a kernel that clearly fills smem** (cp.async / `LDGSTS` supply) | [`smem_traffic_triage`](smem_traffic_triage.md) "The blind spot on sm120": read `sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts` per `LDGSTS` |
| **A GEMM-shaped mainloop at ~70 % of the `mma.sync` peak with ~4 active warps per SM**, or choosing the stage count / K bytes per stage / warps per CTA of a cp.async pipeline | [`cpasync_multistage_pipeline`](cpasync_multistage_pipeline.md): CTAs per SM (smem <= ~49 KiB per CTA for two) moved it 12-13 %, the stage count did not; more warps via smaller warp tiles or `min_blocks_per_mp` (register cap -> spills) lost |
| **The SASS / orchestration tools say "too few instructions" or give no dossier on an sm120 tensor kernel**, or the commit axis does not change when the stage count does | [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotchas 21-22 (200-instruction floor for `mma.sync` pages; `CPWAIT` = cp.async wait depth) |
| **Supplying a GEMM / attention mainloop with TMA**, choosing an in-loop producer vs a dedicated producer warp, or a warp-specialized CTA that unexpectedly runs 1 CTA / SM | [`tma_mbarrier_mainloop`](tma_mbarrier_mainloop.md): in-loop TMA beat cp.async (+2 % e4m3, +6 % bf16); a 5-warp CTA lost its second CTA per SM |
| **The dossier reads `mbar_objs = 0` / every mbarrier site UNRESOLVED on a page that clearly has `SYNCS` waits** | [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) gotcha 23 (follow the `LEA` addend) |

## Interfaces

| Interface | Purpose (one line) | Status |
|------|---------------|------|
| [`device_facts_rtx_pro_5000`](device_facts_rtx_pro_5000.md) | Measured budget of the RTX PRO 5000 (GB202, 110 SMs): `mma.sync` rate/latency per kind, FFMA/MUFU, smem/L2/DRAM bandwidth and latency, launch cost | ✅ |
| [`gpu_probe_ladder`](gpu_probe_ladder.md) | **Interpretation recipe**: instrument engineering of multi-contract GPU probes -- one rung per process, self-contained digest, fact/verdict split, candidate sweep, two-marginal bijection, timing readings; sm120 re-checks of the CUDA-level and DSL-error rules | ✅ |
| [`smem_traffic_triage`](smem_traffic_triage.md) | **Interpretation recipe**: smem traffic over the line -> conflict-type (bankconf over too) vs volume-type (bankconf clean); counters verified on GB202 | ✅ |
| [`reg_chain_latency_vs_issue`](reg_chain_latency_vs_issue.md) | Register reduction chain on sm120: latency-bound at 1 warp/SMSP (2 chains suffice), issue-bound above; `.f32x2` = two scalar ops; full tree = N/2-way in SASS; `fma.rn.f32.bf16` cuts the widening | ✅ |
| [`launch_identity_stamp`](launch_identity_stamp.md) | **Instrument**: the kernel writes trace-time constants + launch special registers (`%nctaid`, `%cluster_nctaid` vs `%nclusterid`, `%dynamic_smem_size`) into a stamp table checked cell by cell; sm120 smem ceiling behaviour | ✅ |
| [`cute_dsl_trace_scope`](cute_dsl_trace_scope.md) | Trace-scope rules of CuTe DSL 4.8.0: `@cute.jit` on every helper with runtime control flow (decorator syntax, not a call), explicit arguments instead of closure capture, Constexpr transcription, API drift, `range_constexpr` needs the preprocessor, no subscript stores through object attributes in staged regions | ✅ |
| [`cute_dsl_const_expr_if`](cute_dsl_const_expr_if.md) | Compile-time branches: `cutlass.const_expr` folds an `if` into Python so bindings escape; bare `if` is a runtime region whose new names do not escape; folded branches land nothing | ✅ |
| [`ptx_escape`](ptx_escape.md) | `llvm.inline_asm` inside `@dsl_user_op` (no operands / address operands / a return value), the A -> B -> C expressibility ladder, the SASS footprint check with DSL 4.8.0's artifact switches | ✅ |
| [`warp_mma_fp8_ldsm`](warp_mma_fp8_ldsm.md) | FP8 `mma.sync` on sm120: cp.async -> swizzled K-major smem -> `ldmatrix` (`LDSM.16.M88.4`) -> `QMMA.16832`; census, swizzle vs linear bank conflicts, the cp.async fill counter blind spot | ✅ |
| [`ptx_sass_census_criteria`](ptx_sass_census_criteria.md) | **Interpretation recipe**: writing PTX/SASS footprint criteria that do not mistake an instrument fault for a kernel defect; sm120 worked example on a `mma.sync` tile | ✅ |
| [`cpasync_multistage_pipeline`](cpasync_multistage_pipeline.md) | The sm120 mainloop: cp.async into S stages, `wait_group(S-2)` + `__syncthreads`, ldmatrix + `mma.sync`; measured bf16 / e4m3 throughput vs stages, K bytes per stage, CTAs per SM | ✅ |
| [`tma_mbarrier_mainloop`](tma_mbarrier_mainloop.md) | TMA (`make_tiled_tma_atom` + `tma_partition`) into a full / empty mbarrier ring (`PipelineTmaAsync`) feeding ldmatrix + `mma.sync`; in-loop producer vs producer warp, measured | ✅ |

Status: ⬜ TODO / 🟡 draft / ✅ Verified (the example runs on sm_120 and its source is cited).
