# Provenance: hw_kb/cutedsl/sm120/setmaxnreg_role_split.md (+ ptx_sass_census_criteria.md gotcha 25)

For human review only; not in any agent read set.

## Run

- Run `sm120_attn_fwd_261107-073131-f10f` (2026-11-07), operator `sm120_attn_fwd` (causal bf16 attention forward,
  288-thread CTA: 8 `mma.sync` consumer warps + 1 TMA producer warp, 1 CTA / SM).
- Harvest trigger: contract-axis change (`rings`) on two KEEPs of milestone decision D2.1 (register split between roles).

## Narrative

1. iter 7, D2.1, the student's first attempt was producer 24 / consumer 184 (pool arithmetic: 4,096 asked <= 4,608
   released). Smoke probe round 1 (`notes/d21_smoke_probe.py`, job 1794048037): `smoke-s5 rc=timeout` (300 s), so
   every other rung was skipped.
2. Hang bisection `notes/d21_regsplit_probe.py` (commit c38c6c8, job 1794048427): 0_0, 24_0, 24_168 complete;
   0_176, 24_176, 24_184 time out.
3. The kernel was set to 24 / 168 (commit b71d940); smoke probe round 2 (job 1794048954) passed 6/6, with a same-job
   A/B of x1.1066 / x1.1041 against the tree without the split (`notes/kernel_head_d22.py`).
4. D2.1 was committed as efec17d4f4 and KEPT (eval job 1794048999, NCU job 1794049092): gate 5.2769 -> 5.7694 ms,
   spill 0/0 -> 106,962,496 / 96,600,064 sectors, tensor_pct 84.84 -> 74.8. The milestone gate admitted it as a KEEP
   despite the regression.
5. iter 8: the student re-ruled D2.1 "not triggered" (spill 0 and regs below the 224 cap on the D2.2 tree) and
   restored kernel.py byte for byte to 986bece (commit 892bb42c10, eval job 1794049160, NCU job 1794049253):
   gate 5.7694 -> 5.2881 ms, spills 0/0, the `rings` axis back to its pre-split reading.

## What was not established

- (Established on 2026-11-14, see below.) Why `.inc` above 168 hangs when the pool arithmetic fits.
- Which role's code spilled under the 24 / 168 split.
- Whether ptxas drops `setmaxnreg` on sm_120a without `min_blocks_per_mp` (every rung had the attribute).

## 2026-11-14: the quota rule, re-certified with a repository probe

The two probes above lived in the run's `notes/` and are gone. `probes/sm120/mechanisms/setmaxnreg_split_check.py`
re-earns the hang and separates its cause on a minimal TMA-fed `mma.sync` GEMM (RTX PRO 5000, GPU 3, CuTe DSL 4.8.0,
`min_blocks_per_mp=1`), one subprocess per rung, 90 s without a return = hang:

| CTA (consumer + producer-role warps) | compiled registers / thread | producer `.dec` / consumer `.inc` | outcome |
|---|---|---|---|
| 4 + 4 (256 threads) | 178 | none / 232 | done, exact |
| 4 + 4 | 178 | 40 / 232 | done, exact |
| 4 + 1 (160 threads) | 178 | 24 / 208 | done, exact |
| 8 + 1 (288 threads) | 118 | none / none | done, exact |
| 8 + 1 | 118 | 24 / 168 | done, exact |
| 8 + 1 | 118 | 24 / 176 | hang |
| 8 + 1 | 118 | none / 176 | hang |
| 8 + 4 (384 threads) | 118 | none / 176 | hang |
| 8 + 4 | 118 | none / 200 | hang |
| 8 + 4 | 118 | 40 / 200 | done, exact |
| 8 + 4 | 118 | 40 / 232 | done, exact |

(Seven of these rungs are the probe's contracts, `PROBE PASS setmaxnreg-split contracts=7`; the other four were run
once while narrowing the rule.)

What the table rules out: "a lone producer warp hangs" (4 + 1 at 24 / 208 completes), "the wall is the compiled
register count" (118 -> 168 completes), and "an `.inc` nobody pays for always hangs" (4 + 4 at none / 232 completes).
What fits every rung: the per-thread quota at launch is 65536 / (128 x warpgroups of the CTA) -- 256 for two
warpgroups, 170 for three -- and above it an `.inc` needs registers from a `.dec` executed by a whole warpgroup.
The 288-thread CTA of the original run launched at 168, which is that share rounded down to a multiple of 8, so
"above the launch value" and "above the share" were the same wall there.

Not established: what a split buys in time on a register-starved consumer; whether ptxas drops `setmaxnreg` without
`min_blocks_per_mp` (every rung had it).
