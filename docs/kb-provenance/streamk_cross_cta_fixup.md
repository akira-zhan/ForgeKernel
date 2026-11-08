# Provenance: hw_kb/cutedsl/sm120/streamk_cross_cta_fixup.md (+ ptx_sass_census_criteria.md gotcha 26)

For human review only; not in any agent read set.

## Run

- Run `sm120_fp8_decode_gemm-g2_261108-022350-5048` (2026-11-08), operator `sm120_fp8_decode_gemm-g2` (FP8 e4m3
  decode GEMM with per-row / per-column f32 scales, bf16 out; workloads M = 1 / 16 / 64 at N 14336, K 4096 and
  M = 16 at N 4096, K 14336; RTX PRO 5000, 110 SMs).
- Harvest trigger: contract-axis change (`rings`, `commit`) on the KEEP of decision D0.1+D0.3+D0.4+D0.5+D0.6+D0.7+D0.8+D0.9
  (iter 2, sha 6f296347c6, closes M0-gate). No expensive probe battle in this run (threshold 6 rounds).

## Narrative

1. iter 0 (sha cf94e9e01f): one CTA per N tile, grid (1, N/128, 1) = 112 CTAs on 110 SMs for the up shapes,
   1 CTA / SM (82944 B smem). KEEP at 0.0597 ms (vs_ref_device 0.9998). Census: `rings []`,
   `commit CPWAIT {LE 0x1 x4}`.
2. Lever pricing on that tree (jobs 1794110033, 1794111935, 1794112259, 1794112592): ablation `no_tail_wave` +15.24 %
   (NCU clock, correctness fail) -- the tail wave was priced; listed levers were null or harmful.
3. Meta diagnosis R1 @ D0.3 (decomposition); milestone patched to stream-K.
4. iter 2: probe `notes/streamk_fixup_probe.py` (commit a4e8130, job 1794113745) passed 24/24 in its first round.
   D0.x bundle committed as 6f296347c6 and KEPT: eval job 1794113794, census job 1794113821, watermark confirmation
   job 1794113839; gate 0.0597 -> 0.0548 ms (-8.21 %); paired confirmations 1794113866 / 1794113893 / 1794113920,
   median same-job ratio 1.0884; correctness monitor OK over 10 profiles.
5. The two kernel bulk-store waits (`wait_group(EPI_B - 1, read=True)` per sub-tile, `wait_group(0, read=True)` after
   the outer loop) are byte-identical in both trees; the commit-axis `LE 0x0 x1` term appears only in the iter-2
   census.

## What was not established

- Why the iter-0 census did not count the final drain in a consumer region while the iter-2 census did (region
  partition after ptxas scheduling; no raw-SASS diff was taken).
- Which element of the fixup's ordering chain is individually necessary (no ablation probe).
- How much of the -8.2 % comes from removing the tail wave vs better DRAM streaming on the K-heavy shape
  (32 tiles x 112 k-tiles moved only -4.5 % despite 32 -> 110 active CTAs).
