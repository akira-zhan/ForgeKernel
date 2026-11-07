# Out-of-run gap queue (recon-table rows this run did not verify: sm120_nvfp4_gemm)

Rows with VERIFY `none` have no machine-read acceptance; rows that say "may not be expressible" without an `[EXPR:]` ladder verdict are unverified guesses; rows with VERIFY `inferred` are structural readings whose discriminating observation gave no positive reading (the two readings are byte-identical). None of the three may be used as a conclusion. These rows are the recon's reading of the reference's structure (narrated provenance) and **do not enter hw_kb**: only a probe-certified recipe carries knowledge across runs; to verify one, start a separate probe, with the evidence under `docs/kb-provenance/`.

- **A14** = wait alignment — VERIFY `none`; notes: the hiding window is ~24 OMMAs per warp (no machine-read verification in this run)
- **A15** = tile scheduler — VERIFY `none`; notes: `[EXPR: A=cute.arch.issue_clc_query(mbar_ptr, clc_response_ptr, multicast=True) / cutlass.utils.ClcDynamicPersistentTileScheduler (sig r2 log:843-848; lowering for sm_120a not measured); B=not measured; C=->]`. Unpriced. (no machine-read verification in this run)
- **A20** = descriptor prefetch — VERIFY `none`; notes: `[EXPR: A=not measured; B=not measured; C=->]`; ≤1 µs effect expected, unpriced (no machine-read verification in this run)
