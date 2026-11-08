# Out-of-run gap queue (recon-table rows this run did not verify: sm120_fp8_decode_gemm)

Rows with VERIFY `none` have no machine-read acceptance; rows that say "may not be expressible" without an `[EXPR:]` ladder verdict are unverified guesses; rows with VERIFY `inferred` are structural readings whose discriminating observation gave no positive reading (the two readings are byte-identical). None of the three may be used as a conclusion. These rows are the recon's reading of the reference's structure (narrated provenance) and **do not enter hw_kb**: only a probe-certified recipe carries knowledge across runs; to verify one, start a separate probe, with the evidence under `docs/kb-provenance/`.

- **Element-wise inside the K loop** = none (only `UIADD3 URZ` filler + stage-index arithmetic). No in-loop scaling — VERIFY `none` (no machine-read verification in this run)
