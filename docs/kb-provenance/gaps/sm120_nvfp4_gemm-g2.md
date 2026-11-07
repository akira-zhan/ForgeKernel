# Out-of-run gap queue (recon-table rows this run did not verify: sm120_nvfp4_gemm-g2)

Rows with VERIFY `none` have no machine-read acceptance; rows that say "may not be expressible" without an `[EXPR:]` ladder verdict are unverified guesses; rows with VERIFY `inferred` are structural readings whose discriminating observation gave no positive reading (the two readings are byte-identical). None of the three may be used as a conclusion. These rows are the recon's reading of the reference's structure (narrated provenance) and **do not enter hw_kb**: only a probe-certified recipe carries knowledge across runs; to verify one, start a separate probe, with the evidence under `docs/kb-provenance/`.

- **schedule** = static persistent: CTA b takes tiles b, b+110, …; tm = tile >> log2(tiles_n), tn = tile & (tiles_n−1); **linear row-major raster, no swizzle / group-M** — VERIFY `none` (no machine-read verification in this run)
- **live registers** = peak live 151 of 232; accumulators 64 per thread; non-accumulator working set ≈ 87 — VERIFY `none` (no machine-read verification in this run)
- **epilogue overlap** = the producer pre-fills k-tiles 0–1 of the next tile during the epilogue; the epilogue does **not** overlap MMA (same warps) — VERIFY `none` (no machine-read verification in this run)
