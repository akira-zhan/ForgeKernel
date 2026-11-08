# sm120_fp8_decode_gemm

FP8 GEMM with per-row (activation) and per-output-channel (weight) scales, at decode batch sizes: a linear layer of
a quantized LLM during token generation. The weight read dominates the time.

## Signature

- inputs, in order:
  - `a`: `[m, k]`, `torch.float8_e4m3fn` -- activations, K-major (row-major)
  - `b`: `[n, k]`, `torch.float8_e4m3fn` -- weights, K-major (one output channel per row)
  - `scale_a`: `[m, 1]`, float32 -- per-row activation scale, positive
  - `scale_b`: `[n, 1]`, float32 -- per-output-channel weight scale, positive
- output:
  - `y`: `[m, n]`, bfloat16

## Semantics

`y[i, j] = scale_a[i] * scale_b[j] * sum_k a[i, k] * b[j, k]`, products of the e4m3 values accumulated in float32
(at least), the scales applied to the float32 sum, rounded to bfloat16 once.

## Legal inputs

e4m3 values of quantized activations and weights (finite, |x| <= 448; typically roughly normal with a scale chosen so
the row max maps near 448); scales positive finite float32. `m` in 1..128, `n` and `k` multiples of 128.
