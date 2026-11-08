# sm120_nvfp4_decode_gemm

NVFP4 block-scaled GEMM at decode batch sizes, `y = (A x SA) . (B x SB)^T`: an FP4-quantized linear layer during
token generation. The weight read dominates the time.

## Signature

Axes: `m, n, k` plus the derived `kh = k/2` (packed bytes per row), `ks = k/16` (scales per row) and
`mp = ceil(m/128)*128` (scale rows of A, padded) -- given per shape in shapes.json.


- inputs, in order:
  - `a`: `[m, kh]`, `torch.float4_e2m1fn_x2` -- activations, K-major, two e2m1 values per byte (element `2i` in the
    low nibble)
  - `sa`: `[mp, ks]`, `torch.float8_e4m3fn` with `mp = ceil(m/128)*128` -- one positive scale per 16 consecutive K
    elements of a row of A, in the 128x4 blocked layout below; rows `m..mp-1` are padding and must not affect `y`
  - `b`: `[n, kh]`, `torch.float4_e2m1fn_x2` -- weights, K-major, packed like `a`
  - `sb`: `[n, ks]`, `torch.float8_e4m3fn` -- scales of B, same layout
- output:
  - `y`: `[m, n]`, bfloat16

## Semantics

`y[i, j] = sum_k e2m1(a[i, k]) * sa[i, k // 16] * e2m1(b[j, k]) * sb[j, k // 16]`, accumulated in float32 (at
least), rounded to bfloat16 once. e2m1 values: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a sign bit (bit 3).

Blocked scale layout (cuBLASLt / CUTLASS "128x4"): the logical scale `s[r, c]` (row `r`, K-block `c`, `C = k/16`
columns) is stored at byte offset
`((r // 128) * (C // 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)`.

## Legal inputs

`m` in 1..128, `n` a multiple of 128, `k` a multiple of 64. Any finite e2m1 codes; scales positive finite e4m3 values
(quantization scales, roughly 2^-9 .. 2^8).
