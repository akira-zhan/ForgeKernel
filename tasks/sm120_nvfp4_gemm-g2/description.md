# sm120_nvfp4_gemm

NVFP4 block-scaled GEMM, `y = (A x SA) . (B x SB)^T`, as used for FP4 linear layers on RTX Blackwell (sm_120a).

## Signature

- inputs, in order:
  - `a`: `[m, k/2]`, `torch.float4_e2m1fn_x2` -- A, K-major, two e2m1 values per byte (element `2i` in the low nibble)
  - `sa`: `[m, k/16]`, `torch.float8_e4m3fn` -- one positive scale per 16 consecutive K elements of a row of A, stored
    in the 128x4 blocked layout below (the tensor's shape is `[m, k/16]`, its memory is blocked)
  - `b`: `[n, k/2]`, `torch.float4_e2m1fn_x2` -- B, K-major, packed like `a`
  - `sb`: `[n, k/16]`, `torch.float8_e4m3fn` -- scales of B, same layout
- output:
  - `y`: `[m, n]`, bfloat16

## Semantics

`y[i, j] = sum_k e2m1(a[i, k]) * sa[i, k // 16] * e2m1(b[j, k]) * sb[j, k // 16]`, accumulated in float32 (at
least), rounded to bfloat16 once at the end. e2m1 values: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` with a sign bit (bit 3).

Blocked scale layout (cuBLASLt / CUTLASS "128x4"): the logical scale `s[r, c]` (row `r`, K-block `c`, `C = k/16`
columns) is stored at byte offset
`((r // 128) * (C // 4) + c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)`.

## Legal inputs

`m`, `n` multiples of 128, `k` a multiple of 64. Any finite e2m1 codes; scales are positive finite e4m3 values
(in practice powers of two and quantization scales between about 2^-9 and 2^8).
