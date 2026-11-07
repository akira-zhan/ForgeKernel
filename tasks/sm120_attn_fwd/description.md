# sm120_attn_fwd

Causal grouped-query attention forward, as in the prefill of a decoder-only transformer.

## Signature

- inputs, in order:
  - `q`: `[b, hq, s, d]`, bfloat16 -- queries
  - `k`: `[b, hk, s, d]`, bfloat16 -- keys; `hq` is a multiple of `hk`, query head `h` uses key/value head `h // (hq / hk)`
  - `v`: `[b, hk, s, d]`, bfloat16 -- values
- output:
  - `o`: `[b, hq, s, d]`, bfloat16

## Semantics

For every batch `i`, query head `h` (kv head `g = h // (hq // hk)`) and query row `t`:
`o[i, h, t, :] = sum_{u <= t} softmax_u( q[i, h, t, :] . k[i, g, u, :] / sqrt(d) ) * v[i, g, u, :]` -- causal
(keys after the query position are masked), softmax over `u` in float32 (at least), result rounded to bfloat16 once.

## Legal inputs

Finite bfloat16 values of activation scale (roughly unit variance; occasional outliers several times larger).
`s` a multiple of 128, `d = 128`.
