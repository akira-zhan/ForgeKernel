# reg_chain_latency_vs_issue (register reduction chains on sm120: splitting accumulators vs cutting instructions; `.f32x2` lowers to two scalar ops; a full tree contracts to an N/2-way accumulation; `fma.rn.f32.bf16` saves the unpacking)

> One file per hardware interface. Operator-agnostic: mechanism only. CUDA mode: examples are CUDA C++ / CUTLASS (CuTe C++) / inline PTX.
>
> **Status**: ✅ Verified (GPU probe `probes/sm120/methodology/reg_chain_check.py`, `PROBE PASS reg-chain contracts=5`, RTX PRO 5000:
> numerics of all eight variants against an fp64 sum of squares, max relative error 6.84e-08; SASS contracts -- no
> `FFMA2/FMUL2/FADD2` anywhere, each PTX `.f32x2` op of a linear chain becomes exactly two scalar ops, nothing on the
> uniform datapath, the `fma.rn.f32.bf16` variant issues 2N `FHFMA.BF16` and 32 fewer integer (unpacking) ops).
>
> **Certified surface**: numerics and SASS counts. The cycle table is INFO readings from the same run (median over the
> warps of one launch after a warm-up launch), without a significance test; use it for direction.
>
> **Certification source**: re-measured on sm120. The sm100 entry of the same name measured packed `FFMA2/FADD2`
> chains; those instructions do not exist on sm_120a, so none of its cycle readings carry over.

## Purpose
A thread holds N bf16x2 words that have already landed; they must be widened to fp32, squared and reduced to one scalar
(before a `SHFL` / store). The cycles of this stretch are set by the **larger** of two things:

- **dependency depth x per-dependency latency** (latency-bound): a serial accumulation `A = fma(x, x, A)` over 2N values
  has depth 2N; split into k accumulators it is `2N/k + log2(k)`;
- **instructions of this stretch over all warps on this SMSP / issue rate** (issue-bound): each SMSP issues at most one
  warp instruction per cycle; widening (`SHF` / `LOP3` / `IMAD.U32`), FP, the tail tree and the feedback all take slots.

Splitting the accumulator only lowers the first term and adds tail-tree instructions, so it pays only with "few warps,
deep chain"; once the warps on this SMSP fill each other's dependency waits, the only price left is the instruction count.

## When needed (trigger)
- You want to shorten a "land -> widen -> square -> in-thread reduce" register chain with "multiple accumulators / a tree".
  **Estimate before building**: depth x ~4 clk vs (warps on this SMSP) x (SASS count of the stretch); if the second is
  larger, it is issue-bound and splitting will not be faster.
- A kernel ported from sm100 uses `.f32x2` (`add/mul/fma.rn.f32x2`) for throughput: on sm120 it buys nothing (see (a)).
- Widening instructions (2 integer ops per bf16x2 word) outnumber FP instructions: use `fma.rn.f32.bf16` (see (c)).

## Mechanism

### (a) No packed fp32 on sm120: `.f32x2` is two scalar instructions
`add/mul/fma.rn.f32x2` compile for `sm_120a`, but ptxas emits **two** `FADD/FMUL/FFMA` per PTX op (measured: a 16-op
packed chain -> 33 scalar FP ops including the final `add.rn.f32` of the two halves). Consequences:
- a "packed, 1 accumulator" chain is really **two independent scalar chains** (lo lanes, hi lanes) of depth N each;
- `.f32x2` saves no issue slot; a packed and a scalar spelling of the same k-way reduction issue the same FP count
  (packed 4 accumulators = scalar 8 accumulators = 24 FFMA + 8 FMUL + 7 FADD);
- ptxas does not reassociate, and contracts `mul -> add` into `FFMA`: a **full tree** written as N `mul.rn.f32x2` +
  N-1 `add.rn.f32x2` comes out **identical** to the 8-accumulator packed chain (16 FMUL + 16 FFMA + 15 FADD); write
  k ways directly if you want k ways.

### (b) The timing loop must be one asm block, and its inputs must not be provably uniform
To measure the latency of "land -> scalar", the timing loop (`%clock64` -> loop body -> `%clock64`) goes into **one**
inline-PTX block with `.pragma "nounroll";` at the loop head, and the next iteration's input depends on the previous
result (`t = bits(s) & Z`, Z a kernel parameter that is 0 at run time, `w_i ^= t`). Otherwise the loop is unrolled and the
loop-invariant widening hoisted, and you measure throughput, not latency. **On sm120 also load the words through a
per-lane address** (`win[(lane) * N + i]`, the host may replicate the same values): words the compiler can prove
identical across the warp send the whole chain to the uniform datapath (`UFFMA`, `ULOP3`, `USHF`) and the readings
describe a different pipe.

### (c) Readings (RTX PRO 5000, sm_120a, same run as the certification)
N = 16 bf16x2 words per thread (32 values). Cycles / iteration (median over warps, `REPS=1024`), one block per SM;
columns are warps resident per SMSP; SASS totals are for the whole kernel:

| Variant | Scalar FP ops (SASS) | Depth | SASS total | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|---|---|
| packed, 1 accumulator (= 2 scalar chains) | 33 | N | 144 | 128 | 212 | 375 | 651 |
| packed, 2 accumulators | 35 | N/2+1 | 144 | 130 | 217 | 397 | 684 |
| packed, 4 accumulators + tree | 39 | N/4+2 | 144 | 130 | 221 | 418 | 762 |
| packed, 8 accumulators + tree | 47 | N/8+3 | 152 | 137 | 230 | 454 | 821 |
| packed, full tree (= 8 accumulators in SASS) | 47 | N/8+3 | 152 | 140 | 228 | 448 | 804 |
| scalar fp32, 1 accumulator | 32 | 2N | 136 | **180** | 209 | **363** | **650** |
| scalar fp32, 8 accumulators + tree | 39 | 2N/8+3 | 144 | 129 | 214 | 420 | 738 |
| `fma.rn.f32.bf16`, 8 accumulators + tree | 32 `FHFMA.BF16` + 7 `FADD` | 2N/8+3 | 112 | **95** | **152** | **268** | **511** |

How to read this table:
- **1 warp / SMSP is latency-bound, and two chains are enough.** The depth-32 scalar chain costs 180 clk (about 4 clk
  per dependent FFMA plus the loop); any spelling with two or more independent chains lands at 128-140 clk. Going from
  2 to 8 chains buys nothing.
- **From 2 warps / SMSP on it is issue-bound, and the fewest instructions win.** At 8 warps the serial scalar chain
  (650) ties the packed serial one (651) and beats 8 accumulators by 12-21 % (738 scalar, 821 packed): every extra
  accumulator adds a tail-tree `FADD` and, for packed spellings, the split of each `.f32x2` op.
- **The only gain at every warp count comes from cutting instructions.** `fma.rn.f32.bf16` (SASS `FHFMA.BF16`)
  consumes a bf16 half directly and accumulates in fp32, so the widening `SHF/LOP3/IMAD` disappear (integer ops 56 -> 24):
  26 % faster than the packed serial chain at 1 warp, 22 % at 8 warps; numerics consistent with the widening path.

## Minimal runnable example

```cpp
// fk-cuda-cflags: -lineinfo
// (a) mixed-precision FMA: a, b are bf16 (.b16 registers), c, d are f32. SASS = FHFMA.BF16 (no integer widening).
__device__ __forceinline__ float sumsq_bf16x2_mixed(uint32_t w, float acc) {
  asm("{\n"
      ".reg .b16 h0, h1;\n"
      "mov.b32 {h0, h1}, %1;\n"                  // low half word = the first element
      "fma.rn.f32.bf16 %0, h0, h0, %0;\n"
      "fma.rn.f32.bf16 %0, h1, h1, %0;\n"
      "}\n" : "+f"(acc) : "r"(w));
  return acc;
}

// (b) widening path, two scalar accumulators (lo and hi halves): enough to cover FFMA latency at 1 warp / SMSP.
//     On sm120 an fma.rn.f32x2 spelling of the same thing issues the same two FFMA.
__device__ __forceinline__ void sumsq_bf16x2_two_chains(uint32_t w, float& acc_lo, float& acc_hi) {
  float lo = __uint_as_float(w << 16), hi = __uint_as_float(w & 0xffff0000u);
  acc_lo = fmaf(lo, lo, acc_lo);
  acc_hi = fmaf(hi, hi, acc_hi);
}

// (c) timing-loop skeleton: the whole loop in ONE asm block, .pragma "nounroll", inter-round feedback (z is a runtime 0),
//     inputs loaded through a per-lane address so the chain is not moved to the uniform datapath.
//   asm volatile("{\n .reg .b32 sb, tt, cnt;\n .reg .pred q;\n"
//                "mov.u64 %t0, %%clock64;\n mov.u32 cnt, 0;\n"
//                "RC_LOOP:\n .pragma \"nounroll\";\n"
//                "  ...chain body, result lands in sres...\n"
//                "  mov.b32 sb, sres;\n and.b32 tt, sb, %z;\n xor.b32 %w0, %w0, tt; ...(each input word)\n"
//                "  add.u32 cnt, cnt, 1;\n setp.lt.u32 q, cnt, %reps;\n @q bra RC_LOOP;\n"
//                "mov.u64 %t1, %%clock64;\n}\n" : ...);
```

## Gotchas
1. **Depth is not the price; issue may be.** Above 1 warp per SMSP the chain depth is hidden by the other warps and only
   the instruction count is left. Use the table in (c) to decide which column you are in before splitting.
2. **`.f32x2` is not a speed-up on sm120.** It compiles and is numerically identical, but every op is two scalar issues;
   code carried over from sm100 for its `FFMA2` throughput keeps only the register-pair layout.
3. **A full tree in PTX is not a full tree in SASS.** ptxas contracts the first level into `FFMA`; the SASS is an
   N/2-way accumulation, identical to writing that accumulation directly.
4. **Widening usually outnumbers FP.** Each bf16x2 word costs 2 integer ops to widen against 2 FP ops; in the
   issue-bound regime what can be saved is the widening (`fma.rn.f32.bf16`), not the shape of the FP chain.
5. **Uniform inputs move the chain to another pipe.** A probe (or a kernel) whose chain inputs are provably equal
   across the warp gets `UFFMA` / `ULOP3`; check the SASS for vector `FFMA` before reading a cycle count.
6. **Do not time a chain of a few dozen cycles with two C-level clock reads**: ptxas may move independent ALU work
   across them. Use the circular dependency of (b) and average over many iterations.

## Sources
- PTX ISA: `add/mul/fma` `.f32x2` forms, `fma.rn.f32.bf16` (mixed-precision FMA), `mov.b64 {lo, hi}`, `mov.b32 {h0, h1}`,
  `%clock64`, `.pragma "nounroll"`: https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUDA Binary Utilities (`cuobjdump -sass`): https://docs.nvidia.com/cuda/cuda-binary-utilities/
- Certification probe `probes/sm120/methodology/reg_chain_check.py` (variants p_serial / p_acc2 / p_acc4 / p_acc8 /
  p_tree / s_serial / s_acc8 / m_acc8): `PROBE PASS reg-chain contracts=5` (`numerics max_rel_err=6.84e-08`,
  `sass-no-packed`, `sass-split`, `sass-fhfma`, `stage-run`).
