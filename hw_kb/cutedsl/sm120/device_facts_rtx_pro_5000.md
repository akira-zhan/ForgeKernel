# device_facts_rtx_pro_5000 (RTX PRO 5000 72GB Blackwell, `sm_120a`: the measured budget a kernel plan is priced against)

> One file per hardware interface. Operator-agnostic; mechanism only.
>
> **Status**: ✅ Verified (GPU probe `probes/sm120/device_facts/run_facts.py`, `PROBE PASS device-facts contracts=35`:
> the build, the SASS opcode of all nine `mma.sync` kinds, their bit-exact numerics and every stage pass; the GPU was
> idle before and after the run (0 % utilisation, no other compute process), SM clock **unlocked** and read inside each
> timed kernel, no clock-event reason raised; driver 580.178.04, CUDA 13.0).
>
> **Certification source**: every number below was taken on one RTX PRO 5000 72GB (GB202, 110 SMs). Another sm_120 SKU
> (different SM count, memory bus, clocks or L2) has to re-run the probe before borrowing a number; per-clock-per-SM
> rates are the part most likely to carry over, absolute TFLOPS / GB/s are not.

## Purpose

The device budget of this card — tensor-core rate per kind, ALU / SFU rate, shared-memory, L2 and DRAM bandwidth,
load latency per level, launch cost — measured, so that a roofline, a tile plan or a "how far from peak" claim is
priced against this card and not against a datasheet or another architecture.

## When needed (trigger)

- Estimating a roofline / ridge point, or a "% of peak" reading of an NCU profile, on this card.
- Choosing an MMA kind (bf16 vs FP8 vs FP4, block-scaled or not) by its rate.
- Sizing a tile: smem bytes per stage, warps per SM needed to hide `mma.sync` latency, operand reuse needed so that
  shared memory can feed the tensor cores.
- Deciding whether a working set will be served by L2 (footprint vs the 96 MB L2) or by DRAM.
- Comparing a stream launch chain with a CUDA graph for many small kernels.

## Minimal runnable example

```bash
# Re-measure on any sm_120 card (pick an idle GPU; a busy one marks the digest FACTS-CONTENDED).
python3 -I probes/sm120/device_facts/run_facts.py --gpu <nvidia-smi index> --out <dir>
# -> one "PROBE PASS|FAIL <contract>" line per contract, the FACT lines, <dir>/digest.json
# A single stage by hand (same binary):  <dir>/facts mma e4m3_f32acc  |  <dir>/facts l2 32 96 128
```

## Device attributes (`cudaGetDeviceProperties` / `cudaDeviceGetAttribute`)

| Attribute | Value |
|---|---|
| Compute capability | 12.0 (`sm_120a` for the arch-specific features) |
| SMs | 110 |
| L2 | 100663296 B (96 MiB); persisting-L2 max 62914560 B (60 MiB) |
| Shared memory | 102400 B per SM; 101376 B (99 KiB) per block with opt-in; 49152 B default; 1024 B reserved per block |
| Registers | 65536 per SM, 65536 per block |
| Threads / blocks | 1536 threads per SM (48 warps), 1024 per block, 24 blocks per SM |
| Global memory | 76368379904 B, 384-bit bus, memory clock attribute 14001 MHz, ECC on |
| SM clock attribute | 2377 MHz (default application clock; under load the card ran 2.58-2.63 GHz in every stage) |
| Cluster launch | supported |

## Tensor cores: `mma.sync` per kind

Rates are dense, one mma per warp per instruction, all operand registers resident, 8 independent accumulator chains per
warp; "per clk per SM" uses the clock measured inside the same kernel.

| Kind (PTX) | SASS | Shape | Best TFLOPS (at ~2.61 GHz) | Dense flop / clk / SM | Dependent latency (clk) |
|---|---|---|---|---|---|
| bf16, f32 acc | `HMMA.16816.F32.BF16` | m16n8k16 | 293 | ~1016 | 29 |
| f16, f16 acc | `HMMA.16816.F16` | m16n8k16 | 293 | ~1016 | 29 |
| e4m3, f32 acc (`kind::f8f6f4`) | `QMMA.16832.F32.E4M3.E4M3` | m16n8k32 | 587 | ~2030 | 29 |
| e4m3, f16 acc (`kind::f8f6f4`) | `QMMA.16832.F16.E4M3.E4M3` | m16n8k32 | 587 | ~2040 | 29 |
| e2m1 in `kind::f8f6f4` | `QMMA.16832.F32.E2M1.E2M1` | m16n8k32 | 586 | ~2040 | 29 |
| s8, s32 acc | `IMMA.16832.S8.S8` | m16n8k32 | 587 | ~2035 | 27 |
| `kind::mxf8f6f4` block-scaled, `scale_vec::1X` ue8m0 | `QMMA.SF.16832.F32.E4M3.E4M3.E8` | m16n8k32 | 585 | ~2040 | 29 |
| `kind::mxf4` block-scaled, `scale_vec::2X` ue8m0 | `OMMA.SF.16864.F32.E2M1.E2M1.E8` | m16n8k64 | 1173 | ~4090 | 29 |
| `kind::mxf4nvf4` block-scaled, `scale_vec::4X` ue4m3 | `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` | m16n8k64 | 1173 | ~4063 | 29 |

What the table means for a plan:

- **One `mma.sync` per 16 clk per SM sub-partition, for every kind.** 4 sub-partitions x one m16n8k16 (4096 flop) per
  16 clk = 1024 flop/clk/SM for bf16; FP8 doubles only because K doubles (m16n8k32), FP4 doubles again only through
  the `OMMA` m16n8k64 forms.
- **FP4 through `kind::f8f6f4` runs at the FP8 rate.** The 2x FP4 rate needs the block-scaled `kind::mxf4` /
  `kind::mxf4nvf4` forms (`OMMA.SF`).
- **f16 accumulation is not faster than f32 accumulation** on this card (HMMA and QMMA alike); int8 runs at the FP8 rate.
- **Block scaling is free in issue rate**: `QMMA.SF` (mxf8, 1X) matches the unscaled `QMMA`.
- **Latency hiding**: dependent latency (~29 clk) is about 2x the issue interval (16 clk), so each sub-partition
  needs at least 2 independent mma chains in flight. Measured: 1 warp per sub-partition (4 warps/SM) with 8 chains
  reaches ~92-93 % of the best rate; 2 warps per sub-partition (8 warps/SM) reach 99-100 %, 4 add nothing.

## FP32 ALU and SFU

| Op | Measured | Note |
|---|---|---|
| FFMA (vector datapath) | 123 / clk / SM (69.6 TFLOPS at 2.58 GHz) | 128 lanes per SM; the shortfall is loop overhead of the probe |
| MUFU.EX2 (`ex2.approx.ftz.f32`) | 16 / clk / SM | 1/8 of the FFMA rate: an exp per element in a softmax-like pass costs 8 FFMA slots |

## Memory hierarchy

| Level | Bandwidth | Latency (dependent `clock64` chase, 128 B lines) |
|---|---|---|
| Shared memory, conflict-free `LDS.128` | 127.8 B / clk / SM (36.6 TB/s chip-wide at 2.61 GHz) | — |
| L1 hit (`ld.global.ca`, 16 KiB footprint) | — | 46 clk (18 ns) |
| L2 hit (`ld.global.cg`, footprint 16 KiB to 80 MiB) | 6.2-6.4 TB/s chip-wide for footprints 8-96 MiB (~22 B / clk / SM) | 350 clk (134 ns), flat across the range |
| Footprint = 96 MiB (= L2 size) | 6.3 TB/s (streaming read) | 588 clk (partial misses under random access) |
| DRAM (footprint >= 128 MiB) | read 1.23 TB/s; write 1.20 TB/s; copy 1.08 TB/s (read+write bytes) | 895-902 clk (341-344 ns), TLB misses included |

What the table means for a plan:

- **Peak DRAM**: 384-bit x 2 x 14001 MHz = 1344 GB/s; the streaming read reaches 91.6 % of it. 256-bit loads
  (`ld.global.v8.u32` -> `LDG.E.ENL2.256`) read 1.236 TB/s vs 1.232 TB/s for 128-bit: **no bandwidth gain** from the
  wider load in a streaming read.
- **L2 is ~5x DRAM and the whole 96 MiB serves streaming reads**: the cliff sits between 96 and 128 MiB of footprint
  (128 MiB already reads 1.40 TB/s). 1 load in flight per thread at 32 warps/SM and 4 loads in flight at 48
  warps/SM both read ~6.2-6.4 TB/s: this is the L2 ceiling, not a latency x MLP limit of the probe.
- **Ridge points** (dense mma rate / bandwidth): from DRAM, bf16 ~238 flop/B, FP8 ~476 flop/B, block-scaled FP4
  ~950 flop/B; from L2, bf16 ~46 flop/B, FP8 ~93 flop/B.
- **Shared memory cannot feed `mma.sync` without register reuse.** One m16n8k16 (bf16) or m16n8k32 (FP8) needs
  768 B of A+B fragments; 4 sub-partitions issuing one each per 16 clk need 3072 B per 16 clk per SM, while shared
  memory delivers 128 x 16 = 2048 B. A warp tile must reuse each loaded A / B fragment across several mma (larger
  warp-tile M x N) to stay tensor-bound; this is arithmetic on two readings above, not a separate probe.
- **Latency x bandwidth (Little's law)**: L2 at 6.3 TB/s x 134 ns ~ 845 KB in flight chip-wide (~7.7 KB per SM);
  DRAM at 1.23 TB/s x 342 ns ~ 420 KB in flight (~3.8 KB per SM). A memory-bound kernel needs that many bytes
  outstanding per SM (warps x loads in flight x bytes per load).

## Launch cost

| Path | Cost per empty kernel |
|---|---|
| Back-to-back stream launches | 1.65 us |
| Same kernels captured in one CUDA graph | 0.41 us |

## Library reference ceilings (4096^3 GEMM, bf16 output)

INFO (median of 50 launches, clocks unlocked; `probes/sm120/device_facts/ref_ceilings.py`, torch 2.14.1 + CUDA 13.0):
the bar a hand-written sm120 mainloop is compared against.

| Kind | Library | TFLOPS | Best hand-written mainloop here |
|---|---|---|---|
| bf16 | cuBLAS (`torch.matmul`) | 228 | 249 ([`tma_mbarrier_mainloop`](tma_mbarrier_mainloop.md)) |
| e4m3 | cuBLASLt per-tensor / rowwise (`torch._scaled_mm`) | 420 / 447 | 468 ([`tma_mbarrier_mainloop`](tma_mbarrier_mainloop.md)) |
| e4m3 | CUTLASS sm120 (`sgl_kernel.fp8_scaled_mm`, rowwise) | 450 | |
| mxf8 (e8m0 / 32) | cuBLASLt (`torch._scaled_mm`) | 416 | 275 ([`blockscaled_mma`](blockscaled_mma.md)) |
| nvf4 (ue4m3 / 16) | cuBLASLt (`torch._scaled_mm`) | 741 | 537 ([`blockscaled_mma`](blockscaled_mma.md)) |
| mxf4 (e8m0 / 32) | `torch._scaled_mm` | rejected ("Invalid scaling configuration") | 518 |

How to read it: the plain bf16 / e4m3 mainloops already match the libraries (the hand-written ones write fp32 C and
apply no scale, so the comparison is approximate). The block-scaled probe of this tree trails cuBLASLt by 1.4-1.5x,
but that is the probe's per-thread scale loads, not a ceiling: the ForgeKernel pilot run on
`tasks/sm120_nvfp4_gemm` (one student iteration: persistent CTA, TMA of A / B **and** the blocked scales into smem,
`OMMA.SF` from ldmatrix + LDS scales, stmatrix + TMA-store epilogue, proxy fence before each stage release) matched
cuBLASLt bit for bit and ran 1.016x it in the same job (square4k 0.165 ms vs 0.168 ms, wide8k 0.672 vs 0.684 ms).

## Decode GEMM (small M) on this card -- what the library leaves

INFO, from the forge run on `tasks/sm120_fp8_decode_gemm` (FP8 rowwise, N x K = 14336 x 4096 / 4096 x 14336, M = 1-64,
timed L2-cold: an FP8 weight of 58.7 MB fits the 96 MB L2, so warm repeated calls overstate a decode GEMM):

- **The CUTLASS sm120 FP8 kernel (sgl_kernel.fp8_scaled_mm) tiles M at 128 for every M** (128 x 128 x 128), so at
  decode it loads and multiplies up to 127 zero rows per tile: 46-54 % of DRAM bandwidth, i.e. FP8 decode barely
  beats the bf16 cuBLAS one (85-93 %). Sizing the CTA's M extent to the workload (bM = the smallest of 16 / 32 / 64 /
  128 that is >= M; the A TMA box shrinks with it and the freed smem deepens the ring) was worth, as an ablation
  upper bound, +36 % (M = 1), +29 % (M = 16), +8 % (M = 64).
- **N = 14336 at bN = 128 is 112 tiles on 110 SMs**: one CTA per SM leaves a 2-tile second wave (ablation upper bound
  +17.5 % on the up shapes). Not addressed by the run below; the next lever for these shapes.
- Result: 1.49x the library in forge's device window (3 same-card jobs; M = 1 / 16 up 1.54 / 1.50x, M = 64 up 1.27x,
  M = 16 down 1.67x), 1.13-1.48x per call with launch latency included (my cold re-timing); bit-exact against an
  fp32 reference within bf16 rounding and exact on selector inputs for M in 1..128.

## Gotchas

- **Clocks are not fixed**: the attribute says 2377 MHz, the card ran 2.58-2.63 GHz under these loads (max SM clock
  3090 MHz, 350 W power limit). Compare kernels in per-clock units or at a recorded clock; a TFLOPS figure without
  its clock is not portable across runs.
- **Microbenchmark traps seen while building this probe** (each one produced a plausible-looking but wrong number):
  - values identical across threads (`x + j`) let the compiler move an FFMA chain onto the uniform datapath
    (`UFFMA`) — the "FFMA rate" read half of the truth; seed per-thread values and check SASS for `FFMA`;
  - a shared-memory read loop over a few fixed addresses is hoisted by ptxas even inside `asm volatile`
    (`ld.shared`): the timed loop held no `LDS`; `ld.volatile.shared` keeps one `LDS` per iteration;
  - a grid larger than one resident wave turns a "footprint sweep" into per-wave slices that fit L1/L2 (a read of
    a 256 MiB buffer reported 5.5 TB/s); size the grid with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`;
  - default `ld.global` caches in L1, so small "L2" footprints read L1 bandwidth; use `ld.global.cg` for L2;
  - a pointer chase capped at fewer steps than the footprint has lines only touches part of it (a 1 GiB chase read
    L2 latency); walk the whole cycle.
- **Arch-specific build**: the block-scaled forms only exist on the arch-specific target; build with
  `-gencode arch=compute_120a,code=sm_120a`. A plain `-arch=sm_120a` executable build of the same source failed in
  ptxas with "not supported on .target sm_120".
- **FP4 operand container in `kind::f8f6f4`**: each e2m1 element occupies one byte with the value in bits [5:2]
  (1.0 = `0x08`); the packed two-per-byte layout belongs to the `mxf4` / `mxf4nvf4` kinds (1.0 pair = `0x22`).
- The latency rows include the address arithmetic of the chase loop (a few clk); the DRAM row includes TLB misses
  of a random walk over up to 1 GiB, which a streaming kernel does not pay.

## Sources

1. CUDA C++ Programming Guide — Compute Capabilities (12.x), Shared Memory, L2 cache set-aside:
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/
2. PTX ISA — `mma` (`kind::f8f6f4`, `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4`, `.block_scale`,
   `scale_vec`), `ld` (`.cg`, `.ca`, `.volatile`, `.v8`): https://docs.nvidia.com/cuda/parallel-thread-execution/
3. `nvidia-smi -q` (max clocks, power limit, PCIe link) and CUDA Runtime API — `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`,
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor`: https://docs.nvidia.com/cuda/cuda-runtime-api/
4. GPU probe `probes/sm120/device_facts/run_facts.py` (+ `facts.cu`): `PROBE PASS device-facts contracts=35`
   (`PROBE PASS sass-<kind>` and `PROBE PASS numerics-<kind>` for all nine kinds).
