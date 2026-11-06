// SM120 device-facts probe: one measurement per invocation (`./facts <mode> [args]`).
// Every line the parent reads starts with "FACT " (key=value pairs) or "CHECK " (a
// correctness contract with ok=0|1). Built for the arch-specific target only:
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 -lineinfo -o facts facts.cu
// (the block-scaled mma.sync forms do not exist on the generic compute_120 target).
//
// Clocks are NOT locked: each timed kernel also measures the SM clock it actually ran
// at (clock64 delta / globaltimer delta on one thread), so per-clock rates are derived
// from the clock of that very run.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>

#define CK(x)                                                                      \
  do {                                                                             \
    cudaError_t e_ = (x);                                                          \
    if (e_ != cudaSuccess) {                                                       \
      printf("ERROR %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);  \
      exit(2);                                                                     \
    }                                                                              \
  } while (0)

__device__ __forceinline__ uint64_t gtimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// one record per timed kernel: {clock64 start, clock64 end, globaltimer start, end}
struct ClockRec { unsigned long long c0, c1, t0, t1; };

__device__ __forceinline__ void clk_begin(ClockRec* r) {
  if (r && blockIdx.x == 0 && threadIdx.x == 0) { r->c0 = clock64(); r->t0 = gtimer(); }
}
__device__ __forceinline__ void clk_end(ClockRec* r) {
  if (r && blockIdx.x == 0 && threadIdx.x == 0) { r->c1 = clock64(); r->t1 = gtimer(); }
}

static int g_sms = 0;

static double run_timed(void (*launch)(void*), void* arg, ClockRec* d_rec, double* mhz, int reps) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  launch(arg);  // warm-up
  CK(cudaDeviceSynchronize());
  std::vector<double> ms(reps), clk(reps);
  for (int i = 0; i < reps; ++i) {
    CK(cudaEventRecord(a));
    launch(arg);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float m; CK(cudaEventElapsedTime(&m, a, b));
    ms[i] = m;
    ClockRec h; CK(cudaMemcpy(&h, d_rec, sizeof h, cudaMemcpyDeviceToHost));
    clk[i] = (h.t1 > h.t0) ? double(h.c1 - h.c0) * 1e3 / double(h.t1 - h.t0) : 0.0;
  }
  // report the median run (by time) and the clock of that run
  std::vector<int> idx(reps);
  for (int i = 0; i < reps; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int x, int y) { return ms[x] < ms[y]; });
  int med = idx[reps / 2];
  *mhz = clk[med];
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms[med];
}

// ------------------------------------------------------------------ attrs
static void mode_attrs() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  int v;
  printf("FACT name=\"%s\" cc=%d.%d\n", p.name, p.major, p.minor);
  printf("FACT sms=%d\n", p.multiProcessorCount);
  printf("FACT l2_bytes=%d persisting_l2_max_bytes=%d\n", p.l2CacheSize, p.persistingL2CacheMaxSize);
  printf("FACT smem_per_sm=%zu smem_per_block_optin=%zu smem_per_block_default=%zu reserved_smem_per_block=%zu\n",
         p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin, p.sharedMemPerBlock, p.reservedSharedMemPerBlock);
  printf("FACT regs_per_sm=%d regs_per_block=%d\n", p.regsPerMultiprocessor, p.regsPerBlock);
  printf("FACT max_threads_per_sm=%d max_threads_per_block=%d max_blocks_per_sm=%d warp=%d\n",
         p.maxThreadsPerMultiProcessor, p.maxThreadsPerBlock, p.maxBlocksPerMultiProcessor, p.warpSize);
  printf("FACT global_mem_bytes=%zu mem_bus_width_bits=%d\n", p.totalGlobalMem, p.memoryBusWidth);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, 0)); printf("FACT mem_clock_khz=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, 0)); printf("FACT sm_clock_attr_khz=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, 0)); printf("FACT cluster_launch=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrAsyncEngineCount, 0)); printf("FACT async_engines=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrEccEnabled, 0)); printf("FACT ecc=%d\n", v);
  printf("FACT pcie_bus=%d pcie_device=%d\n", p.pciBusID, p.pciDeviceID);
}

// ------------------------------------------------------------------ memory streaming
struct MemArg { const uint4* src; uint4* dst; size_t n16; int blocks; int threads; ClockRec* rec; int passes; int wide; };

__global__ void k_read(const uint4* __restrict__ src, size_t n16, int passes, int wide, uint4* sink, ClockRec* rec) {
  clk_begin(rec);
  uint32_t acc = 0;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int p = 0; p < passes; ++p) {
    if (wide) {
      // 256-bit loads: two 16 B halves per thread per step (LDG.E.ENL2.256)
      size_t n32 = n16 / 2;
      for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n32; i += stride) {
        uint32_t r[8];
        asm volatile("ld.global.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                     : "l"(src + 2 * i));
        acc ^= r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[5] ^ r[6] ^ r[7];
      }
    } else {
      // ld.global.cg (L1 bypassed: a footprint that fits L1 cannot inflate the L2 reading),
      // 4 independent loads in flight per thread so the reading is not latency x MLP bound
      size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
      for (; i + 3 * stride < n16; i += 4 * stride) {
        uint4 v0 = __ldcg(src + i), v1 = __ldcg(src + i + stride);
        uint4 v2 = __ldcg(src + i + 2 * stride), v3 = __ldcg(src + i + 3 * stride);
        acc ^= v0.x ^ v0.y ^ v0.z ^ v0.w ^ v1.x ^ v1.y ^ v1.z ^ v1.w;
        acc ^= v2.x ^ v2.y ^ v2.z ^ v2.w ^ v3.x ^ v3.y ^ v3.z ^ v3.w;
      }
      for (; i < n16; i += stride) {
        uint4 v = __ldcg(src + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    }
  }
  if (acc == 0x9e3779b9u) sink[0] = make_uint4(acc, 0, 0, 0);
  clk_end(rec);
}

__global__ void k_write(uint4* __restrict__ dst, size_t n16, ClockRec* rec) {
  clk_begin(rec);
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride)
    dst[i] = make_uint4((uint32_t)i, 1, 2, 3);
  clk_end(rec);
}

__global__ void k_copy(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16, ClockRec* rec) {
  clk_begin(rec);
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) dst[i] = src[i];
  clk_end(rec);
}

// one resident wave: blocks per SM from the occupancy API, so a read pass sweeps the whole
// footprint instead of each wave of blocks re-reading only its own slice
static int resident_blocks(const void* fn, int threads) {
  int per_sm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, threads, 0));
  return g_sms * std::max(per_sm, 1);
}

static void launch_read(void* a) {
  MemArg* m = (MemArg*)a;
  k_read<<<m->blocks, m->threads>>>(m->src, m->n16, m->passes, m->wide, m->dst, m->rec);
}
static void launch_write(void* a) { MemArg* m = (MemArg*)a; k_write<<<m->blocks, m->threads>>>(m->dst, m->n16, m->rec); }
static void launch_copy(void* a) { MemArg* m = (MemArg*)a; k_copy<<<m->blocks, m->threads>>>(m->src, m->dst, m->n16, m->rec); }

// dram <MiB>: read (128-bit and 256-bit), write, copy of a buffer far larger than L2
static void mode_dram(size_t mib) {
  size_t bytes = mib << 20, n16 = bytes / 16;
  uint4 *src, *dst; ClockRec* rec;
  CK(cudaMalloc(&src, bytes)); CK(cudaMalloc(&dst, bytes)); CK(cudaMalloc(&rec, sizeof(ClockRec)));
  CK(cudaMemset(src, 1, bytes)); CK(cudaMemset(dst, 0, bytes));
  MemArg m{src, dst, n16, resident_blocks((const void*)k_read, 512), 512, rec, 1, 0};
  double mhz, ms;
  ms = run_timed(launch_read, &m, rec, &mhz, 7);
  printf("FACT dram_read_128_gbs=%.1f mib=%zu sm_mhz=%.0f\n", bytes / ms / 1e6, mib, mhz);
  m.wide = 1;
  ms = run_timed(launch_read, &m, rec, &mhz, 7);
  printf("FACT dram_read_256_gbs=%.1f mib=%zu sm_mhz=%.0f\n", bytes / ms / 1e6, mib, mhz);
  ms = run_timed(launch_write, &m, rec, &mhz, 7);
  printf("FACT dram_write_gbs=%.1f mib=%zu sm_mhz=%.0f\n", bytes / ms / 1e6, mib, mhz);
  ms = run_timed(launch_copy, &m, rec, &mhz, 7);
  printf("FACT dram_copy_gbs=%.1f mib=%zu (read+write bytes) sm_mhz=%.0f\n", 2.0 * bytes / ms / 1e6, mib, mhz);
  CK(cudaFree(src)); CK(cudaFree(dst)); CK(cudaFree(rec));
}

// l2 <MiB list...>: repeated read passes over a working set; bandwidth vs footprint
static void mode_l2(int argc, char** argv) {
  for (int k = 0; k < argc; ++k) {
    size_t mib = strtoull(argv[k], 0, 10), bytes = mib << 20, n16 = bytes / 16;
    uint4 *src, *dst; ClockRec* rec;
    CK(cudaMalloc(&src, bytes)); CK(cudaMalloc(&dst, 64)); CK(cudaMalloc(&rec, sizeof(ClockRec)));
    CK(cudaMemset(src, 1, bytes));
    int passes = (int)std::max<size_t>(1, (size_t)(4096ull << 20) / bytes);  // ~4 GiB of reads per launch
    MemArg m{src, dst, n16, resident_blocks((const void*)k_read, 512), 512, rec, passes, 0};
    double mhz, ms = run_timed(launch_read, &m, rec, &mhz, 5);
    printf("FACT read_footprint_mib=%zu gbs=%.1f passes=%d blocks=%d sm_mhz=%.0f\n", mib,
           (double)bytes * passes / ms / 1e6, passes, m.blocks, mhz);
    CK(cudaFree(src)); CK(cudaFree(dst)); CK(cudaFree(rec));
  }
}

// ------------------------------------------------------------------ latency (pointer chase)
// cg = ld.global.cg (L1 bypassed: L2 or DRAM), ca = ld.global.ca (L1 allocating)
template <bool CG>
__global__ void k_chase(const uint32_t* __restrict__ next, uint32_t start, long long steps, uint32_t* out,
                        unsigned long long* cyc) {
  uint32_t j = start;
  unsigned long long c0 = clock64();
  for (long long i = 0; i < steps; ++i) j = CG ? __ldcg(next + j) : __ldca(next + j);
  unsigned long long c1 = clock64();
  out[0] = j;
  cyc[0] = c1 - c0;
}

// lat <KiB list...>: dependent loads over ONE random cyclic permutation of 128 B lines.
// A warm-up launch walks the full cycle once, then the timed launch walks it again (at least
// 20000 loads), so every line of the footprint is touched in the timed pass: footprints that
// fit L2 read L2-hit latency, larger ones read DRAM (+ TLB) latency. Footprints <= 64 KiB are
// also walked with ld.global.ca (L1 hit).
template <bool CG>
static void chase_once(const uint32_t* d, size_t lines, size_t kib, const char* level) {
  uint32_t* o; unsigned long long* c;
  CK(cudaMalloc(&o, 4)); CK(cudaMalloc(&c, 8));
  long long steps = (long long)std::max<size_t>(lines, 20000);
  k_chase<CG><<<1, 1>>>(d, 0, (long long)lines, o, c);  // warm-up: one full cycle
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  k_chase<CG><<<1, 1>>>(d, 0, steps, o, c);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b));
  unsigned long long cyc; CK(cudaMemcpy(&cyc, c, 8, cudaMemcpyDeviceToHost));
  printf("FACT latency_footprint_kib=%zu load=%s cycles_per_load=%.1f ns_per_load=%.1f steps=%lld\n", kib, level,
         (double)cyc / steps, ms * 1e6 / steps, steps);
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  CK(cudaFree(o)); CK(cudaFree(c));
}

static void mode_lat(int argc, char** argv) {
  for (int k = 0; k < argc; ++k) {
    size_t kib = strtoull(argv[k], 0, 10), bytes = kib << 10;
    size_t lines = bytes / 128, words = bytes / 4;
    std::vector<uint32_t> perm(lines), h(words, 0);
    for (size_t i = 0; i < lines; ++i) perm[i] = (uint32_t)i;
    std::mt19937 rng(1234);
    std::shuffle(perm.begin() + 1, perm.end(), rng);
    for (size_t i = 0; i < lines; ++i) h[perm[i] * 32] = perm[(i + 1) % lines] * 32;
    uint32_t* d;
    CK(cudaMalloc(&d, bytes));
    CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    if (kib <= 64) chase_once<false>(d, lines, kib, "ca");
    chase_once<true>(d, lines, kib, "cg");
    CK(cudaFree(d));
  }
}

// ------------------------------------------------------------------ shared memory bandwidth
__global__ void k_smem(int iters, uint32_t* sink, ClockRec* rec) {
  extern __shared__ uint4 s[];
  int n = blockDim.x * 4;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = make_uint4(i, i + 1, i + 2, i + 3);
  __syncthreads();
  clk_begin(rec);
  uint32_t acc = 0;
  int base = threadIdx.x;
  uint32_t sbase = (uint32_t)__cvta_generic_to_shared(s + base);
  uint32_t step = blockDim.x * 16;
#pragma unroll 8
  for (int it = 0; it < iters; ++it) {
    // consecutive lanes -> consecutive 16 B: conflict-free LDS.128; volatile keeps it in the loop
    uint32_t x, y, z, w;
    asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                 : "r"(sbase + (it & 3) * step));
    acc += x ^ y ^ z ^ w;
  }
  clk_end(rec);
  if (acc == 0x9e3779b9u) sink[0] = acc;
}
struct SmemArg { int blocks, threads, iters; uint32_t* sink; ClockRec* rec; };
static void launch_smem(void* a) {
  SmemArg* m = (SmemArg*)a;
  k_smem<<<m->blocks, m->threads, m->threads * 4 * 16>>>(m->iters, m->sink, m->rec);
}
static void mode_smem() {
  uint32_t* sink; ClockRec* rec;
  CK(cudaMalloc(&sink, 4)); CK(cudaMalloc(&rec, sizeof(ClockRec)));
  SmemArg m{g_sms * 2, 512, 1 << 16, sink, rec};
  double mhz, ms = run_timed(launch_smem, &m, rec, &mhz, 5);
  double bytes = (double)m.blocks * m.threads * m.iters * 16;
  printf("FACT smem_lds128_gbs=%.1f bytes_per_clk_per_sm=%.1f sm_mhz=%.0f\n", bytes / ms / 1e6,
         bytes / (ms * 1e-3 * mhz * 1e6) / g_sms, mhz);
  CK(cudaFree(sink)); CK(cudaFree(rec));
}

// ------------------------------------------------------------------ tensor-core mma.sync
// Each op: D(acc) += A x B with the fragment registers of one thread.
#define MMA_F32ACC(STR)                                                                         \
  asm volatile(STR " {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                        \
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])                                \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
#define MMA_F16ACC(STR)                                                                         \
  asm volatile(STR " {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"                                    \
               : "+r"(*(uint32_t*)&c[0]), "+r"(*(uint32_t*)&c[1])                              \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
#define MMA_S32ACC(STR)                                                                         \
  asm volatile(STR " {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                        \
               : "+r"(*(int*)&c[0]), "+r"(*(int*)&c[1]), "+r"(*(int*)&c[2]), "+r"(*(int*)&c[3]) \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]))
#define MMA_SCALED(STR)                                                                         \
  asm volatile(STR " {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};" \
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])                                \
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),             \
                 "r"(sf), "r"(sf), "h"((uint16_t)0))

enum MmaKind { BF16_F32, F16_F16, E4M3_F32, E4M3_F16, E2M1_F8F6F4_F32, S8_S32, MXF8_1X, MXF4_2X, NVF4_4X, NKIND };
struct MmaInfo { const char* name; int k; const char* sass; uint32_t one; uint32_t sf; int acc16; int acc_int; };
// one = the packed fragment word whose every element is 1.0 (or int 1); sf = scale word = 1.0
static const MmaInfo kInfo[NKIND] = {
    {"bf16_f32acc", 16, "HMMA.16816.F32.BF16", 0x3F803F80u, 0, 0, 0},
    {"f16_f16acc", 16, "HMMA.16816.F16", 0x3C003C00u, 0, 1, 0},
    {"e4m3_f32acc", 32, "QMMA.16832.F32.E4M3.E4M3", 0x38383838u, 0, 0, 0},
    {"e4m3_f16acc", 32, "QMMA.16832.F16.E4M3.E4M3", 0x38383838u, 0, 1, 0},
    {"e2m1_f8f6f4_f32acc", 32, "QMMA.16832.F32.E2M1.E2M1", 0x08080808u, 0, 0, 0},
    {"s8_s32acc", 32, "IMMA.16832.S8.S8", 0x01010101u, 0, 0, 1},
    {"mxf8_1x_f32acc", 32, "QMMA.SF.16832.F32.E4M3.E4M3.E8", 0x38383838u, 0x7F7F7F7Fu, 0, 0},
    {"mxf4_2x_f32acc", 64, "OMMA.SF.16864.F32.E2M1.E2M1.E8", 0x22222222u, 0x7F7F7F7Fu, 0, 0},
    {"nvf4_4x_f32acc", 64, "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", 0x22222222u, 0x38383838u, 0, 0},
};

template <int KIND>
__device__ __forceinline__ void mma_op(float* c, const uint32_t* a, const uint32_t* b, uint32_t sf) {
  if constexpr (KIND == BF16_F32) MMA_F32ACC("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32");
  if constexpr (KIND == F16_F16) MMA_F16ACC("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16");
  if constexpr (KIND == E4M3_F32) MMA_F32ACC("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32");
  if constexpr (KIND == E4M3_F16) MMA_F16ACC("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16");
  if constexpr (KIND == E2M1_F8F6F4_F32) MMA_F32ACC("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32");
  if constexpr (KIND == S8_S32) MMA_S32ACC("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32");
  if constexpr (KIND == MXF8_1X)
    MMA_SCALED("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0");
  if constexpr (KIND == MXF4_2X)
    MMA_SCALED("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0");
  if constexpr (KIND == NVF4_4X)
    MMA_SCALED("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3");
}

constexpr int NACC = 8;  // independent accumulator chains per warp

template <int KIND>
__global__ void k_mma(const uint32_t* in, int iters, int chains, float* out, ClockRec* rec) {
  uint32_t a[4] = {in[0], in[1], in[2], in[3]}, b[2] = {in[4], in[5]}, sf = in[6];
  float c[NACC][4];
#pragma unroll
  for (int j = 0; j < NACC; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
  clk_begin(rec);
  if (chains == NACC) {
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int j = 0; j < NACC; ++j) mma_op<KIND>(c[j], a, b, sf);
    }
  } else {  // one dependent chain: latency
    for (int it = 0; it < iters; ++it) mma_op<KIND>(c[0], a, b, sf);
  }
  clk_end(rec);
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < NACC; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (blockIdx.x == 0 && threadIdx.x < 32) {  // raw accumulator words of chain 0 for the correctness check
    uint32_t* w = (uint32_t*)(out + gridDim.x * blockDim.x);
    for (int r = 0; r < 4; ++r) w[threadIdx.x * 4 + r] = *(uint32_t*)&c[0][r];
  }
}

struct MmaArg { int kind, blocks, threads, iters, chains; const uint32_t* in; float* out; ClockRec* rec; };
template <int KIND>
static void launch_mma_t(MmaArg* m) { k_mma<KIND><<<m->blocks, m->threads>>>(m->in, m->iters, m->chains, m->out, m->rec); }
static void launch_mma(void* p) {
  MmaArg* m = (MmaArg*)p;
  switch (m->kind) {
    case BF16_F32: launch_mma_t<BF16_F32>(m); break;
    case F16_F16: launch_mma_t<F16_F16>(m); break;
    case E4M3_F32: launch_mma_t<E4M3_F32>(m); break;
    case E4M3_F16: launch_mma_t<E4M3_F16>(m); break;
    case E2M1_F8F6F4_F32: launch_mma_t<E2M1_F8F6F4_F32>(m); break;
    case S8_S32: launch_mma_t<S8_S32>(m); break;
    case MXF8_1X: launch_mma_t<MXF8_1X>(m); break;
    case MXF4_2X: launch_mma_t<MXF4_2X>(m); break;
    case NVF4_4X: launch_mma_t<NVF4_4X>(m); break;
  }
}

static float half_to_float(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float v = (e == 0) ? ldexpf((float)m, -24) : ldexpf((float)(m | 1024), (int)e - 25);
  return s ? -v : v;
}

// mma <kind>: correctness of one accumulator chain, then throughput (warps/SM sweep) and latency
static void mode_mma(const char* name) {
  int kind = -1;
  for (int i = 0; i < NKIND; ++i) if (!strcmp(kInfo[i].name, name)) kind = i;
  if (kind < 0) { printf("ERROR unknown mma kind %s\n", name); exit(2); }
  const MmaInfo& I = kInfo[kind];
  uint32_t h_in[8] = {I.one, I.one, I.one, I.one, I.one, I.one, I.sf, 0};
  uint32_t* in; float* out; ClockRec* rec;
  int max_threads = g_sms * 16 * 32;
  CK(cudaMalloc(&in, sizeof h_in)); CK(cudaMalloc(&out, (max_threads + 128) * 4)); CK(cudaMalloc(&rec, sizeof(ClockRec)));
  CK(cudaMemcpy(in, h_in, sizeof h_in, cudaMemcpyHostToDevice));

  // correctness: all-ones operands (scale = 1.0): every accumulator element = iters * K
  const int citers = I.acc16 ? 8 : 64;
  MmaArg m{kind, 1, 32, citers, 1, in, out, rec};
  launch_mma(&m);
  CK(cudaDeviceSynchronize());
  uint32_t w[128];
  CK(cudaMemcpy(w, out + 32, sizeof w, cudaMemcpyDeviceToHost));
  double expect = (double)citers * I.k, worst = 0;
  int nvals = I.acc16 ? 64 : 128;
  for (int i = 0; i < nvals; ++i) {
    double v;
    if (I.acc16) v = half_to_float((uint16_t)(w[(i / 2) % 32 * 4 + (i / 2) / 32] >> (16 * (i & 1))));
    else if (I.acc_int) v = (double)(int32_t)w[i];
    else { float f; memcpy(&f, &w[i], 4); v = f; }
    worst = std::max(worst, std::fabs(v - expect));
  }
  if (I.acc16) {  // only acc words 0,1 of each thread are live for an f16 accumulator
    worst = 0;
    for (int t = 0; t < 32; ++t)
      for (int r = 0; r < 2; ++r)
        for (int hh = 0; hh < 2; ++hh)
          worst = std::max(worst, std::fabs(half_to_float((uint16_t)(w[t * 4 + r] >> (16 * hh))) - expect));
  }
  printf("CHECK mma=%s ok=%d expect=%.0f max_abs_err=%.3g\n", I.name, worst == 0.0 ? 1 : 0, expect, worst);

  // throughput: NACC independent chains per warp, sweep resident warps per SM
  double flop_per_mma = 2.0 * 16 * 8 * I.k;
  double best = 0, best_clk = 0, best_mhz = 0; int best_w = 0;
  for (int wps : {4, 8, 16}) {
    MmaArg t{kind, g_sms * (wps / 4), 128, 4096, NACC, in, out, rec};
    double mhz, ms = run_timed(launch_mma, &t, rec, &mhz, 5);
    double mmas = (double)t.blocks * (t.threads / 32) * t.iters * NACC;
    double tflops = mmas * flop_per_mma / (ms * 1e-3) / 1e12;
    double per_clk_sm = mmas * flop_per_mma / (ms * 1e-3 * mhz * 1e6) / g_sms;
    printf("FACT mma=%s warps_per_sm=%d tflops=%.1f dense_flop_per_clk_per_sm=%.0f sm_mhz=%.0f\n", I.name, wps,
           tflops, per_clk_sm, mhz);
    if (tflops > best) { best = tflops; best_clk = per_clk_sm; best_w = wps; best_mhz = mhz; }
  }
  printf("FACT mma=%s best_tflops=%.1f best_flop_per_clk_per_sm=%.0f at_warps_per_sm=%d sm_mhz=%.0f\n", I.name,
         best, best_clk, best_w, best_mhz);
  // latency: one warp, one dependent chain
  MmaArg l{kind, 1, 32, 4096, 1, in, out, rec};
  double mhz, ms = run_timed(launch_mma, &l, rec, &mhz, 5);
  ClockRec h; CK(cudaMemcpy(&h, rec, sizeof h, cudaMemcpyDeviceToHost));
  printf("FACT mma=%s dep_latency_cycles=%.1f\n", I.name, (double)(h.c1 - h.c0) / l.iters);
  (void)ms;
  CK(cudaFree(in)); CK(cudaFree(out)); CK(cudaFree(rec));
}

// ------------------------------------------------------------------ FP32 FFMA and MUFU.EX2
__global__ void k_ffma(float x, int iters, float* out, ClockRec* rec) {
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = x + j + threadIdx.x * 1e-6f;  // per-thread: FFMA, not UFFMA
  const float m = 0.999f, c = 0.001f;
  clk_begin(rec);
#pragma unroll 16
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < 8; ++j) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v[j]) : "f"(m), "f"(c));
  }
  clk_end(rec);
  float s = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) s += v[j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
__global__ void k_ex2(float x, int iters, float* out, ClockRec* rec) {
  float v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = x * (j + 1) * 1e-3f + threadIdx.x * 1e-7f;  // per-thread values
  clk_begin(rec);
#pragma unroll 4
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < 8; ++j) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(v[j]));
  }
  clk_end(rec);
  float s = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) s += v[j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
struct AluArg { int blocks, threads, iters, ex2; float* out; ClockRec* rec; };
static void launch_alu(void* p) {
  AluArg* m = (AluArg*)p;
  if (m->ex2) k_ex2<<<m->blocks, m->threads>>>(0.5f, m->iters, m->out, m->rec);
  else k_ffma<<<m->blocks, m->threads>>>(0.5f, m->iters, m->out, m->rec);
}
static void mode_alu() {
  float* out; ClockRec* rec;
  AluArg m{resident_blocks((const void*)k_ffma, 512), 512, 1 << 14, 0, nullptr, nullptr};
  CK(cudaMalloc(&out, (size_t)m.blocks * m.threads * 4)); CK(cudaMalloc(&rec, sizeof(ClockRec)));
  m.out = out; m.rec = rec;
  double mhz, ms = run_timed(launch_alu, &m, rec, &mhz, 5);
  double ops = (double)m.blocks * m.threads * m.iters * 8;
  printf("FACT ffma_tflops=%.1f ffma_per_clk_per_sm=%.0f sm_mhz=%.0f\n", 2 * ops / ms / 1e9, ops / (ms * 1e-3 * mhz * 1e6) / g_sms, mhz);
  m.ex2 = 1; m.iters = 1 << 12;
  ms = run_timed(launch_alu, &m, rec, &mhz, 5);
  ops = (double)m.blocks * m.threads * m.iters * 8;
  printf("FACT mufu_ex2_per_clk_per_sm=%.1f sm_mhz=%.0f\n", ops / (ms * 1e-3 * mhz * 1e6) / g_sms, mhz);
  CK(cudaFree(out)); CK(cudaFree(rec));
}

// ------------------------------------------------------------------ launch overhead
__global__ void k_empty() {}
static void mode_launch() {
  const int n = 20000;
  cudaStream_t s; CK(cudaStreamCreate(&s));
  for (int i = 0; i < 1000; ++i) k_empty<<<1, 32, 0, s>>>();
  CK(cudaStreamSynchronize(s));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a, s));
  for (int i = 0; i < n; ++i) k_empty<<<1, 32, 0, s>>>();
  CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
  float ms; CK(cudaEventElapsedTime(&ms, a, b));
  printf("FACT stream_launch_us_per_kernel=%.2f\n", ms * 1e3 / n);
  // the same kernels captured in a CUDA graph
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < 1000; ++i) k_empty<<<1, 32, 0, s>>>();
  CK(cudaStreamEndCapture(s, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
  CK(cudaEventRecord(a, s));
  for (int i = 0; i < n / 1000; ++i) CK(cudaGraphLaunch(ge, s));
  CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b));
  CK(cudaEventElapsedTime(&ms, a, b));
  printf("FACT graph_us_per_kernel=%.2f\n", ms * 1e3 / n);
}

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: facts attrs|dram <MiB>|l2 <MiB...>|lat <KiB...>|smem|mma <kind>|alu|launch\n"); return 2; }
  CK(cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0));
  const char* m = argv[1];
  if (!strcmp(m, "attrs")) mode_attrs();
  else if (!strcmp(m, "dram")) mode_dram(argc > 2 ? strtoull(argv[2], 0, 10) : 2048);
  else if (!strcmp(m, "l2")) mode_l2(argc - 2, argv + 2);
  else if (!strcmp(m, "lat")) mode_lat(argc - 2, argv + 2);
  else if (!strcmp(m, "smem")) mode_smem();
  else if (!strcmp(m, "mma")) mode_mma(argv[2]);
  else if (!strcmp(m, "alu")) mode_alu();
  else if (!strcmp(m, "launch")) mode_launch();
  else { printf("ERROR unknown mode %s\n", m); return 2; }
  CK(cudaDeviceSynchronize());
  printf("DONE mode=%s\n", m);
  return 0;
}
