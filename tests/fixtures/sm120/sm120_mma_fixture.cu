// Fixture source for the sm_120a SASS fixtures in this directory.
// Regenerate:  nvcc -gencode arch=compute_120a,code=sm_120a -cubin -o f.cubin sm120_mma_fixture.cu
//              && cuobjdump -sass f.cubin
// (the block-scaled mma.sync forms exist only on the arch-specific target: a plain
//  -arch=sm_120a executable build also emits compute_120 PTX and ptxas rejects them there)
// The first four kernels are minimal k-loops: cp.async (LDGSTS) -> ldmatrix (LDSM) -> warp
// MMA, in the four sm_120a tensor-core spellings: FP8 f32-acc, FP8 f16-acc, MXF8
// block-scaled, NVF4 block-scaled. The fifth is a named-barrier phase construction for
// struct_tools (profiled: qmma_named_barrier_phases.source.csv). Numerics are irrelevant;
// only the instruction stream matters.
#include <cstdint>

__device__ __forceinline__ void cp16(uint32_t s, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g));
}
__device__ __forceinline__ void ldsm4(uint32_t s, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(s));
}
__device__ __forceinline__ void ldsm2(uint32_t s, uint32_t& b0, uint32_t& b1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(s));
}

#define KLOOP_PROLOGUE                                                         \
  __shared__ alignas(128) uint8_t sA[2][16 * 64];                              \
  __shared__ alignas(128) uint8_t sB[2][8 * 64];                               \
  int t = threadIdx.x;                                                         \
  uint32_t a = (uint32_t)__cvta_generic_to_shared(&sA[0][0]);                  \
  uint32_t b = (uint32_t)__cvta_generic_to_shared(&sB[0][0]);

__global__ void qmma_f32acc_loop(const uint8_t* A, const uint8_t* B, float* C, int ktiles) {
  KLOOP_PROLOGUE
  float d[4] = {0, 0, 0, 0};
  for (int k = 0; k < ktiles; ++k) {
    int s = k & 1;
    cp16(a + s * 1024 + t * 16, A + (size_t)k * 1024 + t * 16);
    cp16(b + s * 512 + (t & 31) * 16, B + (size_t)k * 512 + (t & 31) * 16);
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
    __syncthreads();
    uint32_t a0, a1, a2, a3, b0, b1;
    ldsm4(a + s * 1024 + (t & 15) * 64, a0, a1, a2, a3);
    ldsm2(b + s * 512 + (t & 7) * 64, b0, b1);
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    __syncthreads();
  }
  C[t * 4 + 0] = d[0]; C[t * 4 + 1] = d[1]; C[t * 4 + 2] = d[2]; C[t * 4 + 3] = d[3];
}

__global__ void qmma_f16acc_loop(const uint8_t* A, const uint8_t* B, uint32_t* C, int ktiles) {
  KLOOP_PROLOGUE
  uint32_t c0 = 0, c1 = 0;
  for (int k = 0; k < ktiles; ++k) {
    int s = k & 1;
    cp16(a + s * 1024 + t * 16, A + (size_t)k * 1024 + t * 16);
    cp16(b + s * 512 + (t & 31) * 16, B + (size_t)k * 512 + (t & 31) * 16);
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
    __syncthreads();
    uint32_t a0, a1, a2, a3, b0, b1;
    ldsm4(a + s * 1024 + (t & 15) * 64, a0, a1, a2, a3);
    ldsm2(b + s * 512 + (t & 7) * 64, b0, b1);
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                 : "+r"(c0), "+r"(c1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    __syncthreads();
  }
  C[t * 2 + 0] = c0; C[t * 2 + 1] = c1;
}

__global__ void qmma_mxf8_blockscaled_loop(const uint8_t* A, const uint8_t* B, const uint32_t* SF,
                                           float* C, int ktiles) {
  KLOOP_PROLOGUE
  float d[4] = {0, 0, 0, 0};
  for (int k = 0; k < ktiles; ++k) {
    int s = k & 1;
    cp16(a + s * 1024 + t * 16, A + (size_t)k * 1024 + t * 16);
    cp16(b + s * 512 + (t & 31) * 16, B + (size_t)k * 512 + (t & 31) * 16);
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
    __syncthreads();
    uint32_t a0, a1, a2, a3, b0, b1;
    ldsm4(a + s * 1024 + (t & 15) * 64, a0, a1, a2, a3);
    ldsm2(b + s * 512 + (t & 7) * 64, b0, b1);
    uint32_t sfa = SF[k * 64 + t], sfb = SF[k * 64 + 32 + (t & 31)];
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "h"((uint16_t)0));
    __syncthreads();
  }
  C[t * 4 + 0] = d[0]; C[t * 4 + 1] = d[1]; C[t * 4 + 2] = d[2]; C[t * 4 + 3] = d[3];
}

__global__ void omma_nvf4_blockscaled_loop(const uint8_t* A, const uint8_t* B, const uint32_t* SF,
                                           float* C, int ktiles) {
  KLOOP_PROLOGUE
  float d[4] = {0, 0, 0, 0};
  for (int k = 0; k < ktiles; ++k) {
    int s = k & 1;
    cp16(a + s * 1024 + t * 16, A + (size_t)k * 1024 + t * 16);
    cp16(b + s * 512 + (t & 31) * 16, B + (size_t)k * 512 + (t & 31) * 16);
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
    __syncthreads();
    uint32_t a0, a1, a2, a3, b0, b1;
    ldsm4(a + s * 1024 + (t & 15) * 64, a0, a1, a2, a3);
    ldsm2(b + s * 512 + (t & 7) * 64, b0, b1);
    uint32_t sfa = SF[k * 64 + t], sfb = SF[k * 64 + 32 + (t & 31)];
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},%10,{%12,%12},%11,{%12,%12};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "h"((uint16_t)0));
    __syncthreads();
  }
  C[t * 4 + 0] = d[0]; C[t * 4 + 1] = d[1]; C[t * 4 + 2] = d[2]; C[t * 4 + 3] = d[3];
}

// Phase-fenced variant for struct_tools: 4 warps (128 threads) cooperate through a
// NAMED barrier (bar.sync 1, 128). Phase 1: every warp QMMAs its own 16-row slice
// and publishes the f32 fragment as packed bf16 pairs to smem; phase 2: every warp
// reads a neighbour's slice back (the smem hand-off). Unrolled so the page is a
// realistic size (> 500 instructions).
__global__ void qmma_named_barrier_phases(const uint8_t* A, const uint8_t* B, float* C, int ktiles) {
  __shared__ alignas(128) uint8_t sA[4][16 * 64];
  __shared__ alignas(128) uint8_t sB[8 * 64];
  __shared__ alignas(128) uint32_t sX[4][32 * 2];
  int t = threadIdx.x, w = t >> 5, l = t & 31;
  uint32_t a = (uint32_t)__cvta_generic_to_shared(&sA[w][0]);
  uint32_t b = (uint32_t)__cvta_generic_to_shared(&sB[0]);
  float acc = 0.f;
#pragma unroll 1
  for (int k = 0; k < ktiles; k += 24) {
#pragma unroll
    for (int u = 0; u < 24; ++u) {
      cp16(a + l * 16, A + (size_t)(k + u) * 4096 + t * 16);
      if (w == 0) cp16(b + l * 16, B + (size_t)(k + u) * 512 + l * 16);
      asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
      asm volatile("bar.sync 1, 128;");
      uint32_t a0, a1, a2, a3, b0, b1;
      ldsm4(a + (l & 15) * 64, a0, a1, a2, a3);
      ldsm2(b + (l & 7) * 64, b0, b1);
      float d[4] = {0, 0, 0, 0};
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                   : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      uint32_t p0, p1;
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p0) : "f"(d[1]), "f"(d[0]));
      asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p1) : "f"(d[3]), "f"(d[2]));
      sX[w][l * 2 + 0] = p0; sX[w][l * 2 + 1] = p1;
      asm volatile("bar.sync 1, 128;");
      uint32_t q = sX[(w + 1) & 3][l * 2 + (u & 1)];
      acc += __uint_as_float(q << 16) + __uint_as_float(q & 0xffff0000u);
    }
  }
  C[t] = acc;
}
