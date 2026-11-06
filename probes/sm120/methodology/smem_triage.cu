// smem traffic triage probe: three kernels that move the same useful bytes through shared memory.
//   k_base     : conflict-free STS.128 then LDS.128, once per iteration
//   k_conflict : same instruction count and bytes, 128 B lane stride -> same-bank collisions
//   k_volume   : conflict-free, but every value makes an extra smem round trip (2x STS + 2x LDS)
// Built for sm_120a:  nvcc -gencode arch=compute_120a,code=sm_120a -O3 -lineinfo -o smem_triage smem_triage.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("ERROR %s\n", cudaGetErrorString(e_)); return 2; } } while (0)

constexpr int THREADS = 256, ITERS = 64;

__device__ __forceinline__ void sts128(uint32_t a, uint4 v) {
  asm volatile("st.volatile.shared.v4.u32 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}
__device__ __forceinline__ uint4 lds128(uint32_t a) {
  uint4 v;
  asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a));
  return v;
}

// STRIDE16 = distance between consecutive lanes in 16 B units (1 = contiguous, 8 = 128 B apart)
template <int STRIDE16, int TRIPS>
__global__ void k_smem(uint32_t* out) {
  extern __shared__ uint4 s[];
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(s);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  // each warp owns a 32 * STRIDE16 * 16 B window; lane l uses slot l * STRIDE16 in it
  const uint32_t mine = base + (uint32_t)((warp * 32 * STRIDE16 + lane * STRIDE16) * 16);
  uint4 v = make_uint4(threadIdx.x, blockIdx.x, 1, 2);
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int t = 0; t < TRIPS; ++t) {
      sts128(mine, v);
      __syncwarp();
      v = lds128(mine);
      v.x += 1;
      __syncwarp();
    }
  }
  if (v.x == 0x9e3779b9u) out[0] = v.y;
}

int main() {
  uint32_t* out;
  CK(cudaMalloc(&out, 4));
  int blocks = 110 * 4;
  size_t smem_base = THREADS * 16, smem_wide = THREADS * 16 * 8;
  CK(cudaFuncSetAttribute(k_smem<8, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_wide));
  k_smem<1, 1><<<blocks, THREADS, smem_base>>>(out);   // k_base
  k_smem<8, 1><<<blocks, THREADS, smem_wide>>>(out);   // k_conflict
  k_smem<1, 2><<<blocks, THREADS, smem_base>>>(out);   // k_volume
  CK(cudaDeviceSynchronize());
  printf("DONE blocks=%d threads=%d iters=%d\n", blocks, THREADS, ITERS);
  return 0;
}
