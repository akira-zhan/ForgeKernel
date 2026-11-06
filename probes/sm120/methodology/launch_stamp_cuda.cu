// launch_identity_stamp, CUDA C++ form, on sm_120a.
//   nvcc -gencode arch=compute_120a,code=sm_120a -O3 -o launch_stamp_cuda launch_stamp_cuda.cu && ./launch_stamp_cuda
// Contracts (one PROBE line each): stamp_A / stamp_B (every CTA row == expectation table written here from the
// launch parameters, rows beyond the grid stay -1), gate_catches (B's rows vs A's table mismatch),
// cluster_regs (%cluster_nctaid = cluster shape, %nclusterid = number of clusters, %cluster_ctarank in range),
// smem_optin_limit (cudaFuncSetAttribute accepts 101376 B and rejects 101376 + 1024 B).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#define SREG(name, out) asm volatile("mov.u32 %0, %%" name ";" : "=r"(out))
constexpr int NF = 16, ROWS = 128;

template <int VARIANT, int STAGES>
__global__ void __launch_bounds__(128) k_stamp(int* tab) {
  extern __shared__ __align__(16) unsigned char dsmem[];
  if (threadIdx.x != 0) return;
  dsmem[0] = 1;
  uint32_t ncx, ntx, cnx, cny, cnz, cnr, crank, nclx, ncly, nclz, dyn;
  SREG("nctaid.x", ncx);  SREG("ntid.x", ntx);
  SREG("cluster_nctaid.x", cnx); SREG("cluster_nctaid.y", cny); SREG("cluster_nctaid.z", cnz);
  SREG("cluster_nctarank", cnr);
  SREG("cluster_ctarank", crank);
  SREG("nclusterid.x", nclx); SREG("nclusterid.y", ncly); SREG("nclusterid.z", nclz);
  SREG("dynamic_smem_size", dyn);
  const int lin = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
  int* r = tab + lin * NF;
  r[0] = VARIANT; r[1] = STAGES; r[2] = (int)ncx; r[3] = (int)ntx;
  r[4] = (int)cnx; r[5] = (int)cny; r[6] = (int)cnz; r[7] = (int)cnr; r[8] = (int)crank;
  r[9] = (int)nclx; r[10] = (int)ncly; r[11] = (int)nclz; r[12] = (int)dyn; r[13] = 0x5EED;
}

struct Launch { int variant, stages; dim3 grid, cluster; int smem; };

template <int V, int S>
static cudaError_t launch(const Launch& L, int* tab) {
  auto kern = k_stamp<V, S>;
  cudaError_t e = cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, L.smem);
  if (e != cudaSuccess) return e;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = L.grid; cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = L.smem; cfg.stream = 0;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = L.cluster.x; attr[0].val.clusterDim.y = L.cluster.y; attr[0].val.clusterDim.z = L.cluster.z;
  cfg.attrs = attr; cfg.numAttrs = 1;
  e = cudaLaunchKernelEx(&cfg, kern, tab);
  if (e != cudaSuccess) return e;
  return cudaDeviceSynchronize();
}

// expectation table written from the launch parameters; returns max |diff| over the rows and the -1 tail
static long check(const std::vector<int>& got, const Launch& L, bool* tail_ok, bool* rank_ok) {
  int n = L.grid.x * L.grid.y * L.grid.z, csz = L.cluster.x * L.cluster.y * L.cluster.z;
  long md = 0; *rank_ok = true;
  for (int b = 0; b < n; ++b) {
    int exp[14] = {L.variant, L.stages, (int)L.grid.x, 128, (int)L.cluster.x, (int)L.cluster.y, (int)L.cluster.z, csz,
                   -1, (int)(L.grid.x / L.cluster.x), (int)(L.grid.y / L.cluster.y), (int)(L.grid.z / L.cluster.z),
                   L.smem, 0x5EED};
    for (int f = 0; f < 14; ++f) {
      if (f == 8) { int r = got[b * NF + f]; if (r < 0 || r >= csz) *rank_ok = false; continue; }
      long d = got[b * NF + f] - exp[f]; if (d < 0) d = -d; if (d > md) md = d;
    }
  }
  *tail_ok = true;
  for (int i = n * NF; i < ROWS * NF; ++i) if (got[i] != -1) *tail_ok = false;
  return md;
}

int main() {
  int* d; std::vector<int> h(ROWS * NF);
  if (cudaMalloc(&d, ROWS * NF * 4) != cudaSuccess) { printf("ERROR malloc\n"); return 2; }
  const Launch A{1, 7, dim3(1, 96, 1), dim3(1, 8, 1), 65536};
  const Launch B{2, 3, dim3(8, 4, 1), dim3(4, 1, 1), 98304};
  int fails = 0;
  auto run = [&](const char* tag, const Launch& L, cudaError_t (*fn)(const Launch&, int*)) {
    cudaMemset(d, 0xff, ROWS * NF * 4);
    cudaError_t e = fn(L, d);
    cudaMemcpy(h.data(), d, ROWS * NF * 4, cudaMemcpyDeviceToHost);
    bool tail, rank; long md = check(h, L, &tail, &rank);
    bool ok = e == cudaSuccess && md == 0 && tail && rank;
    printf("FACT %s err=%s row0=", tag, cudaGetErrorString(e));
    for (int f = 0; f < 14; ++f) printf("%d%s", h[f], f < 13 ? "," : "\n");
    printf("PROBE %s %s (max_diff=%ld tail=%d ranks=%d)\n", ok ? "PASS" : "FAIL", tag, md, tail, rank);
    fails += !ok;
  };
  run("stamp_A", A, launch<1, 7>);
  run("stamp_B", B, launch<2, 3>);
  // gate: B's rows against A's table must mismatch
  bool tail, rank; long mism = check(h, A, &tail, &rank);
  printf("PROBE %s gate_catches (max_diff_vs_A=%ld)\n", mism ? "PASS" : "FAIL", mism); fails += !mism;
  // cluster registers of B: shape (4,1,1), number of clusters (2,4,1)
  bool creg = h[4] == 4 && h[5] == 1 && h[9] == 2 && h[10] == 4 && h[7] == 4;
  printf("PROBE %s cluster_regs (cluster_nctaid=(%d,%d,%d) nclusterid=(%d,%d,%d) nctarank=%d)\n",
         creg ? "PASS" : "FAIL", h[4], h[5], h[6], h[9], h[10], h[11], h[7]); fails += !creg;
  // opt-in limit
  cudaError_t at = cudaFuncSetAttribute(k_stamp<3, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
  cudaError_t over = cudaFuncSetAttribute(k_stamp<3, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376 + 1024);
  cudaGetLastError();
  bool lim = at == cudaSuccess && over != cudaSuccess;
  printf("FACT smem opt-in 101376 -> %s; 102400 -> %s\n", cudaGetErrorString(at), cudaGetErrorString(over));
  printf("PROBE %s smem_optin_limit\n", lim ? "PASS" : "FAIL"); fails += !lim;
  printf("PROBE %s launch-stamp-cuda contracts=5\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
