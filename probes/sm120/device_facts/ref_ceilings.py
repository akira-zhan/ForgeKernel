#!/usr/bin/env python3
"""Library reference ceilings on the RTX PRO 5000: 4096^3 GEMMs through cuBLAS / cuBLASLt (torch) and CUTLASS sm120
(sgl_kernel.fp8_scaled_mm). Median of 50 launches, clocks unlocked, bf16 output. INFO readings, no contracts: the
bar a hand-written sm120 mainloop is compared against.

    python3 ref_ceilings.py      (GPU; torch >= 2.14 for the block-scaled forms, sgl_kernel optional)
"""
import torch
try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None
M = N = K = 4096
dev = "cuda"
def bench(fn, reps=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = []
    for _ in range(reps):
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); t.append(e0.elapsed_time(e1))
    ms = sorted(t)[reps // 2]
    return ms, 2.0 * M * N * K / ms / 1e9
def run(name, fn):
    try:
        ms, tf = bench(fn); print(f"{name:52s} {ms:.3f} ms  {tf:7.1f} TFLOPS", flush=True)
    except Exception as e:
        print(f"{name:52s} UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:110]}", flush=True)
a16 = torch.randn(M, K, device=dev, dtype=torch.bfloat16); b16 = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
run("bf16 torch.matmul (cuBLAS)", lambda: a16 @ b16.T)
a8 = a16.to(torch.float8_e4m3fn); b8 = b16.to(torch.float8_e4m3fn)
one = torch.ones((), device=dev)
run("e4m3 torch._scaled_mm per-tensor (cuBLASLt)", lambda: torch._scaled_mm(a8, b8.T, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
sa = torch.ones(M, 1, device=dev); sb = torch.ones(N, 1, device=dev)
run("e4m3 torch._scaled_mm rowwise (cuBLASLt/CUTLASS)", lambda: torch._scaled_mm(a8, b8.T, scale_a=sa, scale_b=sb.T, out_dtype=torch.bfloat16))
run("e4m3 sgl_kernel.fp8_scaled_mm (CUTLASS sm120)", lambda: sgl_kernel.fp8_scaled_mm(a8, b8.T, sa, sb, torch.bfloat16))
# block-scaled: scales in the 128x4 blocked layout (sizes are what matter for timing)
e8 = lambda r, c: torch.full((r * c,), 127, device=dev, dtype=torch.uint8).view(torch.float8_e8m0fnu)
run("mxf8 torch._scaled_mm e8m0/32 (cuBLASLt)", lambda: torch._scaled_mm(a8, b8.T, scale_a=e8(M, K // 32), scale_b=e8(N, K // 32), out_dtype=torch.bfloat16))
a4 = torch.randint(0, 255, (M, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
b4 = torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
run("mxf4 torch._scaled_mm e8m0/32 (cuBLASLt)", lambda: torch._scaled_mm(a4, b4.T, scale_a=e8(M, K // 32), scale_b=e8(N, K // 32), out_dtype=torch.bfloat16))
u43 = lambda r, c: torch.ones(r * c, device=dev).to(torch.float8_e4m3fn)
run("nvf4 torch._scaled_mm e4m3/16 (cuBLASLt)", lambda: torch._scaled_mm(a4, b4.T, scale_a=u43(M, K // 16), scale_b=u43(N, K // 16), out_dtype=torch.bfloat16))
