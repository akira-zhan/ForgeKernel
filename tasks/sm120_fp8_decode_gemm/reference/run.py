"""FP8 rowwise-scaled decode GEMM reference: the CUTLASS sm120 kernel shipped in sgl_kernel (bf16 output)."""

import sgl_kernel
import torch


def run(a, b, scale_a, scale_b):
    return sgl_kernel.fp8_scaled_mm(a, b.T, scale_a, scale_b, torch.bfloat16)
