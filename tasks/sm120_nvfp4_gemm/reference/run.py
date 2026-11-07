"""NVFP4 GEMM reference: cuBLASLt block-scaled matmul through torch._scaled_mm (bf16 output)."""

import torch


def run(a, sa, b, sb):
    return torch._scaled_mm(a, b.T, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
