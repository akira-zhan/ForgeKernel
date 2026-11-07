"""Causal GQA attention forward reference: cuDNN through torch SDPA (bf16)."""

import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def run(q, k, v):
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
