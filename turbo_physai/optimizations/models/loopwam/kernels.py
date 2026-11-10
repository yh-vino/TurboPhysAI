# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""LoopWAM replacements preserving the model's tensor and gradient contracts."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rmsnorm_forward(self, x):
    # Preserve the upstream ordering: normalize in fp32, cast, then multiply
    # by the learned weight (which can promote the output dtype).
    return F.rms_norm(x.float(), (x.shape[-1],), None, self.eps).to(x.dtype) * self.weight


def causal_conv3d_forward(self, x, cache_x=None):
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        x = torch.cat((cache_x.to(x.device), x), dim=2)
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding).contiguous(memory_format=torch.channels_last_3d)
    # Convert on every call: inexpensive when already contiguous, and remains
    # correct after load_state_dict, dtype/device moves, or weight replacement.
    # Keep the Parameter object and its optimizer/autograd identity intact.
    with torch.no_grad():
        self.weight.data = self.weight.data.contiguous(memory_format=torch.channels_last_3d)
    return self._conv_forward(x, self.weight, self.bias)


def masked_attention(q, k, v, mask, heads):
    """Adapt SDPA's True=attend mask to DTK's vendor FlashAttention API."""
    from flash_attn import flash_attn_with_mask_func

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("LoopWAM masked attention requires [batch, sequence, hidden] tensors")
    batch, seq_q, hidden = q.shape
    if heads <= 0 or hidden % heads:
        raise ValueError("LoopWAM attention hidden dimension must be divisible by num_heads")
    if k.shape != v.shape or k.shape[0] != batch or k.shape[2] != hidden:
        raise ValueError("LoopWAM attention requires matching key/value and batch/hidden dimensions")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("LoopWAM vendor masked attention requires matching fp16/bf16 tensors")
    if mask.dtype != torch.bool:
        raise ValueError("LoopWAM masked attention requires a boolean mask (True = attend)")
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        # SDPA aligns a 3-D mask with [heads, query, key], not batch.
        mask = mask[None]
    elif mask.ndim != 4:
        raise ValueError("LoopWAM attention mask must have 2, 3, or 4 dimensions")
    seq_k = k.shape[1]
    mask = mask.to(device=q.device).contiguous().expand(batch, heads, seq_q, seq_k)
    shape_q, shape_k = (batch, seq_q, heads, hidden // heads), (batch, seq_k, heads, hidden // heads)
    output = flash_attn_with_mask_func(
        q.reshape(shape_q), k.reshape(shape_k), v.reshape(shape_k), mask,
        dropout_p=0.0, causal=False,
    )
    return output.reshape(batch, seq_q, hidden)


def mixed_attention(self, q_cat, k_cat, v_cat, attention_mask):
    def forward(q, k, v):
        return masked_attention(q, k, v, attention_mask, self.num_heads)

    if self.mot_checkpoint_mixed_attn and self.training:
        return checkpoint(forward, q_cat, k_cat, v_cat, use_reentrant=False)
    return forward(q_cat, k_cat, v_cat)
