# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from SGLang PR #27656, commit 0da18f8d916eb2d99b7382851d19426554a74cf3.
# Zero-grid guard follows #31794, commit 6314e9e4f5bdadd975969e8781dd1edcb27bb4a7.
# Local scope: BF16 Qwen27B HCU TP2 prefill only; original arithmetic retained.
import logging
import os
from typing import Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("SGLANG_QWEN27B_QK_NORM_GATE", "0").lower() in ("1", "true")
_LOGGED = False

@triton.jit
def _fused_qk_gemma_rmsnorm_gate_kernel(
    QG_ptr,
    K_ptr,
    Q_out_ptr,
    K_out_ptr,
    Gate_out_ptr,
    QW_ptr,
    KW_ptr,
    qg_token_stride,
    qg_head_stride,
    k_token_stride,
    k_head_stride,
    num_heads,
    num_kv_heads,
    k_rows,
    HEAD_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_HD)
    mask = cols < HEAD_DIM
    out_dtype = tl.bfloat16

    token_idx = pid // num_heads
    head_idx = pid % num_heads

    base = token_idx * qg_token_stride + head_idx * qg_head_stride

    # Q norm
    q = tl.load(QG_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    w_q = tl.load(QW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    q_var = tl.sum(q * q, axis=0) / HEAD_DIM
    q_normed = (q * tl.rsqrt(q_var + EPS) * (w_q + 1.0)).to(out_dtype)
    out_off = pid * HEAD_DIM + cols
    tl.store(Q_out_ptr + out_off, q_normed, mask=mask)

    # Gate copy
    gate = tl.load(QG_ptr + base + HEAD_DIM + cols, mask=mask, other=0.0)
    tl.store(Gate_out_ptr + out_off, gate, mask=mask)

    # K norm (first k_rows blocks only)
    if pid < k_rows:
        token_idx_k = pid // num_kv_heads
        head_idx_k = pid % num_kv_heads
        k_off = token_idx_k * k_token_stride + head_idx_k * k_head_stride + cols
        k = tl.load(K_ptr + k_off, mask=mask, other=0.0).to(tl.float32)
        w_k = tl.load(KW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        k_var = tl.sum(k * k, axis=0) / HEAD_DIM
        k_normed = (k * tl.rsqrt(k_var + EPS) * (w_k + 1.0)).to(out_dtype)
        k_out_off = pid * HEAD_DIM + cols
        tl.store(K_out_ptr + k_out_off, k_normed, mask=mask)


def fused_qk_gemma_rmsnorm_with_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    head_dim: int,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QK GemmaRMSNorm + gate extraction from interleaved q_gate buffer.

    q_gate: (seq, q_size*2) where q and gate are interleaved per head,
            i.e. [q_h0, gate_h0, q_h1, gate_h1, ...] with q_size = num_heads * head_dim.
            Can be a non-contiguous view from qkv.split().
    k: (seq, kv_size) — same as fused_qk_gemma_rmsnorm.

    Returns (q_out, k_out, gate_out) all contiguous with shape
    (seq*num_heads, head_dim), (seq*num_kv_heads, head_dim), (seq*num_heads, head_dim).
    """
    if q_gate.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("Only BF16 Q/K/gate inputs are supported")
    if q_weight.dtype != torch.bfloat16 or k_weight.dtype != torch.bfloat16:
        raise ValueError("Only BF16 original Gemma weight offsets are supported")
    if head_dim != 256 or num_heads != 12 or k.shape[-1] != 512:
        raise ValueError("Restricted to TP2 Q12/KV2/H256 geometry")
    if q_gate.ndim != 2 or q_gate.shape[-1] != 6144 or k.shape[0] != q_gate.shape[0]:
        raise ValueError("Invalid Q/gate/K shapes")
    if q_gate.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("Last dimension must be contiguous")
    if q_weight.shape != (256,) or k_weight.shape != (256,):
        raise ValueError("Expected head-dimension Gemma weights")
    if not (q_gate.device == k.device == q_weight.device == k_weight.device):
        raise ValueError("Q/K/gate and weights must share one device")
    seq_len = q_gate.shape[0]
    qg_3d = q_gate.view(seq_len, num_heads, 2 * head_dim)
    num_kv_heads = k.shape[-1] // head_dim
    k_3d = k.view(seq_len, num_kv_heads, head_dim)

    q_rows = seq_len * num_heads
    k_rows = seq_len * num_kv_heads

    q_out = torch.empty(q_rows, head_dim, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(k_rows, head_dim, dtype=k.dtype, device=k.device)
    gate_out = torch.empty(q_rows, head_dim, dtype=q_gate.dtype, device=q_gate.device)

    # Official follow-up #31794: never launch a zero-sized HIP grid.
    if q_rows == 0:
        return q_out, k_out, gate_out

    BLOCK_HD = triton.next_power_of_2(head_dim)

    _fused_qk_gemma_rmsnorm_gate_kernel[(q_rows,)](
        qg_3d,
        k_3d,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        qg_3d.stride(0),
        qg_3d.stride(1),
        k_3d.stride(0),
        k_3d.stride(1),
        num_heads,
        num_kv_heads,
        k_rows,
        HEAD_DIM=head_dim,
        BLOCK_HD=BLOCK_HD,
        EPS=eps,
    )

    return q_out, k_out, gate_out


def try_qk_norm_gate(layer, qkv, forward_batch, *, rotary_fusion_enabled=False):
    """Return prepared Q/K/V/gate, or None to retain the original full branch."""
    if not _ENABLED or rotary_fusion_enabled:
        return None
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
    from sglang.srt.utils import is_hcu

    if not (
        is_hcu()
        and getattr(layer.config, "model_type", None) == "qwen3_5_text"
        and layer.hidden_size == 5120
        and layer.attn_tp_size == 2
        and layer.num_heads == 12
        and layer.num_kv_heads == 2
        and layer.head_dim == 256
        and layer.q_size == 3072
        and layer.kv_size == 512
        and layer.attn_output_gate
        and qkv.dtype == torch.bfloat16
        and qkv.is_cuda
        and qkv.ndim == 2
        and qkv.shape[1] == 7168
        and qkv.shape[0] > 0
        and qkv.stride(-1) == 1
        and layer.q_norm.weight.dtype == torch.bfloat16
        and layer.k_norm.weight.dtype == torch.bfloat16
        and layer.q_norm.variance_epsilon == layer.k_norm.variance_epsilon
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and getattr(forward_batch, "spec_info", None) is None
        and not get_is_capture_mode()
    ):
        return None
    q_gate, k, v = qkv.split([6144, 512, 512], dim=-1)
    q, k, gate = fused_qk_gemma_rmsnorm_with_gate(
        q_gate, k, layer.q_norm.weight.data, layer.k_norm.weight.data,
        layer.q_norm.variance_epsilon, 256, 12,
    )
    global _LOGGED
    if not _LOGGED:
        logger.info("Qwen27B HCU TP2 BF16 prefill QK norm + gate extraction active (official #27656)")
        _LOGGED = True
    tokens = qkv.shape[0]
    return q.view(tokens, 3072), k.view(tokens, 512), v, gate.view(tokens, 3072)

