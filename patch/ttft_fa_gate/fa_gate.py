# SPDX-License-Identifier: Apache-2.0
"""Independent BF16 attention-gate candidate with the original sigmoid rounding."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import logging
import os

logger = logging.getLogger(__name__)
_ENABLED = os.environ.get("SGLANG_QWEN27B_FA_GATE", "0").lower() in ("1", "true")
_LOGGED = False


@triton.jit
def _gate(A, G, Y, SIZE: tl.constexpr, LIB_EXP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    mask = offsets < SIZE
    a = tl.load(A+offsets,mask,other=0).to(tl.float32)
    g = tl.load(G+offsets,mask,other=0).to(tl.float32)
    if LIB_EXP:
        sigmoid = 1.0/(1.0+libdevice.exp(-g))
    else:
        sigmoid = tl.sigmoid(g)
    # torch.sigmoid(BF16 gate) materializes BF16 before the multiply.
    sigmoid = sigmoid.to(tl.bfloat16).to(tl.float32)
    out = (a*sigmoid).to(tl.bfloat16)
    tl.store(Y+offsets,out,mask)


def fused_gate(attn, gate, *, libdevice_exp=False):
    if attn.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise ValueError("BF16 tensors required")
    if attn.shape != gate.shape or attn.ndim != 2 or attn.shape[1] != 3072:
        raise ValueError("Restricted to matching [tokens,3072] tensors")
    if not attn.is_cuda or attn.device != gate.device or not attn.is_contiguous() or not gate.is_contiguous():
        raise ValueError("Contiguous tensors on the same device required")
    out = torch.empty_like(attn)
    if attn.numel():
        _gate[(triton.cdiv(attn.numel(),1024),)](attn,gate,out,attn.numel(),libdevice_exp,1024,
                                               num_warps=4,enable_fp_fusion=False)
    return out


def try_fa_gate(layer, attn_output, gate, forward_batch):
    """Guarded serving adapter; None preserves the original two-operation path."""
    if not _ENABLED:
        return None
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.utils import is_hcu

    if not (
        is_hcu()
        and getattr(layer.config, "model_type", None) == "qwen3_5_text"
        and layer.hidden_size == 5120
        and layer.attn_tp_size == 2
        and layer.num_heads == 12
        and layer.num_kv_heads == 2
        and layer.head_dim == 256
        and layer.attn_output_gate
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and getattr(forward_batch, "spec_info", None) is None
        and not get_is_capture_mode()
        and attn_output.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and attn_output.ndim == 2
        and attn_output.shape == gate.shape
        and attn_output.shape[1] == 3072
        and 4096 <= attn_output.shape[0] <= 65536
        and attn_output.is_cuda
        and attn_output.device == gate.device
        and attn_output.is_contiguous()
        and gate.is_contiguous()
    ):
        return None
    result = fused_gate(attn_output, gate, libdevice_exp=False)
    global _LOGGED
    if not _LOGGED:
        logger.info("Qwen27B HCU TP2 prefill FA gate active: sigmoid BF16 -> multiply BF16")
        _LOGGED = True
    return result
