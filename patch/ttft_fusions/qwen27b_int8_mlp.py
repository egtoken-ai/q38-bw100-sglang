# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 SiLU/INT8 quantization fusion for the deployed dense Qwen27B.

Only imported by the patched qwen3_5 model. The existing vendor RowParallelLinear
retains GEMM, TP reduction, and bias handling. Unsupported calls use its original
MLP implementation. This module does not alter Gemma normalization or residuals.
"""
import logging

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as int8_scheme,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var, is_hcu

logger = logging.getLogger(__name__)


class Qwen27BInt8MLP(Qwen2MoeMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheme = getattr(self.down_proj, "scheme", None)
        self._qwen27b_silu_quant = (
            get_bool_env_var("SGLANG_QWEN27B_FUSED_SILU_QUANT")
            and is_hcu()
            and int8_scheme._use_fused_silu_mul_quant
            and type(scheme) is int8_scheme.CompressedTensorsW8A8Int8
            and scheme.strategy == QuantizationStrategy.CHANNEL
            and not scheme.is_static_input_scheme
            and scheme.input_symmetric
            and scheme.w8a8_strategy == 1
            and self.down_proj.input_is_parallel
            and not self.down_proj.use_dp_attention_reduce
            and self.down_proj.tp_size == 2
            and self.down_proj.input_size == 17408
            and self.down_proj.output_size == 5120
            and get_global_server_args().rl_on_policy_target is None
        )
        if self._qwen27b_silu_quant:
            logger.info(
                "Qwen27B BF16 SiLU/INT8 quantization fusion enabled for %s; "
                "minimum flattened token count 256", kwargs.get("prefix", "")
            )

    def forward(self, x, should_allreduce_fusion=False, use_reduce_scatter=False):
        if not (
            self._qwen27b_silu_quant
            and x.dtype == torch.bfloat16
            and x.is_cuda
            and x.ndim == 2
            and x.shape[0] >= 256
        ):
            return super().forward(x, should_allreduce_fusion, use_reduce_scatter)
        gate_up, _ = self.gate_up_proj(x)
        # Rows contain [gate, up], each of width intermediate_size / TP = 8704.
        # The vendor fused operation halves this width before the INT8 GEMM.
        x, _ = self.down_proj(
            gate_up,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
            use_fused_silu_mul_quant=True,
        )
        return x
