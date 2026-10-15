"""Opt-in dense Qwen27B TP2 prefill boundary pipeline.

Attention/GDN and their metadata stay whole-batch. Only the ordinary
post-attention AR + original norm + MLP + output AR boundary is microbatched.
"""
from functools import partial
import logging
import os

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.communicator import (
    CommunicateSummableTensorPairFn, CommunicateWithAllReduceAndLayerNormFn,
    ScatterMode, get_attn_tp_context,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var, is_hcu

logger = logging.getLogger(__name__)
_STREAMS = {}
_HITS = 0


def rejection_reason(layer, hidden_states, residual, forward_batch):
    """No mutation, allocation or collective before every guard succeeds."""
    if not get_bool_env_var('SGLANG_QWEN27B_MLP_PIPELINE', default='false'):
        return 'disabled'
    if not is_hcu():
        return 'not HCU'
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(residual, torch.Tensor):
        return 'tensor input and residual required'
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 5120:
        return 'hidden shape'
    if hidden_states.shape[0] < 32768 or hidden_states.shape[0] % 4:
        return 'minimum32768 and divisible4'
    if residual.shape != hidden_states.shape or residual.device != hidden_states.device:
        return 'residual shape/device'
    if hidden_states.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        return 'BF16 only'
    if not hidden_states.is_contiguous() or not residual.is_contiguous():
        return 'contiguous input/residual required'
    if hidden_states.data_ptr() == residual.data_ptr():
        return 'input/residual alias'
    if forward_batch.batch_size < 2:
        return 'single request stays baseline'
    if not forward_batch.forward_mode.is_extend_without_speculative() or forward_batch.spec_info is not None:
        return 'ordinary non-spec extend only'
    args = get_global_server_args()
    if args.tp_size != 2 or args.dp_size != 1 or args.pp_size != 1:
        return 'TP2 DP1 PP1 only'
    if args.enable_dp_attention or args.attn_cp_size != 1:
        return 'DP/CP attention disabled required'
    if args.enable_torch_compile or not args.disable_piecewise_cuda_graph:
        return 'eager only'
    if args.enable_two_batch_overlap or args.enable_quant_communications:
        return 'nested overlap or quant communications'
    if getattr(args, 'enable_lora', False):
        return 'LoRA unsupported'
    if os.environ.get('TORCH_NCCL_BLOCKING_WAIT', '0') != '0':
        return 'blocking NCCL wait'
    if get_attn_tp_context().input_scattered:
        return 'input scattered'
    comm = layer.layer_communicator
    ctx = comm._context
    if ctx.tp_size != 2 or ctx.attn_tp_size != 2 or ctx.attn_dp_size != 1 or ctx.attn_cp_size != 1:
        return 'communicator topology'
    if ctx.cache is not None:
        return 'communicator side cache'
    prepare = comm._communicate_with_all_reduce_and_layer_norm_fn
    if not isinstance(prepare, partial) or prepare.func is not CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual:
        return 'prepare_mlp branch'
    if prepare.keywords.get('residual_input_mode') != ScatterMode.TP_ATTN_FULL:
        return 'residual scatter mode'
    if comm._communicate_summable_tensor_pair_fn is not CommunicateSummableTensorPairFn._trivial:
        return 'postprocess must be trivial'
    if comm.should_use_reduce_scatter(forward_batch) or comm.should_fuse_mlp_allreduce_with_next_layer(forward_batch):
        return 'reduce scatter or deferred AR fusion'
    if CommunicateWithAllReduceAndLayerNormFn._skip_layernorm(forward_batch):
        return 'fused quant owns norm'
    # Guard the active boundary AR fusion as well as the deferred MLP one.
    from sglang.srt.layers.communicator import apply_aiter_all_reduce_fusion, apply_flashinfer_allreduce_fusion
    if apply_aiter_all_reduce_fusion(hidden_states) or apply_flashinfer_allreduce_fusion(hidden_states.shape[0]):
        return 'boundary AR fusion'
    mlp = layer.mlp
    if type(mlp).__name__ != 'Qwen2MoeMLP' or type(layer.post_attention_layernorm).__name__ != 'GemmaRMSNorm':
        return 'dense MLP and original GemmaRMSNorm required'
    if not mlp.down_proj.reduce_results or not mlp.down_proj.input_is_parallel:
        return 'down projection semantics'
    for projection in (mlp.gate_up_proj, mlp.down_proj):
        if projection.tp_size != 2 or getattr(projection, 'use_dp_attention_reduce', False):
            return 'projection topology'
        scheme = getattr(projection, 'scheme', None)
        if type(scheme).__name__ != 'CompressedTensorsW8A8Int8':
            return 'W8A8 INT8 scheme only'
        if scheme.is_static_input_scheme or not scheme.input_symmetric or str(scheme.strategy).lower().split('.')[-1] != 'channel':
            return 'dynamic symmetric/channel quant required'
        if projection.weight.dtype != torch.int8 or projection.bias is not None:
            return 'INT8 no-bias projection required'
    tp = get_tp_group()
    attn = get_attention_tp_group()
    if tp.world_size != 2 or attn.world_size != 2 or list(tp.ranks) != list(attn.ranks):
        return 'attention and MLP TP ranks must match'
    if get_is_capture_mode() or torch.cuda.is_current_stream_capturing():
        return 'graph capture disabled'
    return None


def try_mlp_pipeline(layer, hidden_states, residual, forward_batch):
    global _HITS
    if rejection_reason(layer, hidden_states, residual, forward_batch) is not None:
        return None
    # Guard success is rank-invariant: topology/configuration and batch shapes
    # arrive through the scheduler's TP broadcast. No rank-local speed heuristic.
    device = hidden_states.device.index
    if device not in _STREAMS:
        _STREAMS[device] = ([torch.cuda.Stream(device=device) for _ in range(4)],
                            [torch.cuda.Event() for _ in range(4)])
    streams, compute_done = _STREAMS[device]
    current = torch.cuda.current_stream(hidden_states.device)
    group = get_tp_group().device_group
    xs, rs = hidden_states.chunk(4, 0), residual.chunk(4, 0)
    input_works, output_works, outputs, residual_outputs = [], [], [], []
    # One process group, same order on both ranks: in0,in1,in2,in3,out0,...out3.
    for i in range(4):
        streams[i].wait_stream(current)
        with torch.cuda.stream(streams[i]):
            hidden_states.record_stream(streams[i])
            residual.record_stream(streams[i])
            input_works.append(dist.all_reduce(xs[i], group=group, async_op=True))
    for i in range(4):
        with torch.cuda.stream(streams[i]):
            input_works[i].wait()
            if i:
                streams[i].wait_event(compute_done[i-1])
            x, r = layer.post_attention_layernorm(xs[i], rs[i])
            # Use production MLP code/quantization/activation, but explicitly
            # skip its internal output AR. The caller owns exactly one AR below.
            y = layer.mlp(x, should_allreduce_fusion=False, use_reduce_scatter=True)
            compute_done[i].record(streams[i])
            output_works.append(dist.all_reduce(y, group=group, async_op=True))
            outputs.append(y)
            residual_outputs.append(r)
    for i in range(4):
        with torch.cuda.stream(streams[i]):
            output_works[i].wait()
        current.wait_stream(streams[i])
        outputs[i].record_stream(current)
        residual_outputs[i].record_stream(current)
    output = torch.cat(outputs, dim=0)
    # Preserve original post-attention residual values and token order even if
    # a backend later returns fresh residual tensors instead of input aliases.
    if all(value.data_ptr() == rs[i].data_ptr()
           and value.shape == rs[i].shape and value.stride() == rs[i].stride()
           for i, value in enumerate(residual_outputs)):
        residual_output = residual
    else:
        residual_output = torch.cat(residual_outputs, dim=0)
    _HITS += 1
    if _HITS == 1:
        logger.info('QWEN27B_MLP_PIPELINE_HIT layer=%s tokens=%s parts=4 dtype=BF16 W8A8_INT8',
                    layer.layer_id, hidden_states.shape[0])
    return layer.layer_communicator.postprocess_layer(output, residual_output, forward_batch)
