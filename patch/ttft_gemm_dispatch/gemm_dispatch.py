"""Opt-in legacy hipBLASLt candidate; no replacement quantization or packing."""
import functools
import logging
import os

_REPORTED_SHAPES = set()

HELPER_MODULE = 'sglang.srt.layers.quantization.compressed_tensors.schemes.q38_bw100_legacy_gemm_dispatch'

SHAPES = {(m, n, k) for m in (4096, 8192, 16384, 32768)
          for n, k in ((17408, 5120), (8192, 5120), (5120, 3072))}


@functools.lru_cache(None)
def approved_shapes(value):
    requested = {tuple(map(int, item.split('x'))) for item in value.split(',') if item}
    if not requested <= SHAPES:
        raise ValueError('Q38_HIPBLASLT_SHAPES contains unreviewed shapes')
    return requested


@functools.lru_cache(None)
def supported_device(index):
    import torch
    props = torch.cuda.get_device_properties(index)
    return (str(getattr(props, 'gcnArchName', '')).split(':')[0] == 'gfx936'
            and props.multi_processor_count == 64)


def try_legacy_mm(scheme, layer, x, x_q, x_scale, bias):
    """Return None for all non-approved calls, leaving original dispatch intact."""
    import torch
    from compressed_tensors.quantization import QuantizationStrategy
    allowed = approved_shapes(os.getenv('Q38_HIPBLASLT_SHAPES', ''))
    if (not allowed or scheme.w8a8_strategy != 1
            or scheme.strategy != QuantizationStrategy.CHANNEL
            or scheme.is_static_input_scheme or not scheme.input_symmetric
            or bias is not None or x.dtype != torch.bfloat16
            or x_q.ndim != 2 or layer.weight.ndim != 2):
        return None
    m, k = x_q.shape
    n = layer.weight.shape[1]
    if (m, n, k) not in allowed:
        return None
    w, ws = layer.weight, layer.weight_scale
    if (x_q.dtype != torch.int8 or w.dtype != torch.int8
            or tuple(w.shape) != (k, n) or w.stride() != (1, k)
            or not x_q.is_contiguous() or x_q.device.type != 'cuda'
            or w.device != x_q.device or x.device != x_q.device
            or x_scale.device != x_q.device or ws.device != x_q.device
            or x_scale.dtype != torch.float32 or ws.dtype != torch.float32
            or tuple(x_scale.shape) != (m, 1) or tuple(ws.shape) != (n, 1)
            or not x_scale.is_contiguous() or not ws.is_contiguous()
            or not supported_device(x_q.device.index)):
        return None
    from lightop import gemm_ops
    # Strategy 1 process_weights_after_loading stores contiguous [N,K].t().
    # Transposing again is a view; no repack/copy or change to original scales.
    result = gemm_ops.hipblaslt_w8a8_gemm(
        x_q, w.t(), x_scale, ws, m, n, k, 'NT', torch.bfloat16)[1]
    if (m, n, k) not in _REPORTED_SHAPES:
        _REPORTED_SHAPES.add((m, n, k))
        logging.getLogger(__name__).info('QWEN27B_HIPBLASLT_HIT M=%s N=%s K=%s BF16 INT8', m, n, k)
    return result


def patch_source(source):
    """Prepare an in-memory candidate, with a unique original-source anchor."""
    anchor = '        if self.w8a8_strategy == 1:\n            m = x_q.shape[0]\n'
    if source.count(anchor) != 1:
        raise ValueError('CompressedTensors apply_weights anchor differs')
    insertion = '''        from sglang.srt.layers.quantization.compressed_tensors.schemes.q38_bw100_legacy_gemm_dispatch import try_legacy_mm
        candidate = try_legacy_mm(self, layer, x, x_q, x_scale, bias)
        if candidate is not None:
            return candidate

'''
    return source.replace(anchor, insertion + anchor)
