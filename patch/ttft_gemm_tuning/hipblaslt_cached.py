"""ctypes bridge for the experimental DTK W8A8 wrapper; not auto-installed."""
import ctypes
from pathlib import Path


class CachedHipblasLt:
    def __init__(self, library):
        self.library_path = str(Path(library).resolve())
        self.lib = ctypes.CDLL(self.library_path)
        self.lib.q38_lt_last_error.restype = ctypes.c_char_p
        self.lib.q38_lt_run.argtypes = [
            ctypes.c_int, ctypes.c_size_t,
            *([ctypes.c_void_p] * 5),
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.q38_lt_run.restype = ctypes.c_int
        self.lib.q38_lt_clear.argtypes = [ctypes.c_int]
        self.lib.q38_lt_clear.restype = ctypes.c_int
        self.last_selection = None

    def _check(self, status):
        if status:
            raise RuntimeError(self.lib.q38_lt_last_error().decode())

    def run(self, a, weight, activation_scale, weight_scale, *, index=-1, workspace_mib=128):
        """a[M,K], weight[N,K], scales[M,1]/[N,1] -> BF16[M,N]."""
        import torch
        if a.ndim != 2 or weight.ndim != 2 or a.shape[1] != weight.shape[1]:
            raise ValueError('Expected A[M,K] and W[N,K]')
        tensors = (a, weight, activation_scale, weight_scale)
        if any(not t.is_cuda or t.device != a.device or not t.is_contiguous() for t in tensors):
            raise ValueError('All inputs must be contiguous on the same HIP device')
        if a.dtype != torch.int8 or weight.dtype != torch.int8:
            raise ValueError('Only INT8 operands are supported')
        if activation_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
            raise ValueError('Per-token/channel scales must be FP32')
        m, k = a.shape
        n = weight.shape[0]
        if activation_scale.numel() != m or weight_scale.numel() != n:
            raise ValueError('Scale counts must equal M and N; no scalar/group rescaling')
        if workspace_mib < 0 or workspace_mib > 128:
            raise ValueError('Workspace limit must be within 0..128 MiB')
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        chosen, required = ctypes.c_int(), ctypes.c_size_t()
        with torch.cuda.device(a.device):
            stream = torch.cuda.current_stream(a.device)
            self._check(self.lib.q38_lt_run(
                a.device.index, stream.cuda_stream,
                a.data_ptr(), weight.data_ptr(), activation_scale.data_ptr(), weight_scale.data_ptr(),
                out.data_ptr(), m, n, k, index, int(workspace_mib * 1024 * 1024),
                ctypes.byref(chosen), ctypes.byref(required)))
        self.last_selection = dict(index=chosen.value, required_workspace_bytes=required.value)
        return out

    def maybe_scaled_mm(self, a, b, sa, sb, out_dtype, bias, *, is_prefill, choices, fallback):
        """Guarded seam only. Caller must supply measured shape->index choices.

        b is the existing column-major [K,N] weight view. Decode and all shapes
        absent from choices call fallback. Errors on admitted candidates propagate
        instead of silently replacing an unsuccessful performance experiment.
        """
        import torch
        if a.ndim != 2 or b.ndim != 2:
            return fallback(a, b, sa, sb, out_dtype, bias)
        shape = (a.shape[0], b.shape[1], a.shape[1])
        approved = {(17408, 5120), (5120, 8704), (8192, 5120), (5120, 3072)}
        eligible = (is_prefill and shape in choices and 4096 <= shape[0] <= 32768
                    and shape[0] % 128 == 0 and shape[1:] in approved
                    and out_dtype == torch.bfloat16 and bias is None
                    and a.is_cuda and a.dtype == b.dtype == torch.int8
                    and a.is_contiguous() and b.t().is_contiguous())
        if eligible:
            arch = getattr(torch.cuda.get_device_properties(a.device), 'gcnArchName', '').split(':')[0]
            eligible = arch == 'gfx936'
        if not eligible:
            return fallback(a, b, sa, sb, out_dtype, bias)
        return self.run(a, b.t(), sa, sb, index=int(choices[shape]), workspace_mib=128)

    def clear(self, device=0):
        """Synchronizes the device; use only outside timed/model execution."""
        self._check(self.lib.q38_lt_clear(device))
