# SPDX-License-Identifier: Apache-2.0
"""CPU-owned metadata for the audited Qwen27B ordinary GDN prefill path."""
import logging
import os


def build_metadata_lists(lengths, prefixes, chunk_size=64):
    """Pure Python reference; never accepts a device tensor or reads GPU values.

    Zero-length/padded requests are deliberately unsupported in this first path.
    Sequence indices match prepare_chunk_indices only with positive lengths.
    """
    if not isinstance(lengths, (list, tuple)) or not isinstance(prefixes, (list, tuple)):
        raise ValueError('Lengths and prefixes must be CPU integer sequences')
    if not lengths or len(lengths) != len(prefixes) or chunk_size != 64:
        raise ValueError('Expected nonempty equal-size metadata and chunk size 64')
    if any(type(x) is not int or x <= 0 for x in lengths):
        raise ValueError('All extend lengths must be positive Python ints')
    if any(type(x) is not int or x < 0 for x in prefixes):
        raise ValueError('All prefix lengths must be nonnegative Python ints')
    indices, offsets, cumulative = [], [0], [0]
    for sequence, length in enumerate(lengths):
        count = (length + chunk_size - 1) // chunk_size
        indices.extend((sequence, chunk) for chunk in range(count))
        offsets.append(offsets[-1] + count)
        cumulative.append(cumulative[-1] + length)
    return indices, offsets, [x > 0 for x in prefixes], cumulative


def prepare_gdn_cpu_metadata(backend, forward_batch):
    """Prepare once per step, using CPU lists created with the GPU length tensors.

    All unsupported modes retain the original path. Metadata lives on this newly
    created ForwardMetadata object, not on a process-global tensor-identity cache.
    """
    metadata = backend.forward_metadata
    metadata.gdn_cpu_chunk_indices = None
    metadata.gdn_cpu_chunk_offsets = None
    metadata.gdn_cpu_has_initial_states = None
    metadata.gdn_cpu_source_buffers = None
    if (os.environ.get('SGLANG_QWEN27B_GDN_CPU_METADATA', '0').lower() not in ('1', 'true')
            or not getattr(backend, '_qwen27b_cpu_metadata_model', False)
            or not forward_batch.forward_mode.is_extend_without_speculative()
            or forward_batch.spec_info is not None
            or getattr(forward_batch, 'can_run_tbo', False)):
        return False
    lengths = forward_batch.extend_seq_lens_cpu
    prefixes = forward_batch.extend_prefix_lens_cpu
    try:
        indices, offsets, has_initial, cumulative = build_metadata_lists(lengths, prefixes)
    except ValueError:
        return False
    total = cumulative[-1]
    query_start_loc = metadata.query_start_loc
    if (total < 64 or total >= 2**31
            or len(lengths) != forward_batch.batch_size
            or total != forward_batch.extend_num_tokens
            or forward_batch.input_ids is None
            or total != forward_batch.input_ids.shape[0]
            or query_start_loc is None
            or query_start_loc.ndim != 1
            or query_start_loc.shape[0] != len(lengths) + 1):
        return False
    import torch
    if query_start_loc.device.type != 'cuda' or query_start_loc.dtype != torch.int32:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    # Retain pinned source buffers with metadata through this step. Integer
    # indices follow query_start_loc dtype; original cumsum offsets are int64.
    cpu_indices = torch.tensor(indices, dtype=torch.int32, pin_memory=True)
    cpu_offsets = torch.tensor(offsets, dtype=torch.int64, pin_memory=True)
    cpu_has_initial = torch.tensor(has_initial, dtype=torch.bool, pin_memory=True)
    device = query_start_loc.device
    metadata.gdn_cpu_chunk_indices = cpu_indices.to(device, non_blocking=True)
    metadata.gdn_cpu_chunk_offsets = cpu_offsets.to(device, non_blocking=True)
    metadata.gdn_cpu_has_initial_states = cpu_has_initial.to(device, non_blocking=True)
    metadata.gdn_cpu_source_buffers = (cpu_indices, cpu_offsets, cpu_has_initial)
    if not getattr(backend, '_qwen27b_cpu_metadata_reported', False):
        logging.getLogger(__name__).info(
            "Qwen27B GDN CPU metadata active: batch=%s extend_tokens=%s chunk=64",
            len(lengths), total,
        )
        backend._qwen27b_cpu_metadata_reported = True
    return True
