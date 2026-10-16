"""Bounded idle-first-batch admission; CPU-only, before the existing TP broadcast.

No model computation, cache behavior, timestamps, or collective calls are changed.
"""
import logging
import math
import os
import time

import zmq

logger = logging.getLogger(__name__)


def _configured_delay():
    raw = os.environ.get('SGLANG_QWEN27B_ADMISSION_DELAY_MS', '0')
    if raw not in ('0', '2', '5', '10'):
        raise ValueError('SGLANG_QWEN27B_ADMISSION_DELAY_MS must be 0, 2, 5 or 10')
    return int(raw)


_DELAY_MS = _configured_delay()
_LOG_COUNT = 0


def _ordinary(req, generate_type):
    # Exact class excludes batch/embedding/control requests and subclasses.
    return (
        type(req) is generate_type
        and isinstance(req.input_ids, list)
        and len(req.input_ids) > 0
        and getattr(req, 'mm_inputs', None) is None
        and getattr(req, 'input_embeds', None) is None
        and getattr(req, 'session_params', None) is None
        and not getattr(req, 'need_wait_for_mm_inputs', False)
    )


def collect_idle_generate_requests(scheduler, recv_reqs, generate_type):
    """Leader-only list extension; caller performs its original single broadcast.

    Poll both existing sockets, never reset the absolute deadline, and retain
    every received object in original receive order. Any nonordinary object
    ends waiting immediately so the caller can broadcast/process it normally.
    """
    global _LOG_COUNT
    if _DELAY_MS == 0 or not recv_reqs:
        return recv_reqs
    args = scheduler.server_args
    if (
        scheduler.pp_rank != 0
        or scheduler.attn_tp_rank != 0
        or scheduler.attn_cp_rank != 0
        or scheduler.tp_size != 2
        or scheduler.dp_size != 1
        or scheduler.pp_size != 1
        or getattr(args, 'attn_cp_size', 1) != 1
        or args.enable_dp_attention
        or args.disaggregation_mode != 'null'
        or args.speculative_algorithm is not None
        or getattr(args, 'enable_priority_scheduling', False)
        or not args.disable_radix_cache
        or scheduler._engine_paused
        or scheduler.input_blocker is not None
        or not scheduler.is_fully_idle()
        or not all(_ordinary(req, generate_type) for req in recv_reqs)
    ):
        return recv_reqs
    chunk = scheduler.chunked_prefill_size
    max_prefill = scheduler.max_prefill_tokens
    if chunk is None or chunk <= 0 or max_prefill <= 0:
        return recv_reqs
    cap = min(chunk, max_prefill)
    tokens = sum(len(req.input_ids) for req in recv_reqs)
    if tokens >= cap or scheduler.recv_limit_reached(len(recv_reqs)):
        return recv_reqs

    started = time.monotonic()
    deadline = started + _DELAY_MS / 1000.0
    initial_count = len(recv_reqs)
    poller = zmq.Poller()
    poller.register(scheduler.recv_from_tokenizer, zmq.POLLIN)
    poller.register(scheduler.recv_from_rpc, zmq.POLLIN)
    reason = 'deadline'
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Millisecond API may overshoot by <1 ms, plus OS scheduling time.
            ready = dict(poller.poll(max(1, math.ceil(remaining * 1000))))
            # RPC control wakes polling immediately; do not spend another
            # admission window draining tokenizer work before servicing it.
            if ready.get(scheduler.recv_from_rpc, 0) & zmq.POLLIN:
                if scheduler.recv_limit_reached(len(recv_reqs)):
                    reason = 'recv_limit'
                    break
                try:
                    req = scheduler.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                except zmq.Again:
                    pass
                else:
                    recv_reqs.append(req)
                    reason = 'rpc'
                    break
            if ready.get(scheduler.recv_from_tokenizer, 0) & zmq.POLLIN:
                while time.monotonic() < deadline:
                    if scheduler.recv_limit_reached(len(recv_reqs)):
                        reason = 'recv_limit'
                        return recv_reqs
                    # Also check RPC between tokenizer receives so a continuous
                    # generation stream cannot starve control processing.
                    if scheduler.recv_from_rpc.poll(0, zmq.POLLIN):
                        try:
                            req = scheduler.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
                        except zmq.Again:
                            pass
                        else:
                            recv_reqs.append(req)
                            reason = 'rpc'
                            return recv_reqs
                    try:
                        req = scheduler.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    recv_reqs.append(req)
                    if not _ordinary(req, generate_type):
                        reason = 'control_or_nonordinary'
                        return recv_reqs
                    tokens += len(req.input_ids)
                    if tokens >= cap:
                        reason = 'token_cap'
                        return recv_reqs
    except zmq.ZMQError:
        # Preserve all objects already received, then let the normal caller
        # broadcast this list. Match the existing recv loop error boundary.
        reason = 'socket_error'
    finally:
        if _LOG_COUNT < 8:
            _LOG_COUNT += 1
            logger.info(
                'Q38_ADMISSION_BATCH before=%d after=%d tokens=%d cap=%d '
                'budget_ms=%d elapsed_ms=%.3f reason=%s',
                initial_count, len(recv_reqs), tokens, cap, _DELAY_MS,
                (time.monotonic() - started) * 1000, reason,
            )
    return recv_reqs
