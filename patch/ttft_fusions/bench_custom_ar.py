"""Screen the existing HIP deterministic 1-stage BF16 all-reduce entry.

Uses SGLANG_USE_1STAGE_ALLREDUCE, never changes topology or dispatch guards.
Checks driver P2P and actual bidirectional HIP IPC reads/writes before creating
the communicator. The experiment owner must reserve GPU 4/5 and impose timeout.
"""
import argparse
import ctypes
import datetime
import json
import os
from pathlib import Path
import re
import statistics

if os.environ.get("HIP_VISIBLE_DEVICES") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["HIP_VISIBLE_DEVICES"]
os.environ["SGLANG_USE_1STAGE_ALLREDUCE"] = "1"

import torch
import torch.distributed as dist


def ipc_probe(rank):
    """Test real peer IPC without the CUDA-only SGLang P2P probe wrapper."""
    header = Path("/opt/dtk/include/hip/hip_runtime_defines.h").read_text()
    match = re.search(r"#define\s+HIP_IPC_HANDLE_SIZE\s+(\d+)", header)
    if match is None or int(match.group(1)) != 64:
        raise RuntimeError("Unverified HIP IPC ABI; refusing probe")

    class Handle(ctypes.Structure):
        _fields_ = [("reserved", ctypes.c_byte * 64)]

    lib = ctypes.CDLL("/opt/dtk/lib/libamdhip64.so")
    signatures = {
        "hipMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        "hipFree": [ctypes.c_void_p],
        "hipMemset": [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
        "hipMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
        "hipDeviceSynchronize": [],
        "hipIpcGetMemHandle": [ctypes.POINTER(Handle), ctypes.c_void_p],
        "hipIpcOpenMemHandle": [ctypes.POINTER(ctypes.c_void_p), Handle, ctypes.c_uint],
        "hipIpcCloseMemHandle": [ctypes.c_void_p],
    }
    for name, argtypes in signatures.items():
        getattr(lib, name).argtypes = argtypes
        getattr(lib, name).restype = ctypes.c_int

    def call(name, *args):
        result = getattr(lib, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed with HIP code {result}")

    local, peer = ctypes.c_void_p(), ctypes.c_void_p()
    call("hipMalloc", ctypes.byref(local), 4096)
    try:
        call("hipMemset", local, rank + 1, 4096)
        call("hipDeviceSynchronize")
        handle = Handle()
        call("hipIpcGetMemHandle", ctypes.byref(handle), local)
        handles = [None, None]
        dist.all_gather_object(handles, bytes(handle))
        call("hipIpcOpenMemHandle", ctypes.byref(peer), Handle.from_buffer_copy(handles[1-rank]), 1)
        host = (ctypes.c_ubyte * 4096)()
        call("hipMemcpy", ctypes.cast(host, ctypes.c_void_p), peer, 4096, 2)
        if bytes(host) != bytes([2-rank]) * 4096:
            raise RuntimeError("Peer IPC read mismatch")
        call("hipMemset", peer, 9 + rank, 4096)
        call("hipDeviceSynchronize")
        dist.barrier()
        call("hipMemcpy", ctypes.cast(host, ctypes.c_void_p), local, 4096, 2)
        if bytes(host) != bytes([10-rank]) * 4096:
            raise RuntimeError("Peer IPC write mismatch")
        dist.barrier()
    finally:
        if peer.value:
            call("hipIpcCloseMemHandle", peer)
        call("hipFree", local)
    return True


def time_collective(fn, reset):
    for _ in range(10):
        reset()
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(5):
        dist.barrier()
        events = []
        for _ in range(20):
            reset()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        means.append(statistics.mean(a.elapsed_time(b) for a, b in events))
    return dict(median_ms=statistics.median(means), batch_mean_ms=means)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=60))
    metadata = dict(status="CHECKING", dtype="bfloat16", world_size=dist.get_world_size(),
                    entry="SGLANG_USE_1STAGE_ALLREDUCE=1", rows=[])

    def save():
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metadata, indent=2) + "\n")

    car = None
    nccl = None
    save()
    try:
        if dist.get_world_size() != 2 or torch.version.hip is None:
            raise RuntimeError("Requires HIP TP2")
        if os.environ.get("HIP_VISIBLE_DEVICES") != "4,5" or os.environ.get("CUDA_VISIBLE_DEVICES") != "4,5":
            raise RuntimeError("Diagnostic restricted to authorized physical GPUs 4,5")
        peer_access = torch.cuda.can_device_access_peer(rank, 1-rank)
        statuses = [None, None]
        dist.all_gather_object(statuses, peer_access)
        metadata["driver_bidirectional_peer_access"] = statuses
        if not all(statuses):
            raise RuntimeError("Driver does not permit bidirectional peer access")
        metadata["bidirectional_ipc_read_write_verified"] = ipc_probe(rank)
        from sglang.srt.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        car = CustomAllreduce(dist.group.WORLD, torch.device("cuda", rank), max_size=256*1024*1024)
        if car.disabled or not car.use_amd_deterministic_impl:
            raise RuntimeError("Existing deterministic entry unavailable")
        metadata.update(full_nvlink=car.full_nvlink, max_size=car.max_size,
                        use_amd_deterministic_impl=car.use_amd_deterministic_impl)
        # The actual topology is retained: do not assign car.full_nvlink.
        nccl = dist.new_group(backend="nccl", timeout=datetime.timedelta(seconds=60))
        save()
        for tokens in (128, 4096, 8192, 16384):
            torch.manual_seed(20261015 + rank)
            inp = torch.randn(tokens, 5120, device="cuda", dtype=torch.bfloat16)
            if inp.numel() * inp.element_size() >= car.max_size or not car.should_custom_ar(inp):
                raise RuntimeError(f"Existing guards rejected {tokens} tokens")
            reference = inp.clone()
            dist.all_reduce(reference, group=nccl)
            actual = car.custom_all_reduce(inp)
            if actual is None or actual.dtype != torch.bfloat16:
                raise RuntimeError("Existing public API did not return BF16 output")
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, reference, atol=0, rtol=0)
            work = inp.clone()
            timings = {
                "rccl_bf16": time_collective(lambda: dist.all_reduce(work, group=nccl), lambda: work.copy_(inp)),
                "deterministic_1stage_bf16": time_collective(lambda: car.custom_all_reduce(work), lambda: work.copy_(inp)),
            }
            row = dict(rank=rank, tokens=tokens, bytes=inp.numel()*2, exact=True, timings=timings)
            gathered = [None, None]
            dist.all_gather_object(gathered, row)
            if rank == 0:
                metadata["rows"].extend(gathered)
                save()
                print(json.dumps(gathered), flush=True)
        metadata["status"] = "MEASURED"
        save()
    except Exception as exc:
        metadata.update(status="UNAVAILABLE_OR_FAILED", reason=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        if car is not None:
            car.close()
        if nccl is not None:
            dist.destroy_process_group(nccl)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
