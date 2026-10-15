"""Bounded host/allocator telemetry. Never synchronizes a GPU or reads tensor data."""
import json
import os
from pathlib import Path
import threading
import time
import torch
import torch.distributed as dist

_BATCH = 0
_FILES = {}
KEYS = (
    'num_alloc_retries','num_sync_all_streams','num_device_alloc','num_device_free',
    'allocated_bytes.all.current','active_bytes.all.current','reserved_bytes.all.current',
    'inactive_split_bytes.all.current','active.all.current','segment.all.current',
)


def stats(device):
    all_stats = torch.cuda.memory_stats(device)
    return {key: all_stats.get(key) for key in KEYS}


class Probe:
    def __init__(self, layer, hidden, batch):
        self.device = hidden.device
        self.start = time.perf_counter_ns()
        self.before = stats(self.device)
        self.events = []
        self.metadata = dict(pid=os.getpid(),rank=dist.get_rank(),batch=batch,
            layer=layer.layer_id,tokens=hidden.shape[0],
            thread_id=threading.get_ident(),thread_name=threading.current_thread().name,
            caller_stream=int(torch.cuda.current_stream(self.device).cuda_stream))
        self.mark('begin')

    def mark(self, label, part=None):
        self.events.append(dict(label=label,part=part,host_us=(time.perf_counter_ns()-self.start)/1000))

    def finish(self):
        self.mark('end')
        after=stats(self.device)
        delta={key:after[key]-self.before[key] for key in KEYS
               if after[key] is not None and self.before[key] is not None}
        row=dict(**self.metadata,host_submit_ms=self.events[-1]['host_us']/1000,
                 before=self.before,after=after,delta=delta,events=self.events)
        directory=Path(os.environ.get('SGLANG_MLP_DIAG_DIR','/bench/result/ttft/mlp-allocator-diagnostic'))
        path=directory/f'pipeline-rank{self.metadata["rank"]}-pid{os.getpid()}.jsonl'
        if str(path) not in _FILES:
            directory.mkdir(parents=True,exist_ok=True)
            _FILES[str(path)]=path.open('a',buffering=1)
        _FILES[str(path)].write(json.dumps(row,separators=(',',':'))+'\n')


def begin(layer,hidden,forward_batch):
    global _BATCH
    if os.environ.get('SGLANG_MLP_DIAG_ENABLED','0').lower() not in ('1','true'):
        return None
    if layer.layer_id == 0:
        _BATCH += 1
    limit=min(max(int(os.environ.get('SGLANG_MLP_DIAG_MAX_BATCHES','2')),1),4)
    if not 1 <= _BATCH <= limit:
        return None
    return Probe(layer,hidden,_BATCH)
