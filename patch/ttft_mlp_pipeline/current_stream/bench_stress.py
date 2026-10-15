"""Continuous real-boundary chain under bounded artificial live-memory pressure.

Constructs actual Qwen3_5 dense decoder classes and their actual W8A8 modules,
using randomized weight shards. Does not run attention or load the checkpoint.
Root must schedule this two-rank GPU diagnostic separately from serving.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist


def check(a,b,exact=False):
    diff=a.float()-b.float()
    result=dict(max_abs=float(diff.abs().max()),mismatches=int(torch.count_nonzero(a!=b)),
                relative_l2=float(diff.norm()/b.float().norm().clamp_min(1e-12)))
    torch.testing.assert_close(a,b,atol=0 if exact else .015625,rtol=0 if exact else .015625)
    if result['relative_l2']>.002:
        raise AssertionError(result)
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--model',default='/model')
    p.add_argument('--tokens',type=int,default=32768)
    p.add_argument('--layers',type=int,default=64,choices=[16,32,64])
    p.add_argument('--headroom-gib',type=float,default=8.0)
    p.add_argument('--modes',nargs='+',default=['baseline','current'],choices=['baseline','current','failed'])
    p.add_argument('--repeats',type=int,default=2)
    args=p.parse_args()
    if not 4<=args.headroom_gib<=32:raise ValueError('Headroom must be between 4 and 32 GiB')
    os.environ['SGLANG_MLP_DIAG_ENABLED']='0'  # This script records its own layer telemetry.
    if args.tokens!=32768:
        raise ValueError('Initial actual-layer validation is restricted to serving M32768')
    rank=int(os.environ['LOCAL_RANK'])
    if rank==0:Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    if int(os.environ['WORLD_SIZE'])!=2:
        raise RuntimeError('TP2 only')
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    os.environ['SGLANG_QWEN27B_MLP_PIPELINE']='true'
    from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
    server=ServerArgs(model_path=args.model,tp_size=2,dtype='bfloat16',device='cuda',
                      disable_cuda_graph=True,disable_piecewise_cuda_graph=True,
                      disable_custom_all_reduce=True)
    set_global_server_args_for_scheduler(server)
    from sglang.srt.distributed import (
        init_distributed_environment,initialize_model_parallel,get_tp_group,set_custom_all_reduce)
    set_custom_all_reduce(False)
    init_distributed_environment(world_size=2,rank=rank,local_rank=rank,backend='nccl')
    initialize_model_parallel(tensor_model_parallel_size=2,backend='nccl')
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    initialize_dp_attention(server,SimpleNamespace(hidden_size=5120,dtype=torch.bfloat16))
    from sglang.srt.layers.moe.utils import initialize_moe_config
    initialize_moe_config(server)
    from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    from sglang.srt.models.qwen3_5 import Qwen3_5LinearDecoderLayer,Qwen3_5AttentionDecoderLayer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.layers.qwen27b_mlp_pipeline import rejection_reason,try_mlp_pipeline
    config_json=json.loads((Path(args.model)/'config.json').read_text())
    config=Qwen3_5TextConfig(**config_json['text_config'])
    config.torch_dtype=torch.bfloat16
    group=get_tp_group()
    cpu_group=group.cpu_group
    batch=SimpleNamespace(batch_size=8,forward_mode=ForwardMode.EXTEND,spec_info=None,
        input_ids=torch.zeros(args.tokens,device='cuda',dtype=torch.int64),
        rms_quant_flag=False,attn_cp_metadata=None,tbo_children=None,
        dp_padding_mode=SimpleNamespace(is_max_len=lambda:False))
    rows=[]
    for cls,layer_id in ((Qwen3_5LinearDecoderLayer,0),):
        quant=CompressedTensorsConfig.from_config(config_json['compression_config'])
        with torch.device(f'cuda:{rank}'):
            layer=cls(config,layer_id=layer_id,quant_config=quant,
                      prefix=f'model.language_model.layers.{layer_id}')
        # Only the real MLP and post-attention norm execute in this diagnostic.
        # Initialize their native parameters and run native weight processing.
        torch.manual_seed(20261015+rank)
        for proj in (layer.mlp.gate_up_proj,layer.mlp.down_proj):
            if type(proj.scheme).__name__!='CompressedTensorsW8A8Int8':
                raise RuntimeError(f'Unexpected quantization scheme: {type(proj.scheme)}')
            proj.weight.data.copy_(torch.randint(-16,16,proj.weight.shape,device='cuda',dtype=torch.int8))
            proj.weight_scale.data.fill_(.001)
            proj.quant_method.process_weights_after_loading(proj)
        torch.manual_seed(50001)
        norm_weight=(torch.randn(5120,device='cuda')*.1).bfloat16()
        layer.post_attention_layernorm.weight.data.copy_(norm_weight)
        layer.post_attention_layernorm.gemma_weight=norm_weight+1
        torch.manual_seed(61000+rank)
        source=torch.randn(args.tokens,5120,device='cuda',dtype=torch.bfloat16)
        torch.manual_seed(62000)
        residual_source=torch.randn_like(source)
        from sglang.srt.layers import qwen27b_mlp_pipeline_failed as failed_helper
        from sglang.srt.layers.qwen27b_mlp_probe import stats
        # ModelRunner uses a non-default forward_stream. Reproduce that caller
        # stream instead of the default-stream/synchronized single-layer micro.
        model_stream=torch.cuda.Stream()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_bytes,total_bytes=torch.cuda.mem_get_info()
        reserve_bytes=max(0,free_bytes-int(args.headroom_gib*(1024**3)))
        pressure=torch.empty(reserve_bytes,device='cuda',dtype=torch.uint8)
        pressure.fill_(0)
        torch.cuda.synchronize()
        metadata=dict(tokens=args.tokens,layers=args.layers,headroom_gib=args.headroom_gib,
                      pressure_bytes=reserve_bytes,device_total_bytes=total_bytes,
                      stream_kind='non-default model stream',
                      attention_substitute='h * 0.5 allocation, then TP2 input AR; no real attention',
                      weight_scope='same real dense decoder MLP reused across layer chain')
        rows=[]

        def boundary(kind,h,r):
            if kind=='baseline':
                h,r=layer.layer_communicator.prepare_mlp(h,r,batch)
                h=layer.mlp(h,should_allreduce_fusion=False,use_reduce_scatter=False)
                return layer.layer_communicator.postprocess_layer(h,r,batch)
            fn=try_mlp_pipeline if kind=='current' else failed_helper.try_mlp_pipeline
            result=fn(layer,h,r,batch)
            if result is None:
                raise RuntimeError('Guard rejected: '+str(rejection_reason(layer,h,r,batch)))
            return result

        def run(kind):
            observations=[]
            model_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(model_stream):
                h=source.clone();r=residual_source.clone()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                whole_start=time.perf_counter()
                for index in range(args.layers):
                    layer.layer_id=index
                    before=stats(h.device)
                    cpu_start=time.perf_counter()
                    # Keep the per-layer incoming activation allocation on the
                    # model stream, approximating a new attention output buffer.
                    partial=h*.5
                    h,r=boundary(kind,partial,r)
                    host_ms=(time.perf_counter()-cpu_start)*1000
                    after=stats(h.device)
                    observations.append(dict(layer=index,host_submit_ms=host_ms,
                        before=before,after=after,
                        delta={key:after[key]-before[key] for key in before
                               if before[key] is not None and after[key] is not None}))
                host_total_ms=(time.perf_counter()-whole_start)*1000
                end.record()
            # One synchronization AFTER the entire chain. Never between layers.
            end.synchronize()
            torch.cuda.current_stream().wait_stream(model_stream)
            return (h,r),dict(mode=kind,gpu_ms=start.elapsed_time(end),
                              host_submit_ms=host_total_ms,layers=observations)

        # Full-chain baseline oracle, not just per-layer randomized inputs.
        reference,_=run('baseline')
        reference=tuple(value.clone() for value in reference)
        torch.cuda.synchronize()
        for kind in args.modes:
            # One full-chain warmup; no synchronization inside that chain.
            warmed,_=run(kind)
            torch.cuda.synchronize()
            for repeat in range(args.repeats):
                actual,measurement=run(kind)
                local=dict(rank=rank,repeat=repeat,**measurement)
                try:
                    local['output_check']=check(actual[0],reference[0],exact=True)
                    local['residual_check']=check(actual[1],reference[1],exact=True)
                    local['status']='PASS'
                except Exception as exc:
                    local['status']='FAIL';local['error']=f'{type(exc).__name__}: {exc}'
                gathered=[None,None];dist.all_gather_object(gathered,local,group=cpu_group)
                rows.append(dict(mode=kind,repeat=repeat,
                    max_rank_gpu_ms=max(v['gpu_ms'] for v in gathered),
                    max_rank_host_submit_ms=max(v['host_submit_ms'] for v in gathered),ranks=gathered))
                if rank==0:
                    Path(args.output).write_text(json.dumps(dict(**metadata,runs=rows),indent=2)+'\n')
                    print(json.dumps(dict(mode=kind,repeat=repeat,
                        max_rank_gpu_ms=rows[-1]['max_rank_gpu_ms'],
                        max_rank_host_submit_ms=rows[-1]['max_rank_host_submit_ms'],
                        statuses=[v['status'] for v in gathered])),flush=True)
                if any(v['status']!='PASS' for v in gathered):
                    raise RuntimeError('Continuous-chain parity failed')
                torch.cuda.synchronize()
        del pressure
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
