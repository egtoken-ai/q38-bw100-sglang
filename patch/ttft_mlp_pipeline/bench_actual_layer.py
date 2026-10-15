"""Production decoder boundary parity and timing, with real SGLang modules.

Constructs actual Qwen3_5 dense decoder classes and their actual W8A8 modules,
using randomized weight shards. Does not run attention or load the checkpoint.
Root must schedule this two-rank GPU diagnostic separately from serving.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
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
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
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
    for cls,layer_id in ((Qwen3_5LinearDecoderLayer,0),(Qwen3_5AttentionDecoderLayer,3)):
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
        x=torch.empty_like(source);r=torch.empty_like(source)

        def reset():
            x.copy_(source);r.copy_(residual_source)

        def baseline():
            reset()
            h,res=layer.layer_communicator.prepare_mlp(x,r,batch)
            out=layer.mlp(h,should_allreduce_fusion=False,use_reduce_scatter=False)
            return layer.layer_communicator.postprocess_layer(out,res,batch)

        def candidate():
            reset()
            out=try_mlp_pipeline(layer,x,r,batch)
            if out is None:
                raise RuntimeError('Pipeline did not run: '+str(rejection_reason(layer,x,r,batch)))
            return out

        reset()
        reason=rejection_reason(layer,x,r,batch)
        reasons=[None,None];dist.all_gather_object(reasons,reason,group=cpu_group)
        if any(v is not None for v in reasons):
            raise RuntimeError(f'Actual-layer guard rejected: {reasons}')
        # Validate critical fallback scopes without launching the candidate.
        for change,expected in (({'batch_size':1},'single request'),
                                ({'forward_mode':ForwardMode.DECODE},'ordinary non-spec')):
            old={key:getattr(batch,key) for key in change}
            for key,value in change.items():setattr(batch,key,value)
            rejected=rejection_reason(layer,x,r,batch)
            if rejected is None or expected not in rejected:
                raise AssertionError(f'Fallback guard failed: {change}, {rejected}')
            for key,value in old.items():setattr(batch,key,value)
        reference=tuple(v.clone() for v in baseline())
        reference_normalized=x.clone()
        actual=candidate()
        torch.cuda.synchronize()
        local=dict(rank=rank)
        try:
            local['output']=check(actual[0],reference[0])
            local['residual']=check(actual[1],reference[1],exact=True)
            local['normalized_inplace_buffer']=check(x,reference_normalized,exact=True)
            local['residual_changed_elements']=int(torch.count_nonzero(reference[1]!=residual_source))
            if local['residual_changed_elements']==0:raise AssertionError('Vacuous residual test')
            local['status']='PASS'
        except Exception as exc:
            local['status']='FAIL';local['error']=f'{type(exc).__name__}: {exc}'
        checks=[None,None];dist.all_gather_object(checks,local,group=cpu_group)
        row=dict(layer=cls.__name__,tokens=args.tokens,checks=checks,timings={})
        rows.append(row)
        if any(c['status']!='PASS' for c in checks):
            if rank==0:Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')
            raise RuntimeError('Actual layer numerical validation failed')
        samples={'baseline':[],'pipeline':[]}
        for fn in (baseline,candidate):
            for _ in range(3):fn()
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            order=[('baseline',baseline),('pipeline',candidate)]
            if repeat%2:order.reverse()
            for name,fn in order:
                torch.cuda.synchronize();dist.barrier(group=cpu_group)
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):out=fn()
                end.record();end.synchronize()
                elapsed=start.elapsed_time(end)/args.iterations
                times=[None,None];dist.all_gather_object(times,elapsed,group=cpu_group)
                samples[name].append(max(times))
        for name in samples:
            row['timings'][name]=dict(samples_ms=samples[name],median_ms=statistics.median(samples[name]))
        row['latency_improvement_pct']=(1-row['timings']['pipeline']['median_ms']/row['timings']['baseline']['median_ms'])*100
        if rank==0:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps(row),flush=True)
        del layer
        torch.cuda.empty_cache()
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
