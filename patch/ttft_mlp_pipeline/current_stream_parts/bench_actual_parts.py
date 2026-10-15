"""Current-compute-stream M32768/M65536 and 4/8-part actual boundary diagnostic.

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
    p.add_argument('--tokens',type=int,default=32768,choices=[32768,65536])
    p.add_argument('--parts',type=int,nargs='+',default=[4],choices=[4,8])
    p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.tokens not in (32768,65536) or len(set(args.parts)) != len(args.parts):
        raise ValueError('Bounded M32768/M65536, unique parts4/8 only')
    rank=int(os.environ['LOCAL_RANK'])
    if rank==0:Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    if int(os.environ['WORLD_SIZE'])!=2:
        raise RuntimeError('TP2 only')
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.bfloat16)
    os.environ['SGLANG_QWEN27B_MLP_PIPELINE']='true'
    os.environ['SGLANG_MLP_DIAG_ENABLED']='0'
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
    from sglang.srt.layers import qwen27b_mlp_pipeline as installed_pipeline
    installed_control_parts=getattr(installed_pipeline,'_PIPELINE_PARTS',4)
    if installed_control_parts not in (4,8):
        raise RuntimeError('Installed helper has an unexpected parts configuration')
    # Reject accidental testing against the old four-private-stream image.
    import inspect,ast
    installed_tree=ast.parse(inspect.getsource(try_mlp_pipeline))
    call_attributes={node.func.attr for node in ast.walk(installed_tree)
                     if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    if call_attributes & {'record_stream','Stream'}:
        raise RuntimeError('Use the repaired current-stream image, not the failed side-stream image')
    from current_parts import try_mlp_pipeline_parts
    config_json=json.loads((Path(args.model)/'config.json').read_text())
    config=Qwen3_5TextConfig(**config_json['text_config'])
    config.torch_dtype=torch.bfloat16
    group=get_tp_group()
    cpu_group=group.cpu_group
    batch=SimpleNamespace(batch_size=args.tokens//4096,forward_mode=ForwardMode.EXTEND,spec_info=None,
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

        def candidate(parts):
            reset()
            out=try_mlp_pipeline_parts(layer,x,r,batch,parts=parts)
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
        # Check the matching selected part count against the installed helper.
        # An ENV=8 image must not be mislabeled as an installed 4-part control.
        reset()
        installed=tuple(v.clone() for v in try_mlp_pipeline(layer,x,r,batch))
        row=dict(layer=cls.__name__,tokens=args.tokens,parts=args.parts,
                 installed_control_parts=installed_control_parts,checks={},timings={},
                 allocation_policy='all compute on actual caller stream; no private compute streams')
        rows.append(row)
        modes=[('baseline',baseline)]
        for parts in args.parts:
            modes.append((f'pipeline_{parts}',lambda parts=parts:candidate(parts)))
        for parts in args.parts:
            actual=candidate(parts)
            torch.cuda.synchronize()
            local=dict(rank=rank)
            try:
                local['output']=check(actual[0],reference[0],exact=True)
                local['residual']=check(actual[1],reference[1],exact=True)
                local['normalized_inplace_buffer']=check(x,reference_normalized,exact=True)
                if parts==installed_control_parts:
                    local['output_vs_installed']=check(actual[0],installed[0],exact=True)
                    local['residual_vs_installed']=check(actual[1],installed[1],exact=True)
                local['residual_changed_elements']=int(torch.count_nonzero(reference[1]!=residual_source))
                if local['residual_changed_elements']==0:raise AssertionError('Vacuous residual test')
                local['status']='PASS'
            except Exception as exc:
                local['status']='FAIL';local['error']=f'{type(exc).__name__}: {exc}'
            checks=[None,None];dist.all_gather_object(checks,local,group=cpu_group)
            row['checks'][f'pipeline_{parts}']=checks
            if any(c['status']!='PASS' for c in checks):
                if rank==0:Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')
                raise RuntimeError('Actual layer numerical validation failed')
        samples={name:[] for name,_ in modes}
        for _,fn in modes:
            for _ in range(3):fn()
        torch.cuda.synchronize()
        for repeat in range(args.repeats):
            offset=repeat%len(modes)
            for name,fn in modes[offset:]+modes[:offset]:
                torch.cuda.synchronize();dist.barrier(group=cpu_group)
                before=torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):out=fn()
                end.record();end.synchronize()
                local=dict(rank=rank,ms=start.elapsed_time(end)/args.iterations,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_increment_bytes=max(0,torch.cuda.max_memory_allocated()-before))
                measurements=[None,None];dist.all_gather_object(measurements,local,group=cpu_group)
                samples[name].append(dict(max_rank_ms=max(v['ms'] for v in measurements),ranks=measurements))
        for name in samples:
            row['timings'][name]=dict(samples=samples[name],
                                      median_ms=statistics.median(v['max_rank_ms'] for v in samples[name]))
        for parts in args.parts:
            row[f'pipeline_{parts}_improvement_vs_single_pct']=(1-row['timings'][f'pipeline_{parts}']['median_ms']/row['timings']['baseline']['median_ms'])*100
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
