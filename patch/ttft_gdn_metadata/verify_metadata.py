#!/usr/bin/env python3
"""CPU metadata checks by default; --device additionally runs GPU equivalence."""
import argparse
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace, ModuleType

from gdn_cpu_metadata import build_metadata_lists, prepare_gdn_cpu_metadata


def cpu_checks():
    cases=[[64],[1,63],[63,64,65,129],[4096]*8,[8192],[32768],[65536],[32768,1]]
    rng=random.Random(20261015)
    cases += [[rng.randrange(1,2049) for _ in range(rng.randrange(1,17))] for _ in range(100)]
    for lengths in cases:
        prefixes=[0 if i%2==0 else 8192 for i in range(len(lengths))]
        indices,offsets,has_initial,cu=build_metadata_lists(lengths,prefixes)
        assert cu[0]==0 and cu[-1]==sum(lengths)
        # Reproduce the old index.py arange -> eq(0).cumsum formulation.
        chunks=[j for length in lengths for j in range((length+63)//64)]
        old=[];seq=-1
        for value in chunks:
            if value==0:seq+=1
            old.append((seq,value))
        assert indices==old
        assert offsets==[sum((l+63)//64 for l in lengths[:i]) for i in range(len(lengths)+1)]
        assert has_initial==[p>0 for p in prefixes]
    for lengths,prefixes in [([],[]),([0],[0]),([64],[-1]),([True],[0]),([64.0],[0]),([64],[0,1])]:
        try:build_metadata_lists(lengths,prefixes)
        except ValueError:pass
        else:raise AssertionError('Invalid metadata accepted')
    # CPU guard checks exit before importing Torch; no GPU or tensor reads.
    metadata=SimpleNamespace()
    backend=SimpleNamespace(forward_metadata=metadata,_qwen27b_cpu_metadata_model=True)
    batch=SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda:False),spec_info=None)
    os.environ['SGLANG_QWEN27B_GDN_CPU_METADATA']='1'
    assert prepare_gdn_cpu_metadata(backend,batch) is False
    assert metadata.gdn_cpu_chunk_indices is None
    return len(cases)


def device_checks():
    import torch
    from sglang.srt.layers.attention.fla.index import prepare_chunk_indices,prepare_chunk_offsets
    import importlib
    chunk=importlib.import_module('sglang.srt.layers.attention.fla.chunk')
    delta_h=importlib.import_module('sglang.srt.layers.attention.fla.chunk_delta_h')
    chunk_o=importlib.import_module('sglang.srt.layers.attention.fla.chunk_o')
    cumsum=importlib.import_module('sglang.srt.layers.attention.fla.cumsum')
    chunk_fwd=importlib.import_module('sglang.srt.layers.attention.fla.chunk_fwd')
    original=ModuleType('q38_original_chunk')
    original.__file__=str(Path(__file__).parent/'vendor/layers/attention/fla/chunk.py')
    exec(compile(Path(original.__file__).read_text(),original.__file__,'exec'),original.__dict__)
    torch.manual_seed(20261015)
    cases=[[64],[1,63],[63,64,65,129],[4096]*8,[8192],[32768],[65536],[32768,1]]
    results=[]
    for lengths in cases:
        prefixes=[0 if i%2==0 else 8192 for i in range(len(lengths))]
        cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32)
        meta=SimpleNamespace(query_start_loc=cu)
        batch=SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda:True),
                              spec_info=None,can_run_tbo=False,extend_seq_lens_cpu=lengths,
                              extend_prefix_lens_cpu=prefixes,batch_size=len(lengths),extend_num_tokens=sum(lengths),
                              input_ids=torch.empty(sum(lengths),device='cuda',dtype=torch.int64))
        backend=SimpleNamespace(forward_metadata=meta,_qwen27b_cpu_metadata_model=True)
        assert prepare_gdn_cpu_metadata(backend,batch)
        assert torch.equal(meta.gdn_cpu_chunk_indices,prepare_chunk_indices(cu,64))
        assert torch.equal(meta.gdn_cpu_chunk_offsets,prepare_chunk_offsets(cu,64))
        assert torch.equal(meta.gdn_cpu_has_initial_states,torch.tensor(prefixes,device='cuda')>0)
        results.append(dict(lengths=lengths,indices_offsets_mask_exact=True))
    for lengths in ([64],[65,127],[128,129,31]):
        t=sum(lengths);b=len(lengths);pool=b+2
        cu=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),device='cuda',dtype=torch.int32)
        indices,offsets,_,_=build_metadata_lists(list(lengths),[0]*b)
        ci=torch.tensor(indices,device='cuda',dtype=torch.int32)
        co=torch.tensor(offsets,device='cuda',dtype=torch.int64)
        q=torch.randn(1,t,8,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn_like(q);v=torch.randn(1,t,24,128,device='cuda',dtype=torch.bfloat16)
        g=-torch.rand(1,t,24,device='cuda',dtype=torch.float32)*.1
        beta=torch.rand(1,t,24,device='cuda',dtype=torch.bfloat16)
        slots=torch.tensor(list(range(b,0,-1)),device='cuda',dtype=torch.int32)
        for state_dtype in (torch.float32,torch.bfloat16):
            initial=torch.randn(pool,24,128,128,device='cuda',dtype=state_dtype)*.01
            old_state=initial.clone();new_state=initial.clone()
            kwargs=dict(q=q,k=k,v=v,g=g,beta=beta,initial_state_indices=slots,cu_seqlens=cu,use_qk_l2norm_in_kernel=True)
            old=original.chunk_gated_delta_rule(initial_state=old_state,**kwargs)
            # Prove new path never calls GPU-length metadata builders anywhere.
            saved=[]
            def forbidden(*args,**kwargs):raise AssertionError('GPU metadata fallback used')
            for module in (chunk,delta_h,chunk_o,cumsum,chunk_fwd):
                for name in ('prepare_chunk_indices','prepare_chunk_offsets'):
                    if hasattr(module,name):
                        saved.append((module,name,getattr(module,name)));setattr(module,name,forbidden)
            try:
                new=chunk.chunk_gated_delta_rule(initial_state=new_state,chunk_indices=ci,chunk_offsets=co,**kwargs)
                torch.cuda.synchronize()
            finally:
                for module,name,value in saved:setattr(module,name,value)
            assert torch.equal(old[0],new[0]),'GDN output changed'
            assert torch.equal(old[2],new[2]),'Intermediate GDN states changed'
            assert torch.equal(old_state,new_state),'SSM pool state changed'
            results.append(dict(lengths=lengths,state_dtype=str(state_dtype),gdn_output_and_states_exact=True,no_fallback=True))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',action='store_true')
    p.add_argument('--output')
    a=p.parse_args()
    result=dict(cpu_cases=cpu_checks(),cpu_pass=True,device_executed=False,
                scope='Metadata/GDN equivalence, not model accuracy or TTFT acceptance')
    if a.device:
        result['device_cases']=device_checks();result['device_executed']=True
    if a.output:
        target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
