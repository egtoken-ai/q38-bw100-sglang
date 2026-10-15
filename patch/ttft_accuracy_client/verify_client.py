#!/usr/bin/env python3
"""CPU-only native evaluator/reader/isolation sanity; never requests a model."""
import argparse
import ast
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    args=p.parse_args()
    assert os.environ.get('Q38_OC_API_ONLY')=='1'
    assert all(os.environ.get(k)=='' for k in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'))
    assert sorted(os.sched_getaffinity(0))==[0,1,2,3]
    assert os.getpriority(os.PRIO_PROCESS,0)==19
    root=args.evidence_dir
    before=json.loads((root/'vendor-stack-before.json').read_text())
    assert all(metadata.version(n)==v for n,v in before['versions'].items())
    assert all(str(metadata.distribution(n).locate_file(''))==v for n,v in before['locations'].items())
    dist=metadata.distribution('opencompass');site=Path(dist.locate_file(''))
    manifest=json.loads((root/'installed-source-expected.json').read_text())
    for relative,expected in manifest.items():
        assert hashlib.sha256((site/relative).read_bytes()).hexdigest()==expected,relative
    catch_sources={}
    for name in ('vllm.py','vllm_with_tf_above_v4_33.py'):
        source=site/'opencompass/models'/name
        tree=ast.parse(source.read_text())
        guarded=False
        for node in ast.walk(tree):
            if isinstance(node,ast.Try):
                imports=any(isinstance(n,ast.ImportFrom) and (n.module or '').split('.')[0]=='vllm' for statement in node.body for n in ast.walk(statement))
                catches=any(isinstance(h.type,ast.Name) and h.type.id=='ImportError' for h in node.handlers)
                guarded|=imports and catches
        assert guarded,name
        catch_sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ('vllm','vllm_hcu'):
        try: __import__(name)
        except ModuleNotFoundError as exc:assert exc.name==name
        else:raise AssertionError(f'{name} was not disabled')
    code="import importlib.util;\ntry: importlib.util.find_spec('vllm')\nexcept ModuleNotFoundError as e: print('INHERITED:'+str(e.name))\nelse: raise SystemExit(9)"
    child=subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True,timeout=20)
    assert 'INHERITED:vllm' in child.stdout
    from opencompass.models import OpenAISDK
    from opencompass.datasets import Gsm8kEvaluator
    from opencompass.openicl.icl_dataset_reader import load_partial_dataset
    from datasets import Dataset
    evaluator=Gsm8kEvaluator()
    sanity=evaluator.score(['42','7','-3'],['42','8','-3'])
    assert abs(sanity['accuracy']-200/3)<1e-10
    assert [x['correct'] for x in sanity['details']]==[True,False,True]
    selected=load_partial_dataset(Dataset.from_list([{'original_id':i} for i in range(1319)]),'[:64]')
    assert selected['original_id']==list(range(64))
    guard=Path(__file__).parent/'import_guard/sitecustomize.py'
    result=dict(status='PASS',blocked_optional_modules=['vllm','vllm_hcu'],guard_sha256=hashlib.sha256(guard.read_bytes()).hexdigest(),upstream_importerror_catches=catch_sources,subprocess_inherits_guard=True,official_evaluator=type(evaluator).__module__+'.'+type(evaluator).__name__,known_answer_sanity=sanity,reader_test_range='[:64]',reader_original_ids=selected['original_id'],opencompass_version=dist.version,source_commit='1543bd270b6fbabb1a118307f78d4144e8296467',installed_python_sha_verified=len(manifest),gpu_stack_unchanged=True,model_requests_sent=False,gpu_kernels_run=False,affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0))
    (root/'api-client-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
