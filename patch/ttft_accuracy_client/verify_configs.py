#!/usr/bin/env python3
"""Parse native configs and local tokenizer; no model/API request."""
import argparse
import hashlib
import json
from pathlib import Path
import socket


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--evidence-dir',type=Path,required=True)
    args=p.parse_args()
    # Detect unexpected network activity during configuration/tokenizer checks.
    def deny_connection(*a,**kw):
        raise AssertionError('CPU configuration verification must not connect to network')
    socket.socket.connect=deny_connection
    from mmengine.config import Config
    from transformers import AutoTokenizer
    rows=[]
    comparable=[]
    for label in ('native','optimized'):
        path=args.output_root/f'gsm8k64-{label}/configs/gsm8k.py'
        cfg=Config.fromfile(str(path))
        assert len(cfg.datasets)==len(cfg.models)==1
        dataset,model=cfg.datasets[0],cfg.models[0]
        assert dataset.reader_cfg.test_range=='[:64]'
        assert model.extra_body.chat_template_kwargs.enable_thinking is False
        assert model.batch_size==model.max_workers==model.query_per_second==32
        assert model.max_out_len==512 and dataset.infer_cfg.inferencer.max_out_len==512
        assert model.temperature==0 and model.openai_extra_kwargs.seed==20261012
        assert model.run_cfg.num_gpus==0
        settings=dict(reader_range=dataset.reader_cfg.test_range,dataset_path=dataset.path,
                      thinking=model.extra_body.chat_template_kwargs.enable_thinking,
                      concurrency=model.batch_size,max_tokens=model.max_out_len,
                      seed=model.openai_extra_kwargs.seed,base_url=model.openai_api_base,
                      tokenizer_path=model.tokenizer_path,model=model.path)
        comparable.append(settings)
        rows.append(dict(label=label,config=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),settings=settings))
    assert comparable[0]==comparable[1]
    tok=AutoTokenizer.from_pretrained(comparable[0]['tokenizer_path'],local_files_only=True)
    sample=tok.encode('What is 2 + 2?',add_special_tokens=False)
    assert sample
    report=dict(status='PASS',configs=rows,tokenizer_class=type(tok).__name__,local_tokenizer_works=True,network_connections_allowed=False,model_requests_sent=False,gpu_kernels_run=False,scope='64-sample paired subset; not full GSM8K or full task accuracy acceptance')
    (args.evidence_dir/'native64-config-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
