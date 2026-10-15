#!/usr/bin/env python3
"""Three M65536 shapes, shared imports, one externally bounded child process."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

SHAPES = [(65536,17408,5120),(65536,8192,5120),(65536,5120,3072)]


def write(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def shared(args):
    from module_worker import worker
    result = dict(scope='M65536 module micro only, not TTFT or model accuracy',
                  rows=[],completed=False,approved_shapes=[],elapsed_s=0)
    start=time.monotonic()
    write(args.output,result)
    for shape in SHAPES:
        output=str(Path(args.output).with_suffix(''))+'-'+'x'.join(map(str,shape))+'.json'
        Path(output).unlink(missing_ok=True)
        try:
            worker(argparse.Namespace(shape=shape,output=output))
            row=json.loads(Path(output).read_text())
            result['rows'].append(row)
            pairs=row.get('paired_improvement_pct',[])
            eligible=(row.get('status')=='PASS' and row.get('full_output_exact') is True
                      and row.get('native_dispatch_hits')==1 and len(pairs)==8
                      and all(v>0 for v in pairs) and row.get('improvement_pct',0)>=2)
            if eligible:result['approved_shapes'].append(shape)
            result['elapsed_s']=time.monotonic()-start
            write(args.output,result)
            gc.collect()
        except Exception as exc:
            result['rows'].append(dict(shape=shape,status='FAIL',error=f'{type(exc).__name__}: {exc}'))
            result['stop_reason']='first failure; no further GPU work'
            result['elapsed_s']=time.monotonic()-start
            write(args.output,result)
            raise
    result['completed']=True
    result['all_exact_passed']=all(r.get('full_output_exact') is True for r in result['rows'])
    result['elapsed_s']=time.monotonic()-start
    write(args.output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--run-shared',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    assert 0<args.timeout<=450
    if args.dry_run:
        print(json.dumps(dict(shapes=SHAPES,timeout_s=args.timeout,gpu_executed=False,
                              existing_shapes_rerun=False,service_allowlist_changed=False)))
        return
    if args.run_shared:
        shared(args)
        return
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    # Fresh suite file prevents stale successful JSON after import timeout.
    write(output,dict(completed=False,rows=[],approved_shapes=[],status='STARTING'))
    cmd=[sys.executable,str(Path(__file__).resolve()),'--run-shared','--output',str(output)]
    with Path(str(output)+'.log').open('w') as log:
        try:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
            status='DONE' if proc.returncode==0 else 'FAILED'
            code=proc.returncode
        except subprocess.TimeoutExpired:
            status='TIMEOUT';code=124
    result=json.loads(output.read_text())
    result.update(process_status=status,process_returncode=code,timeout_s=args.timeout)
    write(output,result)
    print(json.dumps(result),flush=True)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
