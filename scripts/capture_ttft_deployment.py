"""Record actual model argv, selected environment and immutable Docker image ID.

Run on the Docker host. No full environment or Docker configuration is saved.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


PROCESS_PROBE = r'''
import json
from pathlib import Path
allowed = {
    "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
    "HSA_FORCE_FINE_GRAIN_PCIE", "LIGHTOP_GPU_CUS", "ATTENTION_BACKEND",
    "MODEL_PATH", "CHUNKED_PREFILL_SIZE", "MAX_PREFILL_TOKENS",
    "PACK_PAGED_KV_TO_VARLEN", "SGLANG_KV_LAYOUT_HCU_FA",
    "SGLANG_ROCM_USE_AITER_MOE", "SGLANG_QWEN27B_PACKED_PREFILL",
    "SGLANG_QWEN27B_GDN_PREFILL_VIEWS", "SGLANG_QWEN27B_GDN_FUSED_PREPARE",
    "SGLANG_QWEN27B_FUSED_KV_PACK", "SGLANG_QWEN27B_GDN_CPU_METADATA",
}
rows = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if 'sglang.launch_server' not in argv:
            continue
        env = dict(item.split('=', 1) for item in
                   (proc/'environ').read_bytes().decode().split('\0') if '=' in item)
        rows.append({'pid': proc.name, 'argv': argv,
                     'environment': {k: v for k, v in env.items() if k in allowed}})
    except (OSError, UnicodeError):
        continue
print(json.dumps(rows))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite an existing deployment snapshot')
    def docker(*parts):
        return subprocess.check_output(['docker', *parts], text=True)
    info = json.loads(docker('inspect', args.container))[0]
    processes = json.loads(docker('exec', args.container, 'python3', '-c', PROCESS_PROBE))
    if len(processes) != 1:
        raise SystemExit(f'Expected one SGLang launcher, found {len(processes)}')
    report = {
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'container': args.container, 'image': info['Config']['Image'],
        'image_id': info['Image'], 'processes': processes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
