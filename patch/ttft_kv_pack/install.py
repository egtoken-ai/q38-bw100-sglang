"""Install against the exact previously validated packed candidate."""
import hashlib
from pathlib import Path
import py_compile
import shutil

root = Path('/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention')
target = root / 'pack_paged_kv_to_varlen.py'
reference = Path(__file__).with_name('pack_paged_kv_to_varlen.original.py')
assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(reference.read_bytes()).digest(), 'Unexpected parent source'
source = target.read_text()
old = '''    packed_k, packed_v = pack_paged_kv_to_varlen(
        key_cache.view(-1, layer.tp_k_head_num, page_size, layer.head_dim),
        value_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim, page_size),
        metadata.page_table,
        forward_batch.seq_lens_cpu[: forward_batch.batch_size],
        page_size,
    )'''
new = '''    if (
        get_bool_env_var("SGLANG_QWEN27B_FUSED_KV_PACK", default="false")
        and key_cache.dtype == torch.bfloat16
        and value_cache.dtype == torch.bfloat16
        and layer.tp_k_head_num == layer.tp_v_head_num == 2
        and layer.head_dim == layer.v_head_dim == 256
        and page_size == 64
    ):
        from sglang.srt.layers.attention.qwen27b_fused_kv_pack import fused_pack
        packed_k, packed_v = fused_pack(
            key_cache.view(-1, 2, 64, 256),
            value_cache.view(-1, 2, 256, 64),
            metadata.page_table,
            metadata.cu_seqlens_k[:forward_batch.batch_size + 1],
            int(forward_batch.seq_lens_cpu[:forward_batch.batch_size].sum().item()),
            metadata.max_seq_len_k,
        )
    else:
''' + '\n'.join('    ' + line for line in old.splitlines())
assert source.count(old) == 1, 'Expected exactly one packing call'
patched = source.replace(old, new)
compile(patched, str(target), 'exec')
helper = Path(__file__).with_name('fused_kv_pack.py')
py_compile.compile(str(helper), doraise=True)
backup = target.with_name('pack_paged_kv_to_varlen_before_fused.py')
assert not backup.exists(), 'Already installed'
shutil.copyfile(target, backup)
shutil.copyfile(helper, root / 'qwen27b_fused_kv_pack.py')
target.write_text(patched)
print('Installed opt-in direct paged-KV pack')
