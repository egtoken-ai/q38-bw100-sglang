"""Install only against the audited SGLang source; retain an A/B baseline."""

import hashlib
import json
import py_compile
from pathlib import Path

BASE_SHA256 = "76680de79e5b078e2321fc243382285af8396c91ab47bafa7be0a0738ad91a1d"
root = Path("/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/fla")
target = root / "chunk_fwd.py"
baseline = root / "chunk_fwd_baseline.py"
source = Path(__file__).with_name("chunk_fwd.py")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if digest(target) != BASE_SHA256:
    raise SystemExit("Refusing to patch: base chunk_fwd.py fingerprint changed")
if baseline.exists():
    raise SystemExit("Refusing to overwrite an existing baseline module")
py_compile.compile(str(source), doraise=True)
baseline.write_bytes(target.read_bytes())
target.write_bytes(source.read_bytes())
manifest = {
    "base_sha256": digest(baseline),
    "patched_sha256": digest(target),
    "rollback_env": "SGLANG_GDN_ZERO_INIT_BASELINE=1",
}
Path(__file__).with_name("installed.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print(json.dumps(manifest))
