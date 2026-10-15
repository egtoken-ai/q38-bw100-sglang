"""Process-local isolation of optional local GPU backends for native API eval."""
import importlib.abc
import os
import sys

BLOCKED_ROOTS = frozenset({'vllm', 'vllm_hcu'})


class Q38ApiOnlyOptionalBackendFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in BLOCKED_ROOTS:
            raise ModuleNotFoundError(
                f'Q38 API-only evaluator disables optional local backend {fullname}',
                name=fullname,
            )
        return None


if os.environ.get('Q38_OC_API_ONLY') == '1':
    sys.meta_path.insert(0, Q38ApiOnlyOptionalBackendFinder())
    print('Q38_OC_API_ONLY: optional local backends disabled: vllm,vllm_hcu', file=sys.stderr)
