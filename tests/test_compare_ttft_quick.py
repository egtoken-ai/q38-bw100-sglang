"""Guard the per-case acceptance boundary and paired protocol requirement."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('compare_ttft_quick',
    Path(__file__).resolve().parents[1] / 'scripts/compare_ttft_quick.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot(short, long, seed=1):
    meta = {key: None for key in module.MATCH_FIELDS}
    meta['seed'] = seed
    rows = {key: dict(ttft_p50_s=value, output_tps=100., tpot_p50_ms=10.)
            for key, value in zip(module.CASES, (short, long))}
    return meta, rows, 300., {'image_id': 'sha256:test'}


class ComparisonTest(unittest.TestCase):
    def test_long_gain_cannot_hide_short_failure(self):
        with patch.object(module, 'read_run', side_effect=[snapshot(10, 10), snapshot(9.5, 1)]):
            result = module.compare('base', 'candidate')
        self.assertFalse(result['all_cases_ttft_20pct_met'])
        self.assertEqual([r['ttft_20pct_met'] for r in result['cases']], [False, True])

    def test_exact_twenty_percent_is_only_a_single_pair_pass(self):
        with patch.object(module, 'read_run', side_effect=[snapshot(10, 10), snapshot(8, 8)]):
            result = module.compare('base', 'candidate')
        self.assertTrue(result['all_cases_ttft_20pct_met'])
        self.assertFalse(result['final_goal_verified'])

    def test_unmatched_seed_rejected_even_with_large_gain(self):
        with patch.object(module, 'read_run', side_effect=[snapshot(10, 10, 1), snapshot(1, 1, 2)]):
            with self.assertRaisesRegex(ValueError, 'seed'):
                module.compare('base', 'candidate')


if __name__ == '__main__':
    unittest.main()
