"""Native command generation, output auditing and admission integration tests."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import accuracy_native as native


def arguments(folder):
    args = native.parser().parse_args(['run'])
    args.data_dir = Path(folder) / 'data'
    args.output_dir = Path(folder) / 'output'
    args.output_dir.mkdir()
    args.extra_body = {}
    args.judge_model = 'fixed-judge'
    args.judge_base_url = 'http://judge/v1'
    return args


def tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


class NativeTests(unittest.TestCase):
    def test_configs_and_real_cli_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            configs = native.generate_configs(args)
            self.assertIn('type=OpenAISDK', configs['gsm8k'].read_text())
            self.assertIn('OpenICLEvalTask', configs['gsm8k'].read_text())
            self.assertIn("datasets[0]['n'] = 1", configs['livecodebench'].read_text())
            compile(configs['gsm8k'].read_text(), str(configs['gsm8k']), 'exec')
            for key in ('gsm8k', 'livecodebench'):
                commands = native.commands(args, key, configs[key], 'all')
                self.assertEqual(len(commands), 1)
                cmd = commands[0]['argv']
                self.assertEqual(Path(cmd[0]).name, 'opencompass')
                self.assertEqual(cmd[-4:], ['--reuse', 'native', '--mode', 'all'])
            commands = native.commands(args, 'mathvision', configs['mathvision'], 'all')
            self.assertEqual([c['stage'] for c in commands], ['infer', 'eval'])
            self.assertIn('--judge-base-url', commands[1]['argv'])
            cfg = json.loads(configs['mathvision'].read_text())
            self.assertEqual(cfg['data']['MathVision_MINI']['class'], 'MathVision')
            self.assertNotIn('key', cfg['model']['q38'])
            self.assertNotIn('tools', cfg['model']['q38'])

    def test_separate_inference_and_judge_credentials(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'ACCURACY_API_KEY': 'subject-secret', 'ACCURACY_JUDGE_API_KEY': 'judge-secret'}):
            args = arguments(folder)
            self.assertEqual(native.child_env(args, 'infer', 'mathvision')['OPENAI_API_KEY'], 'subject-secret')
            self.assertEqual(native.child_env(args, 'eval', 'mathvision')['OPENAI_API_KEY'], 'judge-secret')
            self.assertEqual(native.child_env(args, 'infer', 'mathvision')['PRED_FORMAT'], 'tsv')

    def test_oc_results_require_all_native_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            root = args.output_dir / 'opencompass/livecodebench/native'
            preds = {str(i): {'prediction': '```python\nprint(1)\n```'} for i in range(175)}
            native.shared.write_json(root / 'predictions/q38/livecodebench_v6.json', preds)
            native.shared.write_json(root / 'results/q38/livecodebench_v6.json', {'pass@1': 88.3})
            result = native.audit(args, 'livecodebench')
            self.assertEqual(result['score'], 88.3)  # already percent, not 0..1
            self.assertEqual(native.shared.admission('livecodebench', result)['status'], 'PASS')
            preds.pop('174')
            native.shared.write_json(root / 'predictions/q38/livecodebench_v6.json', preds)
            with self.assertRaises(ValueError):
                native.audit(args, 'livecodebench')

    def test_vlm_unit_conversion_and_latest_run_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            root = args.output_dir / 'vlmeval/realworldqa/q38/T20261012-100000'
            rows = [{'index': str(i), 'prediction': 'A'} for i in range(765)]
            tsv(args.data_dir / 'vlm/RealWorldQA.tsv', rows)
            tsv(root / 'q38_RealWorldQA.tsv', rows)
            status = {'datasets': {'RealWorldQA': dict(status='done', prediction_file='q38_RealWorldQA.tsv',
                                                     metrics={'split=none|Overall': 0.859})}}
            native.shared.write_json(root / 'status.json', status)
            self.assertAlmostEqual(native.audit(args, 'realworldqa')['score'], 85.9)
            newer = root.parent / 'T20261012-110000'
            status['datasets']['RealWorldQA']['status'] = 'eval'
            native.shared.write_json(newer / 'status.json', status)
            with self.assertRaises(ValueError):
                native.audit(args, 'realworldqa')

    def test_mathvision_judge_failures_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            root = args.output_dir / 'vlmeval/mathvision/q38/T20261012-100000'
            rows = [{'index': str(i), 'prediction': 'A'} for i in range(304)]
            tsv(args.data_dir / 'vlm/MathVision_MINI.tsv', rows)
            tsv(root / 'q38_MathVision_MINI.tsv', rows)
            status = {'datasets': {'MathVision_MINI': dict(status='done', prediction_file='q38_MathVision_MINI.tsv',
                                                         metrics={'Subject=Overall|acc': 90.0})}}
            native.shared.write_json(root / 'status.json', status)
            graded = [{'index': str(i), 'res': 'A', 'log': 'Succeed'} for i in range(304)]
            tsv(root / 'q38_MathVision_MINI_fixed-judge.tsv', graded)
            self.assertEqual(native.audit(args, 'mathvision')['score'], 90.0)
            graded[0]['log'] = 'All 5 retries failed.'
            tsv(root / 'q38_MathVision_MINI_fixed-judge.tsv', graded)
            with self.assertRaises(ValueError):
                native.audit(args, 'mathvision')

    def test_adding_initial_judge_is_allowed_but_model_changes_are_not(self):
        with tempfile.TemporaryDirectory() as folder:
            args = arguments(folder)
            current = native.settings(args)
            prior = {**current, 'judge_model': '', 'judge_base_url': ''}
            native.check_suite(prior, current)
            with self.assertRaises(ValueError):
                native.check_suite(current, {**current, 'judge_model': 'different-judge'})
            with self.assertRaises(ValueError):
                native.check_suite(current, {**current, 'model_revision': 'other-weights'})

    def test_native_failure_does_not_abort_other_benchmarks(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            argv = ['run', '--output-dir', folder, '--judge-model', 'judge', '--judge-base-url', 'http://judge/v1']
            def execute(args, key, command):
                if key == 'gsm8k':
                    raise RuntimeError('native process failed')
            def audit(args, key):
                return dict(status='scored', score=100.0, evaluated=native.shared.SPECS[key]['count'])
            with patch.object(native.shared, 'toolkit_versions', return_value={}), patch.object(native, 'prepare', return_value={}), patch.object(native, 'execute', side_effect=execute) as executor, patch.object(native, 'audit', side_effect=audit):
                self.assertEqual(native.main(argv), 2)
                self.assertEqual(executor.call_count, 6)
            report = json.loads((Path(folder) / 'summary.json').read_text())
            self.assertEqual([r['status'] for r in report['results']], ['ERROR', 'PASS', 'PASS', 'PASS'])
            self.assertEqual(report['engine'], 'native-cli')


if __name__ == '__main__':
    unittest.main()
