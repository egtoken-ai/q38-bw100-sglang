"""Protocol glue tests; no model/toolkit installation required."""
import argparse
import base64
import contextlib
import http.server
import importlib.util
import io
import json
from pathlib import Path
import pickle
import tempfile
import threading
import unittest
from unittest.mock import patch
import zlib

MODULE = Path(__file__).resolve().parents[1] / 'scripts/accuracy.py'
spec = importlib.util.spec_from_file_location('accuracy', MODULE)
accuracy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accuracy)


@contextlib.contextmanager
def server():
    class Handler(http.server.BaseHTTPRequestHandler):
        requests = []
        downloads = 0
        response = {'choices': [{'message': {'content': '<think>work</think>42'}, 'finish_reason': 'stop'}]}
        status = 200

        def log_message(self, *args):
            pass

        def do_GET(self):
            type(self).downloads += 1
            self.send_response(200)
            self.end_headers()
            self.wfile.write((json.dumps({'question': '1+1?', 'answer': '#### 2'}) + '\n').encode() * 1319)

        def do_POST(self):
            type(self).requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(self.status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.response).encode())

    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}', Handler
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()


def args_for(folder, url='http://localhost/v1'):
    args = accuracy.parser().parse_args(['run'])
    args.data_dir = Path(folder) / 'data'
    args.output_dir = Path(folder) / 'output'
    args.extra_body = {}
    args.base_url = url
    args.retries = 0
    return args


class AccuracyTests(unittest.TestCase):
    def test_thresholds_without_rounding(self):
        for key, spec in accuracy.SPECS.items():
            result = dict(status='scored', score=spec['threshold'], evaluated=spec['count'])
            self.assertEqual(accuracy.admission(key, result)['status'], 'PASS')
            result['score'] -= 0.000001
            self.assertEqual(accuracy.admission(key, result)['status'], 'FAIL')
            result['score'] = 100
            result['evaluated'] -= 1
            self.assertEqual(accuracy.admission(key, result)['status'], 'SMOKE')
            result['evaluated'] = spec['count']
            result['smoke'] = True
            self.assertEqual(accuracy.admission(key, result)['status'], 'SMOKE')

    def test_report_requires_all_four(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            results = {k: dict(status='scored', score=100, evaluated=v['count']) for k, v in accuracy.SPECS.items()}
            out = Path(folder)
            self.assertEqual(accuracy.report(out, results), 0)
            parsed = json.loads((out / 'summary.json').read_text())
            self.assertEqual(parsed['overall'], 'PASS')
            self.assertEqual(len(parsed['results']), 4)
            results['gsm8k']['score'] = 94.9999
            self.assertEqual(accuracy.report(out, results), 1)
            results.pop('mathvision')
            self.assertEqual(accuracy.report(out, results), 2)
            self.assertTrue((out / 'summary.csv').read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_nonfinite_scores_never_pass(self):
        for score in (float('nan'), float('inf'), -1, 101):
            self.assertEqual(accuracy.admission('gsm8k', dict(status='scored', score=score, evaluated=1319))['status'], 'ERROR')

    def test_download_reuse_and_offline(self):
        with tempfile.TemporaryDirectory() as folder, server() as (url, handler), contextlib.redirect_stdout(io.StringIO()):
            args = args_for(folder)
            args.benchmarks = ['gsm8k']
            source = {'gsm8k': [('gsm8k/test.jsonl', url + '/test.jsonl', None)]}
            with patch.dict(accuracy.SOURCES, source):
                first = accuracy.prepare(args)
                second = accuracy.prepare(args)
                self.assertEqual(first, second)
                self.assertEqual(handler.downloads, 1)
                args.offline = True
                accuracy.prepare(args)
                self.assertEqual(handler.downloads, 1)
                path = args.data_dir / 'gsm8k/test.jsonl'
                path.write_text('{}\n')
                with self.assertRaises(ValueError):
                    accuracy.prepare(args)
                self.assertEqual(handler.downloads, 1)

    def test_failed_download_never_publishes_cache(self):
        with tempfile.TemporaryDirectory() as folder, server() as (url, handler), contextlib.redirect_stdout(io.StringIO()):
            args = args_for(folder)
            args.benchmarks = ['gsm8k']
            with patch.dict(accuracy.SOURCES, {'gsm8k': [('gsm8k/test.jsonl', url, ('sha256', 'bad'))]}):
                with self.assertRaises(ValueError):
                    accuracy.prepare(args)
            self.assertFalse((args.data_dir / 'gsm8k/test.jsonl').exists())

    def test_http_final_answer_and_no_tools(self):
        with tempfile.TemporaryDirectory() as folder, server() as (url, handler):
            args = args_for(folder, url)
            result = accuracy.generate(args, [{'role': 'user', 'content': 'question'}])
            self.assertEqual(result['prediction'], '42')
            self.assertIn('<think>', result['response']['choices'][0]['message']['content'])
            self.assertNotIn('tools', handler.requests[0])
            handler.response['choices'][0]['finish_reason'] = 'length'
            self.assertEqual(accuracy.generate(args, [])['status'], 'invalid')
            handler.response = {'choices': [{'message': {'content': '', 'reasoning_content': '42'}, 'finish_reason': 'stop'}]}
            self.assertEqual(accuracy.generate(args, [])['status'], 'error')
            handler.status = 401
            self.assertEqual(accuracy.generate(args, [])['status'], 'error')

    def test_math_answer_extraction(self):
        self.assertEqual(accuracy.extract_math_answer(r'work \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertEqual(accuracy.extract_math_answer('<think>1</think>The final answer is: 7'), '7')
        self.assertEqual(accuracy.extract_math_answer(r'\boxed{A} then \boxed{B}'), 'B')

    def test_lcb_private_tests_are_included(self):
        private = json.dumps([dict(input='2', output='4')])
        row = dict(question_id='a', public_test_cases=json.dumps([dict(input='1', output='2')]),
                   private_test_cases=private, metadata='{}')
        result = json.loads(accuracy.lcb_sample(row)['input_output'])
        self.assertEqual(result['inputs'], ['1', '2'])
        row['private_test_cases'] = base64.b64encode(zlib.compress(pickle.dumps(private))).decode()
        self.assertEqual(json.loads(accuracy.lcb_sample(row)['input_output']), result)
        row['private_test_cases'] = base64.b64encode(zlib.compress(pickle.dumps(str))).decode()
        with self.assertRaises(pickle.UnpicklingError):
            accuracy.lcb_sample(row)

    def test_resume_retries_only_failed_and_rejects_changed_config(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            args = args_for(folder)
            rows = [dict(question='q', answer='2'), dict(question='q2', answer='3')]
            manifest = {'gsm8k': {'sha256': 'data-v1'}}
            loader = lambda *_: (rows, lambda i: [{'role': 'user', 'content': str(i)}], None)
            replies = [dict(status='ok', prediction='2'), dict(status='error', error='temporary')]
            with patch.object(accuracy, 'load_benchmark', loader), patch.object(accuracy, 'generate', side_effect=replies):
                with self.assertRaises(RuntimeError):
                    accuracy.run_one('gsm8k', args, manifest, {})
            cache = args.output_dir / 'gsm8k/predictions.jsonl'
            with cache.open('ab') as f:
                f.write(b'{"id":')  # a process crash during the final append
            with patch.object(accuracy, 'load_benchmark', loader), patch.object(accuracy, 'generate', return_value=dict(status='ok', prediction='3')) as generate, patch.object(accuracy, 'score_benchmark', return_value=100):
                accuracy.run_one('gsm8k', args, manifest, {})
                self.assertEqual(generate.call_count, 1)
                accuracy.run_one('gsm8k', args, manifest, {})
                self.assertEqual(generate.call_count, 1)
                args.model_revision = 'new-checkpoint'
                with self.assertRaises(ValueError):
                    accuracy.run_one('gsm8k', args, manifest, {})

    def test_suite_continues_on_error_and_combines_only_same_config(self):
        def run(key, *_):
            if key == 'mathvision':
                raise RuntimeError('judge unavailable')
            return dict(status='scored', score=100, evaluated=accuracy.SPECS[key]['count'])

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            argv = ['run', '--output-dir', folder, '--data-dir', folder, '--model-revision', 'weights-A']
            model_list = {'data': [{'id': accuracy.parser().parse_args(['run']).model}]}
            with patch.object(accuracy, 'request_json', return_value=model_list), patch.object(accuracy, 'prepare', return_value={}), patch.object(accuracy, 'toolkit_versions', return_value={}), patch.object(accuracy, 'run_one', side_effect=run) as runner:
                self.assertEqual(accuracy.main(argv), 2)
                self.assertEqual(runner.call_count, 4)
                result = json.loads((Path(folder) / 'summary.json').read_text())
                self.assertEqual([r['status'] for r in result['results']], ['PASS', 'PASS', 'PASS', 'ERROR'])
                runner.side_effect = lambda key, *_: dict(status='scored', score=100, evaluated=accuracy.SPECS[key]['count'])
                self.assertEqual(accuracy.main(argv + ['--benchmarks', 'mathvision']), 0)
                with self.assertRaises(ValueError):
                    accuracy.main(argv + ['--model-revision', 'weights-B'])


if __name__ == '__main__':
    unittest.main()
