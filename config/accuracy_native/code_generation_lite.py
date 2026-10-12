"""Local HF dataset builder; OpenCompass owns prompting, inference and scoring.

This builder deliberately supports only the 175-question test6.jsonl release.
Both LCBCodeGenerationDataset and LCBCodeGenerationEvaluator resolve this builder
through COMPASS_DATA_CACHE/data/code_generation_lite.
"""
import json
import os
from pathlib import Path

import datasets


class Test6Config(datasets.BuilderConfig):
    def __init__(self, version_tag='v6', **kwargs):
        super().__init__(**kwargs)
        if version_tag != 'v6':
            raise ValueError('This local dataset exposes v6/test6.jsonl only')


class CodeGenerationLite(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = Test6Config
    VERSION = datasets.Version('1.0.0')

    def _info(self):
        columns = ('question_title', 'question_content', 'platform', 'question_id',
                   'contest_id', 'contest_date', 'starter_code', 'difficulty',
                   'public_test_cases', 'private_test_cases', 'metadata')
        return datasets.DatasetInfo(features=datasets.Features({
            name: datasets.Value('string') for name in columns
        }))

    def _split_generators(self, dl_manager):
        path = Path(os.environ['ACCURACY_LCB_FILE'])
        if not path.is_file():
            raise FileNotFoundError(path)
        return [datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={'path': str(path)})]

    def _generate_examples(self, path):
        with open(path, encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) != 175 or len({r['question_id'] for r in rows}) != 175:
            raise ValueError('Expected exactly 175 unique test6 questions')
        for i, row in enumerate(rows):
            yield i, {key: row[key] for key in self.info.features}
