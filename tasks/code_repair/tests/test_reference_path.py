"""The reference decision path is fully recorded, so --inputs recorded-forced never hits an unrecorded state."""
import hashlib
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cure_wm import TestCase
from evaluation import public_cache, state_key
from fixtures import execution_from_dict
from io_utils import experiment, lines

ASSETS = (ROOT / '../../assets').resolve()


@unittest.skipUnless((ASSETS / 'public_cache').exists(), 'evaluation assets not exported')
class ReferencePath(unittest.TestCase):
    def test_every_reference_state_is_cached_and_reaches_reference_final_code(self):
        reference = experiment()['reference']
        decisions = {(r['task_id'], r['seed']): r for r in lines(ASSETS / 'reference/decisions.jsonl')}
        matched = 0
        for seed in range(3):
            for item in lines(ASSETS / f'public/seed{seed}.jsonl'):
                tests = tuple(TestCase(**t) for t in item['visible_tests'])
                code, feedback = item['initial_code'], execution_from_dict(item['initial_feedback'])
                ref, steps = decisions[(item['task_id'], seed)], 0
                for step in range(3):
                    if feedback.all_pass:
                        break
                    raw = item['first_candidates'] if step == 0 else public_cache(ASSETS, 'historical_generation', dict(
                        protocol=reference['generation_protocol_sha256'], task_id=item['task_id'], seed=seed, step=step,
                        state_key=state_key(code, feedback, step)))
                    code = ([row['code'] for row in raw] + [code])[ref['choices'][step]]
                    feedback = execution_from_dict(public_cache(ASSETS, 'historical_visible', dict(
                        executor_sha256=reference['executor_source_sha256'], code=code,
                        tests=[asdict(t) for t in tests], timeout=item['timeout'], sandbox='required')))
                    steps += 1
                matched += steps == len(ref['choices']) and hashlib.sha256(code.encode()).hexdigest() == ref['final_code_sha256']
        self.assertEqual(matched, 600)


if __name__ == '__main__':
    unittest.main()
