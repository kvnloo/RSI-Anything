import concurrent.futures
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from artifacts import Cache, atomic_json, checkpoint_manifest, hash_object
from cure_wm import Candidate, TestCase, execute
from fixtures import execution_from_dict
from io_utils import PACKAGE, experiment, lines, read, sha, verify_assets
from public_context import feedback_text
from seed import stable_seed


def state_key(code, feedback, step):
    outcomes = [{k: getattr(row, k) for k in ('test_id', 'status', 'actual', 'expected', 'stderr')}
                for row in feedback.results]
    return hashlib.sha256(json.dumps(dict(code=code, feedback=outcomes, step=step), sort_keys=True).encode()).hexdigest()


def choose(predictions):
    if len(predictions) != 5:
        raise ValueError('Expected four candidates and one current program')
    utility = [row['utility'] for row in predictions]
    if not all(math.isfinite(v) and 0 <= v <= 1.25 for v in utility):
        raise ValueError('Invalid utility')
    best = max(range(4), key=lambda i: utility[i])
    return best if utility[best] > utility[4] else 4


RECORDED = ('recorded', 'recorded-forced')
GAP_BINS = (0, .001, .002, .005, .01, .02, .05, .1, .2, math.inf)


def decide(predictions, forced=None):
    """Return (followed index, our index). Teacher forcing follows the recorded reference choice."""
    ours = choose(predictions)
    return (ours if forced is None else forced), ours


def agreement(rows):
    """Decision agreement of our world model with the followed reference path (recorded-forced runs)."""
    steps, gaps, episodes, stepped = 0, [], 0, 0
    for row in rows:
        same = True
        for t in row['trace']:
            utility = [p['utility'] for p in t['predictions']]
            steps += 1
            if t['our_choice'] != t['selected_index']:
                same = False
                gaps.append(dict(task_id=row['task_id'], seed=row['seed'], step=t['step'], our_choice=t['our_choice'],
                                 reference_choice=t['selected_index'],
                                 gap=utility[t['our_choice']] - utility[t['selected_index']]))
        episodes += same
        stepped += bool(row['trace'])
    labels = [f'[{a},{b})' for a, b in zip(GAP_BINS, GAP_BINS[1:])]
    histogram = {label: sum(a <= g['gap'] < b for g in gaps) for label, a, b in zip(labels, GAP_BINS, GAP_BINS[1:])}
    values = sorted(g['gap'] for g in gaps)
    return dict(decision_steps=steps, steps_agreeing=steps - len(gaps),
                step_agreement=(steps - len(gaps)) / steps if steps else 1.0,
                episodes=len(rows), episodes_agreeing=episodes, episode_agreement=episodes / len(rows) if rows else 1.0,
                episodes_with_decisions=stepped,
                episodes_with_decisions_agreeing=episodes - (len(rows) - stepped),
                disagreements=len(gaps), gap_histogram=histogram,
                gap_max=values[-1] if values else 0.0,
                gap_median=values[len(values) // 2] if values else 0.0,
                gap_p95=values[min(len(values) - 1, int(.95 * len(values)))] if values else 0.0,
                disagreement_details=gaps)


class Predictor:
    def __init__(self, checkpoint, backbone):
        from semantic_world import SemanticExecutionWorld
        self.model = SemanticExecutionWorld.load(checkpoint, backbone, training=False, device='cuda').eval()
        self.identity = self.model.identity
        assert self.model.public_world.target is None

    def predict(self, problem, code, feedback, tests, candidates):
        public = [asdict(test) for test in tests if test.test_id.startswith('public_')]
        observed = asdict(feedback)
        observed['results'] = [row for row in observed['results'] if row['test_id'].startswith('public_')]
        contexts = [(f'Whole-training-suite execution forecast. Only public observations are supplied.\n'
                     f'Problem:\n{problem}\nCurrent program:\n{code}\n'
                     f'Trusted previous public execution:\n{feedback_text(observed)}\n'
                     f'Candidate action:\n{candidate.code}\nOfficial public test suite:\n'
                     + json.dumps(public, ensure_ascii=False)) for candidate in candidates]
        values = self.model.score_suite(contexts)
        return [dict(candidate_id=c.candidate_id, utility=float(values['utility'][i]),
                     full_suite_pass_probability=float(values['full_suite_pass_probability'][i]),
                     predicted_pass_fraction=float(values['predicted_pass_fraction'][i]))
                for i, c in enumerate(candidates)]


def public_cache(assets, namespace, key):
    identity = hash_object(key)
    path = assets / 'public_cache' / namespace / identity[:2] / (identity + '.json')
    if not path.exists():
        raise FileNotFoundError(f'Unrecorded public state {identity}; use --inputs fresh to generate a new trajectory')
    entry = read(path)
    assert entry['key_hash'] == identity and entry['key'] == key
    return entry['value']


def worker(args):
    import torch
    torch.set_num_threads(4)
    assets, output = Path(args.assets).resolve(), Path(args.output).resolve()
    protocol = read(output / 'protocol.json')
    seal = sha(output / 'protocol.json')
    model = Predictor(protocol['checkpoint'], assets / 'models/world')
    assert model.identity == protocol['world_identity']
    coder = None
    if protocol['inputs'] not in RECORDED:
        from coder import FrozenCoder
        coder = FrozenCoder(assets / 'models/policy')
    conf = experiment()
    forced = protocol['inputs'] == 'recorded-forced'
    reference = {row['task_id']: row['choices'] for row in lines(assets / 'reference/decisions.jsonl')
                 if row['seed'] == args.seed} if forced else {}
    items = [row for i, row in enumerate(lines(assets / f'public/seed{args.seed}.jsonl')) if i % 8 == args.shard]
    path = output / 'public' / f'seed{args.seed}.shard{args.shard}.jsonl'
    assert not path.exists() and len(items) == 25
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for number, item in enumerate(items, 1):
            tests = tuple(TestCase(**test) for test in item['visible_tests'])
            code = item['initial_code']
            feedback = execution_from_dict(item['initial_feedback'])
            trace = []
            for step in range(3):
                if feedback.all_pass:
                    break
                key = state_key(code, feedback, step)
                if step == 0:
                    raw = item['first_candidates']
                elif coder is None:
                    raw = public_cache(assets, 'historical_generation', dict(
                        protocol=conf['reference']['generation_protocol_sha256'], task_id=item['task_id'],
                        seed=args.seed, step=step, state_key=key))
                else:
                    generated = coder.refine(item['problem'], code, feedback, tests, k=4,
                        seed=stable_seed(item['task_id'], args.seed, step, key, 'repair'))
                    raw = [dict(candidate_id=f't{step}_k{i}', code=c) for i, (c, _) in enumerate(generated)]
                assert len(raw) == 4
                candidates = tuple(Candidate(row['candidate_id'], row['code']) for row in raw) + (Candidate('keep_current', code),)
                predictions = model.predict(item['problem'], code, feedback, tests, candidates)
                choice, ours = decide(predictions, reference[item['task_id']][step] if forced else None)
                before, previous = code, feedback
                code = candidates[choice].code
                if coder is None:
                    value = public_cache(assets, 'historical_visible', dict(
                        executor_sha256=conf['reference']['executor_source_sha256'], code=code,
                        tests=[asdict(t) for t in tests], timeout=item['timeout'], sandbox='required'))
                    feedback = execution_from_dict(value)
                else:
                    feedback = execute(code, tests, timeout=item['timeout'])
                trace.append(dict(step=step, current_code=before, previous_feedback=asdict(previous),
                    candidates=[asdict(c) for c in candidates], predictions=predictions,
                    selected_index=choice, visible=asdict(feedback), **(dict(our_choice=ours) if forced else {})))
            if forced:
                assert [t['selected_index'] for t in trace] == reference[item['task_id']]
            row = dict(task_id=item['task_id'], seed=args.seed, initial_code=item['initial_code'],
                final_code=code, trace=trace, protocol_sha256=seal, world_identity=model.identity,
                hidden_tests_accessed=False, optimizer_steps=0)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            atomic_json(path.with_suffix('.status.json'), dict(rows=number, total=25, time=time.time()))
    atomic_json(path.with_suffix('.complete.json'), dict(rows=25, sha256=sha(path), protocol_sha256=seal))


def seal_public(assets, output):
    config = experiment()
    protocol_sha = sha(output / 'protocol.json')
    forced = read(output / 'protocol.json')['inputs'] == 'recorded-forced'
    reference = {(row['task_id'], row['seed']): row['choices'] for row in lines(assets / 'reference/decisions.jsonl')}
    rows, files = [], {}
    for seed in range(3):
        originals = {row['task_id']: row for row in lines(assets / f'public/seed{seed}.jsonl')}
        for shard in range(8):
            path = output / 'public' / f'seed{seed}.shard{shard}.jsonl'
            complete = read(path.with_suffix('.complete.json'))
            assert sha(path) == complete['sha256'] and complete['rows'] == 25
            batch = list(lines(path))
            for row in batch:
                assert row['protocol_sha256'] == protocol_sha and not row['hidden_tests_accessed']
                assert row['optimizer_steps'] == 0 and len(row['trace']) <= 3
                code = originals[row['task_id']]['initial_code']
                assert code == row['initial_code']
                for step, transition in enumerate(row['trace']):
                    assert transition['step'] == step and transition['current_code'] == code
                    assert len(transition['candidates']) == 5
                    assert transition['candidates'][4]['code'] == code
                    if forced:
                        assert transition['our_choice'] == choose(transition['predictions'])
                        assert transition['selected_index'] == reference[(row['task_id'], row['seed'])][step]
                    else:
                        assert transition['selected_index'] == choose(transition['predictions'])
                    code = transition['candidates'][transition['selected_index']]['code']
                assert code == row['final_code']
                if forced:
                    assert len(row['trace']) == len(reference[(row['task_id'], row['seed'])])
            files[path.name] = sha(path)
            rows.extend(batch)
    expected = {(tid, seed) for tid in config['evaluation']['tasks'] for seed in range(3)}
    assert len(rows) == 600 and {(row['task_id'], row['seed']) for row in rows} == expected
    atomic_json(output / 'public_seal.json', dict(records=600, files=files, protocol_sha256=protocol_sha, time=time.time()))
    return rows


def judge(assets, output, rows, workers):
    from dataset import load_rows, tests_from_row
    verify_assets(assets, prefix=['judge/'])
    official = load_rows(assets / 'judge/hidden.json')
    inputs = {(row['task_id'], seed): row for seed in range(3) for row in lines(assets / f'public/seed{seed}.jsonl')}
    cache = Cache(output / 'judge_cache')
    sealed = read(output / 'public_seal.json')
    assert sealed['records'] == 600

    def one(row):
        item = official[row['task_id']]
        tests = tests_from_row(item, 'hidden')
        timeout = max(3., min(float(item.get('test_time_limit') or 1) * 2, 15.))
        assert timeout == inputs[(row['task_id'], row['seed'])]['timeout']
        key = dict(executor_sha256=sha(PACKAGE / 'src/cure_wm.py'), code=row['final_code'],
                   tests=[asdict(test) for test in tests], timeout=timeout, sandbox='required')
        value, hit, identity = cache.get('hidden', key, lambda: asdict(execute(row['final_code'], tests, timeout=timeout)))
        outcome = execution_from_dict(value)
        return dict(row, final_hidden=value, hidden_allpass=outcome.all_pass,
                    hidden_execution_cache=identity, hidden_execution_cache_hit=hit)

    completed = []
    with (output / 'results.jsonl').open('x') as stream, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            atomic_json(output / 'judge_status.json', dict(rows=len(completed), total=600, time=time.time()))
    assert len(completed) == 600
    for filename, expected in sealed['files'].items():
        assert sha(output / 'public' / filename) == expected
    per_seed = [sum(row['hidden_allpass'] for row in completed if row['seed'] == seed) for seed in range(3)]
    reference = {(row['task_id'], row['seed']): row for row in lines(assets / 'reference/decisions.jsonl')}
    matches = sum(hashlib.sha256(row['final_code'].encode()).hexdigest() == reference[(row['task_id'], row['seed'])]['final_code_sha256']
                  and [step['selected_index'] for step in row['trace']] == reference[(row['task_id'], row['seed'])]['choices'] for row in rows)
    summary = dict(records=600, independent_tasks=200, successes=sum(per_seed), rate=sum(per_seed) / 600,
        per_seed_successes=per_seed, reference_decisions_matched=matches,
        reference_counts_matched=per_seed == experiment()['reference']['per_seed_successes'],
        hidden_results_recomputed=True, inputs=read(output / 'protocol.json')['inputs'],
        recorded_public_inputs=read(output / 'protocol.json')['inputs'] in RECORDED,
        reference_per_seed_successes=experiment()['reference']['per_seed_successes'],
        protocol_sha256=sha(output / 'protocol.json'), results_sha256=sha(output / 'results.jsonl'),
        all_decisions_sealed_before_hidden_judge=True, time=time.time())
    if summary['inputs'] == 'recorded-forced':
        summary['decision_agreement'] = agreement(rows)
    atomic_json(output / 'summary.json', summary)
    atomic_json(output / 'complete.json', dict(status='COMPLETE', summary_sha256=sha(output / 'summary.json')))
    print(json.dumps(summary), flush=True)


def stop_workers(running):
    """Reap owned workers and their descendants on failure or interruption."""
    for process, *_ in running.values():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process, *_ in running.values():
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process, log, *_ in running.values():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        log.close()


def run_workers(args, assets, output):
    devices = args.gpus.split(',')
    if not devices or len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        raise ValueError('gpus must list unique nonnegative GPU IDs')
    jobs = [(seed, shard) for shard in range(8) for seed in range(3)]
    running = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Evaluation interrupted')

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while jobs or running:
            for device, (process, log, seed, shard) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                if code:
                    raise RuntimeError(f'Worker seed={seed} shard={shard} failed; inspect logs')
                del running[device]
            for device in devices:
                if device in running or not jobs:
                    continue
                seed, shard = jobs.pop(0)
                log = (output / 'logs' / f'seed{seed}.shard{shard}.log').open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, TOKENIZERS_PARALLELISM='false',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                cmd = [sys.executable, '-u', str(PACKAGE / 'reproduce.py'), '_worker', '--assets', str(assets),
                       '--output', str(output), '--seed', str(seed), '--shard', str(shard)]
                try:
                    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                running[device] = (process, log, seed, shard)
                atomic_json(output / 'logs' / f'seed{seed}.shard{shard}.handle.json', dict(pid=process.pid, gpu=device, args=cmd))
            atomic_json(output / 'status.json', dict(phase='public', queued=len(jobs), running={d:p.pid for d,(p,*_) in running.items()}))
            if jobs or running:
                time.sleep(5)
    finally:
        # Finish bounded cleanup even if the caller sends another stop signal.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        try:
            stop_workers(running)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def run(args):
    assets, output = Path(args.assets).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else assets / 'checkpoints/incumbent'
    proof = verify_assets(assets, prefix=['public/', 'public_cache/', 'models/world/', 'reference/'])
    _, identity = checkpoint_manifest(checkpoint, formal=False)
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    atomic_json(output / 'protocol.json', dict(checkpoint=str(checkpoint), world_identity=identity,
        inputs=args.inputs, experiment=experiment(), asset_manifest_sha256=proof['manifest_sha256'],
        source_sha256={str(p.relative_to(PACKAGE)): sha(p) for p in (PACKAGE / 'src').glob('*.py')},
        started_at=time.time()))
    run_workers(args, assets, output)
    rows = seal_public(assets, output)
    atomic_json(output / 'status.json', dict(phase='hidden_judge'))
    judge(assets, output, rows, args.judge_workers)
