"""Fixed internal-state ATOMIC/SPLIT continuations scored at the root."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path('runs/refinement/internal_probe')
NODE = 'N0_p1_prepare'
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'run':
    from .gpu_resource import acquire
    _reservation = acquire(ROOT / 'gpu_resources.json', 'collect')

from .dataset import read_jsonl, file_hash
from .evaluate import write_json
from .refinement_gate_refresh import context
from .refinement_generalization import guarded_json
from .refinement_separate_gate import fingerprints
from .refinement_rollout import run_case
from .refinement_execution_repair import local_checks
from .schema import PolicyOutput, TokenUsage
from .verifier import digest


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'request.json').exists():
        return
    source = Path('runs/refinement/gate_refresh')
    prior = json.loads((source / 'request.json').read_text())
    selected = []
    for rid, rep in [('train_execution_polynomial_matrix_05', 2), ('train_executor_check_paths_matrix_07', 1)]:
        path = source / 'natural/AFTER' / rid / f'repeat_{rep:02d}/runtime/trace.jsonl'
        events = read_jsonl(path)
        root = next(e for e in events if e['event_kind'] == 'POLICY' and e['node_id'] == 'N0')
        internal = next(e for e in events if e['event_kind'] == 'POLICY' and e['node_id'] == NODE)
        assert root['host_result']['code'] == 'PLAN_REGISTERED'
        assert internal['observation_before']['phase'] == 'NEW'
        assert internal['observation_before']['allowed_plan_modes'] == ['ATOMIC', 'SPLIT']
        state = ROOT / rid / 'state.json'
        state.parent.mkdir(parents=True, exist_ok=True)
        write_json(state, {'root_event': root, 'observation': internal['observation_before']})
        selected.append({'root_id': rid, 'source_trace': str(path), 'state_path': str(state)})
    adapters = {'gate': str(source / 'training/final_adapter'), 'body': prior['body_adapter'], 'executor': prior['executor_adapter']}
    guarded_json(ROOT / 'request.json', {'selection': selected, 'framework': prior['framework'], 'config': prior['config'],
        'adapters': adapters, 'hashes': {k: file_hash(Path(p) / 'adapter_model.safetensors') for k, p in adapters.items()},
        'plan_seed': 401000713, 'execution_seed': 402000713, 'repetitions': 4, 'cases': 16,
        'scope': 'Two failure-selected development states; path mod7 is now diagnostic development, not a holdout.',
        'intervention': 'Replay exact registered root output and usage. At the first prepare NEW decision inject ATOMIC or fresh frozen MULTI_BODY SPLIT; assert full observation equality. Subsequent retries and all other decisions follow frozen policies.',
        'evaluation': 'Root private success is primary. Local prepare contract and actual parent use are secondary. Keep every refusal, retry and failure; no parameter or prompt updates.'})
    guarded_json(ROOT / 'gpu_resources.json', {'devices': ['0', '1', '2'], 'min_free_mib': {'collect': 16000}})


class FixedInternal:
    def __init__(self, base, state, gate, candidate, node=NODE):
        self.base = base
        self.cost = base.cost
        self.identity = {**base.identity, 'intervention': 'fixed_root_and_initial_internal_mode'}
        self.state, self.gate, self.candidate = state, gate, candidate
        self.node = node
        self.root_sent = False
        self.internal_sent = False

    def act(self, obs):
        if obs['node_id'] == 'N0' and not self.root_sent:
            self.root_sent = True
            e = self.state['root_event']
            assert digest(obs) == digest(e['observation_before'])
            return PolicyOutput(raw_text=e['raw_output'], token_usage=TokenUsage(**e['usage']),
                model_identity=e['model_identity'], model_input=e['model_input'],
                output_token_ids=e['output_token_ids'], generation_truncated=e['generation_truncated'])
        if obs['node_id'] == self.node and not self.internal_sent:
            assert digest(obs) == digest(self.state['observation']) == self.gate['observation_sha256']
            self.internal_sent = True
            return self.base.as_output(self.gate, self.candidate)
        return self.base.act(obs)


def run(index):
    from transformers import set_seed
    r = json.loads((ROOT / 'request.json').read_text())
    _, policy, executor, rows, tests = context('AFTER')
    before = fingerprints(policy)
    selected = r['selection'][index]
    rid = selected['root_id']
    base = ROOT / rid
    state = json.loads(Path(selected['state_path']).read_text())
    gate = policy.score_modes(state['observation'])
    write_json(base / 'gate.json', gate)
    cases = []
    for rep in range(r['repetitions']):
        set_seed(r['plan_seed'] + index * 100 + rep)
        candidate = policy.sample_split(gate)
        write_json(base / f'candidate_{rep:02d}.json', candidate)
        for arm in ['ATOMIC', 'SPLIT']:
            fixed = FixedInternal(policy, state, gate, policy.atomic(gate) if arm == 'ATOMIC' else candidate)
            c = run_case(rows[rid], tests, r['framework'], r['config'], fixed, executor,
                base / arm / f'repeat_{rep:02d}', r['execution_seed'] + index * 100 + rep)
            assert c['status'] == 'COMPLETE' and fixed.root_sent and fixed.internal_sent
            checks = local_checks(c, rows[rid], tests, r['framework'])
            write_json(Path(c['run_dir']) / 'execution_checks.json', checks)
            events = read_jsonl(Path(c['run_dir']) / 'runtime/trace.jsonl')
            decision = next(e for e in events if e['event_kind'] == 'POLICY' and e['node_id'] == NODE)
            assert decision['model_input']['refinement']['selected_mode'] == arm
            cases.append({'arm': arm, 'rep': rep, 'case': c, 'checks': checks,
                'initial_internal_registered': decision['host_result']['code'] == 'PLAN_REGISTERED'})
    assert fingerprints(policy) == before
    write_json(base / 'result.json', {'status': 'COMPLETE', 'root_id': rid, 'frozen': before, 'cases': cases})


def report():
    r = json.loads((ROOT / 'request.json').read_text())
    results = [json.loads((ROOT / x['root_id'] / 'result.json').read_text()) for x in r['selection']]
    assert results[0]['frozen'] == results[1]['frozen']
    assert all(file_hash(Path(p) / 'adapter_model.safetensors') == r['hashes'][k] for k, p in r['adapters'].items())
    summary = []
    for result in results:
        item = {'root_id': result['root_id']}
        for arm in ['ATOMIC', 'SPLIT']:
            cs = [c for c in result['cases'] if c['arm'] == arm]
            assert len(cs) == 4
            item[arm] = {'root_success': sum(c['case']['final_scores']['private']['status'] == 'PASS' for c in cs),
                'local_private_success': sum(any(v['visibility'] == 'PRIVATE' and v['status'] == 'PASS' for v in c['checks']['local']) for c in cs),
                'parent_called': sum(c['checks']['parent_called'] for c in cs),
                'initial_registered': sum(c['initial_internal_registered'] for c in cs), 'runs': 4}
        summary.append(item)
    out = {'status': 'COMPLETE', 'cases': 16, 'updates': 0, 'frozen': results[0]['frozen'], 'summary': summary, 'results': results}
    write_json(ROOT / 'result.json', out)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'prepare': prepare()
    elif args.command == 'run': run(args.index)
    else: report()
