"""A5 controls differ only at their assigned slots; isolation remains executable."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from audittrainer.a5_data import GROUPS, replacement
from audittrainer.dataset import read_jsonl
from audittrainer.train_sft import build_schedule, load_config, load_identity_and_tokenizer, load_training_data


ROOT = Path(__file__).resolve().parents[1]


def test_replacement_keeps_protocol_and_uses_real_same_root_events():
    root = 'train_x'
    ordinary = {(root, a): {'root_id': root, 'sample_id': 'functional:' + a} for a in ('plan', 'write_code')}
    for action in ('plan', 'write_code', 'consume', 'verify', 'finish', 'abstain'):
        row = {'root_id': root, 'sample_id': 'responsibility:' + action, 'target_action': {'action': action}}
        for d, i in GROUPS.values():
            actual = replacement(row, d, i, ordinary)
            changed = (action == 'plan' and not d) or (action == 'write_code' and not i)
            assert actual is (ordinary[(root, action)] if changed else row)


def test_fixed_schedule_rejects_wrong_root_and_step(tmp_path):
    rows = [{'sample_id': 'x', 'root_id': 'train_x'}]
    path = tmp_path / 'schedule.jsonl'
    slot = {'sample_id': 'x', 'root_id': 'train_x', 'index': 0, 'optimizer_step': 1}
    data = {'sampling_unit': 'root_then_action', 'schedule_path': str(path), 'matched_slot_budget': 2}
    train = {'epoch_limit': 1, 'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 2}
    for mutation in ({'root_id': 'dev_x'}, {'optimizer_step': 2}, {'sample_id': 'other'}):
        path.write_text(json.dumps({**slot, **mutation}) + '\n')
        with pytest.raises(ValueError, match='bind'):
            build_schedule(rows, data, train)
    path.write_text((json.dumps(slot) + '\n') * 2)
    assert build_schedule(rows, data, train)[1] == 1


def test_actual_factorial_slots_and_functional_opt_in():
    config = load_config(ROOT / 'configs/a5/F00.yaml')
    identity, tokenizer = load_identity_and_tokenizer(config)
    blocked = deepcopy(config)
    blocked['data']['allow_function_only'] = False
    with pytest.raises(ValueError, match='FUNCTION_ONLY'):
        load_training_data(blocked, identity, tokenizer)
    arms = {}
    for group in GROUPS:
        cfg = load_config(ROOT / f'configs/a5/{group}.yaml')
        rows, encoded, _ = load_training_data(cfg, identity, tokenizer)
        schedule, steps = build_schedule(rows, cfg['data'], cfg['training'])
        assert steps == 140 and len(schedule) == 2240
        assert len({s['root_id'] for s in schedule}) == 50
        assert len(rows) == len(encoded)
        arms[group] = [(slot, rows[slot['index']]) for slot in schedule]
    for entries in zip(*(arms[g] for g in GROUPS)):
        assert len({e[0]['root_id'] for e in entries}) == 1
        assert len({e[0]['source_sample_id'] for e in entries}) == 1
        action = entries[-1][1]['target_action']['action']
        f00, f10, f01, f11 = [e[1] for e in entries]
        if action != 'plan':
            assert f00 == f10 and f01 == f11
        if action != 'write_code':
            assert f00 == f01 and f10 == f11
        if action == 'write_code':
            assert f00['supervision_scope'] == 'FUNCTION_ONLY'
            assert f00['meta']['root_functional_status'] == 'PASS'
    pilot = load_config(ROOT / 'configs/a5/F11_pilot.yaml')
    rows = read_jsonl(pilot['data']['path'])
    schedule, steps = build_schedule(rows, pilot['data'], pilot['training'])
    assert len(schedule) == 80 and steps == 5


class ReferenceStudent:
    identity = {'backend': 'test_only'}

    def __init__(self, reference):
        self.reference = reference

    def act(self, obs):
        from audittrainer.a5_evaluate import atomic, fixed
        if obs['phase'] == 'NEW':
            return fixed('plan', atomic(obs['incoming_contract']))
        if obs['phase'] == 'RESOLVE_CHILD':
            returned = obs['returned_child']['result']
            a = returned['artifact']
            return fixed('consume', {'child_id': a['node_id'], 'artifact_version': a['version'],
                'artifact_hash': a['source_hash'], 'alias': '_' + a['public_symbol'],
                'evidence_ids': [returned['evidence']['evidence_id']], 'accept_unverified': False})
        name = obs['incoming_contract']['signature']['name']
        source = self.reference['a' if name == 'transform' else 'b']
        if name == 'solve':
            source = 'import json\n'
            deps = obs['approved_dependencies']
            for key, symbol in [('a', 'transform'), ('b', 'summarize')]:
                dep = next((v for v in deps if v['child_id'].endswith('_' + key)), None)
                if dep:
                    source += f"from nodes.{dep['child_id']} import {symbol} as _{symbol}\n"
                else:
                    source += self.reference[key].replace('def ' + symbol, 'def _' + symbol)
            source += "def solve(text):\n    items = json.loads(text)\n    prepared = _transform(items)\n    result = _summarize(prepared)\n    return json.dumps(result, separators=(',', ':'))\n"
        return fixed('write_code', {'source': source, 'contract_version': obs['incoming_contract']['version'],
                                    'declared_dependencies': obs['approved_dependencies']})


@pytest.mark.parametrize('condition', ['local_plan', 'transform', 'summarize', 'parent_both_children', 'parent_transform_child'])
def test_real_isolation_execution_and_legal_reassignment(tmp_path, condition):
    import yaml
    from audittrainer.a5_evaluate import run_case
    from audittrainer.mechanism_tasks import definitions
    d, private, ref = next(definitions())
    d['suites'] += private['suites']
    student = ReferenceStudent(ref)
    framework = yaml.safe_load((ROOT / 'configs/frozen_mechanism.yaml').read_text())
    result = run_case(d, ref, student, student, condition, tmp_path / condition, framework)
    assert result['final_scores']['private']['status'] == 'PASS'
    assert result['child_artifacts_frozen']
    if condition == 'local_plan':
        assert result['student_calls'] == 1 and result['executor_calls'] == 1
    else:
        assert result['executor_calls'] == 0
    if condition == 'parent_both_children':
        assert result['responsibility_review']['status'] == 'SUPPORTED'
        assert result['student_calls'] == 3
    elif condition == 'parent_transform_child':
        assert result['student_calls'] == 2


def test_isolation_host_rejects_child_repair():
    from types import SimpleNamespace
    from audittrainer.a5_evaluate import IsolationRuntime
    from audittrainer.runtime import ProtocolError
    with pytest.raises(ProtocolError, match='frozen'):
        IsolationRuntime._dispatch(None, SimpleNamespace(action='request_child_repair'))


def test_composition_review_rejects_a_helper_shadowing_the_selected_child(tmp_path):
    from types import SimpleNamespace
    from audittrainer.a5_evaluate import composition_review
    source = """import json
from nodes.N0_p1_a import transform as _a
from nodes.N0_p1_b import summarize as _b
def solve(text):
    items = json.loads(text)
    prepared = _a(items)
    result = _b(prepared)
    return json.dumps(result, separators=(',', ':'))
"""
    artifact = SimpleNamespace(module_path='root.py', dependency_bindings=[
        SimpleNamespace(alias='_a', child_id='N0_p1_a'), SimpleNamespace(alias='_b', child_id='N0_p1_b')])
    runtime = SimpleNamespace(run_dir=tmp_path, contexts={
        'N0': SimpleNamespace(artifact=artifact),
        'N0_p1_a': SimpleNamespace(artifact=SimpleNamespace(public_symbol='transform')),
        'N0_p1_b': SimpleNamespace(artifact=SimpleNamespace(public_symbol='summarize'))})
    path = tmp_path / 'root.py'
    path.write_text(source)
    assert composition_review(runtime, 'parent_both_children')['status'] == 'SUPPORTED'
    path.write_text(source + 'def _a(items):\n    return sorted(set(abs(x) for x in items))\n')
    assert composition_review(runtime, 'parent_both_children')['status'] == 'UNKNOWN'
