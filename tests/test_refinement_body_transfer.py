from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.refinement_body_transfer import at_state, reviewed_plans, training_order
from audittrainer.refinement_curriculum_rl import plan_key
from audittrainer.refinement_responsibility_course import allocation, task_pack
from audittrainer.schema import PolicyOutput, TokenUsage
from audittrainer.verifier import canonical


def test_training_schedule_excludes_held_out_states_and_balances_exposure():
    states = [{'split': 'TRAIN', 'target_action': {'args': {'mode': 'SPLIT'}}} for _ in range(24)]
    assert Counter(training_order(states, 480, 42)) == {i: 20 for i in range(24)}
    states[-1]['split'] = 'VALIDATION'
    with pytest.raises(ValueError, match='Only TRAIN'):
        training_order(states, 480, 42)


def test_reviewed_joint_allocation_includes_retained_parent_responsibility():
    definition, _, _ = task_pack('paths', 'paths', 3, 'joint_obligation')
    obs = {'incoming_contract': definition['contract']}
    plans = reviewed_plans(obs, definition)
    partial = allocation(definition['contract'], definition, 'parent_aggregate')
    assert plan_key(partial, definition) not in plans
    complete = deepcopy(partial)
    complete['args']['obligation_map']['RESULT'].append('self')
    assert plans[plan_key(complete, definition)] == 'parent_aggregate'


def test_exported_root_and_internal_states_recreate_without_future_artifacts(tmp_path):
    request_path = Path('runs/refinement/body_transfer/request.json')
    if not request_path.exists():
        pytest.skip('Run the finite transfer preparation first')
    request = json.loads(request_path.read_text())
    states = read_jsonl(Path(request['study']['data_dir'])/'states.jsonl')
    rows = load_root_manifest(request['config']['root_manifest'])
    tests = {r['root_id']: r for r in read_jsonl(request['config']['test_manifest'])}
    train_families = {s['family_id'] for s in states if s['split'] == 'TRAIN'}
    validation_families = {s['family_id'] for s in states if s['split'] == 'VALIDATION'}
    assert not train_families & validation_families
    # Check every exported state, including combined obligations and actual child entry.
    for index, state in enumerate(states):
        row = rows[state['root_id']]
        definition = json.loads(Path(row['fixture_path']).read_text())
        policy = SimpleNamespace(act=lambda obs, s=state: PolicyOutput(
            raw_text=canonical(s['target_action']), token_usage=TokenUsage(), model_identity={'backend': 'test'}))
        runtime = at_state(state, row, tests, request['framework'], definition, tmp_path/str(index), policy)
        assert all(ctx.artifact is None for ctx in runtime.contexts.values())
        assert runtime.step().host_result.code == 'PLAN_REGISTERED'
