from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from audittrainer.a6 import ReassignmentPolicy, reassignment_plan, review_reassignment, revise_slots
from audittrainer.dataset import read_jsonl
from audittrainer.schema import TraceEvent, parse_action
from audittrainer.training_data import execute
from audittrainer.training_tasks import definition
from audittrainer.verifier import canonical


ROOT = Path(__file__).resolve().parents[1]


def test_reassignment_runs_and_review_requires_its_responsibility_boundary(tmp_path):
    d, private, reference = definition('chunks', 3)
    framework = yaml.safe_load((ROOT / 'configs/frozen_mechanism.yaml').read_text())
    policy = ReassignmentPolicy(d, reference, 'helper')
    directory = tmp_path / 'run'
    result = execute(d, private, policy, directory, framework, {'trajectory_source': 'reviewed_reference'})
    assert result['functional_status'] == 'PASS'
    assert all(e['status'] == 'PASS' for e in result['offline_evidence'])
    events = [TraceEvent.model_validate(e) for e in read_jsonl(directory / 'trace.jsonl')]
    root = next(e for e in events if e.node_id == 'N0' and e.parsed_action and e.parsed_action.action == 'write_code')
    assert review_reassignment(root, d, reference, 'helper')[0] == 'SUPPORTED'
    plan = root.observation_before['current_plan']
    assert plan['obligation_map']['AGGREGATE'] == ['self']
    assert [c['local_key'] for c in plan['children']] == ['prepare']
    assert root.parsed_action.args.declared_dependencies[0].alias == '_prepare_child'
    # The same implementation is not valid responsibility evidence under a plan
    # that delegates aggregate away from the parent.
    changed = deepcopy(root)
    changed.observation_before['current_plan']['obligation_map']['AGGREGATE'] = ['child:prepare']
    assert review_reassignment(changed, d, reference, 'helper')[0] == 'UNKNOWN'
    changed = deepcopy(root)
    action = changed.parsed_action.model_dump(mode='json')
    action['args']['source'] = action['args']['source'].replace('result = _aggregate(prepared)', 'result = prepared')
    changed = changed.model_copy(update={'parsed_action': parse_action(canonical(action))})
    assert review_reassignment(changed, d, reference, 'helper')[0] == 'UNKNOWN'
    # An unrelated enlarged incoming contract must not inherit the strong label.
    changed = deepcopy(root)
    changed.observation_before['incoming_contract']['owns'].append('unreviewed responsibility')
    assert review_reassignment(changed, d, reference, 'helper')[0] == 'UNKNOWN'


def test_reassignment_preserves_root_obligations_and_original_definition():
    d, _, _ = definition('ledger', 2)
    before = deepcopy(d)
    plan = reassignment_plan(d['contract'], d)
    assert d == before
    assert set(plan['obligation_map']) == {o['id'] for o in d['contract']['ensures']}
    assert plan['obligation_map']['PREPARE'] == ['child:prepare']
    assert plan['obligation_map']['AGGREGATE'] == ['self']
    assert plan['children'][0]['contract'] == d['local_contracts']['prepare']
    assert 'aggregate' in plan['retained_responsibilities']


def test_only_alternate_parent_write_slots_are_replaced():
    def row(identifier, root, action='write_code', bucket='D_COMPOSE', node='N0'):
        return {'sample_id': identifier, 'root_id': root, 'bucket': bucket,
                'target_action': {'action': action}, 'meta': {'node_id': node}}
    base = [row('a', 'r1'), row('b', 'r1', 'consume'), row('c', 'r1', 'plan', 'D_PLAN'),
            row('d', 'r2'), row('e', 'r2', bucket='D_IMPL', node='N0_p1_prepare')]
    additions = [row('new1', 'r1'), row('new2', 'r2')]
    order = [0, 1, 2, 3, 4, 0, 0, 3]
    slots = [{'index': i, 'sample_id': base[i]['sample_id'], 'root_id': base[i]['root_id'],
              'optimizer_step': 1, 'phase': 'main', 'bucket': base[i]['bucket']} for i in order]
    frozen = deepcopy((base, additions, slots))
    pool, changed = revise_slots(base, additions, slots)
    assert (base, additions, slots) == frozen
    assert [s['a6_replaced'] for s in changed] == [True, False, False, True, False, False, True, False]
    for old, new in zip(slots, changed):
        assert new['root_id'] == old['root_id']
        assert new['optimizer_step'] == old['optimizer_step'] and new['bucket'] == old['bucket']
        selected = pool[new['index']]
        assert selected['sample_id'] == new['sample_id']
        if new['a6_replaced']:
            assert selected['root_id'] == old['root_id']
        else:
            assert selected == base[old['index']]
    with pytest.raises(ValueError, match='Every baseline'):
        revise_slots(base, additions[:1], slots)
    slots[0]['sample_id'] = 'wrong event'
    with pytest.raises(ValueError, match='binding'):
        revise_slots(base, additions, slots)
