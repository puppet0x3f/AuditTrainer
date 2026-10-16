"""Checks for real local-oracle coverage and repeatable calibration contrasts."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from audittrainer.refinement_curriculum import plans, repeated_contrasts, transfer_definition, partition_values
from audittrainer.refinement_policy import atomic_plan
from audittrainer.refinement_pr1 import load_config
from audittrainer.refinement_rollout import runtime_for
from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.schema import PolicyOutput, TokenUsage
from audittrainer.verifier import canonical
from audittrainer.training_tasks import oracle_a, oracle_b


@pytest.mark.parametrize('variant', ['pipeline', 'fused', 'parent_aggregate'])
def test_legal_decompositions_have_exact_oracles_and_atomic_leaves(tmp_path, variant):
    config, framework = load_config('configs/refinement_curriculum.yaml')
    _, old_framework = load_config('configs/refinement_pr1.yaml')
    assert framework['runtime']['max_depth'] == 1
    assert old_framework['runtime']['max_depth'] == 2
    rows = load_root_manifest('data/manifests/roots.jsonl')
    tests = {r['root_id']: r for r in read_jsonl('data/manifests/tests.jsonl')}
    for rid in config['root_ids']:
        row = rows[rid]
        with open(row['fixture_path']) as stream:
            definition = json.load(stream)
        before = deepcopy(definition)
        action = plans(definition)[variant]
        assert definition == before
        output = PolicyOutput(raw_text=canonical(action), token_usage=TokenUsage(), model_identity={'backend': 'test'})
        observations = []
        def act(observation):
            observations.append(observation)
            if observation['node_id'] == 'N0':
                return output
            return PolicyOutput(raw_text=canonical(atomic_plan(observation['incoming_contract'])),
                                token_usage=TokenUsage(), model_identity={'backend': 'test'})
        runtime, _ = runtime_for(row, tests, framework, SimpleNamespace(act=act), tmp_path/rid/variant)
        assert runtime.step().host_result.code == 'PLAN_REGISTERED'
        assert runtime.step().host_result.code == 'CHILD_ENTERED'
        assert runtime.step().host_result.code == 'PLAN_REGISTERED'
        obs = observations[-1]
        assert obs['node_id'] != 'N0'
        assert obs['allowed_plan_modes'] == ['ATOMIC']
        for child in runtime.contexts.values():
            if child.node.node_id != 'N0':
                suite_ids = runtime.verifier.available_suites(rid, child.node.incoming_contract)
                assert suite_ids and all(s in runtime.verifier.suites for s in suite_ids)
        assert all(s.visibility == 'VISIBLE' for s in runtime.verifier.suites.values())


def test_contrast_requires_same_direction_in_both_fixed_seed_blocks():
    values = {'pipeline': {'returns': [1, 1, 1, 1]},
              'fused': {'returns': [0, 0, 1, 1]},
              'parent_aggregate': {'returns': [0, 0, 0, 0]}}
    pairs = {(r['a'], r['b']) for r in repeated_contrasts(values)}
    assert pairs == {('pipeline', 'parent_aggregate')}
    values['fused']['returns'] = [1, 1, 0, 0]
    assert ('pipeline', 'fused') not in {(r['a'], r['b']) for r in repeated_contrasts(values)}


def test_duplicate_execution_noise_is_not_a_distinct_partition_contrast():
    values = partition_values([('fused', 0), ('fused', 1)])
    assert values == {'fused': 0.5}
    assert len(set(values.values())) == 1
    values = partition_values([('fused', 0), ('fused', 1), ('pipeline', 0.5)])
    assert len(set(values.values())) == 1
    values = partition_values([('fused', 0), ('fused', 0), ('pipeline', 1)])
    assert values == {'fused': 0, 'pipeline': 1}


@pytest.mark.parametrize('first,second,items,expected', [
    ('inventory', 'ledger', [['a', 1, 8], ['b', 10, 2], ['a', 2, 0]], {'accounts': ['a'], 'debt': 5}),
    ('ledger', 'inventory', [['z', 7], ['a', 4], ['z', -2], ['b', -4]], {'order': ['z', 'a'], 'units': [5, 4]}),
    ('graph', 'inventory', [['a', 'b'], ['a', 'c'], ['a', 'd'], ['b', 'c'], ['a', 'b'], ['a', 'a']], {'order': ['a'], 'units': [3]}),
    ('inventory', 'graph', [['a', 5, 1], ['b', 1, 4]], {'nodes': ['a'], 'degree_sum': 4}),
    ('ledger', 'graph', [['a', 7], ['a', -3], ['b', 1]], {'nodes': ['a'], 'degree_sum': 4}),
    ('intervals', 'matrix', [[1, 5], [3, 7]], {'indices': [2, 5], 'total': 9}),
    ('calendar', 'matrix', ['00:03', '00:04', '00:03', '00:06'], {'indices': [0, 2], 'total': 9}),
])
def test_component_transfer_against_hand_answers_and_reference(first, second, items, expected):
    original = deepcopy(items)
    d, private, reference = transfer_definition(first, second, 3)
    assert oracle_b(second, oracle_a(first, items), 3) == expected
    namespace = {}
    exec(reference['prepare']+'\n'+reference['aggregate'], namespace)
    assert namespace['aggregate'](namespace['prepare'](items)) == expected
    assert items == original
    assert d['task']['split'] == 'TRAIN'
    assert all(s['visibility'] == 'VISIBLE' for s in d['suites'])
    assert all(s['visibility'] == 'PRIVATE' for s in private['suites'])
