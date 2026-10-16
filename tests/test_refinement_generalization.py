from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.evaluate import load_task
from audittrainer.refinement_curriculum import plans
from audittrainer.refinement_generalization import ARMS, arm_config, budget, evaluation_seed
from audittrainer.refinement_heldout_tasks import INPUTS, TRANSFERS, definition
from audittrainer.refinement_multiround import round_config
from audittrainer.refinement_pr1 import load_config, load_training_groups
from audittrainer.refinement_rollout import runtime_for, split_candidate_count
from audittrainer.schema import PolicyOutput, TokenUsage
from audittrainer.training_tasks import oracle_a, oracle_b
from audittrainer.verifier import canonical


def configuration():
    return yaml.safe_load(Path('configs/refinement_generalization.yaml').read_text())


def test_three_independent_seed_runs_start_common_sft_with_equal_budgets():
    config = configuration(); seeds = config['training_seeds']
    assert len(set(seeds)) == 3
    initial = set(); outputs = set(); rollout_seeds = set()
    for seed in seeds:
        for arm in ARMS:
            c = arm_config(config, arm, seed); b = budget(c)
            assert b['total_continuations'] == 240 and b['optimizer_steps'] == 40
            covered = {r for batch in c['round_root_ids'] for r in batch}
            assert len(covered) == (3 if arm == 'original' else 12)
            first = round_config(c, 'B3', 0)
            initial.add(first['initial_adapter']); assert 'optimizer_state' not in first
            assert c['update']['mode_reward_weight'] == 1
            for i in range(4):
                r = round_config(c, 'B3', i); outputs.add(r['output_dir'])
                assert r['reference_adapter'] == first['initial_adapter']
                assert r['retention_offset'] == i*30
                if i:
                    assert Path(r['optimizer_state']).parent == Path(r['initial_adapter']).parent
                for j, rid in enumerate(sorted(c['round_root_ids'][i])):
                    rollout_seeds.update(r['seed']+300000+j*8+k for k in range(r['candidate_sampling']['continuations_per_candidate']))
                    rollout_seeds.update(r['seed']+900000+j*8+k for k in range(r['independent_baseline_repetitions']))
    assert initial == {'runs/refinement/PR1C_support/adaptation/final_adapter'}
    assert len(outputs) == 24
    eval_seeds = {evaluation_seed(config, i, j) for i in range(12) for j in range(5)}
    assert len(eval_seeds) == 60 and rollout_seeds.isdisjoint(eval_seeds)


def test_breadth_schedule_matches_repeat_total_without_changing_update_objective():
    config = configuration(); original, optimized = [arm_config(config, a, config['training_seeds'][0]) for a in ARMS]
    assert original['update'] == optimized['update']
    assert original['reward'] == optimized['reward']
    assert budget(original)['candidate_continuations'] == 228
    assert budget(optimized)['candidate_continuations'] == 192
    totals = []
    for i, roots in enumerate(original['round_root_ids']):
        r = round_config(original, 'B3', i)
        totals.append(sum((1+split_candidate_count(r, rid))*2+1 for rid in roots))
    assert totals == [59, 61, 59, 61]


@pytest.mark.parametrize('first,second,p,items,expected', [
    ('words', 'paths', 3, [' Hare ', 'HARP', 'hay', 'hare'], {'prefixes': ['h/a/r', 'h/a/y'], 'counts': [3, 1]}),
    ('matrix', 'chunks', 3, [[1, 2, 3, 4], [5, 6, 7, 8]], {'chunks': [[6, 8, 10], [12]], 'length': 4}),
    ('paths', 'chunks', 3, ['a/b', 'c', '/', 'd/e/f'], {'chunks': [[['a', 'b'], ['c'], []], [['d', 'e', 'f']]], 'length': 4}),
    ('chunks', 'words', 1, [['A', 2], [' ', 1], ['a', 1]], {'words': [' ', 'A', 'a'], 'count': 3}),
    ('chunks', 'paths', 3, [['/', 2], ['.', 1]], {'prefixes': ['.', '/'], 'counts': [1, 2]}),
    ('graph', 'words', 3, [['Elm', 'oak'], ['oak', 'x'], ['Elm', 'Elm']], {'words': ['Elm', 'oak'], 'count': 2}),
    ('intervals', 'calendar', 3, [[-29, -28], [28, 30]], {'breaks': [[-29, 28]], 'max_gap': 57}),
    ('calendar', 'intervals', 3, ['01:02', '01:01', '01:00', '02:00'], {'segments': [[60, 63]], 'covered': 3}),
])
def test_new_composition_oracles_against_hand_answers(first, second, p, items, expected):
    before = deepcopy(items); _, _, reference = definition(first, second, p)
    ns = {}; exec(reference['prepare']+'\n'+reference['aggregate'], ns)
    assert ns['aggregate'](ns['prepare'](items)) == oracle_b(second, oracle_a(first, items), p) == expected
    assert items == before


def test_frozen_validation_is_disjoint_and_private_cases_are_absent_online(tmp_path):
    config = configuration(); directory = Path(config['heldout_dir'])
    rows = load_root_manifest(directory/'roots.jsonl')
    tests = {r['root_id']: r for r in read_jsonl(directory/'tests.jsonl')}
    prior = {r['family_id'] for path in Path('data').rglob('roots.jsonl') if directory not in path.parents for r in read_jsonl(path)}
    base, framework = load_config(config['base_config']); framework['allow_validation'] = True
    assert len(rows) == len(TRANSFERS) == 12
    for rid, row in rows.items():
        assert row['family_id'] not in prior and row['split'] == row['original_split'] == 'VALIDATION'
        d, task, _, online, full = load_task(row, tests, framework)
        assert len(online.suites) == 4 and len(full.suites) == 8
        assert all(s.visibility == 'VISIBLE' for s in online.suites.values())
        assert len({canonical(c['expected']) for c in d['suites'][0]['cases']}) > 1  # nonconstant public problem
        for name, action in plans(d).items():
            output = PolicyOutput(raw_text=canonical(action), token_usage=TokenUsage(), model_identity={'backend': 'test'})
            runtime, _ = runtime_for(row, tests, framework, SimpleNamespace(act=lambda obs: output), tmp_path/rid/name)
            assert runtime.step().host_result.code == 'PLAN_REGISTERED'
    base.update(root_manifest=str(directory/'roots.jsonl'), output_dir=str(tmp_path/'no_training'), root_limit=12)
    with pytest.raises(ValueError, match='Missing training states'):
        load_training_groups(base)
