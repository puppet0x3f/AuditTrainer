from pathlib import Path
from collections import Counter

import pytest
import torch

from audittrainer.refinement_curriculum_rl import natural_seed
from audittrainer.refinement_coverage import paired_rows
from audittrainer.refinement_multiround import round_config
from audittrainer.refinement_pr1 import conditional_token_terms, load_config, mode_surrogate


def test_behavior_denominator_remains_distinct_from_fixed_kl_reference():
    behavior = torch.tensor([[2., -1., 0.], [-1., 2., 0.]])
    reference = -behavior
    actor = behavior.clone().requires_grad_()
    labels = [0, 1]
    old = behavior.log_softmax(-1)[torch.arange(2), labels]
    objective, kl, _ = conditional_token_terms(actor, reference, labels, old, .4, .2)
    assert objective.item() == pytest.approx(.8)  # ratio is one after refreshing behavior
    assert kl.item() > 0  # drift from fixed SFT is still penalized
    gradient = torch.autograd.grad(objective, actor, retain_graph=True)[0]
    assert gradient[0, 0] > 0 and gradient[1, 1] > 0
    mode_old = torch.tensor([.8, .2]).log()
    mode_actor = mode_old.clone().requires_grad_()
    surrogate = mode_surrogate(mode_actor, mode_old, [-.1, .4], [0., .6, .6], 'B4', .2)
    mode_gradient = torch.autograd.grad(surrogate, mode_actor)[0]
    assert mode_gradient[0] < 0 < mode_gradient[1]


def test_rounds_refresh_each_methods_behavior_keep_reference_and_pair_seeds():
    config, _ = load_config('configs/refinement_multiround.yaml')
    all_seeds = set()
    for index in range(5):
        b3, b4 = [round_config(config, method, index) for method in ('B3', 'B4')]
        assert b3['reference_adapter'] == b4['reference_adapter'] == config['initial_adapter']
        assert b3['seed'] == b4['seed']
        assert b3['retention_offset'] == b4['retention_offset'] == index*30
        if index:
            assert b3['initial_adapter'] != b4['initial_adapter']
            assert Path(b3['optimizer_state']).parent == Path(b3['initial_adapter']).parent
        else:
            assert b3['initial_adapter'] == b4['initial_adapter'] == config['initial_adapter']
            assert 'optimizer_state' not in b3
        # Include gate, conditional-generation, continuation and independent-baseline seeds.
        seeds = {b3['seed']+offset+i for offset, count in ((100000,3),(200000,9),(300000,6),(900000,3)) for i in range(count)}
        assert all_seeds.isdisjoint(seeds)
        all_seeds.update(seeds)
    evaluation_seeds = {natural_seed(config, i, repeat) for i, (_, count) in enumerate(sorted(config['evaluation']['natural_repetitions'].items())) for repeat in range(count)}
    assert len(evaluation_seeds) == 60
    assert all_seeds.isdisjoint(evaluation_seeds)


def test_coverage_rotation_preserves_budget_and_isolates_round_manifests():
    config, _ = load_config('configs/refinement_coverage.yaml')
    slots = Counter(rid for batch in config['round_root_ids'] for rid in batch)
    assert len(slots) == 12 and sum(slots.values()) == 15
    assert set(slots) == set(config['evaluation']['natural_repetitions'])
    assert all(slots[rid] == 2 for rid in config['round_root_ids'][0])
    assert config['candidate_sampling']['continuations_per_candidate'] == 2
    manifests = set()
    for index, selected in enumerate(config['round_root_ids']):
        current = round_config(config, 'B3', index)
        assert len(set(selected)) == current['root_limit'] == 3
        assert current['reference_adapter'] == config['initial_adapter']
        assert Path(current['root_manifest']).parent == Path(current['output_dir'])
        manifests.add(current['root_manifest'])
    assert len(manifests) == 5 and config['root_manifest'] not in manifests


def test_coverage_paired_outcomes_keep_task_subsets_and_failures():
    left = [{'root_id': r, 'seed': s, 'success': y} for r, s, y in [('old', 1, 0), ('new', 2, 1), ('new', 3, 0)]]
    right = [{'root_id': r, 'seed': s, 'success': y} for r, s, y in [('new', 3, 0), ('old', 1, 1), ('new', 2, 0)]]
    assert paired_rows(left, right, {'old', 'new'}) == {
        'runs': 3, 'left_passes': 1, 'right_passes': 1, 'wins': 1, 'losses': 1, 'ties': 1}
    assert paired_rows(left, right, {'new'})['losses'] == 1
    with pytest.raises(ValueError, match='Missing paired'):
        paired_rows(left, right[:-1], {'new'})
