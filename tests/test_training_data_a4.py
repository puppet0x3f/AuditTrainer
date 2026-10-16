"""A4 verifies new-root isolation, executed supervision and single-factor contrasts."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from audittrainer.dataset import RunData, classify, load_root_manifest
from audittrainer.training_data import (ReferencePolicy, execute, import_roots, plan_args,
    responsibility_check, sources, transformed_source, validate_labels)
from audittrainer.training_tasks import FAMILIES, definition, oracle_a, oracle_b

ROOT = Path(__file__).resolve().parents[1]


def test_all_50_authored_reference_algorithms_agree_with_independent_oracles():
    # These are closed authored reference strings, never model-produced source.
    for family in FAMILIES:
        for p in range(1, 6):
            d, _, ref = definition(family, p)
            for style in ('plain', 'renamed', 'helper', 'alternative'):
                code = sources(ref, 'atomic', style)
                namespace = {}
                exec(code['solve'], namespace)
                for items in FAMILIES[family]['cases']:
                    before = deepcopy(items)
                    expected = oracle_b(family, oracle_a(family, items), p)
                    assert json.loads(namespace['solve'](json.dumps(items))) == expected
                    assert items == before
            assert d['task']['family_id'] == f'a4_{family}_v1'


@pytest.mark.parametrize('kind', ['development', 'family_crossing', 'duplicate'])
def test_import_cannot_relabel_development_or_duplicate_roots(tmp_path, kind):
    d, private, ref = definition('ledger', 1)
    manifest, tests = tmp_path / 'roots.jsonl', tmp_path / 'tests.jsonl'
    rows = []
    if kind == 'development':
        d['task']['split'] = 'DEV_MECHANISM'
    else:
        rows = [{'root_id': d['task']['root_id'] if kind == 'duplicate' else 'already_seen',
                 'family_id': d['task']['family_id'], 'split': 'DEV_ENGINEERING' if kind == 'family_crossing' else 'TRAIN'}]
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    tests.write_text('')
    with pytest.raises(ValueError):
        import_roots(tmp_path / 'tasks', manifest, tests, [{'definition': d, 'private': private, 'reference': ref,
                                                          'source': 'test', 'original_split': 'TRAIN'}])
    assert not (tmp_path / 'tasks').exists()


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    base = tmp_path_factory.mktemp('a4_tests')
    manifest, tests = base / 'roots.jsonl', base / 'tests.jsonl'
    manifest.touch(); tests.touch()
    d, private, ref = definition('ledger', 3)
    imported = import_roots(base / 'tasks', manifest, tests, [{'definition': d, 'private': private, 'reference': ref,
                                                           'source': 'test', 'original_split': 'TRAIN'}])
    config = {'root_manifest': str(manifest), 'test_manifest': str(tests), 'trace_dir': str(base / 'traces'),
              'report_dir': str(base / 'report'), 'selection': str(base / 'tasks/selection.json')}
    framework = yaml.safe_load((ROOT / 'configs/frozen_mechanism.yaml').read_text())
    for variant in ('atomic', 'pipeline', 'fused', 'move_algorithm', 'bypass_child', 'stale_artifact', 'expand_owns', 'omit_output', 'wrong_argument'):
        plan = variant if variant in ('atomic', 'pipeline', 'fused') else 'pipeline'
        transform = None if variant in ('atomic', 'pipeline', 'fused') else variant
        policy = ReferencePolicy(d, ref, plan, transform=transform)
        execute(d, private, policy, base / 'traces' / variant, framework,
                {'trajectory_source': 'controlled_transform' if transform else 'reviewed_reference',
                 'plan': plan, 'style': 'plain', 'transform': transform, 'model_identity': policy.identity})
    validate_labels(config)
    return base, d, ref, imported


def run_data(corpus, variant):
    base = corpus[0]
    return RunData(base / 'traces' / variant / 'trace.jsonl', load_root_manifest(base / 'roots.jsonl'))


def test_three_plans_start_at_same_parent_state_and_receive_supported_execution(corpus):
    base = corpus[0]
    observations = []
    for variant in ('atomic', 'pipeline', 'fused'):
        run = run_data(corpus, variant)
        observations.append(json.loads((base / 'traces' / variant / 'initial_observation.json').read_text()))
        event = next(e for e in run.events if e.parsed_action and e.parsed_action.action == 'plan' and e.node_id == 'N0')
        assert run.result['functional_status'] == 'PASS'
        assert classify(run, event, event.parsed_action.model_dump(mode='json'), False)['supervision_scope'] == 'RESPONSIBILITY_SUPPORTED'
    assert observations[0] == observations[1] == observations[2]
    assert observations[0]['budget_remaining']['child_slots'] == 3
    assert observations[0]['budget_remaining']['nodes'] == 6


@pytest.mark.parametrize('variant', ['move_algorithm', 'bypass_child'])
def test_functionally_passing_responsibility_violations_are_not_sft_positives(corpus, variant):
    run = run_data(corpus, variant)
    assert run.result['functional_status'] == 'PASS'
    root_evidence = [e for e in run.evidence.values() if e.check_kind == 'ROOT']
    assert {e.visibility for e in root_evidence} == {'VISIBLE', 'PRIVATE'}
    assert all(e.status == 'PASS' for e in root_evidence)
    event = next(e for e in run.events if e.node_id == 'N0' and e.host_result.code == 'SOURCE_SAVED')
    assert run.reviews[event.event_id].verdict == 'VIOLATION'
    if variant == 'bypass_child':
        visible = next(e for e in root_evidence if e.visibility == 'VISIBLE')
        assert all(u['called'] for u in visible.details['dependency_usage'])
    from audittrainer.dataset import Rejected
    with pytest.raises(Rejected):
        classify(run, event, event.parsed_action.model_dump(mode='json'), False)


def test_stale_consumption_uses_an_existing_old_artifact_after_real_repair(corpus):
    run = run_data(corpus, 'stale_artifact')
    event = next(e for e in run.events if e.parsed_action and e.parsed_action.action == 'consume')
    assert event.parsed_action.args.artifact_version == 1
    assert event.observation_before['returned_child']['result']['artifact']['version'] == 2
    assert (run.directory / 'artifacts/N0_p1_prepare/v1.py').exists()
    assert (run.directory / 'artifacts/N0_p1_prepare/v2.py').exists()
    assert event.host_result.status == 'REJECTED'
    assert run.reviews[event.event_id].verdict == 'VIOLATION'
    assert any(e.host_result.code == 'CODE_REPAIR_ACCEPTED' for e in run.events)


def test_changed_contract_is_unknown_and_functional_faults_are_distinct(corpus):
    run = run_data(corpus, 'expand_owns')
    assert any(e.status == 'UNKNOWN' for e in run.evidence.values())
    root_plan = next(e for e in run.events if e.parsed_action and e.parsed_action.action == 'plan' and e.node_id == 'N0')
    assert run.reviews[root_plan.event_id].verdict == 'VIOLATION'
    for variant in ('omit_output', 'wrong_argument'):
        assert run_data(corpus, variant).result['functional_status'] == 'FAIL'


def test_private_execution_labels_never_enter_predecision_observations(corpus):
    run = run_data(corpus, 'pipeline')
    private = {e.evidence_id for e in run.evidence.values() if e.visibility == 'PRIVATE'}
    assert private
    for event in run.events:
        obs = json.dumps(event.observation_before)
        assert not any(identifier in obs for identifier in private)
        assert all(e['visibility'] == 'VISIBLE' for e in event.observation_before['visible_evidence'])


def test_unknown_model_source_does_not_receive_a_review_from_its_request_label(corpus):
    run = run_data(corpus, 'atomic')
    event = next(e for e in run.events if e.parsed_action and e.parsed_action.action == 'write_code')
    from audittrainer.schema import parse_action
    action = event.parsed_action.model_dump(mode='json')
    action['args']['source'] = "def solve(text):\n    return '{}'\n"
    changed = event.model_copy(update={'parsed_action': parse_action(json.dumps(action))})
    verdict, _ = responsibility_check(changed, corpus[1], corpus[2], 'atomic', 'plain')
    assert verdict == 'UNKNOWN'
