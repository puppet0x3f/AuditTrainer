"""F6 checks binding, label strength, leakage and raw-trace preservation."""

import json
from pathlib import Path
import shutil

import pytest
from transformers import AutoTokenizer
import yaml

from audittrainer.dataset import (RunData, Rejected, classify, collect_fixture_suite, export_sft,
                                  file_hash, load_root_manifest, read_jsonl)
from audittrainer.prompts import prepare_prompt
from audittrainer.schema import Evidence, parse_action
from audittrainer.verifier import canonical, digest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/manifests/roots.jsonl'


@pytest.fixture(scope='module')
def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT / 'Qwen3-8B', local_files_only=True)


@pytest.fixture(scope='module')
def config():
    return yaml.safe_load((ROOT / 'configs/framework.yaml').read_text())


@pytest.fixture(scope='module')
def corpus(tmp_path_factory, config):
    directory = tmp_path_factory.mktemp('f6') / 'suite'
    collect_fixture_suite(directory, config=config, root_manifest=MANIFEST,
                          review_file=ROOT / 'data/fixtures/f6_reviews.json')
    return directory


@pytest.fixture
def export(tmp_path, config, tokenizer):
    counter = 0
    def run(directory, **kwargs):
        nonlocal counter
        counter += 1
        out = tmp_path / f'samples_{counter}.jsonl'
        result = export_sft(directory, out, root_manifest=MANIFEST, config=config, tokenizer=tokenizer, **kwargs)
        return read_jsonl(out), read_jsonl(out.with_suffix('.rejected.jsonl')), result
    return run


def load_run(path):
    return RunData(path / 'trace.jsonl', load_root_manifest(MANIFEST))


def write_rows(path, rows):
    path.write_text(''.join(canonical(row) + '\n' for row in rows))


def copied(corpus, tmp_path, root='fx01_integer_sum'):
    target = tmp_path / 'copy'
    shutil.copytree(corpus / root, target)
    return target


def write_event(run, node='N0'):
    return next(e for e in run.events if e.node_id == node and e.host_result.code == 'SOURCE_SAVED')


def test_smoke_32_covers_modes_and_preserves_all_raw_events(corpus, export, tokenizer):
    hashes = {p: file_hash(p) for p in corpus.rglob('trace.jsonl')}
    rows, rejected, manifest = export(corpus, limit=32)
    assert len(rows) == 32
    assert {s['mode'] for s in rows} >= {'PLAN', 'IMPLEMENT', 'COMPOSE', 'CONTROL'}
    assert {s['bucket'] for s in rows} == {'D_PLAN', 'D_IMPL', 'D_COMPOSE', 'D_CONTROL'}
    assert {s['supervision_scope'] for s in rows} == {'RESPONSIBILITY_SUPPORTED', 'PROTOCOL_ONLY'}
    assert sum(r['raw_events'] for r in manifest['raw_sources']) == len(rows) + len(rejected) == 79
    assert sum(r['policy_events'] for r in manifest['raw_sources']) == 61
    assert {p: file_hash(p) for p in hashes} == hashes
    assert all(r['reason'] for r in rejected)
    assert len({r['sample_id'] for r in rows}) == 32
    for row in rows:
        action = parse_action(row['target']).model_dump(mode='json')
        assert action == row['target_action']
        prompt_ids = tokenizer.apply_chat_template(row['messages'], tokenize=True,
                       add_generation_prompt=True, enable_thinking=False)
        assert len(prompt_ids) == row['meta']['prefix_tokens']
        assert row['meta']['target_tokens'] == len(tokenizer.encode(row['target'], add_special_tokens=False)) + 1
        assert len(prompt_ids) + row['meta']['target_tokens'] <= 4096
        assert row['split'] == 'SYN_FIXTURE'


def test_failed_root_retains_correct_children_not_bad_parent(corpus, export):
    rows, rejected, _ = export(corpus / 'fx05_failed_parent', max_length=8192)
    implementations = [s for s in rows if s['target_action']['action'] == 'write_code']
    assert {s['meta']['node_id'] for s in implementations} == {
        'N0_p1_prefix', 'N0_p1_query', 'N0_p1_query_p1_difference'}
    assert all(s['meta']['root_functional_status'] == 'FAIL' for s in implementations)
    assert all(s['supervision_scope'] == 'RESPONSIBILITY_SUPPORTED' for s in implementations)
    assert any(s['target_action']['action'] == 'finish' for s in rows)  # Honest submission is protocol-only.
    assert any(r['reason'] == 'REVIEWED_RESPONSIBILITY_VIOLATION' for r in rejected)


def test_called_but_discarded_child_does_not_become_positive(corpus, export):
    run = load_run(corpus / 'fx06_bypassed_child')
    assert run.result['functional_status'] == 'PASS'
    root_evidence = next(e for e in run.evidence.values() if e.binding['artifact']['node_id'] == 'N0')
    assert all(u['called'] and u['linked'] for u in root_evidence.details['dependency_usage'])
    rows, rejected, _ = export(run.directory, max_length=8192, allow_function_only=True)
    bad = write_event(run)
    assert not any(s['meta']['event_id'] == bad.event_id for s in rows)
    assert any(r['event_id'] == bad.event_id and r['reason'] == 'REVIEWED_RESPONSIBILITY_VIOLATION' for r in rejected)


def test_passing_tests_without_review_are_function_only(corpus):
    run = load_run(corpus / 'fx01_integer_sum')
    event = write_event(run); run.reviews.clear()
    action = parse_action(event.raw_output).model_dump(mode='json')
    with pytest.raises(Rejected, match='Passing tests'):
        classify(run, event, action, False)
    assert classify(run, event, action, True)['supervision_scope'] == 'FUNCTION_ONLY'


def test_unknown_never_becomes_responsibility_positive(corpus):
    run = load_run(corpus / 'fx01_integer_sum')
    event = write_event(run)
    run.evidence = {k: e.model_copy(update={'status': 'UNKNOWN'}) for k, e in run.evidence.items()}
    with pytest.raises(Rejected) as failure:
        classify(run, event, parse_action(event.raw_output).model_dump(mode='json'), True)
    assert failure.value.reason == 'FUNCTION_NOT_PASS'


def test_wrong_version_evidence_is_not_reused(corpus):
    run = load_run(corpus / 'fx01_integer_sum'); event = write_event(run)
    evidence = next(iter(run.evidence.values())).model_dump(mode='json')
    evidence['binding']['artifact'].update(version=2, module_path='artifacts/N0/v2.py')
    evidence['target_ref'] = 'artifacts/N0/v2.py'
    evidence['fingerprint'] = digest(evidence['binding'])
    run.evidence = {'E_new': Evidence.model_validate_json(json.dumps(evidence))}
    with pytest.raises(Rejected) as failure:
        classify(run, event, parse_action(event.raw_output).model_dump(mode='json'), True)
    assert failure.value.reason == 'NO_MATCHING_EXECUTION_EVIDENCE'


def test_repaired_v2_pass_does_not_label_failed_v1(make_f4_runtime):
    runtime = make_f4_runtime(); result = runtime.run()
    assert result['functional_status'] == 'PASS' and result['artifact']['version'] == 2
    run = load_run(runtime.run_dir); event = write_event(run)
    with pytest.raises(Rejected) as failure:
        classify(run, event, parse_action(event.raw_output).model_dump(mode='json'), True)
    assert failure.value.reason == 'FUNCTION_NOT_PASS'


def test_post_state_and_private_review_canary_stay_out_of_input(corpus, tmp_path, export):
    directory = copied(corpus, tmp_path)
    run = load_run(directory); event = write_event(run)
    canary = 'F6_PRIVATE_FUTURE_CANARY_a137'
    state_path = directory / event.state_after_ref
    state = json.loads(state_path.read_text()); state['future_private_label'] = canary
    state_path.write_text(json.dumps(state))
    reviews = read_jsonl(directory / 'supervision.jsonl')
    next(r for r in reviews if r['event_id'] == event.event_id)['meta']['private_label'] = canary
    write_rows(directory / 'supervision.jsonl', reviews)
    rows, _, _ = export(directory, max_length=8192)
    row = next(s for s in rows if s['meta']['event_id'] == event.event_id)
    assert canary in canonical(row['meta'])
    assert canary not in canonical(row['messages']) + row['target']
    assert json.loads(row['messages'][1]['content'])['observation']['visible_evidence'] == []
    assert row['evidence_refs']  # Future execution can select a label, without entering the input.


@pytest.mark.parametrize('private', [False, True])
def test_future_or_private_evidence_in_observation_is_rejected(corpus, tmp_path, export, private):
    directory = copied(corpus, tmp_path)
    events = read_jsonl(directory / 'trace.jsonl')
    evidence = json.loads((directory / 'evidence/E0001.json').read_text())
    if private: evidence['visibility'] = 'PRIVATE'
    evidence['details']['future_canary'] = 'FORBIDDEN_FUTURE'
    event = next(e for e in events if e['host_result']['code'] == 'SOURCE_SAVED')
    event['observation_before']['visible_evidence'] = [evidence]
    write_rows(directory / 'trace.jsonl', events)
    rows, rejected, _ = export(directory, max_length=8192)
    assert not any(s['meta']['event_id'] == event['event_id'] for s in rows)
    assert any(r['event_id'] == event['event_id'] and r['reason'] == 'FUTURE_OR_PRIVATE_EVIDENCE_IN_INPUT' for r in rejected)
    assert 'FORBIDDEN_FUTURE' not in canonical([s['messages'] for s in rows])


def test_raw_json_is_never_repaired_to_inherit_a_positive(corpus, tmp_path, export):
    directory = copied(corpus, tmp_path)
    events = read_jsonl(directory / 'trace.jsonl')
    event = next(e for e in events if e['host_result']['code'] == 'SOURCE_SAVED')
    event['raw_output'] = '```json\n' + event['raw_output'] + '\n```'
    write_rows(directory / 'trace.jsonl', events)
    rows, rejected, _ = export(directory, max_length=8192)
    assert not any(s['meta']['event_id'] == event['event_id'] for s in rows)
    assert any(r['event_id'] == event['event_id'] and r['reason'] == 'INVALID_EVENT' for r in rejected)


def test_copied_events_are_not_double_weighted(corpus, tmp_path, export):
    directory = tmp_path / 'copies'
    for name in ['first', 'second']:
        shutil.copytree(corpus / 'fx01_integer_sum', directory / name)
    rows, rejected, manifest = export(directory, max_length=8192)
    assert len(rows) == 4
    assert sum(r['reason'] == 'DUPLICATE_EVENT' for r in rejected) == 4
    assert len(manifest['raw_sources']) == 2


def test_family_partition_and_derived_root_inheritance(tmp_path):
    roots = load_root_manifest(MANIFEST)
    assert roots['fx05_failed_parent']['family_id'] == roots['fx06_bypassed_child']['family_id'] == roots['fx02_range_sums']['family_id']
    rows = list(roots.values())
    next(r for r in rows if r['root_id'] == 'fx05_failed_parent')['split'] = 'TRAIN'
    path = tmp_path / 'roots.jsonl'; write_rows(path, rows)
    with pytest.raises(ValueError, match='Family crosses splits'):
        load_root_manifest(path)


def test_review_must_bind_exact_action_and_contract(corpus):
    run = load_run(corpus / 'fx01_integer_sum'); event = write_event(run)
    review = run.reviews[event.event_id]
    run.reviews[event.event_id] = review.model_copy(update={'contract_hash': '0' * 64})
    with pytest.raises(Rejected) as failure:
        classify(run, event, parse_action(event.raw_output).model_dump(mode='json'), False)
    assert failure.value.reason == 'REVIEW_BINDING_MISMATCH'


def test_saved_model_prefix_is_reused_without_post_state_reconstruction(corpus, tmp_path, config, tokenizer, export):
    directory = copied(corpus, tmp_path)
    events = read_jsonl(directory / 'trace.jsonl')
    for event in events:
        prepared = prepare_prompt(tokenizer, event['observation_before'], context_tokens=8192,
                                  generation_tokens=config['runtime']['generation_tokens'])
        event['model_input'] = prepared.record
    write_rows(directory / 'trace.jsonl', events)
    (directory / 'trace_manifest.json').unlink()  # Pre-F6 F5 traces already contain actual model input.
    rows, _, _ = export(directory, max_length=8192)
    assert len(rows) == 4
    for row in rows:
        original = next(e for e in events if e['event_id'] == row['meta']['event_id'])
        assert row['messages'] == original['model_input']['messages']


def test_scope_split_and_output_guards(corpus, tmp_path, config, tokenizer):
    args = dict(root_manifest=MANIFEST, config=config, tokenizer=tokenizer)
    with pytest.raises(ValueError, match='development/holdout/SEALED'):
        export_sft(corpus, tmp_path / 'sealed.jsonl', allowed_splits=('APPS_SEALED',), **args)
    with pytest.raises(ValueError, match='separate'):
        export_sft(corpus, tmp_path / 'mixed.jsonl', allowed_splits=('SYN_FIXTURE', 'TRAIN'), **args)
    out = tmp_path / 'exists.jsonl'; out.write_text('preserve me')
    with pytest.raises(FileExistsError):
        export_sft(corpus, out, **args)
    assert out.read_text() == 'preserve me'


def test_oversize_actions_are_excluded_whole(corpus, export):
    rows, rejected, _ = export(corpus, max_length=2000)
    assert any(r['reason'] == 'SEQUENCE_TOO_LONG' for r in rejected)
    for row in rows:
        assert row['target_action'] == parse_action(row['target']).model_dump(mode='json')
        assert row['meta']['prefix_tokens'] + row['meta']['target_tokens'] <= 2000
