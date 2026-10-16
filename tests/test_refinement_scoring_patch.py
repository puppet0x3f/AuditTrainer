from copy import deepcopy
import json
from pathlib import Path

from audittrainer.evaluate import load_task
from audittrainer.refinement_scoring_patch import CASES, ROOT, patched_private, prepare, copy_snapshot


def test_patch_preserves_old_cases_and_keeps_answers_offline(tmp_path):
    source = json.loads(Path('data/refinement/b3_coverage/'+ROOT+'/private_tests.json').read_text())
    original = deepcopy(source)
    updated = patched_private(source)
    assert source == original
    for old, new in zip(source['suites'], updated['suites']):
        assert new['cases'][:len(old['cases'])] == old['cases']
        assert len(new['cases']) == len(old['cases']) + 4
    _, framework, row, tests = prepare(tmp_path/'run', tmp_path/'data')
    definition, _, _, online, full = load_task(row, tests, framework)
    assert all(s.visibility == 'VISIBLE' for s in online.suites.values())
    assert all('score_v2' not in c.case_id for s in online.suites.values() for c in s.cases)
    assert sum('score_v2' in c.case_id for s in full.suites.values() for c in s.cases) == 16
    assert definition['task'] == json.loads(Path(row['fixture_path']).read_text())['task']


def test_independent_cases_detect_stride_bug():
    for items, prepared, expected in CASES:
        actual_prepared = sorted({int(t[:2])*60+int(t[3:]) for t in items})
        assert actual_prepared == prepared
        correct_indices = [i for i, x in enumerate(prepared) if x % 3 == 0]
        assert expected == {'indices': correct_indices, 'total': sum(prepared[i] for i in correct_indices)}
    prepared = CASES[0][1]
    buggy_indices = [i for i in range(len(prepared)) if sum(prepared[i::3]) % 3 == 0]
    assert buggy_indices != CASES[0][2]['indices']


def test_snapshot_copy_does_not_copy_or_write_old_evidence(tmp_path):
    source = tmp_path/'old'; (source/'artifacts/N0').mkdir(parents=True)
    (source/'artifacts/N0/v1.py').write_text('source')
    (source/'artifacts/N0/v1.json').write_text('{}')
    (source/'evidence').mkdir(); (source/'evidence/old.json').write_text('old')
    snapshot = {'path': str(source), 'bundle': {'modules': {'N0': {'module_path': 'artifacts/N0/v1.py'}}}}
    copied = copy_snapshot(snapshot, tmp_path/'new')
    assert copied['path'] != snapshot['path']
    assert not (tmp_path/'new/evidence').exists()
    assert (source/'evidence/old.json').read_text() == 'old'
