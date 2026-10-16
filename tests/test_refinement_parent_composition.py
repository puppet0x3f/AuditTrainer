from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from audittrainer.dataset import file_hash, load_root_manifest, read_jsonl
from audittrainer.refinement_parent_composition import (
    composition_summary, framework_for, paired_summary, restore_parent, selected_state, summary_prompt,
)


def settings():
    return yaml.safe_load(Path('configs/refinement_parent_composition.yaml').read_text())


def test_restore_all_twelve_observations_and_current_passed_children_without_future_source(tmp_path):
    s = settings(); rows = load_root_manifest(s['root_manifest'])
    tests = {r['root_id']:r for r in read_jsonl(s['test_manifest'])}
    labels = []
    for i,path in enumerate(s['source_cases']):
        selected = selected_state(path,i); labels.append(selected['source_success'])
        runtime,_ = restore_parent(selected,rows[selected['root_id']],tests,framework_for(s),None,tmp_path/str(i))
        assert runtime.build_observation() == selected['observation']
        assert runtime.node.artifact_ref is None and not (runtime.run_dir/'artifacts/N0').exists()
        for child,sha in selected['child_hashes'].items():
            assert file_hash(runtime.run_dir/runtime.contexts[child].artifact.module_path) == sha
            assert runtime._node_result(runtime.contexts[child]).functional_status == 'PASS'
    assert len(labels) == 12 and sum(labels) == 10


def test_summary_restates_responsibilities_and_io_without_hidden_fields_or_plan_changes(monkeypatch):
    s = settings(); selected = selected_state(s['source_cases'][0],0)
    observation = deepcopy(selected['observation']); observation['teacher_answer'] = 'SECRET'
    original = deepcopy(observation)
    summary = composition_summary(observation)
    assert {o['id'] for o in summary['parent_obligations']} == {'PARSE','AGGREGATE','FORMAT'}
    assert summary['child_inputs_and_outputs'][0]['signature'] == observation['child_interfaces'][0]['contract']['signature']
    assert summary['declared_dataflow'] == observation['current_plan']['dataflow']
    assert 'SECRET' not in json.dumps(summary)
    monkeypatch.setattr('audittrainer.refinement_parent_composition.prepare_prompt',lambda tokenizer,o,**kwargs:o)
    rendered = summary_prompt(None,observation)
    assert observation == original
    assert rendered['root_spec'].startswith(observation['root_spec'])
    assert set(k for k in rendered if rendered[k] != observation[k]) == {'root_spec'}
    assert 'SECRET' not in json.dumps(rendered)
    pipeline = selected_state(s['source_cases'][4],4)['observation']
    assert composition_summary(pipeline)['declared_dataflow'] == pipeline['current_plan']['dataflow'] != []


def test_unknown_children_are_excluded():
    source = 'runs/refinement/B3_first_credit/evaluation/B3_92000713/natural/train_transfer_chunks_chunks_03/repeat_10'
    with pytest.raises(ValueError,match='passed a bound local check'):
        selected_state(source,0)


def test_pairing_preserves_losses_and_requires_identical_state_seed():
    records = [{'state_id':'S00','seed':i,'arm':arm,'success':success}
               for i,a,b in [(1,False,True),(2,True,False),(3,True,True)]
               for arm,success in [('control',a),('summary',b)]]
    assert paired_summary(records) == {'pairs':3,'control_passes':2,'summary_passes':2,'wins':1,'losses':1,'ties':1}
    with pytest.raises(ValueError,match='Incomplete pair'): paired_summary(records[:-1])
    with pytest.raises(ValueError,match='Duplicate'): paired_summary(records+[records[0]])
