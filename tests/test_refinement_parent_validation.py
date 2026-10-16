from copy import deepcopy
import json
from pathlib import Path

import yaml

from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.refinement_parent_composition import restore_parent
from audittrainer.refinement_parent_validation import broader_support, verify_frozen


def test_sixteen_new_states_restore_exactly_with_current_local_passes(tmp_path):
    settings=yaml.safe_load(Path('configs/refinement_parent_validation.yaml').read_text())
    request=verify_frozen(settings)
    old=json.loads((Path(settings['previous_study'])/'request.json').read_text())
    rows=load_root_manifest(settings['root_manifest'])
    tests={r['root_id']:r for r in read_jsonl(settings['test_manifest'])}
    assert len(request['states'])==16 and len(rows)==12
    assert not {s['root_id'] for s in old['states']} & set(rows)
    assert {r['split'] for r in rows.values()}=={'TRAIN','VALIDATION'}
    for i,s in enumerate(request['states']):
        runtime,_=restore_parent(s,rows[s['root_id']],tests,request['framework'],None,tmp_path/str(i))
        assert runtime.build_observation()==s['observation']
        assert not (runtime.run_dir/'artifacts/N0').exists()
        assert all(runtime._node_result(runtime.contexts[n]).functional_status=='PASS' for n in s['child_hashes'])


def test_new_support_rule_allows_losses_but_requires_spread_beyond_one_root():
    settings=yaml.safe_load(Path('configs/refinement_parent_validation.yaml').read_text())
    a={'wins':2,'losses':0};b={'wins':1,'losses':0};c={'wins':0,'losses':1}
    assert broader_support({'wins':3,'losses':1},{'a':a,'b':b,'c':c},{'r1':a,'r2':b,'r3':c},settings['decision'])
    assert not broader_support({'wins':3,'losses':1},{'a':a,'b':b,'c':c},{'r1':{'wins':3,'losses':1}},settings['decision'])
    assert not broader_support({'wins':1,'losses':1},{'a':a,'b':b},{'r1':a,'r2':b},settings['decision'])


def test_new_seed_slots_do_not_overlap_the_design_evaluation():
    settings=yaml.safe_load(Path('configs/refinement_parent_validation.yaml').read_text())
    old=json.loads((Path(settings['previous_study'])/'result.json').read_text())
    seeds={settings['evaluation_seed']+i*1000+j for i in range(16) for j in range(4)}
    assert len(seeds)==64 and not seeds & {r['seed'] for r in old['records']}
