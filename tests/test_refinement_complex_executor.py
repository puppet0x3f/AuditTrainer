import json
from pathlib import Path
from audittrainer.dataset import read_jsonl,load_root_manifest
from audittrainer.refinement_complex_executor import plans


def test_old_new_comparison_changes_only_executor_and_covers_counterexample():
    r=json.loads(Path('runs/refinement/complex_executor/request.json').read_text())
    prior=json.loads(Path('runs/refinement/complex_values/request.json').read_text())
    assert r['framework']==prior['framework'] and r['selection']==prior['selection']
    assert r['cases']==16 and r['repetitions']==2 and r['updates']==0
    assert r['adapters']['BEFORE']=='runs/refinement/handoff_sft/training/final_adapter'
    assert r['adapters']['AFTER']==prior['adapters']['executor']
    assert r['hashes']['BEFORE']!=r['hashes']['AFTER']
    for key in ['gate','body']:assert r['hashes'][key]==prior['hashes'][key]
    tests=read_jsonl(r['config']['test_manifest'])
    graph=next(x for x in tests if x['root_id'].endswith('weighted_routes'))
    suites=json.loads(Path(graph['path']).read_text())['suites']
    edge=next(s for s in suites if s['suite_id'].endswith('edge_order_private'))
    assert edge['visibility']=='PRIVATE' and len(edge['cases'])==2
    assert edge['cases'][0]['expected']==edge['cases'][1]['expected']


def test_fixed_plans_have_real_responsibilities_and_teacher_passes():
    r=json.loads(Path('runs/refinement/complex_executor/request.json').read_text());rows=load_root_manifest(r['config']['root_manifest'])
    for s in r['selection']:
        rid=s['root_id'];d=json.loads(Path(rows[rid]['fixture_path']).read_text());p=plans(d)
        assert p['N0']['args']['children'][0]['local_key']=='compute'
        internal=p['N0_p1_compute']['args']
        assert internal['retained_responsibilities']==['aggregate']
        assert internal['obligation_map']=={'PREPARE':['child:prepare'],'AGGREGATE':['self']}
        teacher=json.loads((Path('runs/refinement/complex_executor/teacher')/rid/'result.json').read_text())
        assert all(v['status']=='PASS' for v in teacher['scores'].values())
