"""Refresh two historical positive A/S anchors with the current executor."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT=Path('runs/refinement/anchor_value_refresh')
DATA=Path('data/refinement/anchor_value_refresh')
SOURCE=Path('runs/refinement/transfer_refresh')
CURRENT=Path('runs/refinement/contract_value_refresh')
EXECUTOR='runs/refinement/contract_correction/training/final_adapter'
SELECTED=[0,3]  # weighted routes and polynomial joint, fixed before results
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')

from . import refinement_contract_value_refresh as base
from .dataset import file_hash,load_root_manifest,read_jsonl
from .evaluate import capture_program,score_program,write_json
from .refinement_responsibility_course import CourseReference
from .refinement_rollout import runtime_for
from .verifier import canonical


@contextmanager
def configured_base():
    old=(base.ROOT,base.DATA,base.EXECUTOR,base.loader.ROOT)
    base.ROOT=ROOT;base.DATA=DATA;base.EXECUTOR=EXECUTOR
    try:
        yield
    finally:
        base.ROOT,base.DATA,base.EXECUTOR,base.loader.ROOT=old


def prepare():
    assert not (ROOT/'request.json').exists(), 'request.json already exists; frozen design will not be overwritten'
    source=json.loads((SOURCE/'request.json').read_text())
    current=json.loads((CURRENT/'request.json').read_text())
    completed=json.loads((CURRENT/'result.json').read_text())
    assert completed['status']=='COMPLETE' and completed['errors']==[]
    assert source['framework']==current['framework']
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    source_rows=load_root_manifest(source['config']['root_manifest'])
    source_tests={x['root_id']:x for x in read_jsonl(source['config']['test_manifest'])}
    rows=[];tests=[];selection=[]
    for index in SELECTED:
        selected=source['selection'][index];rid=selected['root_id']
        row=source_rows[rid];test=source_tests[rid]
        assert file_hash(row['fixture_path'])==row['source_hash']
        assert file_hash(row['reference_path'])==row['reference_hash']
        assert file_hash(test['path'])==test['sha256']
        d=json.loads(Path(row['fixture_path']).read_text())
        ref=json.loads(Path(row['reference_path']).read_text())
        rows.append(row);tests.append(test)
        folder=ROOT/rid;folder.mkdir(parents=True,exist_ok=True)
        teacher,full=runtime_for(row,source_tests,source['framework'],
                                 CourseReference(d,ref,'recursive'),folder/'teacher/runtime')
        teacher.run()
        scores,_=score_program(capture_program(teacher,folder/'teacher/program'),full,rid)
        assert all(x['status']=='PASS' for x in scores.values())
        write_json(folder/'teacher/result.json',scores)
        state=json.loads(Path(selected['state_path']).read_text())
        assert state['observation']['node_id']=='N0_p1_compute'
        assert state['root_event']['host_result']['code']=='PLAN_REGISTERED'
        write_json(folder/'state.json',state)
        selection.append({'root_id':rid,'state_path':str(folder/'state.json'),
                          'contract':selected['variant'],'family_id':row['family_id']})
    for label,values in [('roots',rows),('tests',tests)]:
        (DATA/f'{label}.jsonl').write_text(''.join(canonical(v)+'\n' for v in values))
    adapters={**current['adapters'],'executor':EXECUTOR,'initial':EXECUTOR}
    request={'framework':current['framework'],
             'config':{**source['config'],'root_manifest':str(DATA/'roots.jsonl'),
                       'test_manifest':str(DATA/'tests.jsonl'),'output_dir':str(ROOT)},
             'adapters':adapters,
             'hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},
             'selection':selection,'execution_seed':623000713,'repetitions':4,
             'cases':16,'updates':0,'arms':['ATOMIC','SPLIT'],
             'source_request':str(SOURCE/'request.json'),
             'scope':'Two known historical positive development states. Root plan and first compute observation copied exactly. No new-task or natural-gate claim.',
             'design':'Freeze contract_correction executor and current root/internal gates and MULTI_BODY. Forced A versus freshly sampled BODY S on first compute choice, four new paired seeds per state. Keep every rejection, retry and failure.',
             'decision':'Check whether historical positive split value survives the new executor before mixed-direction gate calibration. Difference <=1/4 uncertain; >=2/4 exploratory direction. No updates.'}
    write_json(ROOT/'request.json',request)
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1'],
                                          'min_free_mib':{'collect':16000}})


def check():
    with configured_base():
        r=base.check()
    assert [s['contract'] for s in r['selection']]==['routes_outside','polynomial_joint']
    assert r['cases']==16 and r['updates']==0
    return r


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','check','run','report'])
    parser.add_argument('--index',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare();check()
    elif args.command=='check':
        r=check();print(json.dumps({'status':'READY','cases':r['cases'],'model_loaded':False}))
    elif args.command=='run':
        check()
        with configured_base():base.run(args.index)
    else:
        check()
        with configured_base():base.report()
