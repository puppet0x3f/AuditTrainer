"""Root return-contract retention: alias-independent composition, fixed budget."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path('runs/refinement/root_retention')
DATA=Path('data/refinement/root_retention')
PREVIOUS=Path('runs/refinement/transfer_sft')
INITIAL=str(PREVIOUS/'training/final_adapter')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')
from . import refinement_transfer_sft as base
from .dataset import read_jsonl,file_hash
from .evaluate import write_json
from .verifier import canonical


def configure():
    base.ROOT=ROOT;base.DATA=DATA;base.INITIAL=INITIAL;base.configure()


def prepare():
    if (ROOT/'request.json').exists():return
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    old=json.loads((PREVIOUS/'request.json').read_text())
    original=read_jsonl('data/refinement/transfer_sft/training.jsonl')
    targets=[]
    for x in original:
        if not (x['root_id'].startswith('train_complex_sft_') and x['observation']['node_id']=='N0' and x['target_action']['action']=='write_code'):continue
        # Change the consumed alias consistently in observation and action, not the child contract.
        y=json.loads(json.dumps(x).replace('_compute_child','_prepare_child'))
        y.update(kind='target',sample_id=x['sample_id']+'_return_contract_alias')
        targets.append(y)
    retention=[{**x,'kind':'retention'} for x in original]
    assert len(targets)==8 and len(retention)==88
    (DATA/'training.jsonl').write_text(''.join(canonical(x)+'\n' for x in targets+retention))
    r=deepcopy(old);r['config']['output_dir']=str(ROOT)
    r.update(steps=44,learning_rate=1e-5,seed=551000713,initial_executor=INITIAL,
        training_sha256=file_hash(DATA/'training.jsonl'),target_examples=8,retention_examples=88,
        new_evaluation_cases=24,cached_baseline_cases=24,evaluation_cases=48,
        design='Executor copy from transfer_sft; 44 fixed updates, 2 root targets +2 retention per update, each target 11 exposures, each of all 88 old examples once. Existing TRAIN root examples with consistent alias renaming only. One final checkpoint.',
        scope='Known development repair check with cached transfer_sft AFTER as BEFORE. Same plans, seeds, tests, observations; no failed development trace supervision. No unseen generalization or fresh A/S claim.',
        decision='Require all 8 new-contract cases and all 16 retention cases to pass. Otherwise keep candidate and report failures; no adaptive checkpoint selection.')
    r['adapters']['initial']=INITIAL;r['adapters']['executor']=INITIAL
    r['source_hashes']={k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in r['adapters'].items()}
    r['cache_hashes']={str(PREVIOUS/'evaluation/AFTER'/f'shard_{i}.json'):file_hash(PREVIOUS/'evaluation/AFTER'/f'shard_{i}.json') for i in range(3)}
    dst=ROOT/'evaluation/BEFORE';dst.mkdir(parents=True,exist_ok=True)
    for i in range(3):shutil.copyfile(PREVIOUS/'evaluation/AFTER'/f'shard_{i}.json',dst/f'shard_{i}.json')
    write_json(ROOT/'request.json',r)
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'train':22000,'collect':16000}})
    print(json.dumps({'targets':8,'retention':88,'steps':44,'new_evaluations':24}))


def report():
    configure();base.study.report()
    r=json.loads((ROOT/'request.json').read_text());result=json.loads((ROOT/'result.json').read_text())
    counts=Counter(i for h in result['training']['history'] for i in h['indices'][:2])
    keeps=Counter(i for h in result['training']['history'] for i in h['indices'][2:])
    assert len(counts)==8 and set(counts.values())=={11}
    assert len(keeps)==88 and set(keeps.values())=={1}
    assert file_hash(DATA/'training.jsonl')==r['training_sha256']
    assert all(file_hash(p)==h for p,h in r['cache_hashes'].items())
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['source_hashes'][k] for k,p in r['adapters'].items())
    def keyed(m):return {(x['case']['root_id'],x['case']['seed']):x for x in result['evaluation'][m]['results']}
    before,after=keyed('BEFORE'),keyed('AFTER');pairs=[];errors=[]
    for key,x in after.items():
        def first(z):return next(e['observation_before'] for e in read_jsonl(Path(z['case']['run_dir'])/'runtime/trace.jsonl') if e['observation_before']['phase']=='IMPLEMENT')
        assert first(x)==first(before[key])
        pairs.append({'root_id':key[0],'seed':key[1],'first_observation_identical':True})
        if x['case']['errors']:errors.append({'root_id':key[0],'seed':key[1],'errors':x['case']['errors']})
    s=result['summary']['AFTER']
    result['proceed_to_fresh_values']=not errors and all(s[g]['root_success']==s[g]['n'] for g in s)
    result.update(new_evaluation_cases=24,cached_baseline_cases=24)
    write_json(ROOT/'result.json',result)
    write_json(ROOT/'validation.json',{'paired':pairs,'target_exposures':dict(counts),'retention_examples_exposed_once':len(keeps),'original_adapters_unchanged':True,'cached_baseline_unchanged':True,'training_data_unchanged':True,'errors':errors})
    print(json.dumps({'summary':result['summary'],'paired':result['paired'],'proceed':result['proceed_to_fresh_values']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate','report']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    configure()
    if a.command=='prepare':prepare()
    elif a.command=='train':base.study.train(ROOT,DATA,INITIAL)
    elif a.command=='evaluate':base.evaluate('AFTER',a.shard)
    else:report()
