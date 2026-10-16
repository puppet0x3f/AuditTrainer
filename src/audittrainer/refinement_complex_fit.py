"""Equal-exposure positive control for the existing complete-algorithm curriculum."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT=Path('runs/refinement/complex_fit')
PREVIOUS=Path('runs/refinement/complex_sft')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1] in {'train','evaluate'}:
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','train' if sys.argv[1]=='train' else 'collect')
from . import refinement_complex_sft as study
from .dataset import file_hash,read_jsonl
from .evaluate import write_json


def prepare():
    if (ROOT/'request.json').exists():return
    ROOT.mkdir(parents=True,exist_ok=True)
    old=json.loads((PREVIOUS/'request.json').read_text());r=deepcopy(old)
    assert file_hash(study.DATA/'training.jsonl')==old['training_sha256']
    r.update(steps=112,design='Same initial compute_sft, seed, data, loss, learning rate and cyclic schedule as 32-step control. 112 fixed updates, exactly 8 exposures per new target; 224 target +224 retention exposures. Unique final checkpoint; no adaptive stopping or checkpoint selection.',
        evaluation='24 new autonomous AFTER cases paired with the exact 24 cached BEFORE cases from complex_sft. Also report cached 32-step model on the same observations/plans/seeds. Same-family development only; no new heldout claim.',
        decision='FIT 8/8 is the diagnostic target. Fresh value estimation requires FIT and development improvement over baseline plus retention protection, not perfect execution. No gate training in this experiment.',
        cached_before=str(PREVIOUS/'evaluation/BEFORE'),prior_32_step_result=str(PREVIOUS/'result.json'),new_evaluation_cases=24,cached_baseline_cases=24)
    r['config']['output_dir']=str(ROOT)
    r['cache_hashes']={str(PREVIOUS/'evaluation/BEFORE'/f'shard_{i}.json'):file_hash(PREVIOUS/'evaluation/BEFORE'/f'shard_{i}.json') for i in range(3)}
    write_json(ROOT/'request.json',r)
    write_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'train':22000,'collect':16000}})
    dst=ROOT/'evaluation/BEFORE';dst.mkdir(parents=True,exist_ok=True)
    for i in range(3):shutil.copyfile(PREVIOUS/'evaluation/BEFORE'/f'shard_{i}.json',dst/f'shard_{i}.json')
    print(json.dumps({'steps':112,'target_exposures':224,'retention_exposures':224,'new_evaluations':24,'cached_baseline':24}))


def report():
    study.report()
    r=json.loads((ROOT/'request.json').read_text());result=json.loads((ROOT/'result.json').read_text())
    prior=json.loads((PREVIOUS/'result.json').read_text());history=result['training']['history']
    counts=Counter(i for h in history for i in h['indices'][:2]);assert len(counts)==28 and set(counts.values())=={8}
    assert file_hash(study.DATA/'training.jsonl')==r['training_sha256']
    assert all(file_hash(p)==h for p,h in r['cache_hashes'].items())
    def keyed(arm):return {(x['group'],x['case']['root_id'],x['case']['seed']):x for x in arm['results']}
    before=keyed(result['evaluation']['BEFORE']);after=keyed(result['evaluation']['AFTER']);old=keyed(prior['evaluation']['AFTER'])
    assert before.keys()==after.keys()==old.keys() and len(after)==24
    comparisons=[];paired={'wins':0,'losses':0,'ties':0};rejects=Counter()
    for key,x in after.items():
        def first(item):return next(e['observation_before'] for e in read_jsonl(Path(item['case']['run_dir'])/'runtime/trace.jsonl') if e['observation_before']['phase']=='IMPLEMENT')
        assert first(x)==first(before[key])==first(old[key])
        y=lambda item:int(item['case']['final_scores']['private']['status']=='PASS')
        a,b=y(x),y(old[key]);paired['wins' if a>b else 'losses' if a<b else 'ties']+=1
        assert not x['case']['errors'];rejects.update(x['case']['protocol_rejections'])
        comparisons.append({'group':key[0],'root_id':key[1],'seed':key[2],'baseline':y(before[key]),'steps32':b,'steps112':a,'first_observation_identical':True})
    s=result['summary']['AFTER']
    result.update(prior_32_step_summary=prior['summary']['AFTER'],paired_vs_32=paired,new_evaluation_cases=24,cached_baseline_cases=24,
        fit_target_reached=s['fit']['root_success']==8,
        proceed_to_fresh_values=s['fit']['root_success']>result['summary']['BEFORE']['fit']['root_success'] and s['retention']['root_success']==8 and s['development']['root_success']>result['summary']['BEFORE']['development']['root_success'])
    write_json(ROOT/'result.json',result)
    write_json(ROOT/'validation.json',{'target_exposure_counts':dict(counts),'paired_observations':comparisons,'new_case_rejections':dict(rejects),'new_case_infra_errors':0,'training_data_unchanged':True,'cached_baseline_unchanged':True,'tests_log':str(ROOT/'tests.log')})
    print(json.dumps({'summary':result['summary'],'paired_vs_32':paired,'proceed':result['proceed_to_fresh_values']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train','evaluate','report']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    study.ROOT=ROOT
    if a.command=='prepare':prepare()
    elif a.command=='train':study.train(ROOT,study.DATA,study.INITIAL)
    elif a.command=='evaluate':study.evaluate('AFTER',a.shard)
    else:report()
