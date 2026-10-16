"""Frozen A/S/natural transfer on prespecified new internal contracts."""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter
ROOT=Path('runs/refinement/internal_transfer')
DATA=Path('data/refinement/internal_transfer')
if __name__=='__main__' and len(sys.argv)>1 and sys.argv[1]=='run':
    from .gpu_resource import acquire
    _reservation=acquire(ROOT/'gpu_resources.json','collect')
from . import refinement_internal_gate as gate_study
from .refinement_internal_transfer_tasks import VARIANTS, task_pack
from .refinement_responsibility_course import CourseReference,allocation
from .refinement_interface_gate import FixedPlanner
from .refinement_rollout import runtime_for,run_case
from .refinement_body_transfer import reviewed_plans
from .refinement_curriculum_rl import plan_key
from .refinement_post_handoff_values import same_contract_delegation
from .dataset import read_jsonl,load_root_manifest,file_hash
from .evaluate import write_json,capture_program,score_program
from .refinement_generalization import guarded_json
from .verifier import canonical,digest


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    if (ROOT/'request.json').exists():return
    old=json.loads((gate_study.ROOT/'request.json').read_text());rows=[];tests={};selection=[]
    assert json.loads((gate_study.ROOT/'result.json').read_text())['status']=='COMPLETE'
    trained=[json.loads(Path(s['state_path']).read_text())['observation']['incoming_contract'] for s in old['selection']]
    for name in VARIANTS:
        d,private,ref=task_pack(name);t=d['task'];rid=t['root_id'];folder=DATA/rid;folder.mkdir(exist_ok=True)
        assert d['local_contracts']['compute'] not in trained
        for label,obj in [('task',d),('private_tests',private),('reference',ref)]:guarded_json(folder/f'{label}.json',obj)
        row={**{k:t[k] for k in ['root_id','family_id','split']},'source':'new_contract_known_algorithm_transfer','original_split':'VALIDATION','io_type':'solve_text',
             'fixture_path':str(folder/'task.json'),'source_hash':file_hash(folder/'task.json'),'reference_path':str(folder/'reference.json'),'reference_hash':file_hash(folder/'reference.json')}
        rows.append(row);tests[rid]={'root_id':rid,'path':str(folder/'private_tests.json'),'sha256':file_hash(folder/'private_tests.json')}
        teacher,full=runtime_for(row,tests,old['framework'],CourseReference(d,ref,'recursive'),ROOT/'teacher'/rid/'runtime')
        outcome=teacher.run();scores,_=score_program(capture_program(teacher,ROOT/'teacher'/rid/'program'),full,rid)
        assert all(v['status']=='PASS' for v in scores.values());write_json(ROOT/'teacher'/rid/'result.json',{'scores':scores,'runtime':outcome})
        base=ROOT/rid;base.mkdir(exist_ok=True)
        planner=FixedPlanner(type('Base',(),{'cost':{},'identity':{}})(),{'N0':allocation(d['contract'],d,'fused')})
        runtime,_=runtime_for(row,tests,old['framework'],planner,base/'decision')
        assert runtime.step().host_result.code=='PLAN_REGISTERED' and runtime.step().host_result.code=='CHILD_ENTERED'
        obs=runtime.build_observation();assert obs['node_id']==gate_study.NODE and obs['phase']=='NEW'
        event=next(e for e in read_jsonl(runtime.trace.path) if e['event_kind']=='POLICY')
        write_json(base/'state.json',{'root_event':event,'observation':obs})
        selection.append({'root_id':rid,'state_path':str(base/'state.json'),'variant':name,'known_algorithm_family':VARIANTS[name][0]})
    for label,values in [('roots',rows),('tests',list(tests.values()))]:
        (DATA/f'{label}.jsonl').write_text(''.join(canonical(v)+'\n' for v in values))
    adapters={**old['adapters'],'internal_gate':str(gate_study.ROOT/'training/final_adapter')}
    guarded_json(ROOT/'request.json',{'framework':old['framework'],'config':{**old['config'],'split':'VALIDATION','output_dir':str(ROOT),'root_manifest':str(DATA/'roots.jsonl'),'test_manifest':str(DATA/'tests.jsonl')},
        'selection':selection,'adapters':adapters,'hashes':{k:file_hash(Path(p)/'adapter_model.safetensors') for k,p in adapters.items()},
        'execution_seed':523000713,'repetitions':4,'cases':48,'updates':0,
        'scope':'Four newly authored aggregation contracts and inputs, absent from internal-gate training and executor supervision. Algorithm families and preparation semantics are known. New-contract transfer, not unseen-algorithm or sealed-benchmark generalization.',
        'selection_rule':'All four variants fixed before any model outcome: reachable distances outside threshold, positive inventory surplus, redundant coverage, joint original polynomial and derivative evaluation. No outcome filtering.',
        'design':'Fixed root delegates compute. First compute choice forced ATOMIC, forced fresh SPLIT, or natural calibrated gate. All later decisions/retries use frozen original BODY; root gate/BODY/executor/internal gate frozen. Common per-replicate RNG seed and mode draw in every arm; fresh conditional body generation inside each run. Keep all failures.',
        'decision':'Differences <=1/4 uncertain; >=2/4 exploratory directional evidence. Compare average A/S root success with calibrated p and natural root outcomes. No labels or updates in this round.',
        'oracle_validation':'Independent oracle checked against reference on every visible/private input; authored recursive teacher passes root suites. Teacher outputs never substituted for model actions.'})
    guarded_json(ROOT/'gpu_resources.json',{'devices':['0','1','2'],'min_free_mib':{'collect':16000}})
    print(json.dumps({'cases':48,'states':4,'teacher_root_pass':4}))


class BranchInternal(gate_study.NaturalInternal):
    def __init__(self,base,state,arm):
        super().__init__(base,state);self.arm=arm
        self.identity={**self.identity,'intervention':'fixed_root_first_compute_'+arm}
    def act(self,obs):
        if obs['node_id']==self.node and not self.internal_sent:
            import torch
            assert digest(obs)==digest(self.state['observation']);self.internal_sent=True
            gate=self.base.score_modes(obs);modes=list(gate['mode_probabilities'])
            sampled=modes[int(torch.multinomial(torch.tensor([gate['mode_probabilities'][m] for m in modes]),1))]
            mode=sampled if self.arm=='NATURAL' else self.arm
            candidate=self.base.atomic(gate) if mode=='ATOMIC' else self.base.sample_split(gate)
            return self.base.as_output(gate,candidate)
        return super().act(obs)


def run(index):
    r=json.loads((ROOT/'request.json').read_text());_,policy,executor,_,_=gate_study.context('AFTER');before=gate_study.fingerprints(policy)
    rows=load_root_manifest(r['config']['root_manifest']);tests={x['root_id']:x for x in read_jsonl(r['config']['test_manifest'])}
    selected=r['selection'][index];rid=selected['root_id'];state=json.loads(Path(selected['state_path']).read_text());base=ROOT/rid
    definition=json.loads(Path(rows[rid]['fixture_path']).read_text());reviewed=reviewed_plans(state['observation'],definition)
    gate=policy.score_modes(state['observation']);prior=policy.body.score_modes(state['observation'])
    write_json(base/'gate.json',gate);write_json(base/'body_gate.json',prior);cases=[]
    for rep in range(4):
        for arm in ['ATOMIC','SPLIT','NATURAL']:
            fixed=BranchInternal(policy,state,arm);directory=base/arm/f'repeat_{rep:02d}';cached=(directory/'case_result.json').exists()
            c=run_case(rows[rid],tests,r['framework'],r['config'],fixed,executor,directory,r['execution_seed']+index*100+rep)
            assert c['status']=='COMPLETE' or c['errors']==['SCORING_ERROR']
            if not cached:assert fixed.root_sent and fixed.internal_sent
            events=read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl')
            first=next(e for e in events if e['event_kind']=='POLICY' and e['node_id']==gate_study.NODE)
            assert first['observation_before']==state['observation']
            mode=first['model_input']['refinement']['selected_mode'];assert arm=='NATURAL' or mode==arm
            variant=reviewed.get(plan_key(first['parsed_action'],definition),'UNCONFIRMED') if mode=='SPLIT' else 'ATOMIC'
            cases.append({'arm':arm,'rep':rep,'mode':mode,'case':c,'first_registered':first['host_result']['code']=='PLAN_REGISTERED','variant':variant,
                          'same_contract_delegation':mode=='SPLIT' and same_contract_delegation({'parsed_action':first['parsed_action']},state['observation']['incoming_contract'])})
            print(json.dumps({'root':rid,'arm':arm,'rep':rep,'mode':mode,'root_score':c['final_scores']['private']['status']}),flush=True)
    assert gate_study.fingerprints(policy)==before
    write_json(base/'result.json',{'status':'COMPLETE','root_id':rid,'frozen':before,'p_split':gate['mode_probabilities']['SPLIT'],'body_prior_p_split':prior['mode_probabilities']['SPLIT'],'cases':cases})


def report():
    r=json.loads((ROOT/'request.json').read_text());groups=[json.loads((ROOT/s['root_id']/'result.json').read_text()) for s in r['selection']];summary=[]
    for g in groups:
        item={k:g[k] for k in ['root_id','p_split','body_prior_p_split']};pair={}
        for arm in ['ATOMIC','SPLIT','NATURAL']:
            chosen=[x for x in g['cases'] if x['arm']==arm];assert len(chosen)==4
            pair[arm]={x['case']['seed']:x['case']['final_scores']['private']['status']=='PASS' for x in chosen}
            item[arm]={'success':sum(pair[arm].values()),'n':4,'split':sum(x['mode']=='SPLIT' for x in chosen),
                'mean_tokens':sum(x['case']['usage']['input_tokens']+x['case']['usage']['output_tokens'] for x in chosen)/4,
                'variants':dict(Counter(x['variant'] for x in chosen)),'same_contract_delegations':sum(x['same_contract_delegation'] for x in chosen),
                'errors':[x['case']['errors'] for x in chosen if x['case']['errors']]}
        assert pair['ATOMIC'].keys()==pair['SPLIT'].keys()==pair['NATURAL'].keys()
        delta=(item['SPLIT']['success']-item['ATOMIC']['success'])/4
        item['delta_success']=delta;item['direction']='UNCERTAIN' if abs(delta)<=.25 else 'SPLIT' if delta>0 else 'ATOMIC'
        item['natural_vs_atomic']={'wins':sum(pair['NATURAL'][k]>v for k,v in pair['ATOMIC'].items()),'losses':sum(pair['NATURAL'][k]<v for k,v in pair['ATOMIC'].items())}
        summary.append(item)
    assert all(g['frozen']==groups[0]['frozen'] for g in groups)
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k] for k,p in r['adapters'].items())
    totals={arm:{'n':16,'success':sum(s[arm]['success'] for s in summary),'split':sum(s[arm]['split'] for s in summary),'mean_tokens':sum(s[arm]['mean_tokens'] for s in summary)/4} for arm in ['ATOMIC','SPLIT','NATURAL']}
    write_json(ROOT/'result.json',{'status':'COMPLETE','cases':48,'updates':0,'summary':summary,'totals':totals,'groups':groups})
    print(json.dumps({'summary':summary,'totals':totals}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','report']);p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='run':run(a.index)
    else:report()
