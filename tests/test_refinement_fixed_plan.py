from collections import Counter
import json
from pathlib import Path
import random
from types import SimpleNamespace

import yaml

from audittrainer.dataset import load_root_manifest, read_jsonl
from audittrainer.refinement_curriculum import plans
from audittrainer.refinement_fixed_plan import FixedRootPlanner, prepare, repeated_advantage, settings_for
from audittrainer.refinement_rollout import run_case
from audittrainer.schema import PolicyOutput, TokenUsage
from audittrainer.verifier import canonical


def test_six_fixed_plans_register_with_local_coverage():
    study=yaml.safe_load(Path('configs/refinement_fixed_plan.yaml').read_text())
    request=prepare(study)
    checks=json.loads((Path(study['output_dir'])/'plan_checks.json').read_text())
    assert len(checks)==6 and all(c['registered'] for c in checks)
    assert request['planned_continuations']==2*3*12==72
    assert request['presentation']==json.loads(Path('runs/refinement/B3_summary_training/request.json').read_text())['presentation']


def test_reset_occurs_after_planner_random_consumption(tmp_path):
    study=yaml.safe_load(Path('configs/refinement_fixed_plan.yaml').read_text());config,framework=settings_for(study)
    row=load_root_manifest(config['root_manifest'])[study['root_ids'][0]]
    tests={r['root_id']:r for r in read_jsonl(config['test_manifest'])}
    action=plans(json.loads(Path(row['fixture_path']).read_text()))['atomic']
    base=SimpleNamespace(cost=Counter(),identity={})
    observed=[]
    def execute(o):
        observed.append(random.random())
        return PolicyOutput(raw_text=canonical({'action':'write_code','args':{
            'contract_version':1,'source':'def solve(text):\n    return text\n','declared_dependencies':[]}}),token_usage=TokenUsage(),
                            model_identity={'backend':'test_executor'})
    executor=SimpleNamespace(cost=Counter(),llm=SimpleNamespace(identity={}),act=execute)
    for reset in (False,True):
        planner=FixedRootPlanner(base,action);original=planner.act
        def consume(o):
            for _ in range(17):random.random()
            return original(o)
        planner.act=consume
        case=run_case(row,tests,framework,config,planner,executor,tmp_path/str(reset),987654,
                      reset_seed_after_root_plan=reset)
        assert case['status']=='COMPLETE' and case['root_first_plan_registered']
        if reset:assert case['continuation_seed_reset_after_root_plan']==987654
    assert observed[0]!=observed[1]==random.Random(987654).random()


def test_repeated_advantage_requires_both_seed_blocks():
    a=[{'root_id':'r','seed':i,'success':0} for i in range(12)]
    b=[{**r,'success':int(r['seed'] in [0,1])} for r in a]
    assert not repeated_advantage(a,b,'r',6)['repeated_advantage']
    b[6]['success']=1
    assert repeated_advantage(a,b,'r',6)['repeated_advantage']
