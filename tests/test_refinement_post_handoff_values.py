from copy import deepcopy
from audittrainer.refinement_post_handoff_values import same_contract_delegation


def test_same_contract_is_not_merely_same_function_name():
    c={'signature':{'name':'prepare'},'ensures':[{'id':'PREPARE','text':'full responsibility'}],'contract_id':'C_1','version':1}
    child={k:v for k,v in c.items() if k not in {'contract_id','version'}}
    candidate={'parsed_action':{'args':{'children':[{'contract':deepcopy(child)}]}}}
    assert same_contract_delegation(candidate,c)
    candidate['parsed_action']['args']['children'][0]['contract']['ensures'][0]['text']='smaller responsibility'
    assert not same_contract_delegation(candidate,c)
    assert not same_contract_delegation({'parsed_action':None},c)
