from copy import deepcopy
import json
from pathlib import Path
import pytest
from audittrainer.refinement_handoff import direct_binding


def observation():
    p=Path('runs/refinement/internal_probe/train_execution_polynomial_matrix_05/SPLIT/repeat_00/runtime/trace.jsonl')
    return next(e['observation_before'] for e in map(json.loads,p.read_text().splitlines()) if e['node_id']=='N0' and e['observation_before']['phase']=='RESOLVE_CHILD')


def test_complete_binding_uses_direct_child_not_nested_dependency():
    o=observation();b=direct_binding(o);a=o['returned_child']['result']['artifact'];d=a['dependency_bindings'][0]
    assert b['child_id']==a['node_id']!=d['child_id']
    assert b['artifact_hash']==a['source_hash']!=d['artifact_hash']
    assert b['artifact_version']==a['version']
    assert b['evidence_ids']==[o['returned_child']['result']['evidence']['evidence_id']]


@pytest.mark.parametrize('change',['missing','wrong_hash','wrong_evidence_node'])
def test_no_fabricated_or_mixed_binding(change):
    o=deepcopy(observation());r=o['returned_child']['result']
    if change=='missing':r['artifact']=None
    elif change=='wrong_hash':r['artifact']['source_hash']='wrong'
    else:r['evidence']['binding']['artifact']['node_id']='descendant'
    with pytest.raises(AssertionError):direct_binding(o)
