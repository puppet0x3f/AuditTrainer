from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from audittrainer.refinement_internal_probe import FixedInternal
from audittrainer.verifier import digest

@pytest.mark.parametrize('rid', ['train_execution_polynomial_matrix_05', 'train_executor_check_paths_matrix_07'])
def test_injection_is_bound_to_state_and_only_first_decision(rid):
    state=json.loads((Path('runs/refinement/internal_probe')/rid/'state.json').read_text())
    base=SimpleNamespace(cost={},identity={},as_output=lambda g,c:'injected',act=lambda obs:'natural_retry')
    gate={'observation_sha256':digest(state['observation'])}
    policy=FixedInternal(base,state,gate,{})
    changed=deepcopy(state['observation']);changed['budget_remaining']['tokens']-=1
    with pytest.raises(AssertionError):policy.act(changed)
    root=policy.act(state['root_event']['observation_before'])
    assert root.raw_text==state['root_event']['raw_output']
    assert root.token_usage.model_dump()==state['root_event']['usage']
    assert policy.act(state['observation'])=='injected'
    assert policy.act(state['observation'])=='natural_retry'
