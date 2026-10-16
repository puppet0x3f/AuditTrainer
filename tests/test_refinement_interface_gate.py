import json
from pathlib import Path
from audittrainer.refinement_interface_gate import interface_note,shape,FixedPlanner


def test_public_interface_shapes_preserve_native_string_and_object():
    definition=json.loads(Path('data/refinement/responsibility_course/train_course_chunks_chunks_04/task.json').read_text())
    observation={'incoming_contract':{**definition['local_contracts']['compute'],'contract_id':'x','version':1}}
    note=interface_note(observation,definition)
    assert note['current_function']['returns']=='dict {chunks: list, length: int}'
    interfaces={e['function']:e for e in note['public_interfaces']}
    assert interfaces['prepare']['returns']=='str (native string data)'
    assert interfaces['aggregate']['parameters']['prepared']=='str (native string data)'
    assert 'aaabbbb' not in json.dumps(note)
    # A private suite must never change the prompt.
    definition['suites'].append({'visibility':'PRIVATE','expected':'SECRET'})
    assert interface_note(observation,definition)==note


def test_shapes_do_not_promote_dynamic_keys_or_sample_values():
    assert shape([{'x':1},{'y':2}])=='dict'
    assert shape([[1,2],[]])=='list'
