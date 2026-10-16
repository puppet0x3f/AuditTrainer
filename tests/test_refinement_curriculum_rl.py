"""The preference probe must match responsibilities as well as child count."""
from copy import deepcopy

from audittrainer.refinement_curriculum import plans, transfer_definition
from audittrainer.refinement_curriculum_rl import plan_key


def test_probe_matches_complete_allocation_not_only_child_count():
    definition, _, _ = transfer_definition('inventory', 'ledger', 3)
    reviewed = plans(definition)
    good = reviewed['parent_aggregate']
    renamed = deepcopy(good)
    renamed['args']['children'][0]['local_key'] = 'component'
    renamed['args']['obligation_map']['PREPARE'] = ['child:component']
    assert plan_key(renamed, definition) == plan_key(good, definition)
    assert len(good['args']['children']) == len(reviewed['fused']['args']['children']) == 1
    assert plan_key(good, definition) != plan_key(reviewed['fused'], definition)
    renamed['args']['obligation_map']['PREPARE'] = ['self']
    assert plan_key(renamed, definition) != plan_key(good, definition)
    renamed['args']['children'][0]['contract']['requires'].append('Different input assumption')
    assert plan_key(renamed, definition) is None
