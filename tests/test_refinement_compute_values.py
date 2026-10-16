import json
from pathlib import Path
from audittrainer.refinement_body_transfer import reviewed_plans
from audittrainer.refinement_curriculum_rl import plan_key
from audittrainer.refinement_responsibility_course import allocation
from audittrainer.refinement_post_handoff_values import same_contract_delegation
from audittrainer.dataset import load_root_manifest


def test_review_distinguishes_responsibility_allocation_from_delegation():
    r=json.loads(Path('runs/refinement/compute_values/request.json').read_text());rows=load_root_manifest(r['config']['root_manifest'])
    for s in r['selection']:
        obs=json.loads(Path(s['state_path']).read_text())['observation'];d=json.loads(Path(rows[s['root_id']]['fixture_path']).read_text())
        assert {o['id'] for o in obs['incoming_contract']['ensures']}=={'PREPARE','AGGREGATE'}
        reviewed=reviewed_plans(obs,d)
        for variant in ['parent_aggregate','parent_prepare','pipeline','fused']:
            a=allocation(obs['incoming_contract'],d,variant)
            assert reviewed[plan_key(a,d)]==variant
            assert same_contract_delegation({'parsed_action':a},obs['incoming_contract'])==(variant=='fused')
