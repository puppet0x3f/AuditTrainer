import ast
from copy import deepcopy
from pathlib import Path
import pytest
from audittrainer.refinement_transfer_repair import repair_prepare,repair_aggregate,NODE,PREPARE
from audittrainer.refinement_internal_transfer_tasks import CASES,oracle

BASE=Path('runs/refinement/internal_transfer')


@pytest.mark.parametrize('rep',range(4))
def test_prepare_repair_fulfills_product_contract_without_input_mutation(rep):
    source=(BASE/'validation_internal_transfer_polynomial_joint/SPLIT'/f'repeat_{rep:02d}'/'runtime/artifacts'/PREPARE/'v1.py').read_text()
    fixed=repair_prepare(source,rep);ns={};exec(fixed,ns)
    for raw in CASES['polynomial_joint']:
        items=deepcopy(raw);prepared,_=oracle('polynomial_joint',deepcopy(raw))
        assert ns['prepare'](items)==prepared and items==raw


@pytest.mark.parametrize('variant',['coverage_redundancy','polynomial_joint'])
@pytest.mark.parametrize('rep',range(4))
def test_aggregate_repair_fulfills_new_contract_and_preserves_composition(variant,rep):
    source=(BASE/('validation_internal_transfer_'+variant)/'SPLIT'/f'repeat_{rep:02d}'/'runtime/artifacts'/NODE/'v1.py').read_text()
    fixed=repair_aggregate(source,variant,rep);tree=ast.parse(fixed)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_aggregate')
    ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),'<aggregate>','exec'),ns)
    for raw in CASES[variant]:
        prepared,expected=oracle(variant,deepcopy(raw));saved=deepcopy(prepared)
        assert ns['_aggregate'](prepared)==expected and prepared==saved
    old_compute=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='compute')
    new_compute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute')
    if variant=='polynomial_joint' and rep==2:old_compute.body.pop(2)
    assert ast.dump(old_compute)==ast.dump(new_compute)
