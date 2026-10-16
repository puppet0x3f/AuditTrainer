import ast
import json
from pathlib import Path
from audittrainer import refinement_cross_repair as study


def load_scenarios():
    r=study.check()
    return [json.loads(Path(x['scenario_path']).read_text()) for x in r['selection']]


def without_function(source,name):
    tree=ast.parse(source)
    tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!=name]
    return ast.dump(tree)


def test_fixed_selection_and_no_model_experiment():
    r=study.check()
    assert [(s['variant'],s['rep']) for s in r['selection']]==study.SELECTION
    assert r['model_calls']==r['updates']==0 and r['program_conditions']==12
    assert [s['original_root_present'] for s in r['selection']]==[True,False,True]


def test_only_designated_functions_change():
    for s in load_scenarios():
        assert without_function(s['sources'][study.PREPARE],'prepare')==without_function(s['patches'][study.PREPARE],'prepare')
        assert without_function(s['sources'][study.NODE],'_aggregate')==without_function(s['patches'][study.NODE],'_aggregate')
        for condition,nodes in study.CONDITIONS.items():
            sources=study.condition_sources(s,condition)
            assert sources['N0']==s['sources']['N0']
            assert {n for n in sources if sources[n]!=s['sources'][n]}<=set(nodes)
            for source in sources.values():ast.parse(source)


def test_wrapper_and_dependency_graph_preserved():
    for s in load_scenarios():
        if not s['original_root_present']:assert s['sources']['N0']==study.WRAPPER
        assert s['dependencies'][study.PREPARE]==[]
        assert [d['child_id'] for d in s['dependencies']['N0']]==[study.NODE]
        assert [d['child_id'] for d in s['dependencies'][study.NODE]]==[study.PREPARE]
        for n,deps in s['dependencies'].items():
            imports=[a.asname for stmt in ast.parse(s['sources'][n]).body if isinstance(stmt,ast.ImportFrom) and stmt.module.startswith('nodes.') for a in stmt.names]
            assert imports==[d['alias'] for d in deps]


def test_reference_replacement_matches_saved_reference():
    for s in load_scenarios():
        ref=json.loads(Path(s['row']['reference_path']).read_text())
        for node,name,code in [(study.PREPARE,'prepare',ref['prepare']),(study.NODE,'_aggregate',ref['aggregate'].replace('def aggregate(','def _aggregate(',1))]:
            actual=next(n for n in ast.parse(s['patches'][node]).body if isinstance(n,ast.FunctionDef) and n.name==name)
            expected=ast.parse(code).body[0]
            assert ast.dump(actual)==ast.dump(expected)
