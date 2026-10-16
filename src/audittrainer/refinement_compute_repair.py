"""Offline, fixed-plan causal repairs; no model training or autonomous claims."""
import difflib
import json
from pathlib import Path

from .artifacts import ArtifactStore
from .dataset import load_root_manifest, read_jsonl, file_hash
from .evaluate import load_task, score_program, write_json
from .schema import Contract, Dependency

ROOT = Path('runs/refinement/compute_repair')
PRIOR = Path('runs/refinement/compute_values')
NODE = 'N0_p1_compute'
WRAPPER = '''import json
from nodes.N0_p1_compute import compute as _compute_child

def solve(text):
    items = json.loads(text)
    result = _compute_child(items)
    return json.dumps(result, separators=(',', ':'))
'''


def repair(rid, rep, source):
    if 'calendar' in rid:
        changes = [('sum(_aggregate_child(prepared))', 'sum(value for value in prepared if value % 4 == 0)')] if rep in (0, 2) else [
            ('return {i for i, value in enumerate(prepared) if value % 4 == 0}',
             "return {'indices': [i for i, value in enumerate(prepared) if value % 4 == 0], 'total': sum(value for value in prepared if value % 4 == 0)}")]
    elif 'intervals' in rid:
        changes = {
            0: [("return {'indices': totals[0], 'total': totals[1]}", 'return totals'),
                ('sum(_base(prepared))', 'sum(value for value in prepared if value % 4 == 0)')],
            1: [('return [sum(row[i] for row in prepared) for i in range(len(prepared[0]))]', 'return prepared')],
            2: [('list(zip(prepared, _base(prepared)))', 'list(enumerate(prepared))')],
            3: [('list(zip(prepared, _get_column_sums(prepared)))', 'list(enumerate(_get_column_sums(prepared)))'),
                ('for i, (value, _) in totals', 'for i, value in totals'),
                ('return [sum(row[j] for row in items) for j in range(len(items[0]))]', 'return items')],
        }[rep]
    elif 'chunks' in rid:
        changes = [('return [prepared[i:i+4] for i in range(0, len(prepared), 4)]',
                    "return {'chunks': [prepared[i:i+4] for i in range(0, len(prepared), 4)], 'length': len(prepared)}")]
    else:
        assert 'words' in rid
        changes = [('return json.dumps(result)', 'return result')]
    for old, new in changes:
        assert source.count(old) == 1, (rid, rep, old)
        source = source.replace(old, new)
    return source


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    request = json.loads((PRIOR/'request.json').read_text())
    prior = json.loads((PRIOR/'result.json').read_text())
    rows = load_root_manifest(request['config']['root_manifest'])
    tests = {x['root_id']: x for x in read_jsonl(request['config']['test_manifest'])}
    results = []
    for group in prior['groups']:
        rid = group['root_id']
        for case in group['cases']:
            if case['arm'] != 'SPLIT' or case['case']['reward']['y']:
                continue
            rep = case['rep']; original = Path(case['case']['run_dir'])/'runtime'
            events = read_jsonl(original/'trace.jsonl')
            contracts = {e['node_id']: Contract.model_validate(e['observation_before']['incoming_contract'])
                         for e in events if e['host_result']['code'] == 'SOURCE_SAVED'}
            contracts['N0'] = Contract.model_validate(next(e['observation_before']['incoming_contract'] for e in events if e['node_id'] == 'N0'))
            old_store = ArtifactStore(original)
            nodes = sorted(contracts, key=lambda n: (-n.count('_'), n))
            present = {n: old_store.latest_version(n) for n in nodes}
            old_artifacts = {n: old_store.load(n, v) for n, v in present.items() if v}
            sources = {n: (original/a.module_path).read_text() for n, a in old_artifacts.items()}
            root_present = 'N0' in sources
            if not root_present:
                # Identical root-contract harness in BOTH conditions; never claim autonomous completion.
                sources['N0'] = WRAPPER
            target = NODE+'_p1_aggregate' if 'intervals' in rid else NODE
            patched = repair(rid, rep, sources[target])
            base = ROOT/rid/f'repeat_{rep:02d}'
            base.mkdir(parents=True, exist_ok=False)
            (base/'intervention.diff').write_text(''.join(difflib.unified_diff(
                sources[target].splitlines(True), patched.splitlines(True), fromfile=target+'/ORIGINAL', tofile=target+'/PATCHED')))
            _, _, _, _, verifier = load_task(rows[rid], tests, request['framework'])
            conditions = {}
            for condition in ('ORIGINAL', 'PATCHED'):
                out = base/condition; out.mkdir()
                store = ArtifactStore(out); artifacts = {}
                for n in nodes:
                    old_deps = old_artifacts[n].dependency_bindings if n in old_artifacts else [Dependency(
                        child_id=NODE, artifact_version=old_artifacts[NODE].version,
                        artifact_hash=old_artifacts[NODE].source_hash, alias='_compute_child')]
                    deps = [Dependency(child_id=d.child_id, artifact_version=artifacts[d.child_id].version,
                                       artifact_hash=artifacts[d.child_id].source_hash, alias=d.alias) for d in old_deps]
                    code = patched if condition == 'PATCHED' and n == target else sources[n]
                    artifacts[n] = store.save_initial(n, contracts[n], code, deps)
                local = {}
                for n in nodes:
                    if n == 'N0': continue
                    local[n] = []
                    for suite in verifier.suites.values():
                        if suite.check_kind == 'LOCAL' and suite.applies(rid, contracts[n]):
                            ev = verifier.verify(out, rid, contracts[n], artifacts[n], suite.suite_id,
                                                 f'LOCAL_{n}_{suite.visibility}', allow_private=True)
                            local[n].append({'visibility': suite.visibility, 'status': ev.status,
                                             'cases': ev.details['executed_cases']})
                scores, _ = score_program({'path': str(out), 'contract': contracts['N0'].model_dump(mode='json'),
                                           'artifact': artifacts['N0'].model_dump(mode='json')}, verifier, rid)
                usage = []
                for suite in scores['private']['suites']:
                    usage.extend(json.loads(Path(suite['evidence_path']).read_text())['details'].get('dependency_usage', []))
                conditions[condition] = {'local': local, 'root_scores': scores,
                    'root_called_compute': any(x['child_id'] == NODE and x.get('called') for x in usage),
                    'source_hashes': {n: a.source_hash for n, a in artifacts.items()}}
            changed = [n for n in nodes if conditions['ORIGINAL']['source_hashes'][n] != conditions['PATCHED']['source_hashes'][n]]
            assert changed == [target]
            assert all(file_hash(original/a.module_path) == a.source_hash for a in old_artifacts.values())
            results.append({'root_id': rid, 'rep': rep, 'source': str(original), 'target': target,
                            'original_root_present': root_present, 'conditions': conditions,
                            'preserved_nodes': [n for n in nodes if n != target], 'diff': str(base/'intervention.diff')})
    assert len(results) == 11
    summary = []
    for present in (True, False):
        cases = [x for x in results if x['original_root_present'] == present]
        item = {'original_root_present': present, 'n': len(cases)}
        for arm in ('ORIGINAL', 'PATCHED'):
            item[arm] = {'root_pass': sum(x['conditions'][arm]['root_scores']['private']['status']=='PASS' for x in cases),
                         'compute_private_pass': sum(any(v['visibility']=='PRIVATE' and v['status']=='PASS' for v in x['conditions'][arm]['local'][NODE]) for x in cases),
                         'root_called_compute': sum(x['conditions'][arm]['root_called_compute'] for x in cases)}
        summary.append(item)
    write_json(ROOT/'result.json', {'status': 'COMPLETE', 'model_calls': 0, 'updates': 0, 'summary': summary,
        'scope': 'All 11 failed SPLIT artifacts; 7 unchanged existing roots, 4 missing roots tested using the same explicit harness in both conditions. Offline human causal diagnosis, not autonomous protocol replay or model score.',
        'results': results})
    print(json.dumps(summary))


if __name__ == '__main__':
    run()
