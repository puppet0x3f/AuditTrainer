"""Bounded first-compute gate calibration with frozen continuation policies."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path('runs/refinement/internal_gate')
SOURCE = Path('runs/refinement/complex_refresh')
NODE = 'N0_p1_compute'
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] in {'train', 'natural'}:
    from .gpu_resource import acquire
    _reservation = acquire(ROOT / 'gpu_resources.json', 'train' if sys.argv[1] == 'train' else 'collect')

from .dataset import file_hash, load_root_manifest, read_jsonl
from .evaluate import write_json
from .refinement_generalization import guarded_json
from .refinement_policy import RefinementPolicy, load_policies, active_adapter, adapter_fingerprint, mode_log_probs
from .refinement_internal_probe import FixedInternal
from .refinement_rollout import run_case
from .verifier import digest


def target_for(delta, prior, scoring_error=False, beta=0.1):
    if scoring_error or abs(delta) <= 0.25:
        return prior
    return 1 / (1 + math.exp(-(math.log(prior) - math.log1p(-prior) + delta / beta)))


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'request.json').exists():
        return
    source = json.loads((SOURCE / 'request.json').read_text())
    result = json.loads((SOURCE / 'result.json').read_text())
    assert result['status'] == 'COMPLETE'
    targets = []
    for summary, group in zip(result['summary'], result['groups']):
        assert summary['root_id'] == group['root_id']
        scoring_error = any(x['case']['errors'] for x in group['cases'])
        delta = summary['delta_success']; prior = summary['p_split']
        targets.append({'root_id': summary['root_id'], 'q_atomic': summary['ATOMIC']['success']/4,
                        'q_split_raw': summary['SPLIT']['success']/4, 'delta_raw': delta,
                        'initial_p_split': prior, 'scoring_error': scoring_error,
                        'target_p_split': target_for(delta, prior, scoring_error),
                        'direction': 'ANCHOR' if scoring_error or abs(delta) <= 0.25 else 'SPLIT' if delta > 0 else 'ATOMIC'})
    guarded_json(ROOT / 'request.json', {**source, 'config': {**source['config'], 'output_dir': str(ROOT)},
        'source': str(SOURCE), 'source_result_sha256': file_hash(SOURCE/'result.json'),
        'targets': targets, 'steps': 32, 'learning_rate': 1e-5, 'kl_beta': 0.1,
        'gate_seed': 510000713, 'natural_seed': 513000713, 'natural_cases': 32,
        'objective': 'Equal exposure soft mode CE. Positive targets maximize p*QS+(1-p)*QA-0.1*KL(p||original BODY prior). Tied/uncertain states anchor original prior; stock scoring ERROR is never a negative reward. Cost weight remains zero.',
        'design': 'Independent BODY copy for the first compute gate only. Replay fixed root plan. Freeze root gate, all SPLIT bodies, executor, all other node decisions and subsequent retries. One fixed 32-step final checkpoint. BEFORE/AFTER each 4 states x 4 fresh paired seeds.',
        'decision': 'Assess positive-state probability changes, anchor drift, natural root success and split locations/costs. Known development fit, not held-out generalization.',
        'updates': 32, 'cases': 32})
    guarded_json(ROOT / 'gpu_resources.json', {'devices': ['0','1','2'], 'min_free_mib': {'train': 22000, 'collect': 16000}})
    print(json.dumps(targets))


class InternalGatePolicy(RefinementPolicy):
    def __init__(self, llm):
        super().__init__(llm)
        self.body = RefinementPolicy(llm, adapter_name='body')
        self.root_gate = RefinementPolicy(llm, adapter_name='root_gate')
        self.body.cost = self.root_gate.cost = self.cost
        self.identity = {**self.identity, 'policy': 'independent_first_compute_gate',
                         'internal_gate_adapter': 'default', 'root_gate_adapter': 'root_gate', 'body_adapter_slot': 'body'}

    def sample_split(self, gate):
        return self.body.sample_split(gate)

    def continuation(self, observation):
        gate = self.root_gate.score_modes(observation) if observation['node_id'] == 'N0' else self.body.score_modes(observation)
        return self.sample_action(gate)

    def sample_action(self, gate):
        import torch
        modes = list(gate['mode_probabilities'])
        mode = modes[int(torch.multinomial(torch.tensor([gate['mode_probabilities'][m] for m in modes]), 1))]
        candidate = self.atomic(gate) if mode == 'ATOMIC' else self.sample_split(gate)
        return self.as_output(gate, candidate)


class NaturalInternal(FixedInternal):
    """Natural first internal choice; later attempts stay on the value-collection policy."""
    def __init__(self, base, state):
        super().__init__(base, state, None, None, node=NODE)
        self.identity = {**base.identity, 'intervention': 'fixed_root_natural_first_compute'}

    def act(self, obs):
        if obs['node_id'] == 'N0' and not self.root_sent:
            return super().act(obs)
        if obs['node_id'] == self.node and not self.internal_sent:
            assert digest(obs) == digest(self.state['observation'])
            self.internal_sent = True
            return self.base.act(obs)
        return self.base.continuation(obs)


def fingerprints(policy):
    return {name: adapter_fingerprint(policy.model, name) for name in ['default', 'root_gate', 'body', 'executor']}


def context(method='BEFORE'):
    r = json.loads((ROOT/'request.json').read_text())
    a = r['adapters']; initial = a['body'] if method == 'BEFORE' else str(ROOT/'training/final_adapter')
    base, executor = load_policies(r['framework'], initial, a['executor'])
    for name, path in [('body', a['body']), ('root_gate', a['gate'])]:
        base.model.load_adapter(path, adapter_name=name, is_trainable=False, local_files_only=True)
    base.model.set_adapter('default'); base.model.requires_grad_(False)
    policy = InternalGatePolicy(base.llm)
    policy.identity['frozen_adapter_paths'] = json.dumps(a, sort_keys=True)
    rows = load_root_manifest(r['config']['root_manifest'])
    tests = {x['root_id']: x for x in read_jsonl(r['config']['test_manifest'])}
    return r, policy, executor, rows, tests


def train():
    import torch
    from transformers import set_seed
    from .refinement_pr1 import enable_planner
    from .train_sft import trainable_snapshots, changed_tensors, prefix_forward
    r, policy, _, _, _ = context(); before = fingerprints(policy)
    set_seed(r['gate_seed']); model = policy.model; tok = policy.tokenizer
    states = [json.loads(Path(s['state_path']).read_text()) for s in r['selection']]
    gates = [policy.score_modes(s['observation']) for s in states]
    root_before = [policy.root_gate.score_modes(s['root_event']['observation_before'])['mode_probabilities'] for s in states]
    for gate, target in zip(gates, r['targets']):
        assert abs(gate['mode_probabilities']['SPLIT']-target['initial_p_split']) < 1e-6
    out = ROOT/'training'; out.mkdir(exist_ok=False)
    write_json(out/'request.json', {'steps': r['steps'], 'lr': r['learning_rate'], 'targets': r['targets']})
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0.
    model.peft_config['default'].lora_dropout = 0.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads(); model.config.use_cache = False; enable_planner(model)
    snapshots, trainable = trainable_snapshots(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=r['learning_rate'], weight_decay=0.)
    history = []
    for step in range(r['steps']):
        enable_planner(model); optimizer.zero_grad(set_to_none=True); losses = []
        for gate, record in zip(gates, r['targets']):
            target = record['target_p_split']; modes = list(gate['mode_log_probs'])
            with policy.llm._autocast():
                lp, _ = mode_log_probs(model, gate['prefix_ids'], gate['mode_token_ids'], modes, tok.pad_token_id)
                loss = -target*lp[modes.index('SPLIT')]-(1-target)*lp[modes.index('ATOMIC')]
            assert torch.isfinite(loss)
            (loss/len(gates)).backward(); losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(params, 1.); assert torch.isfinite(norm) and norm > 0
        optimizer.step(); item = {'step': step+1, 'per_state_ce': losses, 'gradient_norm': float(norm)}
        history.append(item); print(json.dumps(item), flush=True)
    final = out/'final_adapter'; model.save_pretrained(final, selected_adapters=['default']); tok.save_pretrained(final)
    changes = changed_tensors(model, snapshots); original = prefix_forward(model, gates[0]['prefix_ids'])
    model.load_adapter(str(final), adapter_name='reload_check', is_trainable=False, local_files_only=True)
    with active_adapter(model, 'reload_check'): restored = prefix_forward(model, gates[0]['prefix_ids'])
    reload_delta = float((original-restored).abs().max()); assert reload_delta <= 1e-5
    final_probs = {s['root_id']: policy.score_modes(g['observation'])['mode_probabilities']['SPLIT'] for s,g in zip(r['selection'],gates)}
    root_after = [policy.root_gate.score_modes(s['root_event']['observation_before'])['mode_probabilities'] for s in states]
    assert root_before == root_after
    after = fingerprints(policy)
    assert all(before[k] == after[k] for k in ['root_gate','body','executor']) and before['default'] != after['default']
    probes = []
    for i,(s,g) in enumerate(zip(r['selection'], gates)):
        set_seed(r['plan_seed']+i*100); candidate = policy.sample_split(g)
        prior = json.loads((SOURCE/s['root_id']/'candidate_00.json').read_text())
        assert candidate['body_output_ids'] == prior['body_output_ids'] and candidate['raw_action'] == prior['raw_action']
        probes.append({'root_id': s['root_id'], 'tokens_identical': True})
    write_json(out/'result.json', {'status': 'COMPLETE', 'steps': r['steps'], 'exposures': r['steps']*4,
        'history': history, 'final_probs': final_probs, 'frozen_before': before, 'frozen_after': after,
        'root_mode_probabilities_unchanged': True, 'reload_max_abs_difference': reload_delta,
        'parameter_changes': changes, 'trainable_parameters': trainable, 'body_checks': probes,
        'adapter_sha256': file_hash(final/'adapter_model.safetensors')})


def natural(method, shard):
    r, policy, executor, rows, tests = context(method); before = fingerprints(policy); cases = []; gates = []
    for i,s in enumerate(r['selection']):
        if i % 3 != shard: continue
        rid = s['root_id']; state = json.loads(Path(s['state_path']).read_text()); base = ROOT/'natural'/method/rid
        base.mkdir(parents=True, exist_ok=True)
        gate = policy.score_modes(state['observation']); write_json(base/'gate.json', gate)
        gates.append({'root_id': rid, 'p_split': gate['mode_probabilities']['SPLIT']})
        for rep in range(4):
            fixed = NaturalInternal(policy, state)
            directory = base/f'repeat_{rep:02d}'; cached = (directory/'case_result.json').exists()
            c = run_case(rows[rid], tests, r['framework'], r['config'], fixed, executor, directory, r['natural_seed']+i*100+rep)
            assert c['status'] == 'COMPLETE' or c['errors'] == ['SCORING_ERROR']
            if not cached: assert fixed.root_sent and fixed.internal_sent
            events = read_jsonl(Path(c['run_dir'])/'runtime/trace.jsonl')
            e = next(e for e in events if e['event_kind']=='POLICY' and e['node_id']==NODE)
            assert e['observation_before'] == state['observation']
            cases.append({'case': c, 'rep': rep, 'mode': e['model_input']['refinement']['selected_mode'],
                          'first_registered': e['host_result']['code']=='PLAN_REGISTERED',
                          'internal_decisions': [{'node': e['node_id'], 'mode': e['model_input']['refinement']['selected_mode']}
                            for e in events if e['event_kind']=='POLICY' and 'refinement' in (e.get('model_input') or {}) and e['node_id']!='N0']})
            print(json.dumps({'method': method, 'root_id': rid, 'rep': rep, 'mode': cases[-1]['mode'], 'root': c['final_scores']['private']['status']}), flush=True)
    assert fingerprints(policy) == before
    write_json(ROOT/'natural'/method/f'shard_{shard}.json', {'status': 'COMPLETE', 'cases': cases, 'gates': gates, 'frozen': before})


def report():
    r = json.loads((ROOT/'request.json').read_text()); groups = {}; frozen = {}; probs = {}
    for method in ['BEFORE','AFTER']:
        pieces = [json.loads((ROOT/'natural'/method/f'shard_{s}.json').read_text()) for s in range(3)]
        assert all(x['frozen']==pieces[0]['frozen'] for x in pieces)
        frozen[method] = pieces[0]['frozen']; groups[method] = [c for p in pieces for c in p['cases']]
        probs[method] = {g['root_id']:g['p_split'] for p in pieces for g in p['gates']}
        assert len(groups[method]) == 16
    assert all(frozen['BEFORE'][k]==frozen['AFTER'][k] for k in ['root_gate','body','executor'])
    by_root = []
    for s in r['selection']:
        rid = s['root_id']; item = {'root_id':rid}; pairs = {}
        for method, cs in groups.items():
            chosen = [x for x in cs if x['case']['root_id']==rid]; assert len(chosen)==4
            pairs[method] = {x['case']['seed']:x['case']['final_scores']['private']['status']=='PASS' for x in chosen}
            item[method] = {'success':sum(pairs[method].values()), 'split':sum(x['mode']=='SPLIT' for x in chosen),
                'p_split':probs[method][rid], 'mean_tokens':sum(x['case']['usage']['input_tokens']+x['case']['usage']['output_tokens'] for x in chosen)/4,
                'scoring_errors':sum(x['case']['final_scores']['private']['status']=='ERROR' for x in chosen)}
        assert pairs['BEFORE'].keys()==pairs['AFTER'].keys()
        item['wins'] = sum(pairs['AFTER'][k]>v for k,v in pairs['BEFORE'].items())
        item['losses'] = sum(pairs['AFTER'][k]<v for k,v in pairs['BEFORE'].items()); by_root.append(item)
    assert all(file_hash(Path(p)/'adapter_model.safetensors')==r['hashes'][k] for k,p in r['adapters'].items())
    totals = {m:{'success':sum(x[m]['success'] for x in by_root), 'split':sum(x[m]['split'] for x in by_root),
                   'mean_tokens':sum(x[m]['mean_tokens'] for x in by_root)/4, 'runs':16} for m in groups}
    result = {'status':'COMPLETE', 'natural_cases':32, 'by_root':by_root, 'totals':totals, 'choices':groups,
              'frozen':frozen, 'training':json.loads((ROOT/'training/result.json').read_text()),
              'scope':r['scope'], 'targets':r['targets']}
    write_json(ROOT/'result.json',result); print(json.dumps({'by_root':by_root,'totals':totals}))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare','train','natural','report'])
    p.add_argument('--method', choices=['BEFORE','AFTER'], default='BEFORE'); p.add_argument('--shard',type=int,default=0); a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='train': train()
    elif a.command=='natural': natural(a.method,a.shard)
    else: report()
