"""PR0 probability, provenance, protocol, and terminal-reward checks."""
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from audittrainer.policy import ScriptedPolicy, ScriptStep
from audittrainer.refinement_policy import (
    MODES, PLAN_PREFIX, RefinementPolicy, RefinementRouter, atomic_plan,
    mode_log_probs, sequence_log_probs, split_body, two_level_advantages, validate_refinement,
)
from audittrainer.refinement_rollout import reward_record
from audittrainer.runtime import Limits
from audittrainer.train_refinement import load_pr0_config, select_adaptation_rows, representation_loss
from audittrainer.verifier import digest

ROOT = Path(__file__).resolve().parents[1]


def tiny():
    torch.manual_seed(11)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=40, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, attention_dropout=0.0)).eval()


def test_exact_mode_probability_and_causal_body_scores():
    model = tiny()
    prefixes, targets = [[1, 2, 3], [1, 2, 3]], [[4, 5], [6, 7]]
    actual = sequence_log_probs(model, prefixes, targets, 0)
    full = model(input_ids=torch.tensor([p+t for p,t in zip(prefixes, targets)])).logits
    expected = [-F.cross_entropy(full[i, 2:4].float(), torch.tensor(t), reduction="none")
                for i,t in enumerate(targets)]
    for a,b in zip(actual, expected):
        torch.testing.assert_close(a,b)
    mode_ids = dict(zip(MODES, targets))
    gate, _ = mode_log_probs(model, prefixes[0], mode_ids, MODES, 0)
    torch.testing.assert_close(gate, F.log_softmax(torch.stack([x.sum() for x in expected]), dim=0))
    torch.testing.assert_close(gate.exp().sum(), torch.tensor(1.0))
    repeated, _ = mode_log_probs(model, prefixes[0], mode_ids, MODES, 0)
    torch.testing.assert_close((repeated-gate).exp(), torch.ones(2), rtol=0, atol=0)
    one, _ = mode_log_probs(model, prefixes[0], mode_ids, ["ATOMIC"], 0)
    assert one.item() == 0.0


def test_representation_loss_gradients_include_normalized_gate_and_only_body_targets():
    model = tiny()
    example = {"prefix": [1,2,3], "mode_ids": {"ATOMIC": [4,5], "SPLIT": [6,7]},
               "mode": "SPLIT", "body_prefix": [1,2,3,6,7,8], "body_target": [9,10,11]}
    loss, gate, body = representation_loss(model, example, 0)
    logits = model(input_ids=torch.tensor([example["body_prefix"]+example["body_target"]])).logits
    expected_body = F.cross_entropy(logits[0, 5:8].float(), torch.tensor([9,10,11]))
    torch.testing.assert_close(body, expected_body)
    torch.testing.assert_close(loss, gate+expected_body)
    loss.backward()
    assert model.lm_head.weight.grad.norm() > 0
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_stratified_loo_expectation_and_example():
    p, returns = .4, [.85, .50, .45]
    a = two_level_advantages(p, .7, returns)
    assert a["mode"]["SPLIT"] == pytest.approx(-.06)
    assert a["mode"]["ATOMIC"] == pytest.approx(.04)
    assert a["body"] == pytest.approx([.375, -.15, -.225])
    assert a["body_weight"] == pytest.approx(p/3)
    expected, actual = [p/3*(r-.6) for r in returns], [0.0]*3
    for ids in product(range(3), repeat=3):
        advantages = two_level_advantages(p, .7, [returns[i] for i in ids])
        for i, chosen in enumerate(ids):
            for j in range(3):
                actual[j] += 1/27*advantages["body_weight"]*advantages["body"][i]*((chosen == j)-1/3)
    assert actual == pytest.approx(expected)
    for gate_p in (1e-5, .4, 1-1e-5):
        logit = torch.logit(torch.tensor(gate_p, dtype=torch.float64)).requires_grad_()
        advantages = two_level_advantages(gate_p, .7, returns)
        objective = ((1-gate_p)*advantages["mode"]["ATOMIC"]*F.logsigmoid(-logit)
                     +gate_p*advantages["mode"]["SPLIT"]*F.logsigmoid(logit))
        gradient, = torch.autograd.grad(objective, logit)
        assert gradient.item() == pytest.approx(gate_p*(1-gate_p)*(.6-.7))
    assert two_level_advantages(0, .7, returns)["body_weight"] == 0
    with pytest.raises(ValueError):
        two_level_advantages(.4, .7, [.5])


def test_real_tokenizer_and_reencoded_plan_roundtrip():
    tokenizer = AutoTokenizer.from_pretrained(ROOT/"Qwen3-8B", local_files_only=True)
    assert all(len(tokenizer.encode(m, add_special_tokens=False)) > 1 for m in MODES)
    config, _ = load_pr0_config(ROOT/"configs/refinement.yaml")
    rows, _ = select_adaptation_rows(config)
    assert len(rows) == 200
    assert Counter(r["target_action"]["args"]["mode"] for r in rows) == {"ATOMIC": 100, "SPLIT": 100}
    assert len({r["root_id"] for r in rows}) == 50
    row = next(r for r in rows if r["target_action"]["args"]["mode"] == "SPLIT")
    action = row["target_action"]
    raw = PLAN_PREFIX+'SPLIT",'+split_body(action)
    assert validate_refinement(raw, "SPLIT").model_dump(mode="json") == action
    duplicate = raw.replace('"mode":"SPLIT",', '"mode":"SPLIT","mode":"ATOMIC",')
    with pytest.raises(ValueError, match="Duplicate"):
        validate_refinement(duplicate, "SPLIT")


@pytest.mark.parametrize("kind,code", [
    ("missing", "MISSING_OBLIGATION_OWNER"),
    ("unknown", "UNKNOWN_OBLIGATION_OWNER"),
    ("duplicate", "DUPLICATE_JSON_KEY"),
])
def test_required_coverage_rejects_without_committing_plan(make_runtime, fixture_data, kind, code):
    action = deepcopy(fixture_data["script"][0]["output"])
    if kind == "missing":
        action["args"]["obligation_map"] = {}
    elif kind == "unknown":
        first = next(iter(action["args"]["obligation_map"]))
        action["args"]["obligation_map"][first] = ["child:invented"]
    raw = json.dumps(action)
    if kind == "duplicate":
        raw = raw.replace('"mode": "ATOMIC"', '"mode": "ATOMIC", "mode": "ATOMIC"')
    runtime = make_runtime(script=[{"node_id": "N0", "phase": "NEW", "output": raw}])
    runtime.require_obligation_coverage = True
    event = runtime.step()
    assert event.host_result.code == code
    assert runtime.plan is None and runtime.budget.nodes_created == 1
    assert runtime.budget.policy_calls == 1


def test_old_protocol_still_allows_missing_map(make_runtime, fixture_data):
    action = deepcopy(fixture_data["script"][0]["output"])
    action["args"]["obligation_map"] = {}
    runtime = make_runtime(script=[{"node_id": "N0", "phase": "NEW", "output": action}])
    assert runtime.step().host_result.code == "PLAN_REGISTERED"


def test_gate_and_raw_body_provenance_keeps_invalid_samples():
    planner = object.__new__(RefinementPolicy)
    planner.identity = {"policy": "test"}
    gate = {"gate_usage": {"input_tokens": 100, "output_tokens": 4},
            "mode_log_probs": {"SPLIT": -.2}}
    candidate = planner._candidate(gate, "SPLIT", PLAN_PREFIX+'SPLIT",nonsense',
                                  [1,2,3], [4,5], [-.1,-.3], False, "MODEL_CONDITIONAL_BODY")
    assert candidate["parsed_action"] is None and candidate["parse_error"]
    assert candidate["body_output_ids"] == [4,5]
    assert candidate["body_old_log_probs"] == [-.1,-.3]
    assert candidate["logical_usage"] == {"input_tokens": 103, "output_tokens": 6}


def test_split_sampling_does_not_inherit_model_temperature_defaults(monkeypatch):
    model = tiny()
    model.generation_config.update(temperature=.6, top_p=.95, top_k=20,
                                   transformers_version="4.57.1", pad_token_id=0)
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "invalid body", pad_token_id=0)
    llm = SimpleNamespace(model=model, tokenizer=tokenizer, identity={}, _autocast=nullcontext)
    planner = RefinementPolicy(llm)
    monkeypatch.setattr("audittrainer.refinement_policy.active_adapter", lambda *args: nullcontext())
    gate = {"prefix_ids": [1,2,3], "mode_token_ids": {"SPLIT": [6,7]}, "separator_ids": [8],
            "mode_probabilities": {"SPLIT": .4}, "mode_log_probs": {"SPLIT": -.916},
            "gate_usage": {"input_tokens": 6, "output_tokens": 4}, "max_body_tokens": 4}
    candidate = planner.sample_split(gate)
    expected = sequence_log_probs(model, [candidate["body_input_ids"]],
                                  [candidate["body_output_ids"]], 0)[0]
    torch.testing.assert_close(torch.tensor(candidate["body_old_log_probs"]), expected,
                               atol=1e-6, rtol=1e-6)
    assert model.generation_config.temperature == .6
    assert model.generation_config.top_p == .95
    assert len(candidate["body_output_ids"]) == 4
    assert candidate["generation_truncated"] and candidate["parse_error"]


def test_root_candidate_replay_preserves_state_outcome_and_single_budget_charge(make_f2_runtime, fixture_data):
    original, branch = make_f2_runtime(), make_f2_runtime()
    for runtime in (original, branch):
        runtime.require_obligation_coverage = True
    observation = original.build_observation()
    assert digest(branch.build_observation()) == digest(observation)
    planner = object.__new__(RefinementPolicy)
    planner.identity = {"backend": "scripted_plan"}
    gate = {"observation": observation, "observation_sha256": digest(observation),
            "gate_usage": {"input_tokens": 24, "output_tokens": 4},
            "mode_log_probs": {"ATOMIC": -.2}, "model_input": {"messages": []}}
    candidate = planner.atomic(gate)
    planner.act = lambda obs: planner.as_output(gate, candidate)
    results = []
    for runtime, initial in ((original, None), (branch, (gate, candidate))):
        code = ScriptedPolicy([ScriptStep.model_validate_json(json.dumps(fixture_data["script"][1]))])
        executor = SimpleNamespace(llm=SimpleNamespace(identity={"backend": "scripted_code"}), act=code.act)
        runtime.policy = RefinementRouter(planner, executor, initial=initial)
        results.append(runtime.run())
    assert {k:v for k,v in results[0]["usage"].items() if k != "execution_seconds"} == {
        k:v for k,v in results[1]["usage"].items() if k != "execution_seconds"}
    usage = results[0]["usage"]
    assert usage["input_tokens"] == 24
    assert usage["output_tokens"] == 4
    assert usage["policy_calls"] == 4
    assert results[0]["functional_status"] == results[1]["functional_status"] == "PASS"
    assert original.artifact.source_hash == branch.artifact.source_hash
    assert original.store.run_dir != branch.store.run_dir
    changed = deepcopy(observation)
    changed["budget_remaining"]["policy_calls"] -= 1
    router = RefinementRouter(planner, executor, initial=(gate, candidate))
    with pytest.raises(ValueError, match="not bound"):
        router.act(changed)


def test_unknown_is_not_negative_and_only_bound_violation_changes_reward():
    scores = {"private": {"status": "PASS", "suites": [{"responsibility_status": "UNKNOWN", "evidence_path": "e1"}]},
              "visible": {"status": "PASS", "suites": []}}
    budget = {"input_tokens": 100, "output_tokens": 50, "policy_calls": 4, "verify_calls": 1}
    config = {"reward": {"cost_weight": 0.0, "confirmed_violation_weight": .2}}
    result = reward_record(scores, budget, Limits(), config)
    assert result["reward"] == 1 and result["v"] == 0
    assert result["responsibility_status"] == "UNKNOWN"
    scores["private"]["suites"][0]["responsibility_status"] = "VIOLATION"
    result = reward_record(scores, budget, Limits(), config)
    assert result["reward"] == -.2 and result["violation_evidence"] == ["e1"]
    scores["private"]["suites"][0]["responsibility_status"] = "UNKNOWN"
    scores["private"]["status"] = "UNKNOWN"
    assert reward_record(scores, budget, Limits(), config)["reward"] == 0
