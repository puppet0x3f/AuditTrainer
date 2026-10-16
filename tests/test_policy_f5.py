"""Interface tests use a marked fake backend; actual Qwen smoke is reported separately."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig

from audittrainer.policy import LLMPolicy
from audittrainer.prompts import PromptOverflow, prepare_prompt, public_observation, render_messages


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(Path(__file__).resolve().parents[1] / "Qwen3-8B", local_files_only=True)


@pytest.fixture
def generation_tokens():
    return {"plan": 2048, "write_code": 3072, "repair": 3072, "control": 512, "localize": 768}


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, tokenizer, outputs):
        self.tokenizer = tokenizer
        self.outputs = iter(outputs)
        self.calls = []
        self.generation_config = GenerationConfig(eos_token_id=[151645, 151643], pad_token_id=151643)

    def eval(self):
        return self

    def generate(self, *, input_ids, attention_mask, generation_config):
        self.calls.append((input_ids.tolist()[0], generation_config.max_new_tokens))
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert not torch.is_grad_enabled()
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            value = json.dumps(value)
        ids = self.tokenizer.encode(value, add_special_tokens=False) + [151645]
        ids = ids[:generation_config.max_new_tokens]
        return torch.cat([input_ids, torch.tensor([ids])], dim=1)


@pytest.fixture
def make_policy(tokenizer, generation_tokens):
    def make(outputs, context_tokens=8192, quotas=None):
        return LLMPolicy(FakeModel(tokenizer, outputs), tokenizer, identity={"backend": "test_fake"},
                         context_tokens=context_tokens, generation_tokens=quotas or generation_tokens,
                         decoding={"do_sample": False})
    return make


def test_template_exact_prefix_max_allowed_reserve_and_public_surface(make_runtime, tokenizer, generation_tokens):
    obs = make_runtime().build_observation()
    obs.update(script="GOLD_SECRET", reference_code="TEACHER_SECRET", hidden_tests="PRIVATE_SECRET",
               future_result="FUTURE_SECRET")
    obs["allowed_actions"] = ["write_code", "abstain", "localize"]
    prepared = prepare_prompt(tokenizer, obs, context_tokens=8192, generation_tokens=generation_tokens)
    assert prepared.max_new_tokens == 3072
    assert prepared.token_ids == tokenizer.apply_chat_template(
        prepared.record["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert prepared.record["prompt"].endswith("<think>\n\n</think>\n\n")
    assert all(secret not in prepared.record["prompt"] for secret in
               ["GOLD_SECRET", "TEACHER_SECRET", "PRIVATE_SECRET", "FUTURE_SECRET"])
    body = json.loads(render_messages(obs)[1]["content"])
    assert body["observation"]["incoming_contract"] == obs["incoming_contract"]
    assert "WriteCodeAction" in body["action_schema"]["$defs"]
    assert "PlanAction" not in body["action_schema"]["$defs"]


def test_overflow_preserves_entire_contract_source_and_ticket(make_runtime, tokenizer, generation_tokens):
    obs = make_runtime().build_observation()
    obs["own_code"] = "def solve(text):\n    # " + "large " * 10000
    obs["repair_ticket"] = {"ticket_id": "T1", "editable_files": ["nodes/N0.py"]}
    with pytest.raises(PromptOverflow) as failure:
        prepare_prompt(tokenizer, obs, context_tokens=8192, generation_tokens=generation_tokens)
    assert failure.value.code == "CONTEXT_OVERFLOW"
    visible = json.loads(failure.value.model_input["messages"][1]["content"])["observation"]
    for key in ["incoming_contract", "own_code", "repair_ticket"]:
        assert visible[key] == obs[key]


def test_compression_removes_duplicate_binding_but_keeps_failed_case(make_runtime, tokenizer, generation_tokens):
    obs = make_runtime().build_observation()
    evidence = {"evidence_id": "E1", "fingerprint": "f", "visibility": "VISIBLE", "status": "FAIL",
                "binding": {"duplicate": "metadata " * 12000},
                "details": {"cases": [{"input": "-1", "actual": "1", "expected": "-1"}]}}
    obs["visible_evidence"] = [evidence]
    obs["last_action_result"] = {"code": "VERIFIED", "details": {"evidence": deepcopy(evidence)}}
    original = deepcopy(obs)
    prepared = prepare_prompt(tokenizer, obs, context_tokens=8192, generation_tokens=generation_tokens)
    assert prepared.record["compression"] == "evidence_bindings_and_duplicates"
    visible = json.loads(prepared.record["messages"][1]["content"])["observation"]
    assert visible["visible_evidence"][0]["details"] == evidence["details"]
    assert visible["last_action_result"]["details"]["evidence"] == {"evidence_id": "E1", "see": "visible_evidence"}
    assert obs == original


def test_private_evidence_is_rejected(make_runtime):
    obs = make_runtime().build_observation()
    obs["visible_evidence"] = [{"visibility": "PRIVATE", "expected": "secret"}]
    with pytest.raises(ValueError, match="Private evidence"):
        public_observation(obs)


@pytest.mark.parametrize("raw", ["```json\n{}\n```", '{"action":"finish","args":{"artifact_version":1}} {}',
                                '<think>text</think>{"action":"finish","args":{"artifact_version":1}}'])
def test_policy_does_not_extract_or_repair_json(make_runtime, make_policy, raw):
    policy = make_policy([raw])
    output = policy.act(make_runtime().build_observation())
    assert output.raw_text == raw and output.parsed_action is None
    assert output.token_usage.source == "model"
    assert output.token_usage.input_tokens == len(policy.model.calls[0][0])
    assert output.token_usage.output_tokens == len(output.output_token_ids)
    assert output.output_token_ids[-1] == 151645
    assert output.model_identity == {"backend": "test_fake"}


def test_truncation_is_preserved_and_not_dispatched(make_runtime, make_policy):
    quotas = {"plan": 3, "control": 3}
    policy = make_policy(['{"action":"plan","args":{}}'], quotas=quotas)
    runtime = make_runtime(policy=policy)
    event = runtime.step()
    assert event.host_result.code == "TRUNCATED_OUTPUT"
    assert event.generation_truncated and event.parsed_action is None
    assert event.usage.output_tokens == 3 and runtime.plan is None


@pytest.mark.parametrize("kind", ["context", "root_budget", "backend"])
def test_model_failure_records_input_identity_and_terminal_result(make_runtime, make_policy, kind):
    from audittrainer.runtime import Limits
    policy = make_policy([RuntimeError("test generation failed")], context_tokens=100 if kind == "context" else 8192)
    runtime = make_runtime(policy=policy, limits=Limits(max_total_tokens=100 if kind == "root_budget" else 196608))
    result = runtime.run()
    expected = {"context": "CONTEXT_OVERFLOW", "root_budget": "TOKEN_BUDGET_EXHAUSTED", "backend": "MODEL_ERROR"}[kind]
    assert result["stop_reason"] == expected and result["phase"] == "TERMINAL"
    event = json.loads(runtime.trace.path.read_text().splitlines()[0])
    assert event["model_input"]["prompt"] and event["model_identity"]["backend"] == "test_fake"
    assert event["usage"]["source"] == "unavailable"
    assert policy.generation_calls == 0
    assert len(policy.model.calls) == (1 if kind == "backend" else 0)
    assert (runtime.run_dir / "result.json").is_file()


def test_one_error_feedback_retry_without_scripted_fallback(make_runtime, make_policy):
    policy = make_policy(["invalid first", "invalid retry", "must not be used"])
    runtime = make_runtime(policy=policy)
    result = runtime.run()
    assert result["stop_reason"] == "PROTOCOL_RETRIES_EXHAUSTED"
    assert policy.generation_calls == 2
    events = [json.loads(line) for line in runtime.trace.path.read_text().splitlines()]
    feedback = json.loads(events[1]["model_input"]["messages"][1]["content"])["observation"]["last_action_result"]
    assert feedback["code"] == "INVALID_ACTION_JSON"
    assert events[0]["raw_output"] == "invalid first"


@pytest.mark.parametrize("correct", [True, False])
def test_llm_interface_saves_submitted_pass_and_fail(make_f2_runtime, fixture_data, make_policy, correct):
    steps = deepcopy(fixture_data["script"] + fixture_data["completion_script"])
    if not correct:
        steps[1]["output"]["args"]["source"] = "def solve(text): return 'wrong'"
    runtime = make_f2_runtime()
    runtime.policy = make_policy([s["output"] for s in steps])
    result = runtime.run()
    assert result["functional_status"] == ("PASS" if correct else "FAIL")
    assert result["stop_reason"] == "FINISHED"
    events = [json.loads(line) for line in runtime.trace.path.read_text().splitlines()]
    assert result["usage"]["input_tokens"] == sum(e["usage"]["input_tokens"] for e in events)
    assert all(e["model_input"] and e["output_token_ids"] for e in events)
    assert all(e["usage"]["source"] == "model" for e in events)


def test_same_model_generates_all_recursive_node_decisions(make_f3_runtime, recursive_fixture, make_policy):
    runtime = make_f3_runtime()
    policy = make_policy([s["output"] for s in recursive_fixture["script"]])
    runtime.policy = policy
    result = runtime.run()
    assert result["node_count"] == 4 and result["functional_status"] == "PASS"
    assert policy.generation_calls == 19 and runtime.policy is policy
    events = [json.loads(line) for line in runtime.trace.path.read_text().splitlines()]
    assert {e["node_id"] for e in events if e["model_input"]} == set(runtime.contexts)


def test_rollout_base_never_constructs_scripted_policy(tmp_path, monkeypatch, make_policy):
    from audittrainer import cli
    abstain = {"action": "abstain", "args": {"reason_code": "OTHER", "evidence_ids": []}}
    policy = make_policy([abstain, abstain])
    monkeypatch.setattr(cli.LLMPolicy, "from_pretrained", lambda *a, **kw: policy)
    def forbidden(*args, **kwargs):
        raise AssertionError("Base rollout must not read a script")
    monkeypatch.setattr(cli, "ScriptedPolicy", forbidden)
    directory = tmp_path / "rollout"
    assert cli.main(["rollout", "--policy", "base", "--split", "SYN_FIXTURE", "--limit", "2",
                     "--run-dir", str(directory)]) == 0
    result = json.loads((directory / "result.json").read_text())
    assert result["generation_calls"] == 2
    assert [r["stop_reason"] for r in result["results"]] == ["ABSTAINED", "ABSTAINED"]
    assert {json.loads((Path(r["run_dir"]) / "result.json").read_text())["stage"] for r in result["results"]} == {"F5"}


def test_unregistered_development_split_does_not_load_model(monkeypatch, tmp_path):
    from audittrainer import cli
    monkeypatch.setattr(cli.LLMPolicy, "from_pretrained", lambda *a, **kw: pytest.fail("must not load"))
    manifest = tmp_path / "empty_roots.jsonl"
    manifest.write_text("")
    with pytest.raises(SystemExit) as failure:
        cli.main(["rollout", "--policy", "base", "--split", "DEV_ENGINEERING", "--manifest", str(manifest)])
    assert failure.value.code == 2
