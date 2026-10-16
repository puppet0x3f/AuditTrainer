"""PR0 factorized plan policy and frozen execution routing.

The runtime still receives one complete PlanAction. Gate scores and sampled body
tokens are recorded separately; host prefixes/atomic payloads are never presented
as tokens sampled from the language model.
"""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .policy import LLMPolicy, PolicyError, load_adapter_metadata
from .prompts import PromptOverflow, prepare_prompt
from .schema import PolicyOutput, TokenUsage, parse_action
from .verifier import canonical, digest

PLAN_PREFIX = '{"action":"plan","args":{"base_plan_version":0,"mode":"'
MODES = ("ATOMIC", "SPLIT")
BODY_KEYS = ("children", "obligation_map", "retained_responsibilities", "dataflow", "ticket_id")


def atomic_plan(contract):
    return {"action": "plan", "args": {
        "base_plan_version": 0, "mode": "ATOMIC", "children": [], "dataflow": [],
        "obligation_map": {o["id"]: ["self"] for o in contract["ensures"]},
        "retained_responsibilities": list(contract["owns"]), "ticket_id": None,
    }}


def split_body(action):
    args = action["args"]
    if action["action"] != "plan" or args["mode"] != "SPLIT" or args["base_plan_version"] != 0:
        raise ValueError("Expected an initial SPLIT demonstration")
    # This is an explicitly re-encoded SFT target, never a rewritten RL sample.
    body = {key: args.get(key) for key in BODY_KEYS}
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))[1:] + "}"


def unique_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def validate_refinement(raw, forced_mode=None):
    unique_json(raw)
    action = parse_action(raw)
    if action.action != "plan" or action.args.base_plan_version != 0:
        raise ValueError("Expected an initial plan")
    if forced_mode and action.args.mode != forced_mode:
        raise ValueError("Sampled body changes the selected mode")
    return action


def codec_input(tokenizer, prepared):
    prefix = prepared.token_ids + tokenizer.encode(PLAN_PREFIX, add_special_tokens=False)
    modes = {m: tokenizer.encode(m, add_special_tokens=False) for m in MODES}
    separator = tokenizer.encode('",', add_special_tokens=False)
    return {"prefix": prefix, "mode_ids": modes, "separator": separator}


def sequence_log_probs(model, prefixes, targets, pad_token_id):
    """Exact causal scores on saved IDs; gradients are retained for SFT."""
    lengths = [len(p) + len(t) for p, t in zip(prefixes, targets)]
    size = max(lengths)
    if any(not p or not t for p, t in zip(prefixes, targets)):
        raise ValueError("Empty scoring prefix or target")
    ids = torch.full((len(prefixes), size), pad_token_id, dtype=torch.long, device=model.device)
    attention = torch.zeros_like(ids)
    positions = sorted({i for p, t in zip(prefixes, targets) for i in range(len(p)-1, len(p)+len(t)-1)})
    pos = torch.tensor(positions, device=model.device)
    lookup = {position: i for i, position in enumerate(positions)}
    for row, (p, t) in enumerate(zip(prefixes, targets)):
        ids[row, :len(p)+len(t)] = torch.tensor(p+t, device=model.device)
        attention[row, :len(p)+len(t)] = 1
    logits = model(input_ids=ids, attention_mask=attention, logits_to_keep=pos, use_cache=False).logits
    values = []
    for row, (p, t) in enumerate(zip(prefixes, targets)):
        indices = [lookup[i] for i in range(len(p)-1, len(p)+len(t)-1)]
        selected = logits[row, indices].float()
        labels = torch.tensor(t, device=model.device)
        values.append(-F.cross_entropy(selected, labels, reduction="none"))
    return values


def mode_log_probs(model, prefix, mode_ids, modes, pad_token_id):
    scores = sequence_log_probs(model, [prefix]*len(modes), [mode_ids[m] for m in modes], pad_token_id)
    totals = torch.stack([x.sum() for x in scores])
    return F.log_softmax(totals, dim=0), scores


def two_level_advantages(p_split, atomic, splits):
    if len(splits) < 2:
        raise ValueError("At least two conditionally sampled SPLIT bodies are needed")
    q_split = sum(splits)/len(splits)
    value = (1-p_split)*atomic + p_split*q_split
    return {"q_atomic": atomic, "q_split": q_split, "value": value,
            "mode": {"ATOMIC": atomic-value, "SPLIT": q_split-value},
            "body": [r-(sum(splits)-r)/(len(splits)-1) for r in splits],
            "mode_weights": {"ATOMIC": 1-p_split, "SPLIT": p_split},
            "body_weight": p_split/len(splits)}


@contextmanager
def active_adapter(model, name):
    previous = model.active_adapters[0]
    model.set_adapter(name)
    model.requires_grad_(False)
    model.eval()
    try:
        yield
    finally:
        model.set_adapter(previous)
        model.requires_grad_(False)
        model.eval()


def adapter_fingerprint(model, name):
    from peft import get_peft_model_state_dict
    from .train_sft import tensor_hash
    tensors = get_peft_model_state_dict(model, adapter_name=name)
    return digest({key: tensor_hash(value) for key, value in sorted(tensors.items())})


class RefinementPolicy:
    def __init__(self, llm, adapter_name="default"):
        self.llm, self.model, self.tokenizer = llm, llm.model, llm.tokenizer
        self.adapter_name = adapter_name
        self.identity = {**llm.identity, "policy": "PR0_factorized_plan",
                         "codec_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.cost = Counter()

    def prepare(self, observation):
        prepared = prepare_prompt(self.tokenizer, observation, context_tokens=self.llm.context_tokens,
                                  generation_tokens=self.llm.generation_tokens)
        encoded = codec_input(self.tokenizer, prepared)
        extra = len(encoded["prefix"])-len(prepared.token_ids)
        max_body = prepared.max_new_tokens-extra-max(map(len, encoded["mode_ids"].values()))-len(encoded["separator"])
        if max_body < 1:
            raise PromptOverflow("CONTEXT_OVERFLOW", prepared.record)
        return prepared, encoded, max_body

    def score_modes(self, observation):
        prepared, encoded, max_body = self.prepare(observation)
        modes = list(observation["allowed_plan_modes"])
        if not modes:
            raise ValueError("No legal planning mode")
        forced = len(modes) == 1
        if forced:
            logps, tokens = [0.0], [[]]
        else:
            with active_adapter(self.model, self.adapter_name), torch.inference_mode(), self.llm._autocast():
                lp, scores = mode_log_probs(self.model, encoded["prefix"], encoded["mode_ids"],
                                           modes, self.tokenizer.pad_token_id)
                logps = lp.float().cpu().tolist()
                tokens = [x.float().cpu().tolist() for x in scores]
        gate_input = 0 if forced else len(encoded["prefix"])*len(modes)
        gate_output = 0 if forced else sum(len(encoded["mode_ids"][m]) for m in modes)
        self.cost.update(input_tokens=gate_input, output_tokens=gate_output,
                         gate_scoring_batches=int(not forced), gate_scored_sequences=0 if forced else len(modes),
                         model_forward_calls=int(not forced))
        probabilities = {m: float(torch.tensor(lp, dtype=torch.float64).exp()) for m, lp in zip(modes, logps)}
        return {"observation": deepcopy(observation), "observation_sha256": digest(observation),
                "model_input": prepared.record, "prefix_ids": encoded["prefix"],
                "mode_token_ids": encoded["mode_ids"], "separator_ids": encoded["separator"],
                "mode_log_probs": dict(zip(modes, logps)), "mode_probabilities": probabilities,
                "mode_token_log_probs": dict(zip(modes, tokens)), "forced_by_host": forced,
                "max_body_tokens": max_body,
                "gate_usage": {"input_tokens": gate_input, "output_tokens": gate_output},
                "model_identity": self.identity,
                "generation_config": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0},
                "mask_rule": "score mode candidates; body only is sampled; prompt and host prefixes excluded"}

    def atomic(self, gate):
        return self._candidate(gate, "ATOMIC", canonical(atomic_plan(gate["observation"]["incoming_contract"])),
                               [], [], [], False, "HOST_ATOMIC_BODY")

    def sample_split(self, gate):
        if "SPLIT" not in gate["mode_probabilities"]:
            raise ValueError("SPLIT is not legal at this decision")
        prefix = gate["prefix_ids"] + gate["mode_token_ids"]["SPLIT"] + gate["separator_ids"]
        ids = torch.tensor([prefix], device=self.model.device)
        generation = deepcopy(self.model.generation_config)
        generation.update(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                          num_return_sequences=1, num_beams=1, use_cache=True,
                          max_new_tokens=gate["max_body_tokens"], return_dict_in_generate=True,
                          output_scores=True)
        with active_adapter(self.model, self.adapter_name), torch.inference_mode(), self.llm._autocast():
            result = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         generation_config=generation, use_model_defaults=False)
            output = result.sequences[0, len(prefix):].tolist()
            logps = [float(F.log_softmax(score[0].float(), dim=-1)[token].cpu())
                     for token, score in zip(output, result.scores)]
        eos = generation.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        ended = bool(output and output[-1] in eos_ids)
        body = self.tokenizer.decode(output[:-1] if ended else output, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        truncated = len(output) >= gate["max_body_tokens"] and not ended
        raw = PLAN_PREFIX + 'SPLIT",' + body
        self.cost.update(input_tokens=len(prefix), output_tokens=len(output),
                         generation_calls=1, model_forward_calls=len(output))
        return self._candidate(gate, "SPLIT", raw, prefix, output, logps, truncated, "MODEL_CONDITIONAL_BODY")

    def _candidate(self, gate, mode, raw, input_ids, output_ids, logps, truncated, body_source):
        parsed, error = None, None
        try:
            if truncated:
                raise ValueError("TRUNCATED_OUTPUT")
            parsed = validate_refinement(raw, mode).model_dump(mode="json")
        except ValueError as exc:
            error = str(exc)
        usage = {"input_tokens": gate["gate_usage"]["input_tokens"]+len(input_ids),
                 "output_tokens": gate["gate_usage"]["output_tokens"]+len(output_ids)}
        record = {"mode": mode, "raw_action": raw, "parsed_action": parsed, "parse_error": error,
                  "generation_truncated": truncated, "body_source": body_source,
                  "body_input_ids": input_ids, "body_output_ids": output_ids,
                  "body_old_log_probs": logps, "body_mask": [1]*len(output_ids),
                  "gate_old_log_prob": gate["mode_log_probs"][mode],
                  "logical_usage": usage}
        # The full gate remains in the group, and in runtime model_input for standalone actions.
        return record

    def as_output(self, gate, candidate):
        model_input = {"messages": gate["model_input"]["messages"],
                       "refinement": {"gate": {k: v for k, v in gate.items() if k not in {"observation", "model_input"}},
                                      "body_input_ids": candidate["body_input_ids"],
                                      "body_old_log_probs": candidate["body_old_log_probs"],
                                      "body_source": candidate["body_source"],
                                      "selected_mode": candidate["mode"]}}
        return PolicyOutput(raw_text=candidate["raw_action"],
                            token_usage=TokenUsage(**candidate["logical_usage"], source="model"),
                            model_identity=self.identity, model_input=model_input,
                            output_token_ids=candidate["body_output_ids"],
                            generation_truncated=candidate["generation_truncated"])

    def act(self, observation):
        try:
            gate = self.score_modes(observation)
            modes = list(gate["mode_probabilities"])
            probabilities = torch.tensor([gate["mode_probabilities"][m] for m in modes])
            mode = modes[int(torch.multinomial(probabilities, 1))]
            candidate = self.atomic(gate) if mode == "ATOMIC" else self.sample_split(gate)
            return self.as_output(gate, candidate)
        except PromptOverflow as error:
            raise PolicyError(error.code, str(error), PolicyOutput(
                raw_text="", model_input=error.model_input, model_identity=self.identity,
                token_usage=TokenUsage(source="unavailable"))) from error


class FrozenExecutor:
    def __init__(self, llm, adapter_name="executor"):
        self.llm, self.model, self.adapter_name = llm, llm.model, adapter_name
        self.cost = Counter()

    def act(self, observation):
        with active_adapter(self.model, self.adapter_name):
            output = self.llm.act(observation)
        self.cost.update(input_tokens=output.token_usage.input_tokens,
                         output_tokens=output.token_usage.output_tokens, generation_calls=1,
                         model_forward_calls=len(output.output_token_ids))
        return output


class RefinementRouter:
    """Only initial planning learns/refines. PR0 has no repair policy."""
    def __init__(self, planner, executor, initial=None):
        self.planner, self.executor, self.initial = planner, executor, initial
        self.identity = {"planner": planner.identity, "executor": executor.llm.identity, "repair": "disabled"}
        self.calls = Counter()

    def act(self, observation):
        phase = observation["phase"]
        if phase == "NEW":
            self.calls["planner"] += 1
            if self.initial is not None:
                gate, candidate = self.initial
                if digest(observation) != gate["observation_sha256"]:
                    raise ValueError("Injected candidate is not bound to this decision state")
                self.initial = None
                return self.planner.as_output(gate, candidate)
            return self.planner.act(observation)
        if phase in {"IMPLEMENT", "RESOLVE_CHILD"}:
            self.calls["executor"] += 1
            return self.executor.act(observation)
        self.calls["host_control"] += 1
        if phase == "READY_VERIFY":
            action = {"action": "verify", "args": {"target_ref": observation["current_artifact"]["module_path"],
                                                 "suite_id": observation["available_suites"][0]}}
        elif phase == "REVIEW":
            action = {"action": "finish", "args": {"artifact_version": observation["current_artifact"]["version"]}}
        else:
            action = {"action": "abstain", "args": {"reason_code": "INSUFFICIENT_EVIDENCE", "evidence_ids": []}}
        return PolicyOutput(raw_text=canonical(action), token_usage=TokenUsage(),
                            model_identity={"backend": "pr0_fixed_control"})


def load_policies(framework, adapter_path, executor_path):
    llm = LLMPolicy.from_pretrained(framework, adapter_path=Path(adapter_path))
    tokenizer, identity = load_adapter_metadata(Path(executor_path), framework["model"], llm.tokenizer)
    llm.model.load_adapter(executor_path, adapter_name="executor", is_trainable=False, local_files_only=True)
    llm.model.requires_grad_(False)
    llm.model.set_adapter("default")
    llm.model.requires_grad_(False)
    executor_llm = LLMPolicy(llm.model, tokenizer,
        identity={**llm.identity, **identity, "role": "frozen_F01_executor", "active_adapters": '["executor"]'},
        context_tokens=llm.context_tokens, generation_tokens=llm.generation_tokens, decoding=llm.decoding)
    if framework.get('parent_composition_summary', False):
        from .refinement_parent_composition import summary_prompt
        executor_llm.prompt_builder = summary_prompt
    return RefinementPolicy(llm), FrozenExecutor(executor_llm)
