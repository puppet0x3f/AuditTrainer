"""F7 verifies the learned positions, objective, accumulation and finite sampling."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from audittrainer.dataset import file_hash, read_jsonl
from audittrainer.tokenize_sft import ActionCollator, encode_action_example
from audittrainer.train_sft import (ActionSFTTrainer, GradientCheck, build_schedule, changed_tensors,
    load_config, load_identity_and_tokenizer, load_training_data, per_example_target_loss, trainable_snapshots)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(ROOT / "Qwen3-8B", local_files_only=True)


@pytest.fixture
def rows():
    return read_jsonl(ROOT / "data/sft/smoke.jsonl")


def test_exact_prefix_full_target_eos_and_no_metadata(rows, tokenizer):
    row = deepcopy(rows[0])
    row["meta"]["future_or_private"] = "must never enter input"
    encoded = encode_action_example(row, tokenizer, 4096)
    prefix = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
    target = tokenizer.encode(row["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
    assert encoded["input_ids"] == prefix + target
    assert encoded["labels"] == [-100] * len(prefix) + target
    assert set(encoded) == {"input_ids", "labels", "attention_mask"}
    assert "must never enter input" not in tokenizer.decode(encoded["input_ids"])
    assert encoded["labels"][-1] == tokenizer.eos_token_id


def test_overlong_rejected_whole_and_exact_boundary_accepted(rows, tokenizer):
    row = rows[0]
    encoded = encode_action_example(row, tokenizer, 4096)
    size = len(encoded["input_ids"])
    assert encode_action_example(row, tokenizer, size) == encoded
    with pytest.raises(ValueError, match="never truncate"):
        encode_action_example(row, tokenizer, size - 1)


@pytest.mark.parametrize("mutation", ["target", "counts", "assistant", "eos"])
def test_invalid_action_or_prefix_rejected(rows, tokenizer, mutation):
    row = rows[0]
    if mutation == "target":
        row["target"] = row["target"][:-1]
    elif mutation == "counts":
        row["meta"]["prefix_tokens"] += 1
    elif mutation == "assistant":
        row["messages"].append({"role": "assistant", "content": "future"})
    else:
        tokenizer = SimpleNamespace(apply_chat_template=tokenizer.apply_chat_template,
                                    encode=tokenizer.encode, eos_token_id=None)
    with pytest.raises(ValueError):
        encode_action_example(row, tokenizer, 4096)


def test_right_padding_learns_eos_even_when_pad_equals_eos():
    a = {"input_ids": [1, 2, 9], "attention_mask": [1, 1, 1], "labels": [-100, 2, 9]}
    b = {"input_ids": [1, 9], "attention_mask": [1, 1], "labels": [-100, 9]}
    batch = ActionCollator(9)([a, b])
    assert batch["input_ids"].tolist() == [[1, 2, 9], [1, 9, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 9], [-100, 9, -100]]
    assert all(t.dtype == torch.long for t in batch.values())
    with pytest.raises(ValueError, match="Empty batch"):
        ActionCollator(9)([])
    with pytest.raises(ValueError, match="no next-token targets"):
        ActionCollator(9)([{**b, "labels": [-100, -100]}])


def test_example_average_differs_from_token_average_and_has_no_prompt_gradient():
    logits = torch.tensor([[[3., 0.], [3., 0.], [3., 0.]],
                           [[3., 0.], [3., 0.], [3., 0.]]], requires_grad=True)
    labels = torch.tensor([[0, -100, -100], [1, 1, 1]])
    loss = per_example_target_loss(logits, labels)
    expected = (F.cross_entropy(logits[0, :1], labels[0, :1])
                + F.cross_entropy(logits[1], labels[1])) / 2
    torch.testing.assert_close(loss, expected)
    assert not torch.isclose(loss, F.cross_entropy(logits[labels != -100], labels[labels != -100]))
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 1:]) == 0
    with pytest.raises(ValueError, match="no target tokens"):
        per_example_target_loss(logits, torch.full_like(labels, -100))


def tiny_model():
    torch.manual_seed(19)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, attention_dropout=0.0))


def cpu_args(tmp_path, **kwargs):
    return TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[],
        remove_unused_columns=False, save_strategy="no", disable_tqdm=True,
        learning_rate=0.03, lr_scheduler_type="constant", optim="sgd", max_grad_norm=0,
        **kwargs)


def toy_examples(count=6):
    rows = []
    for i in range(count):
        ids = [1, 2, 3 + i, 10, 11, 12][:4 + i % 3]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                     "labels": [-100] * (2 + i % 2) + ids[2 + i % 2:]})
    return rows


def test_selected_logits_match_full_causal_shift(tmp_path):
    model = tiny_model().eval()
    batch = ActionCollator(0)(toy_examples(2))
    trainer = ActionSFTTrainer(model=model, args=cpu_args(tmp_path))
    loss = trainer.compute_loss(model, batch)
    full = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    torch.testing.assert_close(loss, per_example_target_loss(full[:, :-1], batch["labels"][:, 1:]))


def test_saved_tensor_cpu_offload_preserves_loss_and_gradients(tmp_path):
    reference = tiny_model().train()
    offloaded = deepcopy(reference).train()
    batch = ActionCollator(0)(toy_examples(2))
    trainers = [ActionSFTTrainer(model=model, args=cpu_args(tmp_path / str(index)))
                for index, model in enumerate((reference, offloaded))]
    trainers[1].activation_offload = True
    losses = [trainer.compute_loss(model, batch) for trainer, model in zip(trainers, (reference, offloaded))]
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for loss in losses:
        loss.backward()
    for p, q in zip(reference.parameters(), offloaded.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize("batch,accumulation", [(1, 4), (2, 2), (3, 2)])
def test_real_trainer_gradient_accumulation_including_partial_batches(tmp_path, batch, accumulation):
    examples = toy_examples(5)
    actual = tiny_model()
    reference = deepcopy(actual)
    effective_batch = batch * accumulation
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
    for start in range(0, len(examples), effective_batch):
        inputs = ActionCollator(0)(examples[start:start + effective_batch])
        logits = reference(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        loss = per_example_target_loss(logits[:, :-1], inputs["labels"][:, 1:])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    trainer = ActionSFTTrainer(model=actual, args=cpu_args(tmp_path,
        per_device_train_batch_size=batch, gradient_accumulation_steps=accumulation,
        max_steps=(len(examples) + effective_batch - 1) // effective_batch),
        train_dataset=examples, data_collator=ActionCollator(0))
    trainer.train()
    assert trainer.model_accepts_loss_kwargs is False
    for p, q in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)


def test_config_data_and_deterministic_curriculum(rows, tokenizer):
    config = load_config(ROOT / "configs/sft_smoke.yaml")
    identity, tokenizer = load_identity_and_tokenizer(config)
    loaded, encoded, manifest = load_training_data(config, identity, tokenizer)
    assert len(loaded) == len(encoded) == 32
    assert manifest["output_sha256"] == file_hash(ROOT / "data/sft/smoke.jsonl")
    schedule, steps = build_schedule(rows, config["data"], config["training"])
    assert build_schedule(rows, config["data"], config["training"]) == (schedule, steps)
    assert len(schedule) == 40 and steps == 10
    assert all(s["phase"] == "warmup" and s["bucket"] in {"D_IMPL", "D_CONTROL"} for s in schedule[:8])
    assert all(s["phase"] == "main" for s in schedule[8:])
    assert {rows[s["index"]]["target_action"]["action"] for s in schedule if s["bucket"] == "D_COMPOSE"} == {"consume", "write_code"}
    config["data"]["allowed_splits"] = ["TRAIN"]
    with pytest.raises(ValueError, match="separate"):
        load_training_data(config, identity, tokenizer)


def test_root_sampling_not_proportional_to_number_of_actions(rows):
    config = load_config(ROOT / "configs/sft_smoke.yaml")
    # One root has one action, the other has 100. Root choice must still be 50/50.
    small = {**rows[0], "root_id": "small", "bucket": "D_IMPL"}
    big = [{**small, "root_id": "big", "sample_id": str(i)} for i in range(100)]
    data = {**config["data"], "warmup_mix": {"D_IMPL": 1.0}, "main_mix": {"D_IMPL": 1.0}}
    training = {**config["training"], "max_steps": 20, "gradient_accumulation_steps": 100}
    schedule, _ = build_schedule([small, *big], data, training)
    counts = Counter(s["root_id"] for s in schedule)
    assert 850 < counts["small"] < 1150
    formal = {**data, "allowed_splits": ["TRAIN"]}
    schedule, _ = build_schedule([small, *big], formal, training)
    assert len(schedule) == 101  # No endless reuse to satisfy a formal step budget.


def test_missing_bucket_or_composition_code_rejected(rows):
    config = load_config(ROOT / "configs/sft_smoke.yaml")
    with pytest.raises(ValueError, match="missing bucket"):
        build_schedule([r for r in rows if r["bucket"] != "D_PLAN"], config["data"], config["training"])
    with pytest.raises(ValueError, match="parent write_code"):
        build_schedule([r for r in rows if not (r["bucket"] == "D_COMPOSE" and r["target_action"]["action"] == "write_code")],
                       config["data"], config["training"])


def test_tiny_lora_update_save_and_disk_reload(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    base = tiny_model()
    fresh_base = deepcopy(base)
    model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                           lora_dropout=0, task_type="CAUSAL_LM"))
    before, count = trainable_snapshots(model)
    assert count > 0
    with pytest.raises(RuntimeError, match="No tracked"):
        changed_tensors(model, before)
    check = GradientCheck()
    trainer = ActionSFTTrainer(model=model, args=cpu_args(tmp_path, max_steps=1,
        per_device_train_batch_size=2), train_dataset=toy_examples(2),
        data_collator=ActionCollator(0), callbacks=[check])
    trainer.train()
    assert len(check.steps) == 1
    assert any(r["delta_norm"] > 0 for r in changed_tensors(model, before))
    model.save_pretrained(tmp_path / "adapter")
    restored = PeftModel.from_pretrained(fresh_base, tmp_path / "adapter", is_trainable=False)
    inputs = torch.tensor([[1, 2, 3]])
    model.eval(); restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs).logits, restored(inputs).logits, rtol=0, atol=0)
