"""Append-only decisions and the corresponding host state snapshots."""

import json
from pathlib import Path
from uuid import uuid4

from .schema import TraceEvent


def renderer_identity() -> dict:
    """Pin the code used to reconstruct scripted decision inputs offline."""
    import hashlib
    from . import prompts, schema
    return {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in (("prompts_sha256", prompts), ("schema_sha256", schema))}


class TraceWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.states_dir = run_dir / "states"
        self.states_dir.mkdir()
        self.path = run_dir / "trace.jsonl"
        self.path.touch(exist_ok=False)
        self.count = 0

    def initialize(self, task, protocol_version: str):
        with (self.run_dir / "trace_manifest.json").open("x", encoding="utf-8") as stream:
            json.dump({"trace_version": 1, "run_id": uuid4().hex, "task": task.model_dump(mode="json"),
                       "protocol_version": protocol_version, "renderer": renderer_identity()}, stream, indent=2)
            stream.write("\n")

    def snapshot(self, state: dict) -> str:
        relative = f"states/S{self.count:04d}.json"
        with (self.run_dir / relative).open("x", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        return relative

    def append(self, **fields) -> TraceEvent:
        event = TraceEvent(event_id=f"E{self.count:04d}", **fields)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")
        return event
