"""I/O helpers for evaluation artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from eval.schemas import EvaluationResult


def load_json_file(path: str | Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json_file(path: str | Path, payload: Any, indent: int = 2) -> None:
    ensure_parent_dir(path)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=indent, default=str)


def load_evaluation_results(path: str | Path) -> List[EvaluationResult]:
    """Load an evaluation results file (top-level list or {'results': [...]})."""
    data = load_json_file(path)
    rows = data.get("results", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError(f"Expected list or {{'results': [...]}} in {path}")
    return [EvaluationResult.model_validate(r) for r in rows]
