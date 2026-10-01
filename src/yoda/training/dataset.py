"""Dataset loaders and batch collation routines for Yoda training pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "YodaDecisionDataset",
    "collate_decision_batch",
]


class YodaDecisionDataset(Dataset[dict[str, Any]]):
    """Dataset for System 1 choice decision samples with Belnap constraint standardization."""

    def __init__(
        self,
        file_path: Path | str,
        source: str | None = "n4ze3m_synth",
        question_type: str | None = "choice",
        max_samples: int | None = None,
        max_choices: int = 5,
    ) -> None:
        """Initializes YodaDecisionDataset by parsing records from a JSONL file.

        Args:
            file_path: Path to the JSONL dataset file.
            source: Dataset source filter (e.g. "n4ze3m_synth"), or None to disable.
            question_type: Task question type filter (e.g. "choice"), or None to disable.
            max_samples: Maximum number of matched samples to load.
            max_choices: Standardized number of choice constraints per sample.
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.source = source
        self.question_type = question_type
        self.max_samples = max_samples
        self.max_choices = max_choices
        self._samples: list[dict[str, Any]] = []

        self._load_data()

    def _match_target(self, target: Any, raw_constraints: list[str]) -> int | None:
        """Matches a gold target label to an index in the raw constraints list."""
        if target is None or not raw_constraints:
            return None

        target_str = str(target).strip()

        # 1. Exact match or prefix match before colon (e.g. "A1" matches "A1: Option A1")
        for idx, constraint in enumerate(raw_constraints):
            c_str = str(constraint).strip()
            if c_str == target_str:
                return idx
            prefix = c_str.split(":", 1)[0].strip()
            if prefix.lower() == target_str.lower():
                return idx

        # 2. Case-insensitive exact match
        for idx, constraint in enumerate(raw_constraints):
            c_str = str(constraint).strip()
            if c_str.lower() == target_str.lower():
                return idx

        # 3. Constraint starts with target
        for idx, constraint in enumerate(raw_constraints):
            c_str = str(constraint).strip()
            if c_str.startswith(target_str):
                return idx

        return None

    def _load_data(self) -> None:
        """Parses lines from the JSONL file and filters / standardizes items."""
        if not self.file_path.exists():
            logger.warning(
                "training.dataset.file_not_found",
                extra={"file_path": str(self.file_path)},
            )
            return

        with self.file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    record = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}

                # Filter by source if specified
                if self.source is not None and metadata.get("source") != self.source:
                    continue

                # Filter by question_type if specified
                if (
                    self.question_type is not None
                    and metadata.get("question_type") != self.question_type
                ):
                    continue

                # Match target to constraints
                raw_constraints = record.get("constraints", [])
                if not isinstance(raw_constraints, list):
                    continue

                target = metadata.get("target")
                matched_idx = self._match_target(target, raw_constraints)
                if matched_idx is None or matched_idx >= self.max_choices:
                    # Unmatched target or target choice falls beyond max_choices
                    continue

                # Standardize constraints: pad or truncate to max_choices
                constraints: list[str] = [str(c) for c in raw_constraints[: self.max_choices]]
                while len(constraints) < self.max_choices:
                    constraints.append("none: Unused option")

                # Extract symbolic state
                context = record.get("context")
                if isinstance(context, dict) and "symbolic_state" in context:
                    state = context["symbolic_state"]
                elif "state" in record and isinstance(record["state"], dict):
                    state = record["state"]
                elif isinstance(context, dict):
                    state = context
                else:
                    state = {}

                query = str(record.get("query", ""))

                self._samples.append(
                    {
                        "query": query,
                        "state": state if isinstance(state, dict) else {},
                        "constraints": constraints,
                        "target_idx": matched_idx,
                    }
                )

                if self.max_samples is not None and len(self._samples) >= self.max_samples:
                    break

        logger.info(
            "training.dataset.loaded",
            extra={
                "file_path": str(self.file_path),
                "loaded_samples": len(self._samples),
                "max_choices": self.max_choices,
            },
        )

    def __len__(self) -> int:
        """Returns the number of loaded and standardized samples."""
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Retrieves a standardized sample by index."""
        return self._samples[idx]


def collate_decision_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates a list of decision sample dictionaries into batched tensors and lists.

    Args:
        batch: List of sample dictionaries containing 'query', 'state',
            'constraints', and 'target_idx'.

    Returns:
        Dictionary with:
        - `queries`: List of query strings of length `batch_size`.
        - `states`: List of symbolic state dictionaries of length `batch_size`.
        - `constraints`: List of constraint string lists of shape `[batch_size, max_choices]`.
        - `target_indices`: Torch tensor of shape `(batch_size,)` and dtype `torch.long`.
    """
    queries = [item["query"] for item in batch]
    states = [item["state"] for item in batch]
    constraints = [item["constraints"] for item in batch]
    target_indices = torch.tensor([item["target_idx"] for item in batch], dtype=torch.long)

    return {
        "queries": queries,
        "states": states,
        "constraints": constraints,
        "target_indices": target_indices,
    }
