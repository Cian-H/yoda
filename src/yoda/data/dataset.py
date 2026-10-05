"""Dataset loaders and batch collation routines for Yoda training pipelines."""

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
        offset: int = 0,
        max_choices: int = 5,
        shuffle_choices: bool = False,
    ) -> None:
        """Initializes YodaDecisionDataset by parsing records from a JSONL file.

        Args:
            file_path: Path to the JSONL dataset file.
            source: Dataset source filter (e.g. "n4ze3m_synth"), or None to disable.
            question_type: Task question type filter (e.g. "choice"), or None to disable.
            max_samples: Maximum number of matched samples to load.
            offset: Number of initial matched samples to skip.
            max_choices: Standardized number of choice constraints per sample.
            shuffle_choices: Whether to randomly permute active candidate choices in __getitem__.
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.source = source
        self.question_type = question_type
        self.max_samples = max_samples
        self.offset = int(offset)
        self.max_choices = max_choices
        self.shuffle_choices = shuffle_choices
        self._samples: list[dict[str, Any]] = []

        self._load_data()

    def _match_target(self, target: Any, raw_constraints: list[str]) -> int | None:
        """Matches a gold target label to an index in the raw constraints list."""
        if target is None or not raw_constraints:
            return None

        # Handle probability distribution targets (e.g. {"rbp": 0.912, "ma": 0.034})
        if isinstance(target, dict):
            if not target:
                return None
            target = max(target.items(), key=lambda kv: kv[1])[0]

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

        matched_count = 0
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

                # Truncate raw constraints to max_choices
                active_choices: list[str] = [str(c) for c in raw_constraints[: self.max_choices]]

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

                matched_count += 1
                if matched_count <= self.offset:
                    continue

                self._samples.append(
                    {
                        "query": query,
                        "state": state if isinstance(state, dict) else {},
                        "active_choices": active_choices,
                        "target_idx": matched_idx,
                        "question_type": metadata.get("question_type", "null"),
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Retrieves a standardized sample by index, optionally shuffling choices."""
        sample = self._samples[index]
        active = sample["active_choices"]
        target_idx = sample["target_idx"]
        question_type = sample.get("question_type", "null")
        num_active = len(active)

        if self.shuffle_choices and num_active > 1:
            perm = torch.randperm(num_active).tolist()
            shuffled_active = [active[p] for p in perm]
            new_target_idx = perm.index(target_idx)
        else:
            shuffled_active = list(active)
            new_target_idx = target_idx

        constraints = shuffled_active + ["none: Unused option"] * (self.max_choices - num_active)

        task_scalar_map = {
            "null": -1.0,
            "score": 0.0,
            "choice": 1.0,
            "multi_choice": 0.5,
            "multilabel": 0.5,
        }
        task_scalar = task_scalar_map.get(question_type, -1.0)

        return {
            "query": sample["query"],
            "state": sample["state"],
            "constraints": constraints,
            "target_idx": new_target_idx,
            "task_scalar": task_scalar,
            "num_active": num_active,
        }


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
        - `task_scalars`: Torch tensor of shape `(batch_size, 1)` and dtype `torch.float32`.
        - `num_active`: Torch tensor of shape `(batch_size,)` indicating active choice counts.
        - `active_mask`: Torch boolean tensor of shape `(batch_size, max_choices)`.
    """
    queries = [item["query"] for item in batch]
    states = [item["state"] for item in batch]
    constraints = [item["constraints"] for item in batch]
    target_indices = torch.tensor([item["target_idx"] for item in batch], dtype=torch.long)
    task_scalars = torch.tensor(
        [item.get("task_scalar", -1.0) for item in batch], dtype=torch.float32
    ).unsqueeze(-1)

    num_actives = [
        item.get(
            "num_active",
            sum(1 for c in item["constraints"] if not str(c).startswith("none:")),
        )
        for item in batch
    ]
    num_active_tensor = torch.tensor(num_actives, dtype=torch.long)

    max_choices = len(constraints[0]) if constraints else 0
    if max_choices > 0:
        choice_idx = torch.arange(max_choices, dtype=torch.long).unsqueeze(0).expand(len(batch), -1)
        active_mask = choice_idx < num_active_tensor.unsqueeze(1)
    else:
        active_mask = torch.zeros((len(batch), 0), dtype=torch.bool)

    cand_queries: list[str] = []
    cand_states: list[dict[str, Any]] = []
    candidates: list[str] = []
    cand_labels: list[float] = []
    cand_group_ids: list[int] = []
    cand_task_scalars: list[float] = []
    hierarchy_edges: list[tuple[int, int]] = []

    cand_offset = 0
    for b_idx, item in enumerate(batch):
        q = item["query"]
        s = item["state"]
        raw_c = item["constraints"]
        t_idx = item.get("target_idx", -1)
        n_act = num_actives[b_idx]
        task_sc = float(item.get("task_scalar", -1.0))
        active_c = raw_c[:n_act]
        for c_idx, c_str in enumerate(active_c):
            cand_queries.append(q)
            cand_states.append(s)
            candidates.append(str(c_str))
            cand_labels.append(1.0 if c_idx == t_idx else 0.0)
            cand_group_ids.append(b_idx)
            cand_task_scalars.append(task_sc)

        edges = item.get("hierarchy_edges")
        if edges:
            for c_edge, p_edge in edges:
                hierarchy_edges.append((c_edge + cand_offset, p_edge + cand_offset))

        cand_offset += len(active_c)

    batch_dict: dict[str, Any] = {
        "queries": queries,
        "states": states,
        "constraints": constraints,
        "target_indices": target_indices,
        "task_scalars": task_scalars,
        "num_active": num_active_tensor,
        "active_mask": active_mask,
        "candidates": candidates,
        "candidate_queries": cand_queries,
        "candidate_states": cand_states,
        "candidate_labels": torch.tensor(cand_labels, dtype=torch.float32),
        "candidate_group_ids": torch.tensor(cand_group_ids, dtype=torch.long),
        "candidate_task_scalars": (
            torch.tensor(cand_task_scalars, dtype=torch.float32).unsqueeze(-1)
        ),
    }
    if hierarchy_edges:
        batch_dict["hierarchy_edges"] = hierarchy_edges
    return batch_dict
