"""ETL pipeline extracting, cleaning, and standardizing JSONL datasets into columnar Parquet."""

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "ParquetETLPipeline",
    "match_target_to_constraints",
]


def match_target_to_constraints(target: Any, raw_constraints: list[str]) -> int | None:
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


class ParquetETLPipeline:
    """Extracts, cleans, and transforms raw decision JSONL datasets into columnar Parquet tables."""

    def __init__(self, max_choices: int = 5) -> None:
        self.max_choices = max_choices

    def transform_jsonl_to_parquet(
        self,
        input_path: Path | str,
        output_path: Path | str,
        source: str | None = None,
        question_type: str | None = None,
        max_samples: int | None = None,
    ) -> pl.DataFrame:
        """Processes a JSONL file into a standardized, compressed Parquet table."""
        src_path = Path(input_path)
        dst_path = Path(output_path)
        logger.info(
            "data.etl.started",
            extra={"input_path": str(src_path), "output_path": str(dst_path)},
        )

        if not src_path.exists():
            msg = f"Input file not found: {src_path}"
            logger.error("data.etl.error", extra={"error": msg})
            raise FileNotFoundError(msg)

        queries: list[str] = []
        state_jsons: list[str] = []
        choices_list: list[list[str]] = []
        num_active_list: list[int] = []
        target_indices: list[int] = []
        task_scalars: list[float] = []
        question_types: list[str] = []
        sources: list[str] = []

        task_scalar_map = {"null": -1.0, "score": 0.0, "choice": 1.0}

        with src_path.open("r", encoding="utf-8") as f:
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

                # Source filtering
                rec_source = metadata.get("source", "unknown")
                if source is not None and rec_source != source:
                    continue

                # Question type filtering
                q_type = str(metadata.get("question_type", "null"))
                if question_type is not None and q_type != question_type:
                    continue

                raw_constraints = record.get("constraints", [])
                if not isinstance(raw_constraints, list):
                    continue

                target = metadata.get("target")
                matched_idx = match_target_to_constraints(target, raw_constraints)
                if matched_idx is None or matched_idx >= self.max_choices:
                    continue

                active_choices = [str(c) for c in raw_constraints[: self.max_choices]]
                num_active = len(active_choices)
                padded_choices = active_choices + ["none: Unused option"] * (
                    self.max_choices - num_active
                )

                # Extract state
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
                task_scalar = task_scalar_map.get(q_type, -1.0)

                queries.append(query)
                state_jsons.append(json.dumps(state if isinstance(state, dict) else {}))
                choices_list.append(padded_choices)
                num_active_list.append(num_active)
                target_indices.append(matched_idx)
                task_scalars.append(task_scalar)
                question_types.append(q_type)
                sources.append(rec_source)

                if max_samples is not None and len(queries) >= max_samples:
                    break

        df = pl.DataFrame(
            {
                "query": queries,
                "state_json": state_jsons,
                "choices": choices_list,
                "num_active": num_active_list,
                "target_idx": target_indices,
                "task_scalar": task_scalars,
                "question_type": question_types,
                "source": sources,
            },
            schema={
                "query": pl.Utf8,
                "state_json": pl.Utf8,
                "choices": pl.List(pl.Utf8),
                "num_active": pl.Int32,
                "target_idx": pl.Int64,
                "task_scalar": pl.Float32,
                "question_type": pl.Utf8,
                "source": pl.Utf8,
            },
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dst_path, compression="zstd")
        logger.info(
            "data.etl.completed",
            extra={"output_path": str(dst_path), "rows": len(df)},
        )
        return df
