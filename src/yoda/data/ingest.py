"""Dataset ingestion, licensing audit, and schema normalization pipeline.

Downloads and transforms vetted open-source System 1 decision datasets (Nimble, Kev,
Dwidlee Lite Phase 2, Dwidlee Lite General, N4ze3m Synth, and Mghafiri Scenarios)
into canonical DecisionPayload instances, enforcing strict licensing and schema criteria.
"""

import json
import logging
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TextIO

import polars as pl

from yoda.architecture.schema import DecisionPayload, QueryContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _IngestionTask:
    path: Path
    is_train: bool
    fmt: str
    parse_fn: Callable[..., list[DecisionPayload] | DecisionPayload | None]
    extra_arg: str | None = None


class DatasetIngestionPipeline:
    """Manages downloading, auditing, and normalizing System 1 datasets."""

    NIMBLE_BASE_URL: str = "https://raw.githubusercontent.com/bespokelabsai/nimble/main/data"
    KEV_BASE_URL: str = "https://raw.githubusercontent.com/jaredpalmer/kev/main/evals"
    DWIDLEE_GEN_URL: str = (
        "https://huggingface.co/datasets/dwidlee/systemone-lite-general/resolve/main/data"
    )
    DWIDLEE_P2_URL: str = (
        "https://huggingface.co/datasets/dwidlee/systemone-lite-phase2/resolve/main/data"
    )
    N4ZE3M_URL: str = (
        "https://huggingface.co/datasets/n4ze3m/typed-decisions-synth/resolve/main/data"
    )
    MGHAFIRI_URL: str = (
        "https://huggingface.co/datasets/mghafiri/decision-model-scenarios/resolve/main/data"
    )

    ACCEPTED_DATASETS: ClassVar[dict[str, dict[str, str]]] = {
        "kev": {
            "license": "Apache-2.0",
            "repo": "https://github.com/jaredpalmer/kev",
            "posture": "permissive-clean",
        },
        "nimble": {
            "license": "unlicensed-research-fair-use",
            "repo": "https://github.com/bespokelabsai/nimble",
            "posture": "research-fair-use (experimental; retrainable on clean subset)",
        },
        "dwidlee_phase2": {
            "license": "Apache-2.0",
            "repo": "https://huggingface.co/datasets/dwidlee/systemone-lite-phase2",
            "posture": "permissive-clean",
        },
        "dwidlee_general": {
            "license": "MIT",
            "repo": "https://huggingface.co/datasets/dwidlee/systemone-lite-general",
            "posture": "permissive-clean",
        },
        "n4ze3m_synth": {
            "license": "MIT",
            "repo": "https://huggingface.co/datasets/n4ze3m/typed-decisions-synth",
            "posture": "permissive-clean",
        },
        "mghafiri_scenarios": {
            "license": "MIT",
            "repo": "https://huggingface.co/datasets/mghafiri/decision-model-scenarios",
            "posture": "permissive-clean",
        },
    }

    REJECTED_DATASETS: ClassVar[dict[str, dict[str, str]]] = {
        "tasksource": {
            "license": "other",
            "reason": "Mixed non-permissive upstream licenses across 500+ tasks",
        },
        "roskosmos19": {
            "license": "not stated",
            "reason": "Missing grant of rights",
        },
        "pngwn": {
            "license": "CC-BY-SA-4.0",
            "reason": "Viral share-alike copyleft condition restricts model weights",
        },
        "FaroukMoc2_image_beans": {
            "license": "mit",
            "reason": "Image dataset does not fit text DecisionPayload schema",
        },
    }

    def __init__(self, data_root: Path | str = "data") -> None:
        """Initializes the ingestion pipeline.

        Args:
            data_root: Root directory where raw and processed data reside.
        """
        self.data_root = Path(data_root)
        self.raw_nimble = self.data_root / "raw" / "nimble"
        self.raw_kev = self.data_root / "raw" / "kev"
        self.raw_dwidlee_gen = self.data_root / "raw" / "dwidlee_gen"
        self.raw_dwidlee_p2 = self.data_root / "raw" / "dwidlee_p2"
        self.raw_n4ze3m = self.data_root / "raw" / "n4ze3m"
        self.raw_mghafiri = self.data_root / "raw" / "mghafiri"
        self.processed_dir = self.data_root / "processed"

        for p in (
            self.raw_nimble,
            self.raw_kev,
            self.raw_dwidlee_gen,
            self.raw_dwidlee_p2,
            self.raw_n4ze3m,
            self.raw_mghafiri,
            self.processed_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, target: Path) -> Path:
        """Downloads a remote file with standard user-agent header if not present."""
        if target.exists() and target.stat().st_size > 0:
            logger.debug("data.ingest.cache_hit", extra={"path": str(target)})
            return target

        logger.info("data.ingest.download_start", extra={"url": url, "target": str(target)})
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (yoda-ingest/1.0)"})
        with urllib.request.urlopen(req, timeout=90) as resp, open(target, "wb") as f:
            shutil.copyfileobj(resp, f)

        logger.info(
            "data.ingest.download_complete",
            extra={"target": str(target), "bytes": target.stat().st_size},
        )
        return target

    def fetch_raw_datasets(self) -> dict[str, list[Path]]:
        """Fetches all vetted raw source files for accepted datasets."""
        files: dict[str, list[Path]] = {
            "nimble": [],
            "kev": [],
            "dwidlee_gen": [],
            "dwidlee_p2": [],
            "n4ze3m": [],
            "mghafiri": [],
        }

        downloads = [
            ("nimble", f"{self.NIMBLE_BASE_URL}/train.jsonl", self.raw_nimble / "train.jsonl"),
            ("nimble", f"{self.NIMBLE_BASE_URL}/eval.jsonl", self.raw_nimble / "eval.jsonl"),
            (
                "kev",
                f"{self.KEV_BASE_URL}/decision-v1/train.jsonl",
                self.raw_kev / "decision_v1_train.jsonl",
            ),
            (
                "kev",
                f"{self.KEV_BASE_URL}/decision-v1/test.jsonl",
                self.raw_kev / "decision_v1_test.jsonl",
            ),
            (
                "kev",
                f"{self.KEV_BASE_URL}/decision-v2/train.jsonl",
                self.raw_kev / "decision_v2_train.jsonl",
            ),
            (
                "kev",
                f"{self.KEV_BASE_URL}/decision-v2/test.jsonl",
                self.raw_kev / "decision_v2_test.jsonl",
            ),
            (
                "dwidlee_gen",
                f"{self.DWIDLEE_GEN_URL}/train-00000-of-00001.parquet",
                self.raw_dwidlee_gen / "train.parquet",
            ),
            (
                "dwidlee_gen",
                f"{self.DWIDLEE_GEN_URL}/test-00000-of-00001.parquet",
                self.raw_dwidlee_gen / "test.parquet",
            ),
            (
                "dwidlee_p2",
                f"{self.DWIDLEE_P2_URL}/train-00000-of-00001.parquet",
                self.raw_dwidlee_p2 / "train.parquet",
            ),
            (
                "dwidlee_p2",
                f"{self.DWIDLEE_P2_URL}/test-00000-of-00001.parquet",
                self.raw_dwidlee_p2 / "test.parquet",
            ),
            ("n4ze3m", f"{self.N4ZE3M_URL}/train.jsonl", self.raw_n4ze3m / "train.jsonl"),
            ("n4ze3m", f"{self.N4ZE3M_URL}/validation.jsonl", self.raw_n4ze3m / "validation.jsonl"),
            ("mghafiri", f"{self.MGHAFIRI_URL}/train.jsonl", self.raw_mghafiri / "train.jsonl"),
            (
                "mghafiri",
                f"{self.MGHAFIRI_URL}/validation.jsonl",
                self.raw_mghafiri / "validation.jsonl",
            ),
            ("mghafiri", f"{self.MGHAFIRI_URL}/test.jsonl", self.raw_mghafiri / "test.jsonl"),
        ]

        for category, url, target in downloads:
            files[category].append(self.download_file(url, target))

        return files

    @staticmethod
    def _safe_json_loads(data: Any, fallback: Any = None) -> Any:
        """Safely parses JSON string or returns fallback if invalid."""
        if isinstance(data, (dict, list)):
            return data
        if isinstance(data, str):
            text = data.strip()
            if (text.startswith("{") and text.endswith("}")) or (
                text.startswith("[") and text.endswith("]")
            ):
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    return fallback
            return fallback
        return fallback

    @classmethod
    def _try_parse_json_dict(cls, text: str) -> dict[str, Any] | None:
        if not (text.startswith("{") and text.endswith("}")):
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    @classmethod
    def _try_parse_jsonl_events(cls, text: str) -> dict[str, Any] | None:
        if not (("\n" in text or "\\n" in text) and text.startswith("{")):
            return None
        try:
            lines = [
                json.loads(line)
                for line in text.replace("\\n", "\n").splitlines()
                if line.strip().startswith("{")
            ]
            return {"events": lines} if lines else None
        except json.JSONDecodeError:
            return None

    @classmethod
    def _normalize_state(cls, raw_state: Any) -> dict[str, Any]:
        """Normalizes state payload to a dictionary representation."""
        if isinstance(raw_state, dict):
            return raw_state
        if isinstance(raw_state, list):
            return {"dialogue_or_list": raw_state}
        if isinstance(raw_state, str):
            text = raw_state.strip()

            parsed_dict = cls._try_parse_json_dict(text)
            if parsed_dict is not None:
                return parsed_dict

            parsed_events = cls._try_parse_jsonl_events(text)
            if parsed_events is not None:
                return parsed_events

            return {"text": raw_state}

        return {"raw": str(raw_state)}

    @staticmethod
    def _normalize_instructions(raw_inst: Any, default_key: str) -> str:
        """Normalizes instruction/query content from strings, dicts, or lists."""
        if isinstance(raw_inst, str):
            text = raw_inst.strip()
            if text:
                return text
        elif isinstance(raw_inst, dict):
            candidates = [
                str(raw_inst.get(k)).strip()
                for k in ("question", "instruction", "prompt", "query", "text", "rubric")
                if raw_inst.get(k)
            ]
            if candidates:
                return " | ".join(c for c in candidates if c)
            joined = " | ".join(f"{k}: {v}" for k, v in raw_inst.items() if v)
            if joined:
                return joined
        elif isinstance(raw_inst, (list, tuple)):
            joined = " ".join(str(item).strip() for item in raw_inst if str(item).strip())
            if joined:
                return joined
        return f"Decide outcome for {default_key}"

    @staticmethod
    def _format_criteria(criteria: Any) -> list[str]:
        """Converts diverse criteria formats (dict, list) into structured constraint strings."""
        if isinstance(criteria, dict):
            return [f"{k}: {v}" if v else f"{k}" for k, v in criteria.items()]
        if isinstance(criteria, list):
            return [str(c) for c in criteria]
        if criteria:
            return [str(criteria)]
        return []

    def _build_payload(
        self,
        q_key: str,
        q_data: dict[str, Any],
        state_dict: dict[str, Any],
        target_val: Any,
        base_metadata: dict[str, Any],
    ) -> DecisionPayload:
        """Helper to construct a DecisionPayload from normalized question data."""
        raw_inst = q_data.get("instructions")
        instructions = self._normalize_instructions(raw_inst, q_key)
        constraints = self._format_criteria(q_data.get("criteria"))

        metadata = dict(base_metadata)
        metadata.update(
            {
                "question_id": q_key,
                "question_type": q_data.get("type", "choice"),
                "target": target_val,
            }
        )

        return DecisionPayload(
            query=instructions,
            context=QueryContext(
                semantic_embedding=[],
                symbolic_state=state_dict,
                history=[],
            ),
            constraints=constraints,
            metadata=metadata,
        )

    def parse_nimble_record(self, raw_row: dict[str, Any]) -> list[DecisionPayload]:
        """Converts a raw Nimble JSONL record into one or more DecisionPayloads."""
        payloads: list[DecisionPayload] = []
        raw_input = raw_row.get("input", {})
        state_dict = self._normalize_state(raw_input.get("state", ""))
        questions = raw_input.get("questions", {})

        base_metadata = {
            "source": "nimble",
            "record_id": raw_row.get("id", "nimble_anon"),
            "domain": raw_row.get("domain", "general"),
            "family": raw_row.get("family", "unknown"),
        }
        teacher_answers = raw_row.get("teacher", {}).get("answers", {})

        for q_key, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue

            target_val = None
            if q_key in teacher_answers:
                t_ans = teacher_answers[q_key]
                target_val = t_ans.get("choice") or t_ans.get("noul") or t_ans.get("score")

            payloads.append(
                self._build_payload(q_key, q_data, state_dict, target_val, base_metadata)
            )
        return payloads

    def parse_kev_record(self, raw_row: dict[str, Any], file_tag: str) -> list[DecisionPayload]:
        """Converts a raw Kev JSONL record into one or more DecisionPayloads."""
        payloads: list[DecisionPayload] = []
        state_dict = self._normalize_state(raw_row.get("state", ""))
        questions = raw_row.get("questions", {})
        base_metadata = {"source": "kev", "file_tag": file_tag}

        for q_key, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue
            payloads.append(
                self._build_payload(q_key, q_data, state_dict, q_data.get("label"), base_metadata)
            )
        return payloads

    def parse_dwidlee_row(self, row: dict[str, Any], source_tag: str) -> DecisionPayload:
        """Converts a dwidlee tabular record into a DecisionPayload."""
        state_dict = self._normalize_state(row.get("state", "{}"))
        meta_dict = self._safe_json_loads(row.get("meta", "{}"), fallback={})

        criteria_values = row.get("criteria_values")
        constraints = (
            [str(c) for c in criteria_values]
            if isinstance(criteria_values, list)
            else self._format_criteria(criteria_values)
        )

        return DecisionPayload(
            query=str(row.get("instructions", "")),
            context=QueryContext(
                semantic_embedding=[],
                symbolic_state=state_dict,
                history=[],
            ),
            constraints=constraints,
            metadata={
                "source": source_tag,
                "task": row.get("task"),
                "target": row.get("label_key"),
                "label_alias": row.get("label_alias"),
                "criteria_keys": row.get("criteria_keys"),
                "meta": meta_dict,
            },
        )

    def parse_n4ze3m_record(self, raw_row: dict[str, Any]) -> list[DecisionPayload]:
        """Converts an n4ze3m synthetic review record into one or more DecisionPayloads."""
        payloads: list[DecisionPayload] = []
        state_dict = self._normalize_state(raw_row.get("state", "{}"))
        questions = self._safe_json_loads(raw_row.get("questions", "{}"), fallback={})
        gold = self._safe_json_loads(raw_row.get("gold", "{}"), fallback={})

        if not isinstance(questions, dict):
            return payloads

        base_metadata = {
            "source": "n4ze3m_synth",
            "record_id": raw_row.get("state_id", "n4ze3m_anon"),
            "domain": raw_row.get("domain", "reviews"),
        }

        for q_key, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue
            target_val = gold.get(q_key) if isinstance(gold, dict) else None
            payloads.append(
                self._build_payload(q_key, q_data, state_dict, target_val, base_metadata)
            )
        return payloads

    def parse_mghafiri_record(self, raw_row: dict[str, Any]) -> list[DecisionPayload]:
        """Converts an mghafiri scenario record into one or more DecisionPayloads."""
        payloads: list[DecisionPayload] = []
        state_dict = self._normalize_state(raw_row.get("state", "{}"))
        questions = self._safe_json_loads(raw_row.get("questions", "{}"), fallback={})
        targets = self._safe_json_loads(raw_row.get("targets", "{}"), fallback={})

        if not isinstance(questions, dict):
            return payloads

        base_metadata = {
            "source": "mghafiri_scenarios",
            "record_id": raw_row.get("id", "mghafiri_anon"),
            "domain": raw_row.get("domain", "general"),
            "pattern": raw_row.get("pattern", "unknown"),
        }

        for q_key, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue

            target_val = None
            if isinstance(targets, dict) and q_key in targets:
                t_entry = targets[q_key]
                if isinstance(t_entry, dict):
                    target_val = (
                        t_entry.get("noul") if "noul" in t_entry else t_entry.get("probabilities")
                    )
                else:
                    target_val = t_entry

            payloads.append(
                self._build_payload(q_key, q_data, state_dict, target_val, base_metadata)
            )
        return payloads

    @staticmethod
    def _write_payloads(
        payloads: list[DecisionPayload] | DecisionPayload, target_file: TextIO
    ) -> int:
        """Writes serialized DecisionPayload record(s) to target text stream."""
        if isinstance(payloads, DecisionPayload):
            target_file.write(payloads.model_dump_json() + "\n")
            return 1
        count = 0
        for p in payloads:
            target_file.write(p.model_dump_json() + "\n")
            count += 1
        return count

    def _get_ingestion_tasks(self) -> list[_IngestionTask]:
        """Constructs the list of ingestion tasks for supported raw dataset files."""
        return [
            # Nimble
            _IngestionTask(
                self.raw_nimble / "train.jsonl", True, "jsonl", self.parse_nimble_record
            ),
            _IngestionTask(
                self.raw_nimble / "eval.jsonl", False, "jsonl", self.parse_nimble_record
            ),
            # Kev
            _IngestionTask(
                self.raw_kev / "decision_v1_train.jsonl",
                True,
                "jsonl",
                self.parse_kev_record,
                "v1_train",
            ),
            _IngestionTask(
                self.raw_kev / "decision_v1_test.jsonl",
                False,
                "jsonl",
                self.parse_kev_record,
                "v1_test",
            ),
            _IngestionTask(
                self.raw_kev / "decision_v2_train.jsonl",
                True,
                "jsonl",
                self.parse_kev_record,
                "v2_train",
            ),
            _IngestionTask(
                self.raw_kev / "decision_v2_test.jsonl",
                False,
                "jsonl",
                self.parse_kev_record,
                "v2_test",
            ),
            # Dwidlee
            _IngestionTask(
                self.raw_dwidlee_gen / "train.parquet",
                True,
                "parquet",
                self.parse_dwidlee_row,
                "dwidlee_gen",
            ),
            _IngestionTask(
                self.raw_dwidlee_gen / "test.parquet",
                False,
                "parquet",
                self.parse_dwidlee_row,
                "dwidlee_gen",
            ),
            _IngestionTask(
                self.raw_dwidlee_p2 / "train.parquet",
                True,
                "parquet",
                self.parse_dwidlee_row,
                "dwidlee_p2",
            ),
            _IngestionTask(
                self.raw_dwidlee_p2 / "test.parquet",
                False,
                "parquet",
                self.parse_dwidlee_row,
                "dwidlee_p2",
            ),
            # N4ze3m
            _IngestionTask(
                self.raw_n4ze3m / "train.jsonl", True, "jsonl", self.parse_n4ze3m_record
            ),
            _IngestionTask(
                self.raw_n4ze3m / "validation.jsonl",
                False,
                "jsonl",
                self.parse_n4ze3m_record,
            ),
            # Mghafiri
            _IngestionTask(
                self.raw_mghafiri / "train.jsonl", True, "jsonl", self.parse_mghafiri_record
            ),
            _IngestionTask(
                self.raw_mghafiri / "validation.jsonl",
                False,
                "jsonl",
                self.parse_mghafiri_record,
            ),
            _IngestionTask(
                self.raw_mghafiri / "test.jsonl", False, "jsonl", self.parse_mghafiri_record
            ),
        ]

    def _execute_task(self, task: _IngestionTask, target_file: TextIO) -> int:
        """Executes a single file parsing task and streams payloads into target_file."""
        if not task.path.exists():
            return 0

        count = 0
        if task.fmt == "jsonl":
            with open(task.path, encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    record = json.loads(line_str)
                    parsed = (
                        task.parse_fn(record, task.extra_arg)
                        if task.extra_arg is not None
                        else task.parse_fn(record)
                    )
                    if parsed:
                        count += self._write_payloads(parsed, target_file)
        elif task.fmt == "parquet":
            df = pl.read_parquet(task.path)
            for row in df.iter_rows(named=True):
                payload = (
                    task.parse_fn(row, task.extra_arg)
                    if task.extra_arg is not None
                    else task.parse_fn(row)
                )
                if payload:
                    count += self._write_payloads(payload, target_file)

        return count

    def _stream_datasets(self, f_train: TextIO, f_eval: TextIO) -> tuple[int, int]:
        """Streams raw dataset files into train and evaluation JSONL files."""
        train_count, eval_count = 0, 0
        tasks = self._get_ingestion_tasks()

        for task in tasks:
            target_file = f_train if task.is_train else f_eval
            records_written = self._execute_task(task, target_file)
            if task.is_train:
                train_count += records_written
            else:
                eval_count += records_written

        return train_count, eval_count

    def process_and_aggregate(self) -> dict[str, int]:
        """Executes full ingestion, validation, and split aggregation via streaming writes."""
        logger.info("data.ingest.pipeline_start")
        self.fetch_raw_datasets()

        train_out = self.processed_dir / "aggregated_train.jsonl"
        eval_out = self.processed_dir / "aggregated_eval.jsonl"

        with (
            open(train_out, "w", encoding="utf-8") as f_train,
            open(eval_out, "w", encoding="utf-8") as f_eval,
        ):
            train_count, eval_count = self._stream_datasets(f_train, f_eval)

        manifest = {
            "version": "2.0.0",
            "total_train_records": train_count,
            "total_eval_records": eval_count,
            "accepted_sources": self.ACCEPTED_DATASETS,
            "rejected_sources": self.REJECTED_DATASETS,
            "schema": "DecisionPayload(query, context, constraints, metadata)",
        }

        manifest_out = self.processed_dir / "manifest.json"
        with open(manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        stats = {
            "train_records": train_count,
            "eval_records": eval_count,
        }
        logger.info("data.ingest.pipeline_complete", extra=stats)
        return stats

if __name__ == "__main__":
    pipeline = DatasetIngestionPipeline()
    pipeline.process_and_aggregate()
