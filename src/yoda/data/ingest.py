"""Dataset ingestion, licensing audit, and schema normalization pipeline.

Downloads and transforms vetted open-source System 1 decision datasets (Nimble, Kev,
Dwidlee Lite Phase 2, Dwidlee Lite General, N4ze3m Synth, and Mghafiri Scenarios)
into canonical DecisionPayload instances, enforcing strict licensing and schema criteria.
"""

from __future__ import annotations

import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from yoda.architecture.schema import DecisionPayload, QueryContext

logger = logging.getLogger(__name__)


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

        # 1. Nimble train and eval (research posture)
        files["nimble"].extend(
            [
                self.download_file(
                    f"{self.NIMBLE_BASE_URL}/train.jsonl",
                    self.raw_nimble / "train.jsonl",
                ),
                self.download_file(
                    f"{self.NIMBLE_BASE_URL}/eval.jsonl",
                    self.raw_nimble / "eval.jsonl",
                ),
            ]
        )

        # 2. Kev decision suites (Apache-2.0 clean posture)
        files["kev"].extend(
            [
                self.download_file(
                    f"{self.KEV_BASE_URL}/decision-v1/train.jsonl",
                    self.raw_kev / "decision_v1_train.jsonl",
                ),
                self.download_file(
                    f"{self.KEV_BASE_URL}/decision-v1/test.jsonl",
                    self.raw_kev / "decision_v1_test.jsonl",
                ),
                self.download_file(
                    f"{self.KEV_BASE_URL}/decision-v2/train.jsonl",
                    self.raw_kev / "decision_v2_train.jsonl",
                ),
                self.download_file(
                    f"{self.KEV_BASE_URL}/decision-v2/test.jsonl",
                    self.raw_kev / "decision_v2_test.jsonl",
                ),
            ]
        )

        # 3. Dwidlee general (MIT)
        files["dwidlee_gen"].extend(
            [
                self.download_file(
                    f"{self.DWIDLEE_GEN_URL}/train-00000-of-00001.parquet",
                    self.raw_dwidlee_gen / "train.parquet",
                ),
                self.download_file(
                    f"{self.DWIDLEE_GEN_URL}/test-00000-of-00001.parquet",
                    self.raw_dwidlee_gen / "test.parquet",
                ),
            ]
        )

        # 4. Dwidlee phase 2 (Apache-2.0)
        files["dwidlee_p2"].extend(
            [
                self.download_file(
                    f"{self.DWIDLEE_P2_URL}/train-00000-of-00001.parquet",
                    self.raw_dwidlee_p2 / "train.parquet",
                ),
                self.download_file(
                    f"{self.DWIDLEE_P2_URL}/test-00000-of-00001.parquet",
                    self.raw_dwidlee_p2 / "test.parquet",
                ),
            ]
        )

        # 5. N4ze3m synth (MIT)
        files["n4ze3m"].extend(
            [
                self.download_file(
                    f"{self.N4ZE3M_URL}/train.jsonl",
                    self.raw_n4ze3m / "train.jsonl",
                ),
                self.download_file(
                    f"{self.N4ZE3M_URL}/validation.jsonl",
                    self.raw_n4ze3m / "validation.jsonl",
                ),
            ]
        )

        # 6. Mghafiri scenarios (MIT)
        files["mghafiri"].extend(
            [
                self.download_file(
                    f"{self.MGHAFIRI_URL}/train.jsonl",
                    self.raw_mghafiri / "train.jsonl",
                ),
                self.download_file(
                    f"{self.MGHAFIRI_URL}/validation.jsonl",
                    self.raw_mghafiri / "validation.jsonl",
                ),
                self.download_file(
                    f"{self.MGHAFIRI_URL}/test.jsonl",
                    self.raw_mghafiri / "test.jsonl",
                ),
            ]
        )

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
                except Exception:
                    pass
        return fallback if fallback is not None else data

    @classmethod
    def _normalize_state(cls, raw_state: Any) -> dict[str, Any]:
        """Normalizes state payload to a dictionary representation."""
        if isinstance(raw_state, dict):
            return raw_state
        if isinstance(raw_state, list):
            return {"dialogue_or_list": raw_state}
        if isinstance(raw_state, str):
            text = raw_state.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            if ("\\n" in text or "\n" in text) and text.startswith("{"):
                try:
                    norm = text.replace("\\n", "\n")
                    lines = [
                        json.loads(line)
                        for line in norm.splitlines()
                        if line.strip().startswith("{")
                    ]
                    if lines:
                        return {"events": lines}
                except Exception:
                    pass
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

    def parse_nimble_record(self, raw_row: dict[str, Any]) -> list[DecisionPayload]:
        """Converts a raw Nimble JSONL record into one or more DecisionPayloads."""
        payloads: list[DecisionPayload] = []
        raw_input = raw_row.get("input", {})
        state_dict = self._normalize_state(raw_input.get("state", ""))
        questions = raw_input.get("questions", {})

        domain = raw_row.get("domain", "general")
        family = raw_row.get("family", "unknown")
        record_id = raw_row.get("id", "nimble_anon")
        teacher = raw_row.get("teacher", {})
        teacher_answers = teacher.get("answers", {})

        for q_key, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue

            raw_inst = q_data.get("instructions")
            instructions = self._normalize_instructions(raw_inst, q_key)
            constraints = self._format_criteria(q_data.get("criteria"))
            q_type = q_data.get("type", "choice")

            target_val = None
            if q_key in teacher_answers:
                t_ans = teacher_answers[q_key]
                target_val = t_ans.get("choice") or t_ans.get("noul") or t_ans.get("score")

            payload = DecisionPayload(
                query=instructions,
                context=QueryContext(
                    semantic_embedding=[],
                    symbolic_state=state_dict,
                    history=[],
                ),
                constraints=constraints,
                metadata={
                    "source": "nimble",
                    "record_id": record_id,
                    "domain": domain,
                    "family": family,
                    "question_id": q_key,
                    "question_type": q_type,
                    "target": target_val,
                },
            )
            payloads.append(payload)

        return payloads

    def parse_kev_record(self, raw_row: dict[str, Any], file_tag: str) -> list[DecisionPayload]:
        """Converts a raw Kev JSONL record into one or more DecisionPayloads."""
        payloads: list[DecisionPayload] = []
        state_dict = self._normalize_state(raw_row.get("state", ""))
        questions = raw_row.get("questions", {})

        for q_key, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue

            raw_inst = q_data.get("instructions")
            instructions = self._normalize_instructions(raw_inst, q_key)
            constraints = self._format_criteria(q_data.get("criteria"))
            q_type = q_data.get("type", "choice")
            target_label = q_data.get("label")

            payload = DecisionPayload(
                query=instructions,
                context=QueryContext(
                    semantic_embedding=[],
                    symbolic_state=state_dict,
                    history=[],
                ),
                constraints=constraints,
                metadata={
                    "source": "kev",
                    "file_tag": file_tag,
                    "question_id": q_key,
                    "question_type": q_type,
                    "target": target_label,
                },
            )
            payloads.append(payload)

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

        state_id = raw_row.get("state_id", "n4ze3m_anon")
        domain = raw_row.get("domain", "reviews")

        for q_key, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue

            raw_inst = q_data.get("instructions")
            instructions = self._normalize_instructions(raw_inst, q_key)
            constraints = self._format_criteria(q_data.get("criteria"))
            q_type = q_data.get("type", "choice")
            target_val = gold.get(q_key) if isinstance(gold, dict) else None

            payload = DecisionPayload(
                query=instructions,
                context=QueryContext(
                    semantic_embedding=[],
                    symbolic_state=state_dict,
                    history=[],
                ),
                constraints=constraints,
                metadata={
                    "source": "n4ze3m_synth",
                    "record_id": state_id,
                    "domain": domain,
                    "question_id": q_key,
                    "question_type": q_type,
                    "target": target_val,
                },
            )
            payloads.append(payload)

        return payloads

    def parse_mghafiri_record(self, raw_row: dict[str, Any]) -> list[DecisionPayload]:
        """Converts an mghafiri scenario record into one or more DecisionPayloads."""
        payloads: list[DecisionPayload] = []
        state_dict = self._normalize_state(raw_row.get("state", "{}"))
        questions = self._safe_json_loads(raw_row.get("questions", "{}"), fallback={})
        targets = self._safe_json_loads(raw_row.get("targets", "{}"), fallback={})

        if not isinstance(questions, dict):
            return payloads

        record_id = raw_row.get("id", "mghafiri_anon")
        domain = raw_row.get("domain", "general")
        pattern = raw_row.get("pattern", "unknown")

        for q_key, q_data in questions.items():
            if not isinstance(q_data, dict):
                continue

            raw_inst = q_data.get("instructions")
            instructions = self._normalize_instructions(raw_inst, q_key)
            constraints = self._format_criteria(q_data.get("criteria"))
            q_type = q_data.get("type", "choice")

            target_val = None
            if isinstance(targets, dict) and q_key in targets:
                t_entry = targets[q_key]
                if isinstance(t_entry, dict):
                    target_val = (
                        t_entry.get("noul")
                        if "noul" in t_entry
                        else t_entry.get("probabilities")
                    )
                else:
                    target_val = t_entry

            payload = DecisionPayload(
                query=instructions,
                context=QueryContext(
                    semantic_embedding=[],
                    symbolic_state=state_dict,
                    history=[],
                ),
                constraints=constraints,
                metadata={
                    "source": "mghafiri_scenarios",
                    "record_id": record_id,
                    "domain": domain,
                    "pattern": pattern,
                    "question_id": q_key,
                    "question_type": q_type,
                    "target": target_val,
                },
            )
            payloads.append(payload)

        return payloads

    @staticmethod
    def _write_payloads(payloads: list[DecisionPayload], target_file: Any) -> int:
        count = 0
        for p in payloads:
            target_file.write(p.model_dump_json() + "\n")
            count += 1
        return count

    def _stream_nimble(self, f_train: Any, f_eval: Any) -> tuple[int, int]:
        train_count, eval_count = 0, 0
        nimble_train_path = self.raw_nimble / "train.jsonl"
        if nimble_train_path.exists():
            with open(nimble_train_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        train_count += self._write_payloads(
                            self.parse_nimble_record(json.loads(line)), f_train
                        )
        nimble_eval_path = self.raw_nimble / "eval.jsonl"
        if nimble_eval_path.exists():
            with open(nimble_eval_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        eval_count += self._write_payloads(
                            self.parse_nimble_record(json.loads(line)), f_eval
                        )
        return train_count, eval_count

    def _stream_kev(self, f_train: Any, f_eval: Any) -> tuple[int, int]:
        train_count, eval_count = 0, 0
        kev_mappings = [
            (self.raw_kev / "decision_v1_train.jsonl", f_train, "v1_train", True),
            (self.raw_kev / "decision_v1_test.jsonl", f_eval, "v1_test", False),
            (self.raw_kev / "decision_v2_train.jsonl", f_train, "v2_train", True),
            (self.raw_kev / "decision_v2_test.jsonl", f_eval, "v2_test", False),
        ]
        for pth, target_file, tag, is_train in kev_mappings:
            if not pth.exists():
                continue
            with open(pth, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = self._write_payloads(
                            self.parse_kev_record(json.loads(line), tag), target_file
                        )
                        if is_train:
                            train_count += c
                        else:
                            eval_count += c
        return train_count, eval_count

    def _stream_dwidlee(self, f_train: Any, f_eval: Any) -> tuple[int, int]:
        train_count, eval_count = 0, 0
        dwidlee_mappings = [
            (self.raw_dwidlee_gen / "train.parquet", f_train, "dwidlee_gen", True),
            (self.raw_dwidlee_gen / "test.parquet", f_eval, "dwidlee_gen", False),
            (self.raw_dwidlee_p2 / "train.parquet", f_train, "dwidlee_p2", True),
            (self.raw_dwidlee_p2 / "test.parquet", f_eval, "dwidlee_p2", False),
        ]
        for pth, target_file, tag, is_train in dwidlee_mappings:
            if not pth.exists():
                continue
            df = pl.read_parquet(pth)
            for row in df.iter_rows(named=True):
                p = self.parse_dwidlee_row(row, tag)
                target_file.write(p.model_dump_json() + "\n")
                if is_train:
                    train_count += 1
                else:
                    eval_count += 1
        return train_count, eval_count

    def _stream_synth_and_scenarios(self, f_train: Any, f_eval: Any) -> tuple[int, int]:
        train_count, eval_count = 0, 0
        jsonl_mappings = [
            (self.raw_n4ze3m / "train.jsonl", f_train, self.parse_n4ze3m_record, True),
            (self.raw_n4ze3m / "validation.jsonl", f_eval, self.parse_n4ze3m_record, False),
            (self.raw_mghafiri / "train.jsonl", f_train, self.parse_mghafiri_record, True),
            (self.raw_mghafiri / "validation.jsonl", f_eval, self.parse_mghafiri_record, False),
            (self.raw_mghafiri / "test.jsonl", f_eval, self.parse_mghafiri_record, False),
        ]
        for pth, target_file, parse_fn, is_train in jsonl_mappings:
            if not pth.exists():
                continue
            with open(pth, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        c = self._write_payloads(parse_fn(json.loads(line)), target_file)
                        if is_train:
                            train_count += c
                        else:
                            eval_count += c
        return train_count, eval_count

    def process_and_aggregate(self) -> dict[str, int]:
        """Executes full ingestion, validation, and split aggregation via streaming writes."""
        logger.info("data.ingest.pipeline_start")
        self.fetch_raw_datasets()

        train_out = self.processed_dir / "aggregated_train.jsonl"
        eval_out = self.processed_dir / "aggregated_eval.jsonl"

        train_count = 0
        eval_count = 0

        with open(train_out, "w", encoding="utf-8") as f_train, open(
            eval_out, "w", encoding="utf-8"
        ) as f_eval:
            streamers = [
                self._stream_nimble,
                self._stream_kev,
                self._stream_dwidlee,
                self._stream_synth_and_scenarios,
            ]
            for streamer in streamers:
                t_c, e_c = streamer(f_train, f_eval)
                train_count += t_c
                eval_count += e_c

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
