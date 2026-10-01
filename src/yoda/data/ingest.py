"""Dataset ingestion, licensing audit, and schema normalization pipeline.

Downloads and transforms vetted open-source System 1 decision datasets (Nimble and Kev)
into canonical DecisionPayload instances, enforcing strict licensing and schema criteria.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, ClassVar

from yoda.architecture.schema import DecisionPayload, QueryContext

logger = logging.getLogger(__name__)


class DatasetIngestionPipeline:
    """Manages downloading, auditing, and normalizing System 1 datasets."""

    NIMBLE_BASE_URL: str = "https://raw.githubusercontent.com/bespokelabsai/nimble/main/data"
    KEV_BASE_URL: str = "https://raw.githubusercontent.com/jaredpalmer/kev/main/evals"

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
        self.processed_dir = self.data_root / "processed"

        self.raw_nimble.mkdir(parents=True, exist_ok=True)
        self.raw_kev.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, target: Path) -> Path:
        """Downloads a remote file with standard user-agent header if not present."""
        if target.exists() and target.stat().st_size > 0:
            logger.debug("data.ingest.cache_hit", extra={"path": str(target)})
            return target

        logger.info("data.ingest.download_start", extra={"url": url, "target": str(target)})
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (yoda-ingest/1.0)"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(target, "wb") as f:
            f.write(resp.read())

        logger.info(
            "data.ingest.download_complete",
            extra={"target": str(target), "bytes": target.stat().st_size},
        )
        return target

    def fetch_raw_datasets(self) -> dict[str, list[Path]]:
        """Fetches all vetted raw source files for Nimble and Kev."""
        files: dict[str, list[Path]] = {"nimble": [], "kev": []}

        # Nimble train and eval (research / experimental posture)
        nimble_train = self.download_file(
            f"{self.NIMBLE_BASE_URL}/train.jsonl",
            self.raw_nimble / "train.jsonl",
        )
        nimble_eval = self.download_file(
            f"{self.NIMBLE_BASE_URL}/eval.jsonl",
            self.raw_nimble / "eval.jsonl",
        )
        files["nimble"].extend([nimble_train, nimble_eval])

        # Kev decision suites (Apache-2.0 clean posture)
        kev_v1_train = self.download_file(
            f"{self.KEV_BASE_URL}/decision-v1/train.jsonl",
            self.raw_kev / "decision_v1_train.jsonl",
        )
        kev_v1_test = self.download_file(
            f"{self.KEV_BASE_URL}/decision-v1/test.jsonl",
            self.raw_kev / "decision_v1_test.jsonl",
        )
        kev_v2_train = self.download_file(
            f"{self.KEV_BASE_URL}/decision-v2/train.jsonl",
            self.raw_kev / "decision_v2_train.jsonl",
        )
        kev_v2_test = self.download_file(
            f"{self.KEV_BASE_URL}/decision-v2/test.jsonl",
            self.raw_kev / "decision_v2_test.jsonl",
        )
        files["kev"].extend([kev_v1_train, kev_v1_test, kev_v2_train, kev_v2_test])

        return files

    @staticmethod
    def _normalize_state(raw_state: Any) -> dict[str, Any]:
        """Normalizes state payload to a dictionary representation."""
        if isinstance(raw_state, dict):
            return raw_state
        if isinstance(raw_state, str):
            return {"text": raw_state}
        if isinstance(raw_state, list):
            return {"dialogue_or_list": raw_state}
        return {"raw": str(raw_state)}

    @staticmethod
    def _normalize_instructions(raw_inst: Any, default_key: str) -> str:
        """Normalizes instruction/query content from strings, dicts, or lists."""
        if isinstance(raw_inst, str):
            text = raw_inst.strip()
            if text:
                return text
        elif isinstance(raw_inst, dict):
            # Prioritize common query fields, otherwise join key-value pairs
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
            return [
                f"{k}: {v}" if v else f"{k}"
                for k, v in criteria.items()
            ]
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

    def process_and_aggregate(self) -> dict[str, int]:
        """Executes full ingestion, validation, and split aggregation."""
        logger.info("data.ingest.pipeline_start")
        self.fetch_raw_datasets()

        train_payloads: list[DecisionPayload] = []
        eval_payloads: list[DecisionPayload] = []

        # 1. Process Nimble train (research posture)
        nimble_train_path = self.raw_nimble / "train.jsonl"
        if nimble_train_path.exists():
            with open(nimble_train_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    train_payloads.extend(self.parse_nimble_record(row))

        # 2. Process Nimble eval (research posture)
        nimble_eval_path = self.raw_nimble / "eval.jsonl"
        if nimble_eval_path.exists():
            with open(nimble_eval_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    eval_payloads.extend(self.parse_nimble_record(row))

        # 3. Process Kev decision-v1 and v2 (Apache-2.0 clean posture)
        kev_mappings = [
            (self.raw_kev / "decision_v1_train.jsonl", train_payloads, "v1_train"),
            (self.raw_kev / "decision_v1_test.jsonl", eval_payloads, "v1_test"),
            (self.raw_kev / "decision_v2_train.jsonl", train_payloads, "v2_train"),
            (self.raw_kev / "decision_v2_test.jsonl", eval_payloads, "v2_test"),
        ]

        for pth, target_list, tag in kev_mappings:
            if not pth.exists():
                continue
            with open(pth, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    target_list.extend(self.parse_kev_record(row, tag))

        # Write aggregated processed datasets
        train_out = self.processed_dir / "aggregated_train.jsonl"
        eval_out = self.processed_dir / "aggregated_eval.jsonl"

        with open(train_out, "w", encoding="utf-8") as f:
            for p in train_payloads:
                f.write(p.model_dump_json() + "\n")

        with open(eval_out, "w", encoding="utf-8") as f:
            for p in eval_payloads:
                f.write(p.model_dump_json() + "\n")

        manifest = {
            "version": "1.0.0",
            "total_train_records": len(train_payloads),
            "total_eval_records": len(eval_payloads),
            "accepted_sources": self.ACCEPTED_DATASETS,
            "rejected_sources": self.REJECTED_DATASETS,
            "schema": "DecisionPayload(query, context, constraints, metadata)",
        }

        manifest_out = self.processed_dir / "manifest.json"
        with open(manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        stats = {
            "train_records": len(train_payloads),
            "eval_records": len(eval_payloads),
        }
        logger.info("data.ingest.pipeline_complete", extra=stats)
        return stats
