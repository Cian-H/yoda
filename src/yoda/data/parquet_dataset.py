"""High-performance columnar Parquet dataset for Yoda training pipelines."""

import json
import logging
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "YodaParquetDataset",
]


class YodaParquetDataset(Dataset[dict[str, Any]]):
    """High-throughput PyTorch Dataset backed by an Apache Parquet columnar table."""

    def __init__(
        self,
        file_path: Path | str,
        max_samples: int | None = None,
        shuffle_choices: bool = False,
    ) -> None:
        """Initializes dataset by memory-mapping or reading a Parquet table.

        Args:
            file_path: Path to the .parquet dataset file.
            max_samples: Maximum rows to retain.
            shuffle_choices: Whether to randomly permute active candidate choices in __getitem__.
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.shuffle_choices = shuffle_choices

        if not self.file_path.exists():
            msg = f"Parquet file not found: {self.file_path}"
            logger.error("data.parquet_dataset.not_found", extra={"file_path": str(self.file_path)})
            raise FileNotFoundError(msg)

        # Read columnar table via Polars
        df = pl.read_parquet(self.file_path)
        if max_samples is not None and len(df) > max_samples:
            df = df.slice(0, max_samples)

        self._queries: list[str] = df["query"].to_list()
        self._state_jsons: list[str] = df["state_json"].to_list()
        self._choices: list[list[str]] = df["choices"].to_list()
        self._num_actives: list[int] = df["num_active"].to_list()
        self._target_indices: list[int] = df["target_idx"].to_list()
        self._task_scalars: list[float] = df["task_scalar"].to_list()

        logger.info(
            "data.parquet_dataset.loaded",
            extra={"file_path": str(self.file_path), "samples": len(self._queries)},
        )

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        choices = self._choices[idx]
        num_active = self._num_actives[idx]
        target_idx = self._target_indices[idx]

        if self.shuffle_choices and num_active > 1:
            active = choices[:num_active]
            unused = choices[num_active:]
            perm = torch.randperm(num_active).tolist()
            shuffled_active = [active[p] for p in perm]
            new_target_idx = perm.index(target_idx)
            final_choices = shuffled_active + unused
        else:
            final_choices = list(choices)
            new_target_idx = target_idx

        # Parse lightweight JSON state
        try:
            state = json.loads(self._state_jsons[idx])
        except (json.JSONDecodeError, TypeError):
            state = {}

        return {
            "query": self._queries[idx],
            "state": state if isinstance(state, dict) else {},
            "constraints": final_choices,
            "target_idx": new_target_idx,
            "task_scalar": self._task_scalars[idx],
        }
