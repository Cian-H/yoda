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
    "compute_split_boundaries",
    "split_parquet_dataset",
]


class YodaParquetDataset(Dataset[dict[str, Any]]):
    """High-throughput PyTorch Dataset backed by an Apache Parquet columnar table."""

    def __init__(
        self,
        file_path: Path | str | pl.DataFrame,
        max_samples: int | None = None,
        offset: int = 0,
        shuffle_choices: bool = False,
    ) -> None:
        """Initializes dataset by memory-mapping or reading a Parquet table.

        Args:
            file_path: Path to the .parquet dataset file, or an in-memory Polars DataFrame.
            max_samples: Maximum rows to retain.
            offset: Number of initial rows to skip before retaining max_samples.
            shuffle_choices: Whether to randomly permute active candidate choices in __getitem__.
        """
        super().__init__()
        self.offset = int(offset)
        self.shuffle_choices = shuffle_choices

        if isinstance(file_path, pl.DataFrame):
            self.file_path: Path | None = None
            df = file_path
        else:
            self.file_path = Path(file_path)
            if not self.file_path.exists():
                msg = f"Parquet file not found: {self.file_path}"
                logger.error(
                    "data.parquet_dataset.not_found",
                    extra={"file_path": str(self.file_path)},
                )
                raise FileNotFoundError(msg)
            df = pl.read_parquet(self.file_path)

        if self.offset > 0 or max_samples is not None:
            df = df.slice(self.offset, max_samples)

        self._queries: list[str] = df["query"].to_list()
        self._state_jsons: list[str] = df["state_json"].to_list()
        self._choices: list[list[str]] = df["choices"].to_list()
        self._num_actives: list[int] = df["num_active"].to_list()
        self._target_indices: list[int] = df["target_idx"].to_list()
        self._task_scalars: list[float] = df["task_scalar"].to_list()

        logger.info(
            "data.parquet_dataset.loaded",
            extra={
                "file_path": str(self.file_path) if self.file_path else "in_memory_dataframe",
                "samples": len(self._queries),
            },
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
            "num_active": num_active,
        }


def compute_split_boundaries(
    total_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Calculates non-overlapping (offset, count) slices for train, val, test splits.

    Ensures that sum of counts equals total_samples (or 0 if total_samples == 0)
    and slices are strictly contiguous and non-overlapping.

    Args:
        total_samples: Total number of rows/samples to partition.
        train_ratio: Fraction allocated to training (default: 0.8).
        val_ratio: Fraction allocated to validation/eval (default: 0.1).
        test_ratio: Fraction allocated to test (default: 0.1).

    Returns:
        Tuple of (train_slice, val_slice, test_slice) where each slice is (offset, count).
    """
    if total_samples <= 0:
        return (0, 0), (0, 0), (0, 0)

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-4:
        msg = f"Split ratios must sum to 1.0, got {ratio_sum:.4f}"
        raise ValueError(msg)

    if total_samples < 3:
        n_eval = 1 if total_samples > 1 else 0
        n_test = 1 if total_samples > 2 else 0
        n_train = total_samples - n_eval - n_test
    else:
        n_eval = max(1, round(total_samples * val_ratio))
        n_test = max(1, round(total_samples * test_ratio))
        if n_eval + n_test >= total_samples:
            n_eval = 1
            n_test = 1
        n_train = total_samples - n_eval - n_test

    train_slice = (0, n_train)
    val_slice = (n_train, n_eval)
    test_slice = (n_train + n_eval, n_test)
    return train_slice, val_slice, test_slice


def split_parquet_dataset(
    file_path: Path | str | pl.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: int | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
    shuffle_train_choices: bool = True,
) -> tuple[YodaParquetDataset, YodaParquetDataset, YodaParquetDataset]:
    """Splits a Parquet dataset into non-overlapping train, eval, and test datasets.

    Args:
        file_path: Path to the parquet file or pre-loaded Polars DataFrame.
        train_ratio: Fraction allocated to training (default: 0.8).
        val_ratio: Fraction allocated to validation/eval (default: 0.1).
        test_ratio: Fraction allocated to held-out test (default: 0.1).
        max_samples: Optional cap on total samples before splitting.
        max_train_samples: Optional cap on train split samples.
        max_eval_samples: Optional cap on eval split samples.
        max_test_samples: Optional cap on test split samples.
        shuffle_train_choices: Whether to shuffle active candidate choices in train dataset.

    Returns:
        Tuple of (train_dataset, eval_dataset, test_dataset).
    """
    if isinstance(file_path, pl.DataFrame):
        df = file_path
    else:
        p = Path(file_path)
        if not p.exists():
            msg = f"Parquet file not found: {p}"
            logger.error("data.parquet_dataset.not_found", extra={"file_path": str(p)})
            raise FileNotFoundError(msg)
        df = pl.read_parquet(p)

    total_available = len(df)
    total_samples = (
        min(total_available, max_samples) if max_samples is not None else total_available
    )

    (train_off, n_train), (val_off, n_eval), (test_off, n_test) = compute_split_boundaries(
        total_samples=total_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_count = min(n_train, max_train_samples) if max_train_samples is not None else n_train
    val_count = min(n_eval, max_eval_samples) if max_eval_samples is not None else n_eval
    test_count = min(n_test, max_test_samples) if max_test_samples is not None else n_test

    train_df = df.slice(train_off, train_count)
    eval_df = df.slice(val_off, val_count)
    test_df = df.slice(test_off, test_count)

    train_ds = YodaParquetDataset(train_df, shuffle_choices=shuffle_train_choices)
    eval_ds = YodaParquetDataset(eval_df, shuffle_choices=False)
    test_ds = YodaParquetDataset(test_df, shuffle_choices=False)

    return train_ds, eval_ds, test_ds
