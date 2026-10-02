"""CLI script executing offline ETL from JSONL to standardized columnar Parquet."""

import argparse
import logging
import time
from pathlib import Path

from yoda.data import ParquetETLPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("experiments.run_etl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETL raw JSONL datasets into optimized Parquet tables."
    )
    parser.add_argument("--train-jsonl", type=str, default="data/processed/aggregated_train.jsonl")
    parser.add_argument("--eval-jsonl", type=str, default="data/processed/aggregated_eval.jsonl")
    parser.add_argument("--train-parquet", type=str, default="data/processed/train.parquet")
    parser.add_argument("--eval-parquet", type=str, default="data/processed/eval.parquet")
    parser.add_argument("--max-choices", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()

    pipeline = ParquetETLPipeline(max_choices=args.max_choices)

    # 1. Transform training set
    train_in = Path(args.train_jsonl)
    train_out = Path(args.train_parquet)
    if train_in.exists():
        t0 = time.time()
        logger.info("Transforming %s -> %s...", str(train_in), str(train_out))
        df_train = pipeline.transform_jsonl_to_parquet(
            input_path=train_in,
            output_path=train_out,
            max_samples=args.max_train_samples,
        )
        logger.info("Transformed %d training rows in %.2fs", len(df_train), time.time() - t0)
    else:
        logger.warning("Training file not found: %s", str(train_in))

    # 2. Transform evaluation set
    eval_in = Path(args.eval_jsonl)
    eval_out = Path(args.eval_parquet)
    if eval_in.exists():
        t0 = time.time()
        logger.info("Transforming %s -> %s...", str(eval_in), str(eval_out))
        df_eval = pipeline.transform_jsonl_to_parquet(
            input_path=eval_in,
            output_path=eval_out,
            max_samples=args.max_eval_samples,
        )
        logger.info("Transformed %d evaluation rows in %.2fs", len(df_eval), time.time() - t0)
    else:
        logger.warning("Evaluation file not found: %s", str(eval_in))


if __name__ == "__main__":
    main()
