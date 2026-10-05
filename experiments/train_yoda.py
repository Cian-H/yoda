"""Executable experiment script for multi-dataset training of YodaDecisionEngine."""

import argparse
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from loguru import logger
from torch.utils.data import DataLoader

from yoda.architecture.engine import YodaDecisionEngine
from yoda.data import (
    YodaDecisionDataset,
    YodaParquetDataset,
    collate_decision_batch,
    compute_split_boundaries,
    run_etl,
    split_parquet_dataset,
)
from yoda.training import YodaLightningAdapter, YodaTrainer
from yoda.xai import run_dla_evaluation

# Loguru logger is already configured via import


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train YodaDecisionEngine on unified System 1 datasets."
    )
    parser.add_argument(
        "--text-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=False,
        help="Freeze pretrained text encoder backbone weights (default: False)",
    )
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument(
        "--unfreeze-top-layer",
        action="store_true",
        default=False,
        help="Unfreeze top layer of text encoder for domain adaptation (default: False)",
    )
    default_train = (
        "data/processed/train.parquet"
        if Path("data/processed/train.parquet").exists()
        else "data/processed/aggregated_train.jsonl"
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=default_train,
        help="Path to primary dataset (.parquet or .jsonl). Split dynamically 80/10/10 by default.",
    )
    parser.add_argument(
        "--eval-path",
        type=str,
        default=None,
        help="Optional explicit path to evaluation dataset (default: None, slices from primary)",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Optional explicit path to held-out test dataset (default: None, slices from primary)",
    )
    parser.add_argument("--output-model", type=str, default="models/yoda_system1_v1.pt")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples to load from primary dataset before 80/10/10 split",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Maximum samples for training split (default: None, derived from train-ratio)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Maximum samples for evaluation split (default: None, derived from val-ratio)",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Maximum samples for held-out test split (default: None, derived from test-ratio)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of primary dataset allocated to training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of primary dataset allocated to evaluation/validation (default: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of primary dataset allocated to held-out test (default: 0.1)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for the text backbone.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-4,
        help="Minimum learning rate for scheduler warmup/annealing floor (default: 1e-4)",
    )
    parser.add_argument(
        "--t0-epochs",
        type=int,
        default=2,
        help="Number of epochs for initial cosine cycle length T_0 (default: 2)",
    )
    parser.add_argument(
        "--t-mult",
        type=int,
        default=2,
        help="Cycle lengthening factor for cosine annealing warm restarts (default: 2)",
    )
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.75,
        help="Decay factor for base learning rate upon restart (default: 0.75)",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="runs",
        help="Directory to save TensorBoard scalar telemetry events",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to save per-epoch model checkpoints and history.json",
    )
    parser.add_argument("--belnap-weight", type=float, default=1.0)
    parser.add_argument(
        "--ltn-weight",
        type=float,
        default=0.1,
        help="Weight multiplier for task-conditioned LTN constraint loss",
    )
    parser.add_argument(
        "--assertion-weight",
        type=float,
        default=0.1,
        help="Weight multiplier for assertion loss (default: 0.1)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="Latent embedding dimension (default: None, adopts text backbone native dim)",
    )
    parser.add_argument(
        "--independent-eval",
        action="store_true",
        default=True,
        help="Evaluate candidate options independently with grouped losses (default: True)",
    )
    parser.add_argument(
        "--legacy-eval",
        dest="independent_eval",
        action="store_false",
        help="Evaluate candidates using legacy joint multi-choice mode",
    )
    parser.add_argument("--num-choices", type=int, default=5)
    parser.add_argument(
        "--shuffle-choices",
        action="store_true",
        default=True,
        help="Randomly permute active candidate choices during training",
    )
    parser.add_argument("--no-shuffle-choices", dest="shuffle_choices", action="store_false")
    parser.add_argument(
        "--use-lightning",
        action="store_true",
        default=False,
        help="Train using PyTorch Lightning adapter wrapper",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=False,
        help="Enable pinned memory for faster host-to-device transfers",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def prepare_datasets(
    args: argparse.Namespace,
) -> tuple[
    YodaDecisionDataset | YodaParquetDataset,
    YodaDecisionDataset | YodaParquetDataset,
    YodaDecisionDataset | YodaParquetDataset,
]:
    """Load and partition datasets for training, evaluation, and testing."""

    def _create_dataset(
        path_str: str,
        max_samples: int | None,
        offset: int = 0,
        max_choices: int = 5,
        shuffle_choices: bool = False,
    ) -> YodaDecisionDataset | YodaParquetDataset:
        p = Path(path_str)
        if p.suffix.lower() == ".parquet":
            return YodaParquetDataset(
                file_path=p,
                max_samples=max_samples,
                offset=offset,
                shuffle_choices=shuffle_choices,
            )
        if p.suffix.lower() == ".jsonl":
            parquet_path = run_etl(
                input_path=p,
                max_choices=max_choices,
            )
            return YodaParquetDataset(
                file_path=parquet_path,
                max_samples=max_samples,
                offset=offset,
                shuffle_choices=shuffle_choices,
            )
        return YodaDecisionDataset(
            file_path=p,
            source=None,
            question_type=None,
            max_samples=max_samples,
            offset=offset,
            max_choices=max_choices,
            shuffle_choices=shuffle_choices,
        )

    t0 = time.time()
    train_file = Path(args.train_path)

    if args.eval_path is None:
        logger.info(
            "Partitioning primary dataset {} with dynamic {:.0f}/{:.0f}/{:.0f} "
            "(Train/Val/Test) split...",
            str(train_file),
            args.train_ratio * 100,
            args.val_ratio * 100,
            args.test_ratio * 100,
        )
        if train_file.suffix.lower() == ".parquet":
            actual_parquet = train_file
        elif train_file.suffix.lower() == ".jsonl":
            actual_parquet = run_etl(
                input_path=train_file,
                max_choices=args.num_choices,
            )
        else:
            actual_parquet = None

        if actual_parquet is not None:
            train_dataset, eval_dataset, test_dataset = split_parquet_dataset(
                file_path=actual_parquet,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                max_samples=args.max_samples,
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                max_test_samples=args.max_test_samples,
                shuffle_train_choices=args.shuffle_choices,
            )
        else:
            base_ds = YodaDecisionDataset(
                file_path=train_file,
                max_choices=args.num_choices,
            )
            total_available = len(base_ds)
            total_samples = (
                min(total_available, args.max_samples)
                if args.max_samples is not None
                else total_available
            )
            (tr_off, tr_n), (ev_off, ev_n), (te_off, te_n) = compute_split_boundaries(
                total_samples=total_samples,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )
            tr_cnt = (
                min(tr_n, args.max_train_samples)
                if args.max_train_samples is not None
                else tr_n
            )
            ev_cnt = (
                min(ev_n, args.max_eval_samples)
                if args.max_eval_samples is not None
                else ev_n
            )
            te_cnt = (
                min(te_n, args.max_test_samples)
                if args.max_test_samples is not None
                else te_n
            )
            train_dataset = YodaDecisionDataset(
                file_path=train_file,
                offset=tr_off,
                max_samples=tr_cnt,
                max_choices=args.num_choices,
                shuffle_choices=args.shuffle_choices,
            )
            eval_dataset = YodaDecisionDataset(
                file_path=train_file,
                offset=ev_off,
                max_samples=ev_cnt,
                max_choices=args.num_choices,
                shuffle_choices=False,
            )
            test_dataset = YodaDecisionDataset(
                file_path=train_file,
                offset=te_off,
                max_samples=te_cnt,
                max_choices=args.num_choices,
                shuffle_choices=False,
            )
    else:
        # Explicit evaluation path provided by user
        eval_file = Path(args.eval_path)
        logger.info("Using explicit evaluation dataset from {}", str(eval_file))
        train_dataset = _create_dataset(
            path_str=str(train_file),
            max_samples=args.max_train_samples,
            offset=0,
            max_choices=args.num_choices,
            shuffle_choices=args.shuffle_choices,
        )
        if args.test_path is not None:
            logger.info("Using explicit test dataset from {}", args.test_path)
            eval_dataset = _create_dataset(
                path_str=str(eval_file),
                max_samples=args.max_eval_samples,
                offset=0,
                max_choices=args.num_choices,
                shuffle_choices=False,
            )
            test_dataset = _create_dataset(
                path_str=str(args.test_path),
                max_samples=args.max_test_samples,
                offset=0,
                max_choices=args.num_choices,
                shuffle_choices=False,
            )
        else:
            # Dynamic 50/50 partition on explicit eval_file if no test path specified
            if eval_file.suffix.lower() == ".parquet":
                eval_dataset, test_dataset, _ = split_parquet_dataset(
                    file_path=eval_file,
                    train_ratio=0.5,
                    val_ratio=0.5,
                    test_ratio=0.0,
                    max_samples=args.max_samples,
                    max_train_samples=args.max_eval_samples,
                    max_eval_samples=args.max_test_samples,
                    shuffle_train_choices=False,
                )
            else:
                eval_dataset = _create_dataset(
                    path_str=str(eval_file),
                    max_samples=args.max_eval_samples,
                    offset=0,
                    max_choices=args.num_choices,
                    shuffle_choices=False,
                )
                test_dataset = _create_dataset(
                    path_str=str(eval_file),
                    max_samples=args.max_test_samples,
                    offset=len(eval_dataset),
                    max_choices=args.num_choices,
                    shuffle_choices=False,
                )

    logger.info(
        "Partitioned datasets in {:.2f}s: Train={} samples, Val={} samples, "
        "Test={} samples (Total={})",
        time.time() - t0,
        len(train_dataset),
        len(eval_dataset),
        len(test_dataset),
        len(train_dataset) + len(eval_dataset) + len(test_dataset),
    )
    return train_dataset, eval_dataset, test_dataset


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda:0")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Using GPU: {} (VRAM: {:.2f} GB)", torch.cuda.get_device_name(0), vram_gb)
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available; executing on CPU")

    train_dataset, eval_dataset, test_dataset = prepare_datasets(args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_decision_batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    logger.info(
        "Train DataLoader configured with shuffle=True (sample order randomized every epoch)"
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_decision_batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_decision_batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    logger.info(
        "Instantiating YodaDecisionEngine (backbone={}, embed_dim={}, num_choices={})...",
        args.text_model_name,
        str(args.embed_dim),
        args.num_choices,
    )
    model = YodaDecisionEngine(
        text_model_name=args.text_model_name,
        embed_dim=args.embed_dim,
        num_q_probes=4,
        num_c_probes=8,
        num_k_probes=args.num_choices,
        num_choices=args.num_choices,
        n_heads=4,
        device=device,
    )
    model = model.to(device)

    # Text encoder backbone freezing
    if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
        if args.freeze_backbone:
            for param in model.text_encoder.model.parameters():
                param.requires_grad = False

            if (
                args.unfreeze_top_layer
                and hasattr(model.text_encoder.model, "encoder")
                and hasattr(model.text_encoder.model.encoder, "layer")
            ):
                for param in model.text_encoder.model.encoder.layer[-1].parameters():
                    param.requires_grad = True
                logger.info("Froze text backbone except top layer for domain adaptation")
            else:
                encoder = getattr(model.text_encoder.model, "encoder", None)
                num_layers = len(getattr(encoder, "layer", []))
                logger.info(
                    "Froze text backbone 100% ({} transformer layers fully frozen)",
                    num_layers,
                )
        else:
            logger.info("Text encoder backbone left 100% trainable")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model created: {} trainable / {} total parameters",
        f"{trainable_params:,}",
        f"{total_params:,}",
    )

    test_metrics: dict[str, Any] = {}

    if args.use_lightning:
        logger.info("Starting PyTorch Lightning training run ({} epochs)...", args.epochs)
        adapter = YodaLightningAdapter(
            model=model,
            lr=args.lr,
            belnap_weight=args.belnap_weight,
            ltn_weight=args.ltn_weight,
        )
        accelerator = "gpu" if device.type == "cuda" else "cpu"
        pl_trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator=accelerator,
            devices=1,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )
        start_time = time.time()
        pl_trainer.fit(model=adapter, train_dataloaders=train_loader, val_dataloaders=eval_loader)
        total_time = time.time() - start_time
        if len(test_dataset) > 0:
            pl_trainer.test(model=adapter, dataloaders=test_loader)
        history: list[dict[str, Any]] = []
    else:
        trainer = YodaTrainer(
            model=model,
            lr=args.lr,
            backbone_lr=args.backbone_lr,
            min_lr=args.min_lr,
            t0_epochs=args.t0_epochs,
            t_mult=args.t_mult,
            lr_decay=args.lr_decay,
            belnap_weight=args.belnap_weight,
            ltn_weight=args.ltn_weight,
            assertion_weight=args.assertion_weight,
            independent_eval=args.independent_eval,
            tensorboard_dir=args.tensorboard_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=device,
        )

        logger.info("Starting {}-epoch training run with native YodaTrainer...", args.epochs)
        start_time = time.time()
        history = trainer.fit(
            train_loader=train_loader,
            eval_loader=eval_loader,
            epochs=args.epochs,
        )
        total_time = time.time() - start_time

        # Post-training evaluation on held-out test split
        logger.info("Running evaluation on held-out test split ({} samples)...", len(test_dataset))
        test_metrics = trainer.evaluate(test_loader)
        logger.info(
            "Test Results | Loss: {:.4f} | CE: {:.4f} | Belnap: {:.4f} | "
            "LTN: {:.4f} | Assertion: {:.4f} | Accuracy: {:.2f}% | Knowledge: {:.4f}",
            test_metrics["loss"],
            test_metrics["ce_loss"],
            test_metrics["belnap_loss"],
            test_metrics["ltn_loss"],
            test_metrics.get("assertion_loss", 0.0),
            test_metrics["accuracy"] * 100,
            test_metrics["mean_knowledge"],
        )

        if trainer.writer is not None:
            trainer.writer.add_scalar("test/loss", test_metrics["loss"], args.epochs)
            trainer.writer.add_scalar("test/ce_loss", test_metrics["ce_loss"], args.epochs)
            trainer.writer.add_scalar("test/belnap_loss", test_metrics["belnap_loss"], args.epochs)
            trainer.writer.add_scalar("test/ltn_loss", test_metrics["ltn_loss"], args.epochs)
            trainer.writer.add_scalar(
                "test/assertion_loss", test_metrics.get("assertion_loss", 0.0), args.epochs
            )
            trainer.writer.add_scalar("test/accuracy", test_metrics["accuracy"], args.epochs)
            trainer.writer.add_scalar(
                "test/mean_knowledge", test_metrics["mean_knowledge"], args.epochs
            )
            trainer.writer.flush()
        trainer.close()

        print("\n" + "=" * 90)
        header = (
            f"{'Epoch':<6} | {'Train Loss':<11} | {'Train CE':<10} | {'Train Belnap':<13} | "
            f"{'Train Acc':<10} | {'Eval Loss':<10} | {'Eval Acc':<10} | {'Knowledge':<9}"
        )
        print(header)
        print("-" * 90)
        for h in history:
            print(
                f"{int(h['epoch']):<6} | "
                f"{h['train_loss']:<11.4f} | "
                f"{h['train_ce_loss']:<10.4f} | "
                f"{h['train_belnap_loss']:<13.4f} | "
                f"{h['train_accuracy'] * 100:<9.1f}% | "
                f"{h.get('eval_loss', 0.0):<10.4f} | "
                f"{h.get('eval_accuracy', 0.0) * 100:<9.1f}% | "
                f"{h.get('eval_mean_knowledge', 0.0):<9.4f}"
            )
        print("=" * 90)
        if len(test_dataset) > 0 and test_metrics:
            print(
                f"TEST SPLIT ({len(test_dataset)} samples) | "
                f"Loss: {test_metrics['loss']:.4f} | "
                f"Accuracy: {test_metrics['accuracy'] * 100:.2f}% | "
                f"Knowledge: {test_metrics['mean_knowledge']:.4f}"
            )
            print("=" * 90)

    sec_per_epoch = total_time / args.epochs if args.epochs > 0 else 0.0
    logger.info("Training complete in {:.2f}s ({:.2f}s per epoch)", total_time, sec_per_epoch)

    # Direct Logit Attribution (DLA) Diagnostic Report
    dla_report = run_dla_evaluation(model=model, eval_loader=eval_loader, device=device)

    # Save model weights
    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "embed_dim": args.embed_dim,
                "num_choices": args.num_choices,
                "epochs": args.epochs,
                "history": history,
                "test_metrics": test_metrics,
                "dla_report": dla_report,
            },
        },
        out_path,
    )
    logger.info("Checkpoint saved to: {}", str(out_path))


if __name__ == "__main__":
    main()
