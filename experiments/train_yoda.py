"""Executable experiment script for multi-dataset training of YodaDecisionEngine."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from yoda.architecture.engine import YodaDecisionEngine
from yoda.training.dataset import YodaDecisionDataset, collate_decision_batch
from yoda.training.trainer import YodaTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("experiments.train_yoda")


def main() -> None:
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
        default=True,
        help="Freeze pretrained text encoder backbone weights",
    )
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--train-path", type=str, default="data/processed/aggregated_train.jsonl")
    parser.add_argument("--eval-path", type=str, default="data/processed/aggregated_eval.jsonl")
    parser.add_argument("--output-model", type=str, default="models/yoda_system1_v1.pt")
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--belnap-weight", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-choices", type=int, default=5)
    parser.add_argument(
        "--shuffle-choices",
        action="store_true",
        default=True,
        help="Randomly permute active candidate choices during training",
    )
    parser.add_argument("--no-shuffle-choices", dest="shuffle_choices", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda:0")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Using GPU: %s (VRAM: %.2f GB)", torch.cuda.get_device_name(0), vram_gb)
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available; executing on CPU")

    logger.info(
        "Loading aggregated training dataset (max_samples=%d, shuffle_choices=%s)...",
        args.max_train_samples,
        args.shuffle_choices,
    )
    t0 = time.time()
    train_dataset = YodaDecisionDataset(
        file_path=args.train_path,
        source=None,
        question_type=None,
        max_samples=args.max_train_samples,
        max_choices=args.num_choices,
        shuffle_choices=args.shuffle_choices,
    )
    logger.info("Loaded %d training samples in %.2fs", len(train_dataset), time.time() - t0)

    logger.info("Loading evaluation dataset (max_samples=%d)...", args.max_eval_samples)
    eval_dataset = YodaDecisionDataset(
        file_path=args.eval_path,
        source=None,
        question_type=None,
        max_samples=args.max_eval_samples,
        max_choices=args.num_choices,
    )
    logger.info("Loaded %d evaluation samples", len(eval_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_decision_batch,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_decision_batch,
        num_workers=0,
    )

    logger.info(
        "Instantiating YodaDecisionEngine (backbone=%s, embed_dim=%d, num_choices=%d)...",
        args.text_model_name,
        args.embed_dim,
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

    # If freeze_backbone is True, unfreeze only the last layer of the text encoder
    if (
        hasattr(model.text_encoder, "model")
        and model.text_encoder.model is not None
        and args.freeze_backbone
    ):
        for param in model.text_encoder.model.parameters():
            param.requires_grad = False
        # Unfreeze the last transformer layer of MiniLM for semantic domain adaptation
        if hasattr(model.text_encoder.model, "encoder") and hasattr(
            model.text_encoder.model.encoder, "layer"
        ):
            for param in model.text_encoder.model.encoder.layer[-1].parameters():
                param.requires_grad = True
            logger.info("Froze text backbone except top layer for domain adaptation")
        else:
            logger.info("Froze pretrained transformer backbone weights")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model created: %s trainable / %s total parameters",
        f"{trainable_params:,}",
        f"{total_params:,}",
    )

    trainer = YodaTrainer(
        model=model,
        lr=args.lr,
        belnap_weight=args.belnap_weight,
        device=device,
    )

    logger.info("Starting %d-epoch training run...", args.epochs)
    start_time = time.time()
    history = trainer.fit(
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=args.epochs,
    )
    total_time = time.time() - start_time

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
            f"{h['train_accuracy']*100:<9.1f}% | "
            f"{h.get('eval_loss', 0.0):<10.4f} | "
            f"{h.get('eval_accuracy', 0.0)*100:<9.1f}% | "
            f"{h.get('eval_mean_knowledge', 0.0):<9.4f}"
        )
    print("=" * 90)
    sec_per_epoch = total_time / args.epochs if args.epochs > 0 else 0.0
    logger.info("Training complete in %.2fs (%.2fs per epoch)", total_time, sec_per_epoch)

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
                "dla_report": dla_report,
            },
        },
        out_path,
    )
    logger.info("Checkpoint saved to: %s", str(out_path))


def run_dla_evaluation(
    model: YodaDecisionEngine,
    eval_loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Runs Direct Logit Attribution (DLA) diagnostic sweep across the evaluation set."""
    model.eval()
    logger.info("Computing Direct Logit Attribution (DLA) diagnostics across eval set...")

    stage_names = [
        "Stage 0 (Post-Pooling)",
        "Stage 1 (Post-Context)",
        "Stage 2 (Post-Constraint)",
    ]
    num_stages = len(stage_names)
    stage_correct = [0.0] * num_stages
    stage_target_logits = [0.0] * num_stages
    stage_knowledge = [0.0] * num_stages
    stage_target_attributions = [0.0] * num_stages
    total_samples = 0

    with torch.no_grad():
        for batch in eval_loader:
            targets = batch["target_indices"].to(device)
            out = model(
                queries=batch["queries"],
                states=batch["states"],
                constraints=batch["constraints"],
                return_diagnostics=True,
            )
            diag = out["diagnostics"]
            s_logits = diag["stage_logits"]  # [num_stages, batch_size, num_choices]
            s_know = diag["stage_knowledge"]
            s_attr = diag["attributions"]

            b_size = targets.size(0)
            total_samples += b_size

            for s in range(num_stages):
                preds = s_logits[s].argmax(dim=-1)
                stage_correct[s] += (preds == targets).sum().item()

                t_logits = s_logits[s].gather(1, targets.unsqueeze(1)).squeeze(1)
                stage_target_logits[s] += t_logits.sum().item()

                t_know = s_know[s].gather(1, targets.unsqueeze(1)).squeeze(1)
                stage_knowledge[s] += t_know.sum().item()

                t_attr = s_attr[s].gather(1, targets.unsqueeze(1)).squeeze(1)
                stage_target_attributions[s] += t_attr.sum().item()

    if total_samples == 0:
        return {}

    stage_accuracies = [c / total_samples for c in stage_correct]
    avg_target_logits = [val / total_samples for val in stage_target_logits]
    avg_knowledge = [k / total_samples for k in stage_knowledge]
    avg_attributions = [a / total_samples for a in stage_target_attributions]

    print("\n" + "=" * 95)
    print("DIRECT LOGIT ATTRIBUTION (DLA) DIAGNOSTIC REPORT (EVAL SET)")
    print("=" * 95)
    header = (
        f"{'Reasoning Stage':<28} | {'Accuracy':<10} | {'Mean Target Logit':<18} | "
        f"{'Marginal Attribution':<22} | {'Mean Target k':<12}"
    )
    print(header)
    print("-" * 95)
    for s in range(num_stages):
        attr_str = (
            f"{avg_attributions[s]:+.4f}" if s > 0 else f"{avg_attributions[s]:.4f} (Base)"
        )
        print(
            f"{stage_names[s]:<28} | "
            f"{stage_accuracies[s]*100:<9.1f}% | "
            f"{avg_target_logits[s]:<18.4f} | "
            f"{attr_str:<22} | "
            f"{avg_knowledge[s]:<12.4f}"
        )
    print("=" * 95 + "\n")

    return {
        "stage_accuracies": stage_accuracies,
        "avg_target_logits": avg_target_logits,
        "avg_knowledge": avg_knowledge,
        "avg_attributions": avg_attributions,
    }


if __name__ == "__main__":
    main()
