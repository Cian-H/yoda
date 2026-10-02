"""Direct Logit Attribution (DLA) diagnostics and mechanistic reasoning audits."""

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from yoda.architecture.engine import YodaDecisionEngine

logger = logging.getLogger(__name__)

STAGE_NAMES: list[str] = [
    "Stage 0 (Post-Pooling)",
    "Stage 1 (Post-Context)",
    "Stage 2 (Post-Constraint)",
]


class DirectLogitAttribution:
    """Mechanistic interpretability tool measuring logit attribution across reasoning stages."""

    def __init__(self, stage_names: list[str] | None = None) -> None:
        self.stage_names = stage_names or STAGE_NAMES

    def evaluate(
        self,
        model: YodaDecisionEngine,
        eval_loader: DataLoader[dict[str, Any]],
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Computes stage-wise marginal logit attribution, accuracy, and knowledge calibration."""
        target_device = torch.device(device) if isinstance(device, str) else device
        model = model.to(target_device)
        model.eval()
        logger.info(
            "xai.dla.evaluation_started",
            extra={"num_stages": len(self.stage_names), "device": str(target_device)},
        )

        num_stages = len(self.stage_names)
        stage_correct = [0.0] * num_stages
        stage_target_logits = [0.0] * num_stages
        stage_knowledge = [0.0] * num_stages
        stage_target_attributions = [0.0] * num_stages
        total_samples = 0

        with torch.no_grad():
            for batch in eval_loader:
                targets = batch["target_indices"].to(target_device).view(-1)
                task_scalars = batch.get("task_scalars")
                if task_scalars is not None:
                    task_scalars = task_scalars.to(target_device)

                out = model(
                    queries=batch["queries"],
                    states=batch["states"],
                    constraints=batch["constraints"],
                    task_scalars=task_scalars,
                    return_diagnostics=True,
                )
                diag = out["diagnostics"]
                s_logits = diag["stage_logits"]
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
            logger.warning("xai.dla.empty_dataset")
            return {}

        results = {
            "stage_names": self.stage_names,
            "total_samples": total_samples,
            "stage_accuracies": [c / total_samples for c in stage_correct],
            "avg_target_logits": [val / total_samples for val in stage_target_logits],
            "avg_knowledge": [k / total_samples for k in stage_knowledge],
            "avg_attributions": [a / total_samples for a in stage_target_attributions],
        }
        logger.info("xai.dla.evaluation_completed", extra={"samples": total_samples})
        return results

    def format_report(self, report_dict: dict[str, Any]) -> str:
        """Formats DLA metrics into an ASCII diagnostic table."""
        if not report_dict or "stage_names" not in report_dict:
            return "No DLA metrics available."

        names = report_dict["stage_names"]
        accs = report_dict["stage_accuracies"]
        logits = report_dict["avg_target_logits"]
        attrs = report_dict["avg_attributions"]
        knows = report_dict["avg_knowledge"]

        lines = [
            "=" * 95,
            "DIRECT LOGIT ATTRIBUTION (DLA) DIAGNOSTIC REPORT",
            "=" * 95,
            (
                f"{'Reasoning Stage':<28} | {'Accuracy':<10} | {'Mean Target Logit':<18} | "
                f"{'Marginal Attribution':<22} | {'Mean Target k':<12}"
            ),
            "-" * 95,
        ]

        for s in range(len(names)):
            attr_str = f"{attrs[s]:+.4f}" if s > 0 else f"{attrs[s]:.4f} (Base)"
            line = (
                f"{names[s]:<28} | "
                f"{accs[s] * 100:<9.1f}% | "
                f"{logits[s]:<18.4f} | "
                f"{attr_str:<22} | "
                f"{knows[s]:<12.4f}"
            )
            lines.append(line)

        lines.append("=" * 95)
        return "\n".join(lines)


def run_dla_evaluation(
    model: YodaDecisionEngine,
    eval_loader: DataLoader[dict[str, Any]],
    device: str | torch.device = "cpu",
    print_report: bool = True,
) -> dict[str, Any]:
    """Convenience function running DLA evaluation and optionally printing report."""
    attributor = DirectLogitAttribution()
    report = attributor.evaluate(model=model, eval_loader=eval_loader, device=device)
    if print_report:
        print("\n" + attributor.format_report(report) + "\n")
    return report
