import math
from typing import Any, Dict, Optional

import wandb
from evaluator.base_evaluator import BaseEvaluator


class LossEvaluator(BaseEvaluator):
    def __init__(self, config: Dict[str, Any], model: Any, tokenizer: Any, test_dataloader: Any, fs_dataloader: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(config, model, tokenizer, test_dataloader, fs_dataloader, **kwargs)
        self.main_metric = "loss"
        self.generation_evaluation = False

    def reset_metrics(self) -> None:
        self.metrics = {
            "loss": 0,
            "continuous_loss": 0,
            "total": 0,
        }

    def get_prediction(self, outputs: Dict[str, Any], inputs: Any) -> Dict[str, Any]:
        return {
            "loss": outputs["loss"].item(),
            "continuous_loss": (
                outputs["continuous_loss"] if outputs.get("continuous_loss") is not None else None
            ),
        }

    def evaluate_item(self, predicted_label: Dict[str, Any], expected_label: str, full_item: Dict[str, Any]) -> Dict[str, Any]:
        self.metrics["loss"] += predicted_label["loss"]
        self.metrics["continuous_loss"] += (
            predicted_label["continuous_loss"]
            if predicted_label.get("continuous_loss") is not None
            else 0
        )
        self.metrics["total"] += 1
        return {
            "loss": predicted_label["loss"],
            "continuous_loss": predicted_label["continuous_loss"],
        }

    def pbar_update(self, pbar: Any) -> None:
        loss_avg = (
            self.metrics["loss"] / self.metrics["total"] if self.metrics["total"] > 0 else math.nan
        )
        pbar.set_description(f"Loss: {loss_avg:.4f}")

    def log_metrics(self, prefix: str = "", step: Optional[int] = None) -> None:
        loss_avg = (
            self.metrics["loss"] / self.metrics["total"] if self.metrics["total"] > 0 else math.nan
        )
        continuous_loss_avg = (
            self.metrics["continuous_loss"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else math.nan
        )
        wandb.log(
            {
                f"{prefix}loss": loss_avg,
                f"{prefix}continuous_loss": continuous_loss_avg,
            },
            step=step,
        )

    def print_metrics(self, prefix: str = "") -> None:
        loss_avg = (
            self.metrics["loss"] / self.metrics["total"] if self.metrics["total"] > 0 else math.nan
        )
        continuous_loss_avg = (
            self.metrics["continuous_loss"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else math.nan
        )
        print(f"{prefix}Loss: {loss_avg:.4f}, Continuous Loss: {continuous_loss_avg:.4f}")

    def finalize_metrics(self) -> None:
        pass

    #     self.metrics["loss"] /= self.metrics["total"]
