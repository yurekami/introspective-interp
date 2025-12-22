from typing import Any, Dict, Optional

import wandb
from evaluator.base_evaluator import BaseEvaluator
from evaluator.causal_exact_match_eval import CausalExactMatchEvaluator
from evaluator.loss_eval import LossEvaluator
from evaluator.semantic_similarity_eval import SemanticSimilarityEvaluator
from evaluator.simulator_correlation_eval import SimulatorCorrelationEvaluator

EVALUATOR_MAPPING: Dict[str, Any] = {
    "loss": LossEvaluator,
    "semantic_similarity": SemanticSimilarityEvaluator,
    "simulator_correlation": SimulatorCorrelationEvaluator,
    "exact_match": CausalExactMatchEvaluator,
}


class MixedTypeEvaluator(BaseEvaluator):
    def __init__(
        self, config: Dict[str, Any], model: Any, tokenizer: Any, test_dataloader: Any, fs_dataloader: Optional[Any] = None, **kwargs: Any
    ) -> None:
        super().__init__(
            config, model, tokenizer, test_dataloader, fs_dataloader, **kwargs
        )
        self.generate_limit = config.get(
            "generate_limit", 100
        )  # Longer for open-ended responses

        self.question_evaluators = {}
        self.any_continuous_output = False
        # Create specialized evaluators for different question types
        for question_type in self.config["question_types"]:
            question_config = self.config["question_types"][question_type]
            if "evaluation_type" not in question_config:
                continue
            self.question_evaluators[question_type] = EVALUATOR_MAPPING[
                question_config["evaluation_type"]
            ](
                question_config,
                model,
                tokenizer,
                test_dataloader,
                fs_dataloader,
                **kwargs,
            )
            self.any_continuous_output |= (
                getattr(self.question_evaluators[question_type], "output_type", None)
                == "continuous"
            )
        self.output_type = "continuous" if self.any_continuous_output else "text"

    def reset_metrics(self) -> None:
        """Reset all metrics for each question type"""
        for evaluator in self.question_evaluators.values():
            evaluator.reset_metrics()

        # Initialize metrics for each question type
        self.metrics = {
            "total_by_type": {qtype: 0 for qtype in self.question_evaluators.keys()},
            "metric_by_type": {qtype: 0 for qtype in self.question_evaluators.keys()},
            "total": 0,
        }

    def get_prediction(self, outputs: Any, inputs: Any) -> tuple[Any, Any]:
        if inputs["question_type"] not in self.question_evaluators:
            return None, None
        return self.question_evaluators[inputs["question_type"]].get_prediction(
            outputs, inputs
        )

    def evaluate_item(self, predicted_label: str, expected_label: str, full_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate based on question type"""
        question_type = full_item["question_type"]
        self.metrics["total"] += 1

        if question_type not in self.question_evaluators:
            return None
        increment_metric = self.question_evaluators[question_type].evaluate_item(
            predicted_label, expected_label, full_item
        )

        if increment_metric is not None:
            self.metrics["total_by_type"][question_type] += 1
            self.metrics["metric_by_type"][question_type] += increment_metric[
                self.question_evaluators[question_type].main_metric
            ]
        # self.metrics["correct"] = self.question_evaluators[question_type].metrics["correct"]

        return increment_metric

    def finalize_metrics(self) -> None:
        """Calculate final metrics"""
        # Calculate main for each question type
        self.metrics["metric"] = sum(self.metrics["metric_by_type"].values())
        for qtype in self.question_evaluators.keys():
            self.question_evaluators[qtype].finalize_metrics()
            if self.metrics["total_by_type"][qtype] > 0:
                self.metrics[f"{qtype}_metric"] = (
                    self.metrics["metric_by_type"][qtype]
                    / self.metrics["total_by_type"][qtype]
                )
            else:
                self.metrics[f"{qtype}_metric"] = 0.0

    def pbar_update(self, pbar: Any) -> None:
        """Update progress bar with current metrics"""
        if self.metrics["total"] > 0:
            # Get main metrics by type
            type_metrics = {
                qtype: (
                    self.metrics["metric_by_type"][qtype]
                    / self.metrics["total_by_type"][qtype]
                    if self.metrics["total_by_type"][qtype] > 0
                    else 0.0
                )
                for qtype in self.question_evaluators.keys()
                if self.metrics["total_by_type"][qtype] > 0
            }

            # Get counts by type
            type_counts = {
                qtype: self.metrics["total_by_type"][qtype]
                for qtype in self.question_evaluators.keys()
                if self.metrics["total_by_type"][qtype] > 0
            }

            pbar.set_description(
                " | ".join(
                    [
                        f"{qtype[:3]}: {metric:.2f} ({count})"
                        for qtype, metric, count in zip(
                            type_metrics.keys(),
                            type_metrics.values(),
                            type_counts.values(),
                        )
                    ]
                )
            )

    def log_metrics(self, prefix: str = "", step: Optional[int] = None) -> None:
        """Log metrics to wandb"""
        # Log overall metrics
        log_metrics = {}

        # Log metrics for each question type
        for qtype in self.question_evaluators.keys():
            if self.metrics["total_by_type"][qtype] > 0:
                log_metrics[prefix + f"{qtype}/metric"] = self.metrics[
                    f"{qtype}_metric"
                ]
            self.question_evaluators[qtype].log_metrics(prefix=prefix + f"{qtype}/")

        wandb.log(log_metrics, step=step)

    def print_metrics(self, prefix: str = "") -> None:
        """Print metrics to console"""
        print(f"{prefix}Results by question type:")
        for qtype in self.question_evaluators.keys():
            if self.metrics["total_by_type"][qtype] > 0:
                print(
                    f"  {qtype}: {self.metrics[f'{qtype}_metric']:.4f} ({self.metrics['metric_by_type'][qtype]}/{self.metrics['total_by_type'][qtype]})"
                )
            self.question_evaluators[qtype].print_metrics(prefix=f"    {qtype} ")
