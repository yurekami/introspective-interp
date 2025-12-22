from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import openai
import wandb
from evaluator.base_evaluator import BaseEvaluator
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

warnings.filterwarnings("ignore")


class SemanticSimilarityEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: Dict[str, Any],
        model: Any,
        tokenizer: Any,
        test_dataloader: Any,
        fs_dataloader: Optional[Any] = None,
        debug_mode: bool = False,
        fast_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, model, tokenizer, test_dataloader, fs_dataloader)
        self.main_metric = "similarity"
        self.greater_is_better = True
        self.debug_mode = debug_mode
        if fast_eval:
            # Use vLLM for evaluation model loading
            self.eval_model = LLM(
                # model="meta-llama/Llama-3.1-8B-Instruct",
                model="google/gemma-3-27b-it",
                dtype="bfloat16",
                gpu_memory_utilization=0.7,
                max_model_len=128,
            )
            self.eval_tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-3-27b-it",
                padding_side="left",
            )
            self.eval_tokenizer.pad_token = self.eval_tokenizer.eos_token
            self.eval_tokenizer.pad_token_id = self.eval_tokenizer.eos_token_id
        else:
            self.openai_model = "gpt-4.1-mini"
            self.client = openai.OpenAI()
        self.fast_eval = fast_eval

    def reset_metrics(self) -> None:
        self.metrics: Dict[str, float] = {
            "similarity": 0.0,
            "total": 0.0,
        }
        self.similarity_scores: list[float] = []

    def get_similarity_score(
        self, predicted_label: str, expected_label: str, full_item: Dict[str, Any]
    ) -> float:
        """Get similarity score between predicted and expected labels using Llama model or OpenAI GPT API."""
        all_labels = full_item["extra_args"]["all_labels"]
        scores = []

        for label in all_labels:
            prompt = f"""Does this feature description accurately describe when this feature activates? 
Rate on a scale of:
- 1 = completely unrelated to expected
- 2 = mostly unrelated
- 3 = somewhat related
- 4 = related and fairly similar
- 5 = same as expected, or highly similar (treat this as a correct match)

If unsure between 4 and 5, choose 5.

Examples:

Predicted: mentions of cooking recipes
Expected: references to financial transactions
Correct rating: 1

Predicted: mentions of dogs and cats
Expected: references to farm animals
Correct rating: 2

Predicted: mentions of sunny weather and rain
Expected: references to climate conditions
Correct rating: 3

Predicted: mentions of jazz musicians and concerts
Expected: references to music
Correct rating: 4

Predicted: mentions of Shakespeare's plays
Expected: references to works by Shakespeare
Correct rating: 5

Now rate the following pair:

Predicted: {predicted_label}
Expected: {expected_label}

Return a number from 1 to 5 and nothing else."""

            if self.fast_eval:
                try:
                    # Use vLLM for generation with proper chat template
                    messages = [{"role": "user", "content": prompt}]

                    sampling_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=10,
                        stop=["\n", " ", "."],  # Stop at common sentence endings
                    )

                    outputs = self.eval_model.chat(
                        [messages],
                        sampling_params,
                        stop_token_ids=[self.eval_tokenizer.eos_token_id],
                    )
                    response = outputs[0].outputs[0].text.strip()

                    try:
                        score = response.strip().lower()
                        assert score in ["1", "2", "3", "4", "5"]
                    except (ValueError, AssertionError):
                        print(f"Invalid response format: {response}")
                        score = "1"  # Default to 1 if invalid response

                except Exception as e:
                    print(f"Error getting similarity score: {e}")
                    score = "1"  # Default to 1 on error
            else:
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip()
                try:
                    score = content.strip().lower()
                    assert score in ["1", "2", "3", "4", "5"]
                except ValueError:
                    print(f"Invalid response format from OpenAI: {content}")
                    score = "1"
            scores.append(score)

        # Decode and extract scores
        print("Predicted:", predicted_label)
        print("Expected:")

        max_similarity = 0.0
        for i, label in enumerate(all_labels):
            score = min(1.0, (int(scores[i]) - 1) / 4.0)
            max_similarity = max(max_similarity, score)
            if (
                "all_label_scores" in full_item["extra_args"]
                and label in full_item["extra_args"]["all_label_scores"]
                and full_item["extra_args"]["all_label_scores"][label] is not None
            ):
                gt_score = full_item["extra_args"]["all_label_scores"][label]
                print(f"    {all_labels[i]} - {score:.4f} - {gt_score:.4f}")
            else:
                print(f"    {all_labels[i]} - {score:.4f}")
        return max_similarity

    def evaluate_item(
        self,
        predicted_label: str | List[str],
        expected_label: str,
        full_item: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        """Evaluate similarity between predicted and expected labels."""
        if predicted_label is None:
            return None
        if isinstance(predicted_label, str):
            predicted_label = [predicted_label]
        max_similarity = 0.0
        print(f"Feature: {full_item['example']}")
        for pred_label in predicted_label:
            if pred_label.strip() == "":
                similarity_score = 0.0
            else:
                similarity_score = self.get_similarity_score(
                    pred_label, expected_label, full_item
                )
            max_similarity = max(max_similarity, similarity_score)
            if similarity_score == 1.0:
                # Stop if we get a perfect match -- we don't need to check the rest
                break
        print("Max similarity:", max_similarity)
        print("*===*")
        self.metrics["similarity"] += max_similarity
        self.metrics["total"] += 1.0
        self.similarity_scores.append(max_similarity)
        return {self.main_metric: max_similarity}

    def finalize_metrics(self) -> None:
        """Calculate final metrics."""
        if self.metrics["total"] > 0:
            self.metrics["avg_similarity"] = (
                self.metrics["similarity"] / self.metrics["total"]
            )
            # Calculate standard error of the mean
            if len(self.similarity_scores) > 1:
                self.metrics["std_error"] = np.std(
                    self.similarity_scores, ddof=1
                ) / np.sqrt(len(self.similarity_scores))
            else:
                self.metrics["std_error"] = 0.0
        else:
            self.metrics["avg_similarity"] = 0.0
            self.metrics["std_error"] = 0.0

    def pbar_update(self, pbar: tqdm) -> None:
        """Update progress bar with current metrics."""
        avg_similarity = (
            self.metrics["similarity"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0.0
        )
        # Calculate current standard error for progress bar
        if len(self.similarity_scores) > 1:
            current_std_error = np.std(self.similarity_scores, ddof=1) / np.sqrt(
                len(self.similarity_scores)
            )
        else:
            current_std_error = 0.0
        pbar.set_description(
            f"Avg Similarity: {avg_similarity:.4f} ± {current_std_error:.4f}"
        )

    def log_metrics(self, prefix: str = "", step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if not self.debug_mode:
            wandb.log(
                {
                    prefix + "avg_similarity": self.metrics["avg_similarity"],
                    prefix + "std_error": self.metrics.get("std_error", 0.0),
                },
                step=step,
            )

    def print_metrics(self, prefix: str = "") -> None:
        """Print metrics to console."""
        avg_similarity = self.metrics.get("avg_similarity", 0.0)
        std_error = self.metrics.get("std_error", 0.0)
        print(f"{prefix}Average Similarity: {avg_similarity:.4f} ± {std_error:.4f}")
