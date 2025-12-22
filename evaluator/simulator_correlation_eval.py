from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from evaluator.base_evaluator import BaseEvaluator
from observatory_utils.general import Subject, get_subject_config, ActivationSign
from observatory_utils.exemplar import ExemplarConfig, ExemplarSplit
from observatory_utils.explanation import ExplanationConfig, ExplanationsWrapper
from observatory_utils.simulator import FinetunedSimulator
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import get_first_available_device


class FeatureDescriptorSubject(Subject):
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            get_subject_config(tokenizer.name_or_path),
            device=device,
        )
        self.model._model = model
        self.tokenizer = tokenizer


# ANSI color codes for terminal highlighting
class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BRIGHT_RED = "\033[31;1m"
    BRIGHT_GREEN = "\033[32;1m"
    BRIGHT_YELLOW = "\033[33;1m"
    BRIGHT_BLUE = "\033[34;1m"
    BRIGHT_MAGENTA = "\033[35;1m"
    BRIGHT_CYAN = "\033[36;1m"

    @staticmethod
    def colorize(text: str, color: str) -> str:
        """Apply color to text."""
        return f"{color}{text}{Colors.RESET}"

    @staticmethod
    def highlight_score(
        score: float, threshold_high: float = 0.8, threshold_medium: float = 0.5
    ) -> str:
        """Color-code a score based on its value."""
        if score >= threshold_high:
            return Colors.colorize(f"{score:.4f}", Colors.BRIGHT_GREEN)
        elif score >= threshold_medium:
            return Colors.colorize(f"{score:.4f}", Colors.BRIGHT_YELLOW)
        else:
            return Colors.colorize(f"{score:.4f}", Colors.BRIGHT_RED)


class SimulatorCorrelationEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: Dict[str, Any],
        model: Any,
        tokenizer: Any,
        test_dataloader: Any,
        fs_dataloader: Optional[Any] = None,
        debug_mode: bool = False,
        # fast_eval: bool = False,
        simulator: Optional[FinetunedSimulator] = None,
        target_model: Optional[torch.nn.Module] = None,
        target_tokenizer: Optional[AutoTokenizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, model, tokenizer, test_dataloader, fs_dataloader)
        self.main_metric = "simulator_correlation_abs"
        self.greater_is_better = True
        self.debug_mode = debug_mode

        if simulator is None:
            simulator_device = get_first_available_device()
            # get index of device
            simulator_device_idx = simulator_device.index
            # load simulator
            self.simulator = FinetunedSimulator.setup(
                model_path=config["simulator_path"],
                add_special_tokens=True,
                gpu_idx=simulator_device_idx,
                tokenizer_path="meta-llama/Llama-3.1-8B",
                cache_dir=config.get("cache_dir", None),
            )
            print("Loaded simulator LLM")
        else:
            self.simulator = simulator

        self.subject = FeatureDescriptorSubject(
            model=target_model,
            tokenizer=target_tokenizer,
            is_chat_model=target_tokenizer.chat_template is not None,
        )
        self.subject.tokenizer.padding_side = "left"
        with open(
            os.path.join(config["explanation_dir"], "exemplar_config.json"), "r"
        ) as f:
            exemplar_config = ExemplarConfig.model_validate_json(f.read())
        explanation_config = ExplanationConfig(
            exemplar_config=exemplar_config,
            exem_slice_for_exp=[0, 20, 1],
            permute_exemplars_for_exp=True,
            num_exem_range_for_exp=[10, 20],
            fix_exemplars_for_exp=False,
            num_examples_for_exp=None,
            fix_examples_for_exp=False,
            explainer_model_name="Transluce/llama_8b_explainer",
            explainer_system_prompt_type="no_cot",
            use_puzzle_for_bills=False,
            examples_placement="no_examples",
            min_tokens_to_highlight=3,
            round_to_int=True,
            num_explanation_samples=50,
            max_new_tokens_for_explanation_generation=2000,
            temperature_for_explanation_generation=1.0,
            save_full_explainer_responses=False,
            exem_slice_to_score=[0, 20, 1],
            simulator_model_name="Transluce/llama_8b_simulator",
            add_special_tokens=True,
            simulator_system_prompt_type="unk_base",
            seed=42,
        )
        self.explanations_wrapper = ExplanationsWrapper(
            exemplar_data_dir=config["explanation_dir"],
            config=explanation_config,
            subject=self.subject,
        )
        self.layer_to_indices_map = {}
        if config["split_explanations_data"]:
            split_explanations_file = os.path.join(
                config["explanation_dir"], config["split_keys"]
            )
            if split_explanations_file.endswith(".json"):
                with open(
                    split_explanations_file.format(split=self.data_split),
                    "r",
                ) as f:
                    layer_indices_map = json.load(f)
            elif split_explanations_file.endswith(".pkl"):
                with open(
                    split_explanations_file.format(split=self.data_split),
                    "rb",
                ) as f:
                    layer_indices_map = pickle.load(f)
            else:
                raise ValueError(
                    f"Unsupported indices file format: {config['split_keys']}"
                )
            neurons = []
            for layer, sae_index in tqdm(
                layer_indices_map, desc="Loading layer indices"
            ):
                if int(layer) not in self.layer_to_indices_map:
                    self.layer_to_indices_map[int(layer)] = []
                neurons.append([int(layer), len(self.layer_to_indices_map[int(layer)])])
                self.layer_to_indices_map[int(layer)].append(int(sae_index))

    def reset_metrics(self):
        self.metrics: Dict[str, float] = {
            "simulator_correlation_abs": 0.0,
            "simulator_correlation_ratio": 0.0,
            "gt_correlation_abs": 0.0,
            "total": 0.0,
        }
        self.simulator_correlation_abs_scores: list[float] = []
        self.simulator_correlation_ratio_scores: list[float] = []
        self.gt_correlation_abs_scores: list[float] = []

    def display_metadata_with_highlights(
        self, metadata: pd.DataFrame | None, top_k: int = 3
    ) -> None:
        """
        Display metadata entries with tokens having highest-scoring true activations highlighted.

        Args:
            metadata: DataFrame containing simulation results with columns:
                     'tokens', 'true_activations', 'simulated_activations', 'score', 'rank'
            top_k: Number of highest-scoring tokens to highlight per sequence

        Example output:
            ================================================================================
            METADATA ANALYSIS WITH HIGHLIGHTED HIGH-ACTIVATION TOKENS
            ================================================================================

            --- Sequence Rank 0 (Score: 0.8500) ---
            Tokens with activations (highlighted = top activation):
            The **cat** sat on the **mat** . (0.123) (0.456) (0.234) (0.789) (0.345) (0.012)

            Full text with high-activation tokens highlighted:
              The **cat** sat on the **mat** .

            Top 3 highest-activation tokens:
              1. 'mat' - True: 0.7890, Simulated: 0.7500
              2. 'cat' - True: 0.4560, Simulated: 0.4200
              3. 'sat' - True: 0.3450, Simulated: 0.3200
              Top-3 correlation: 0.9234

            Overall statistics:
              Max true activation: 0.7890
              Min true activation: 0.0120
              Mean true activation: 0.3295
              Max simulated activation: 0.7500
              Min simulated activation: 0.0100
              Mean simulated activation: 0.3150
              Overall correlation: 0.8956
            ------------------------------------------------------------
        """
        if metadata is None or metadata.empty:
            print(Colors.colorize("No metadata available to display.", Colors.YELLOW))
            return

        print("\n" + "=" * 80)
        print(
            Colors.colorize(
                "METADATA ANALYSIS WITH HIGHLIGHTED HIGH-ACTIVATION TOKENS",
                Colors.BOLD + Colors.BRIGHT_CYAN,
            )
        )
        print("=" * 80)

        for idx, row in metadata.iterrows():
            row["rank"]
            score = row["score"]
            tokens = row["tokens"]
            true_activations = row["true_activations"]
            simulated_activations = row["simulated_activations"]

            # # Color-code the score
            score_value = (
                float(score)
                if isinstance(score, (int, float))
                else float(score.iloc[0])
            )
            colored_score = Colors.highlight_score(score_value)
            # print(
            #     f"\n{Colors.colorize('--- Sequence Rank ' + str(idx) + ' (Score: ', Colors.BRIGHT_BLUE)}{colored_score}{Colors.colorize(') ---', Colors.BRIGHT_BLUE)}"
            # )

            # Find top-k tokens with highest true activations
            if len(true_activations) > 0:
                # Get indices of top-k highest activations (handle case where top_k > len(activations))
                top_k_actual = min(top_k, len(true_activations))
                top_indices = np.argsort(true_activations)[-top_k_actual:][::-1]

                # Display tokens with highlighting
                highlighted_tokens = []
                for i, (token, true_act) in enumerate(zip(tokens, true_activations)):
                    if i in top_indices:
                        # Highlight top tokens with colors and activation values
                        colored_token = Colors.colorize(
                            token, Colors.BRIGHT_GREEN + Colors.BOLD
                        )
                        colored_act = Colors.colorize(
                            f"({true_act:.3f})", Colors.BRIGHT_GREEN
                        )
                        highlighted_tokens.append(f"{colored_token}{colored_act}")
                    else:
                        # Regular tokens with activation values
                        act_color = Colors.highlight_score(true_act, 0.3, 0.1)
                        highlighted_tokens.append(f"{token}({act_color})")

                # print(
                #     Colors.colorize(
                #         "Tokens with activations (highlighted = top activation):",
                #         Colors.BRIGHT_MAGENTA,
                #     )
                # )
                # print(" ".join(highlighted_tokens))

                # Show full text with highlighting
                # print(
                #     f"\n{Colors.colorize('Full text with high-activation tokens highlighted:', Colors.BRIGHT_MAGENTA)}"
                # )
                full_text = ""
                for i, (token, true_act) in enumerate(zip(tokens, true_activations)):
                    if i in top_indices:
                        full_text += Colors.colorize(
                            token, Colors.BRIGHT_GREEN + Colors.BOLD
                        )
                    else:
                        full_text += token
                    # full_text += " "
                full_text = full_text.replace("\n", "\\n")
                print(
                    f"  Actual: {full_text.strip()} {Colors.colorize('(Score: ', Colors.BRIGHT_BLUE)}{colored_score}{Colors.colorize(')', Colors.BRIGHT_BLUE)}"
                )

            if len(simulated_activations) > 0:
                top_k_simulated = min(top_k, len(simulated_activations))
                top_indices_simulated = np.argsort(simulated_activations)[
                    -top_k_simulated:
                ][::-1]
                highlighted_tokens_simulated = []
                for i, (token, simulated_act) in enumerate(
                    zip(tokens, simulated_activations)
                ):
                    if i in top_indices_simulated:
                        colored_token = Colors.colorize(
                            token, Colors.BRIGHT_GREEN + Colors.BOLD
                        )
                        colored_act = Colors.colorize(
                            f"({simulated_act:.3f})", Colors.BRIGHT_GREEN
                        )
                        highlighted_tokens_simulated.append(
                            f"{colored_token}{colored_act}"
                        )
                    else:
                        act_color = Colors.highlight_score(simulated_act, 0.3, 0.1)
                        highlighted_tokens_simulated.append(f"{token}({act_color})")
                full_text_simulated = ""
                for i, (token, simulated_act) in enumerate(
                    zip(tokens, simulated_activations)
                ):
                    if i in top_indices_simulated:
                        full_text_simulated += Colors.colorize(
                            token, Colors.BRIGHT_GREEN + Colors.BOLD
                        )
                    else:
                        full_text_simulated += token
                full_text_simulated = full_text_simulated.replace("\n", "\\n")
                print(
                    f"  Simulated: {full_text_simulated.strip()} {Colors.colorize('(Score: ', Colors.BRIGHT_BLUE)}{colored_score}{Colors.colorize(')', Colors.BRIGHT_BLUE)}"
                )

    def get_simulator_score(
        self, predicted_label: str, full_item: dict
    ) -> Tuple[float | None, pd.DataFrame | None]:
        """Get simulator score between predicted and expected labels."""

        neuron_idx = int(full_item["extra_args"]["feature_idx"])
        if full_item["extra_args"]["layer"] in self.layer_to_indices_map:
            idx = self.layer_to_indices_map[full_item["extra_args"]["layer"]].index(
                neuron_idx
            )
        else:
            idx = neuron_idx

        result = self.explanations_wrapper.score_arbitrary_explanation(
            explanation=predicted_label,
            layer=full_item["extra_args"]["layer"],
            neuron_idx=idx,
            act_sign=ActivationSign.POS,
            simulator=self.simulator,
            exem_splits=[ExemplarSplit.TRAIN],
        )
        score = result.get_preferred_score([ExemplarSplit.TRAIN])
        metadata = result.parse_simulation_results(ExemplarSplit.TRAIN)
        return score, metadata

    def evaluate_item(
        self,
        predicted_label: str | List[str],
        expected_label: str,
        full_item: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate similarity between predicted and expected labels."""
        if predicted_label is None:
            return None
        if isinstance(predicted_label, str):
            predicted_label = [predicted_label]

        if expected_label not in full_item["extra_args"]["all_label_scores"]:
            expected_score = list(full_item["extra_args"]["all_label_scores"].values())[
                0
            ]
        else:
            expected_score = full_item["extra_args"]["all_label_scores"][expected_label]
        if expected_score is None:
            # # get from simulator
            expected_score = 0.0
            # simulator_score, metadata = self.get_simulator_score(
            #     expected_label, full_item
            # )
            # expected_score = simulator_score if simulator_score is not None else 0.0
        # Display metrics
        print(
            f"{Colors.colorize('Feature:', Colors.BRIGHT_CYAN)} {full_item['example']}"
        )
        if "context" in full_item["extra_args"]:
            print(
                f"{Colors.colorize('Context:', Colors.BRIGHT_CYAN)} {full_item['extra_args']['context']}"
            )

        # Color-code expected score
        expected_score_display = (
            Colors.highlight_score(expected_score)
            if expected_score is not None
            else Colors.colorize("None", Colors.YELLOW)
        )
        print(
            f"{Colors.colorize('Expected:', Colors.BRIGHT_GREEN)} {expected_label} {expected_score_display}"
        )

        max_simulator_score = 0.0
        for pred_label in predicted_label:
            if pred_label.strip() == "":
                simulator_score = 0.0
            else:
                simulator_score, metadata = self.get_simulator_score(
                    pred_label, full_item
                )

            # Color-code predicted score
            pred_score_display = (
                Colors.highlight_score(simulator_score)
                if simulator_score is not None
                else Colors.colorize("None", Colors.YELLOW)
            )
            print(
                f"{Colors.colorize('Predicted:', Colors.BRIGHT_MAGENTA)} {pred_label} {pred_score_display}"
            )
            if simulator_score is not None:
                simulator_score = abs(simulator_score)
                max_simulator_score = max(max_simulator_score, simulator_score)

        # Color-code max simulator score
        max_score_display = Colors.highlight_score(max_simulator_score)
        print(
            f"{Colors.colorize('Max simulator score:', Colors.BRIGHT_BLUE)} {max_score_display}"
        )
        # self.display_metadata_with_highlights(metadata, top_k=5)
        print(Colors.colorize("*===*", Colors.BRIGHT_CYAN))

        self.metrics["simulator_correlation_abs"] += max_simulator_score
        if expected_score == 0:
            self.metrics["simulator_correlation_ratio"] += max_simulator_score
            self.metrics["gt_correlation_abs"] += expected_score
        elif expected_score is None or pd.isna(expected_score):
            self.metrics["simulator_correlation_ratio"] += max_simulator_score
        else:
            self.metrics["simulator_correlation_ratio"] += (
                max_simulator_score / expected_score
            )
            self.metrics["gt_correlation_abs"] += expected_score
        self.metrics["total"] += 1.0

        # Store individual scores for error bar calculation
        self.simulator_correlation_abs_scores.append(max_simulator_score)
        if expected_score == 0:
            self.simulator_correlation_ratio_scores.append(max_simulator_score)
            self.gt_correlation_abs_scores.append(expected_score)
        elif expected_score is None or pd.isna(expected_score):
            self.simulator_correlation_ratio_scores.append(max_simulator_score)
        else:
            self.simulator_correlation_ratio_scores.append(
                max_simulator_score / expected_score
            )
            self.gt_correlation_abs_scores.append(expected_score)

        return {self.main_metric: max_simulator_score}

    def finalize_metrics(self) -> None:
        """Calculate final metrics."""
        if self.metrics["total"] > 0:
            self.metrics["avg_simulator_correlation_abs"] = (
                self.metrics["simulator_correlation_abs"] / self.metrics["total"]
            )
            self.metrics["avg_simulator_correlation_ratio"] = (
                self.metrics["simulator_correlation_ratio"] / self.metrics["total"]
            )
            self.metrics["avg_gt_correlation_abs"] = sum(
                self.gt_correlation_abs_scores
            ) / len(self.gt_correlation_abs_scores)
            # Calculate standard errors
            if len(self.simulator_correlation_abs_scores) > 1:
                self.metrics["std_error_abs"] = np.std(
                    self.simulator_correlation_abs_scores, ddof=1
                ) / np.sqrt(len(self.simulator_correlation_abs_scores))
            else:
                self.metrics["std_error_abs"] = 0.0
            if len(self.simulator_correlation_ratio_scores) > 1:
                self.metrics["std_error_ratio"] = np.std(
                    self.simulator_correlation_ratio_scores, ddof=1
                ) / np.sqrt(len(self.simulator_correlation_ratio_scores))
            else:
                self.metrics["std_error_ratio"] = 0.0
            if len(self.gt_correlation_abs_scores) > 1:
                self.metrics["std_error_gt_abs"] = np.std(
                    self.gt_correlation_abs_scores, ddof=1
                ) / np.sqrt(len(self.gt_correlation_abs_scores))
            else:
                self.metrics["std_error_gt_abs"] = 0.0
        else:
            self.metrics["avg_simulator_correlation_abs"] = 0.0
            self.metrics["avg_simulator_correlation_ratio"] = 0.0
            self.metrics["std_error_abs"] = 0.0
            self.metrics["std_error_ratio"] = 0.0
            self.metrics["avg_gt_correlation_abs"] = 0.0
            self.metrics["std_error_gt_abs"] = 0.0

    def pbar_update(self, pbar: tqdm) -> None:
        """Update progress bar with current metrics."""
        avg_simulator_correlation_abs = (
            self.metrics["simulator_correlation_abs"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0.0
        )
        avg_simulator_correlation_ratio = (
            self.metrics["simulator_correlation_ratio"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0.0
        )
        avg_gt_correlation_abs = (
            self.metrics["gt_correlation_abs"] / self.metrics["total"]
            if self.metrics["total"] > 0
            else 0.0
        )
        # Calculate current standard errors for progress bar
        if len(self.simulator_correlation_abs_scores) > 1:
            current_std_error_abs = np.std(
                self.simulator_correlation_abs_scores, ddof=1
            ) / np.sqrt(len(self.simulator_correlation_abs_scores))
        else:
            current_std_error_abs = 0.0
        if len(self.simulator_correlation_ratio_scores) > 1:
            current_std_error_ratio = np.std(
                self.simulator_correlation_ratio_scores, ddof=1
            ) / np.sqrt(len(self.simulator_correlation_ratio_scores))
        else:
            current_std_error_ratio = 0.0
        if len(self.gt_correlation_abs_scores) > 1:
            current_std_error_gt_abs = np.std(
                self.gt_correlation_abs_scores, ddof=1
            ) / np.sqrt(len(self.gt_correlation_abs_scores))
        else:
            current_std_error_gt_abs = 0.0
        pbar.set_description(
            f"Avg Simulator Correlation: {avg_simulator_correlation_abs:.4f} ± {current_std_error_abs:.4f} abs, {avg_simulator_correlation_ratio:.4f} ± {current_std_error_ratio:.4f} ratio, {avg_gt_correlation_abs:.4f} ± {current_std_error_gt_abs:.4f} gt abs"
        )

    def log_metrics(self, prefix: str = "", step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if not self.debug_mode:
            wandb.log(
                {
                    prefix
                    + "avg_simulator_correlation_abs": self.metrics[
                        "avg_simulator_correlation_abs"
                    ],
                    prefix
                    + "avg_simulator_correlation_ratio": self.metrics[
                        "avg_simulator_correlation_ratio"
                    ],
                    prefix + "std_error_abs": self.metrics.get("std_error_abs", 0.0),
                    prefix
                    + "std_error_ratio": self.metrics.get("std_error_ratio", 0.0),
                    prefix
                    + "avg_gt_correlation_abs": self.metrics["avg_gt_correlation_abs"],
                    prefix
                    + "std_error_gt_abs": self.metrics.get("std_error_gt_abs", 0.0),
                },
                step=step,
            )

    def print_metrics(self, prefix: str = "") -> None:
        """Print metrics to console."""
        avg_abs = self.metrics.get("avg_simulator_correlation_abs", 0.0)
        avg_ratio = self.metrics.get("avg_simulator_correlation_ratio", 0.0)
        std_error_abs = self.metrics.get("std_error_abs", 0.0)
        std_error_ratio = self.metrics.get("std_error_ratio", 0.0)
        avg_gt_correlation_abs = self.metrics.get("avg_gt_correlation_abs", 0.0)
        std_error_gt_abs = self.metrics.get("std_error_gt_abs", 0.0)
        print(
            f"{prefix}Average Simulator Correlation: {avg_abs:.4f} ± {std_error_abs:.4f} abs, {avg_ratio:.4f} ± {std_error_ratio:.4f} ratio, {avg_gt_correlation_abs:.4f} ± {std_error_gt_abs:.4f} gt abs"
        )
