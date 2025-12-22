import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.nn import Module
from transformers import PreTrainedTokenizer
from dataloaders.base_dataset import BaseDataCollator
from dataloaders.base_dataset import BaseIterableDataset
from nnsight import LanguageModel

warnings.filterwarnings("ignore")


class CausalBaseDataset(BaseIterableDataset):
    """
    Base class for causal intervention datasets with shared functionality.
    """

    def __init__(
        self,
        split: str,
        predictor_model: Module,
        target_model: Module,
        predictor_tokenizer: PreTrainedTokenizer,
        target_tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        **kwargs,
    ):
        # Initialize causal-specific configuration
        self.self_consistency = config.get("self_consistency", False)

        if self.self_consistency:
            self.nnsight_model = LanguageModel(
                target_model,
                tokenizer=target_tokenizer,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        # Initialize dataset lists
        self.examples_list, self.labels_list = [], []
        self.num_samples = 0

        super().__init__(
            split,
            predictor_model,
            target_model,
            predictor_tokenizer,
            target_tokenizer,
            config,
            **kwargs,
        )

    def load_datasets(self, dataset_names: List[str]):
        self.examples_list, self.labels_list = [], []
        self.num_samples = 0
        self.example_to_dataset_name = {}

        for dataset_name in dataset_names:
            (
                examples,
                labels,
                self.example_to_dataset_name,
            ) = self.load_data_files_for_dataset(self.question_config, self.data_split)

            for i, example in enumerate(examples):
                for qtype in self.question_config["question_types"]:
                    self.example_to_dataset_name[example] = dataset_name
                    self.examples_list.append((example, [qtype]))
                    self.labels_list.append(labels[i])
                    self.num_samples += 1

        print(f"# Total samples for {self.data_split}: {self.num_samples}")

    def _format_causal_change(self, original_continuation: str, ablated_continuation: str) -> Tuple[str, str, str]:
        """Format the causal change description."""
        if not isinstance(original_continuation, str):
            original_response = self.predictor_tokenizer.convert_tokens_to_string(
                original_continuation
            )
        else:
            original_response = original_continuation
        if not isinstance(ablated_continuation, str):
            new_response = self.predictor_tokenizer.convert_tokens_to_string(
                ablated_continuation
            )
        else:
            new_response = ablated_continuation

        if original_response == new_response:
            return (
                original_response,
                new_response,
                f"The output would remain unchanged from <<<{original_response}>>>.",
            )
        else:
            return (
                original_response,
                new_response,
                f"The most likely output would change to <<<{new_response}>>>.",
            )

    def fill_in_prompt_args(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_template: List[Dict[str, str]] | str,
        prompt_args: Dict[str, str],
        expected_label: Optional[str] = None,
        full_assistant_response: bool = False,
    ) -> Tuple[Optional[List[Dict[str, str]]], List[Dict[str, str]], Optional[str]]:
        """Fill in template with args - enhanced for causal datasets."""
        if isinstance(prompt_template, str):
            prompt_without_answer = prompt_template.format(**prompt_args)
            if expected_label is None:
                return None, prompt_without_answer, None
            expected_label = expected_label
            prompt_with_answer = prompt_without_answer + expected_label
            return prompt_with_answer, prompt_without_answer, expected_label

        prompt_with_answer = copy.deepcopy(prompt_template)
        last_user_turn_idx = -1

        for i, message in enumerate(prompt_with_answer):
            if message["role"] == "user":
                last_user_turn_idx = i
            format_args = {}
            for arg in prompt_args:
                if f"{{{arg}}}" in message.get("content", ""):
                    format_args[arg] = prompt_args[arg]

            if format_args:
                prompt_with_answer[i]["content"] = prompt_with_answer[i][
                    "content"
                ].format(**format_args)

        prompt_without_answer = copy.deepcopy(prompt_with_answer)

        if expected_label is None:
            return None, prompt_without_answer, None

        if last_user_turn_idx == len(prompt_template) - 1:
            prompt_with_answer.append({"role": "assistant", "content": expected_label})
            if not full_assistant_response:
                prompt_without_answer.append({"role": "assistant", "content": ""})
            else:
                expected_label = tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": expected_label}], tokenize=False
                )
                # Handle different tokenizer templates
                if "<|start_header_id|>" in expected_label:
                    expected_label_start_idx = expected_label.index(
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                elif "<start_of_turn>" in expected_label:
                    expected_label_start_idx = expected_label.index(
                        "<start_of_turn>model\n"
                    )
                elif "<|im_start|>" in expected_label:
                    expected_label_start_idx = expected_label.index(
                        "<|im_start|>assistant"
                    )
                else:
                    expected_label_start_idx = 0

                expected_label = expected_label[expected_label_start_idx:]
        else:
            if "content" in prompt_with_answer[-1]:
                prompt_with_answer[-1]["content"] += expected_label
            else:
                prompt_with_answer[-1]["content"] = expected_label

        try:
            assert prompt_with_answer[:-1] == prompt_without_answer[:-1]
        except AssertionError:
            print(prompt_with_answer)
            print(prompt_without_answer)
            print(expected_label)

        return prompt_with_answer, prompt_without_answer, expected_label

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

    def __iter__(self):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement __iter__")


class CausalDataCollator(BaseDataCollator):
    """
    Data collator for causal intervention tasks that handles change_match questions
    by preparing intervention arguments for the model.
    """

    def __init__(
        self,
        predictor_tokenizer: PreTrainedTokenizer,
        target_tokenizer: PreTrainedTokenizer,
        question_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(predictor_tokenizer, target_tokenizer)
        self.question_config = question_config or {}

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # First call the parent collator to handle standard processing
        batch = super().__call__(features)

        # Check for change_match questions and prepare intervention arguments
        intervention_mask = torch.zeros(len(features), dtype=torch.bool)
        intervention_layers = []
        intervention_positions = []
        intervention_vectors = []

        for i, feature in enumerate(features):
            if (
                feature.get("question_type") == "change_match"
                and feature.get("extra_args") is not None
                and not feature.get("task", "").startswith("hint")
            ):

                intervention_mask[i] = True
                extra_args = feature["extra_args"]

                # Extract intervention information from extra_args
                layer = extra_args["layer"]
                intervention_layers.append(layer)

                # account for padding
                start_token_pos = 0
                special_token_ids = [
                    self.target_tokenizer.bos_token_id,
                    self.target_tokenizer.eos_token_id,
                    self.target_tokenizer.pad_token_id,
                ]
                while (
                    start_token_pos < len(batch["input_ids"][i])
                    and batch["input_ids"][i][start_token_pos] in special_token_ids
                ):
                    start_token_pos += 1
                if start_token_pos == len(batch["input_ids"][i]):
                    intervention_positions.append([])
                    continue

                # Get ablated token positions
                if "ablated_token_positions" in extra_args:
                    positions = [
                        pos + start_token_pos
                        for pos in extra_args["ablated_token_positions"]
                    ]
                    intervention_positions.append(positions)
                else:
                    intervention_positions.append([])

                # For other intervention types, we might need different handling
                intervention_vectors.append(None)
            else:
                intervention_layers.append(None)
                intervention_positions.append([])
                intervention_vectors.append(None)

        # Add intervention information to batch
        if intervention_mask.any():
            batch["intervention_mask"] = intervention_mask
            batch["intervention_layers"] = intervention_layers
            batch["intervention_positions"] = intervention_positions
            batch["intervention_vectors"] = intervention_vectors

        # Collect question type weights from all features and add to batch
        question_type_weights = []
        for feature in features:
            if feature.get("extra_args") is not None:
                weight = feature["extra_args"].get("question_type_weight", 1.0)
                question_type_weights.append(weight)
            else:
                question_type_weights.append(1.0)  # Default weight

        batch["question_type_weights"] = torch.tensor(
            question_type_weights, dtype=torch.float32
        )

        return batch
