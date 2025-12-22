import copy
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
import re
from datasets import load_dataset
from torch.nn import Module
from transformers import PreTrainedTokenizer

from dataloaders.causal import CausalBaseDataset
from dataloaders.base_dataset import ExampleLabels

warnings.filterwarnings("ignore")


class HintAttributionDataset(CausalBaseDataset):
    """
    Dataset for hint attribution experiments.
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
        super().__init__(
            split,
            predictor_model,
            target_model,
            predictor_tokenizer,
            target_tokenizer,
            config,
            **kwargs,
        )

    def load_data_files_for_dataset(self, question_config: Dict[str, Any], data_split: str) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str]]:
        """Load hint data from Hugging Face dataset."""
        dataset_name = question_config.get("hint_path")
        if not dataset_name:
            raise ValueError("hint_path must be specified in question_config")
        
        print(f"Loading from Hugging Face dataset: {dataset_name}, split: {data_split}")

        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(
                dataset_name,
                split=data_split,
                cache_dir=question_config.get("hf_data_cache_dir", None),
            )

            all_data_list = dataset.to_list()
            print(f"Loaded {len(all_data_list)} rows from Hugging Face dataset")

            examples_list, labels_list = [], []
            example_to_dataset_name = {}
            
            for hint_data in all_data_list:
                hint_data["dataset"] = "hint"
                examples_list.append(hint_data["ablation_prompt"])
                labels_list.append(hint_data)
                example_to_dataset_name[hint_data["ablation_prompt"]] = "hint"
                
            return examples_list, labels_list, example_to_dataset_name

        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace dataset {dataset_name} for split {data_split}: {e}"
            )

    def recompute_labels_hint(
        self,
        with_hint_text: str,
        no_hint_text: str,
        label: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Recompute hint labels using the target model."""
        self.target_model.eval()
        with torch.no_grad():
            # 1. get original continuation
            tokens = self.target_tokenizer(with_hint_text, return_tensors="pt")
            tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}
            original_next_token = self.target_model(**tokens)
            original_next_token = original_next_token.logits[0, -1, :]
            original_next_token = torch.argmax(original_next_token, dim=-1)
            original_continuation = self.target_tokenizer.convert_tokens_to_string(
                self.target_tokenizer.convert_ids_to_tokens([original_next_token])
            ).strip()

            # 2. get no hint continuation
            tokens = self.target_tokenizer(no_hint_text, return_tensors="pt")
            tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}
            zeroshot_next_token = self.target_model(**tokens)
            zeroshot_next_token = zeroshot_next_token.logits[0, -1, :]
            zeroshot_next_token = torch.argmax(zeroshot_next_token, dim=-1)
            zeroshot_prediction = self.target_tokenizer.convert_tokens_to_string(
                self.target_tokenizer.convert_ids_to_tokens([zeroshot_next_token])
            ).strip()

        label["random_hint_prediction"] = original_continuation
        label["zeroshot_prediction"] = zeroshot_prediction
        label["changed_pred_bc_hint"] = original_continuation != zeroshot_prediction

        return label

    def extract_prompt_role(self, input_text: str, role: str = "user") -> str:
        """Extract specific role content from formatted prompt."""
        if f"<|start_header_id|>{role}<|end_header_id|>\\n\\n" in input_text:
            return input_text.split(
                f"<|start_header_id|>{role}<|end_header_id|>\\n\\n"
            )[1].split("<|eot_id|>")[0]
        elif f"<start_of_turn>{role}\\n" in input_text:
            return input_text.split(f"<start_of_turn>{role}\\n")[1].split(
                "<end_of_turn>\\n"
            )[0]
        elif f"<|im_start|>{role}" in input_text:
            return input_text.split(f"<|im_start|>{role}")[1].split("<|im_end|>\\n")[0]
        else:
            return input_text

    def remove_hint(self, input_text: str) -> str:
        """Remove hint text from input."""
        pattern = r"\\nHint: I think the answer is \\([A-D]\\) but I'm curious to hear what you think\\."
        return re.sub(pattern, "", input_text)

    def get_labels_hint(
        self,
        example: str,
        label: Optional[Dict[str, Any]] = None,
        question_type: Optional[str] = None,
    ) -> ExampleLabels:
        """Generate labels for hint attribution examples."""
        question_config = self.question_config["question_types"][question_type]

        chosen_prompt_template = copy.deepcopy(
            random.choice(question_config["prompts"])
        )
        input_text = example
        if isinstance(chosen_prompt_template, str):
            prompt_template = chosen_prompt_template
        else:
            prompt_template = chosen_prompt_template["messages"]

        if question_type == "change_match":
            prompt = label["original_prompt_full"]
        else:
            prompt = label["hint_prompt_full"]

        if isinstance(chosen_prompt_template, str):
            if question_type == "generative_explanation":
                assert prompt.endswith("\\n\\nAssistant: Answer:")
                prompt = prompt[: -len("\\n\\nAssistant: Answer:")]
            prompt_args = {
                "prompt": prompt,
            }
        elif question_type in ["orig_match", "generative_explanation"]:
            prompt_args = {
                "user_prompt": label["hint_user_prompt"],
                "system_prompt": label["system_prompt"],
            }
        elif question_type == "change_match":
            prompt_args = {
                "user_prompt": label["original_user_prompt"],
                "system_prompt": label["system_prompt"],
            }
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        if self.self_consistency:
            label = self.recompute_labels_hint(
                with_hint_text=label["hint_prompt_full"],
                no_hint_text=label["original_prompt_full"],
                label=label,
            )

        if question_type == "generative_explanation":
            original_response, new_response, response = self._format_causal_change(
                "Answer: " + label["random_hint_prediction"],
                "Answer: " + label["zeroshot_prediction"],
            )
            if self.predictor_tokenizer.chat_template is None:
                response = " " + response
        elif question_type == "orig_match":
            original_response = " " + label["random_hint_prediction"].strip()
            new_response = " " + label["zeroshot_prediction"].strip()
            response = original_response
        elif question_type == "change_match":
            original_response = " " + label["random_hint_prediction"].strip()
            new_response = " " + label["zeroshot_prediction"].strip()
            response = new_response
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        try:
            prompt_with_answer, prompt_without_answer, expected_label = (
                self.fill_in_prompt_args(
                    self.predictor_tokenizer,
                    prompt_template,
                    prompt_args,
                    response,
                    full_assistant_response=False,
                )
            )
        except Exception:
            raise

        if label["changed_pred_bc_hint"] is None:
            label["changed_pred_bc_hint"] = original_response != new_response

        other_features = {
            "input": input_text,
            "original_response": original_response,
            "ablated_response": new_response,
            "hint": label["hint"],
            "changed_pred_bc_hint": label["changed_pred_bc_hint"],
            "is_different": original_response != new_response,
            "question_type_weight": question_config.get("weight", 1.0),
        }

        assert label["changed_pred_bc_hint"] == other_features["is_different"]

        return ExampleLabels(
            prompt=prompt_with_answer,
            prompt_without_answer=prompt_without_answer,
            expected_label=expected_label,
            evaluation_type=self.question_config["evaluation_type"],
            task=self.question_config.get("intervention_type", "hint_attribution"),
            question_type=question_type,
            dataset_name=self.example_to_dataset_name[example],
            extra_args=other_features,
        )

    def __iter__(self):
        """Iterate through data and convert to training examples."""
        samples_yielded = 0
        samples_skipped = 0
        pbar = tqdm(
            enumerate(self.examples_list),
            total=len(self.examples_list),
            desc="Generating hint samples",
        )

        for s, (example, question_types) in pbar:
            pbar.set_postfix(
                {
                    "samples_yielded": samples_yielded,
                    "samples_skipped": samples_skipped,
                }
            )
            for question_type in question_types:
                # Check if we've reached the sample limit
                if self.num_samples is not None and samples_yielded >= self.num_samples:
                    break

                example_label = self.labels_list[s]

                if example_label["dataset"] == "hint":
                    training_example = self.get_labels_hint(
                        example=example,
                        label=example_label,
                        question_type=question_type,
                    )
                else:
                    raise ValueError(f"Unknown dataset: {example_label['dataset']}")

                training_example = {
                    "example": example,
                    "prompt": training_example.prompt,
                    "prompt_without_answer": training_example.prompt_without_answer,
                    "expected_label": training_example.expected_label,
                    "evaluation_type": training_example.evaluation_type,
                    "task": training_example.task,
                    "dataset_name": training_example.dataset_name,
                    "question_type": training_example.question_type,
                    "extra_args": training_example.extra_args,
                    "prompt_continuous_tokens": getattr(
                        training_example, "prompt_continuous_tokens", None
                    ),
                    "prompt_without_answer_continuous_tokens": getattr(
                        training_example,
                        "prompt_without_answer_continuous_tokens",
                        None,
                    ),
                    "label_continuous_tokens": getattr(
                        training_example, "label_continuous_tokens", None
                    ),
                }

                if isinstance(training_example["prompt"], list):
                    prompt = self.predictor_tokenizer.apply_chat_template(
                        training_example["prompt"], tokenize=False
                    )
                else:
                    prompt = training_example["prompt"]

                if len(self.predictor_tokenizer.tokenize(prompt)) > 500:
                    samples_skipped += 1
                    continue

                yield training_example
                samples_yielded += 1

                # Break outer loop if we've reached the limit
                if self.num_samples is not None and samples_yielded >= self.num_samples:
                    break

    def __getitem__(self, idx: int) -> None:
        return None
