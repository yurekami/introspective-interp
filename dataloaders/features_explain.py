import copy
import os
import pickle
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple
from torch.nn import Module
from transformers import PreTrainedTokenizer
import torch
from dataloaders.base_dataset import BaseIterableDataset, ExampleLabels
from tqdm import tqdm
from dataloaders.features_explain_utils import load_features_dataset


warnings.filterwarnings("ignore")


class FeaturesExplainDataset(BaseIterableDataset):
    def __init__(
        self,
        split: str,
        predictor_model: Module,
        target_model: Module,
        predictor_tokenizer: PreTrainedTokenizer,
        target_tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        all_features: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
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
        if all_features is None:
            if "sae_explanations" in config["dataset"]:
                assert "sae_save_path" in config
                self.sae_save_path = os.path.join(
                    config["explanation_dir"],
                    config["sae_save_path"],
                )
                self.all_features = {}
                for layer in tqdm(
                    self.question_config["layers"],
                    desc="Loading SAE features",
                ):
                    weights = pickle.load(
                        open(
                            os.path.join(self.sae_save_path, f"layer_{layer}.pkl"), "rb"
                        )
                    )
                    self.all_features[layer] = {
                        feature_idx: weights[:, feature_idx]
                        for feature_idx in range(weights.shape[1])
                    }
            elif "custom_explanations" in config["dataset"]:
                self.vectors_save_path = os.path.join(
                    config["explanation_dir"],
                    config["vectors_save_path"].format(split=split),
                )
                all_vectors = torch.load(self.vectors_save_path)
                self.all_features: dict[int, dict[str, torch.Tensor]] = {
                    layer: {
                        feature_idx: all_vectors[feature_idx]
                        for feature_idx in range(all_vectors.shape[0])
                    }
                    for layer in range(self.target_model.config.num_hidden_layers)
                }
            else:
                raise ValueError(f"Invalid dataset: {config['dataset']}")
        else:
            self.all_features = all_features

    def load_datasets(self, dataset_names: List[str]):
        self.examples_list, self.labels_list = [], []
        self.example_to_dataset_name = {}
        self.task_template = [
            {"role": "user", "content": "{task_prompt}{example}"},
            {"role": "assistant"},
        ]

        for dataset_name in dataset_names:
            num_dataset_samples = (
                self.num_samples // len(dataset_names) // self.num_repeats_per_example
            )
            (
                prompt,
                examples,
                labels,
                extra_args,
            ) = load_features_dataset(
                dataset_name,
                self.question_config,
                self.data_split,
            )
            for k, v in extra_args.items():
                setattr(self, k, v)

            examples = examples[:num_dataset_samples]
            for example in examples:
                self.example_to_dataset_name[example] = dataset_name
            self.examples_list.extend(examples)
            labels = labels[:num_dataset_samples]
            self.labels_list.extend(labels)

            assert len(examples) == len(labels)

            if dataset_name in [
                "sae_explanations",
                "sae_autointerp_explanations",
                "one_hot_explanations",
            ]:
                print(
                    f"{dataset_name} ({self.data_split}) has {len(examples)} samples and {len(set(examples))} unique features"
                )

            self.task_prompt = prompt

    def get_labels(
        self,
        example: Tuple[int, str],
        label: Optional[Dict[str, Any]] = None,
        question_type: Optional[str] = None,
        idx: Optional[int] = None,
    ) -> ExampleLabels:
        layer, feature_idx = example
        question_config = self.question_config["question_types"][question_type]

        chosen_prompt_template = copy.deepcopy(
            random.choice(question_config["prompts"])
        )
        if question_type == "generative_explanation":
            # 1 x hidden_size
            feature_continuous_token = self.all_features[layer][feature_idx]
            if isinstance(chosen_prompt_template, str):
                prompt_template = chosen_prompt_template
            else:
                prompt_template = chosen_prompt_template["messages"]
            prompt_args = {
                "layer": str(layer),
                "begin_continuous": self.special_tokens["begin_continuous"],
                "end_continuous": self.special_tokens["end_continuous"],
                "feature": self.special_tokens["continuous_rep"],
            }
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        prompt_with_answer, prompt_without_answer, expected_label = (
            self.fill_in_prompt_args(
                self.predictor_tokenizer,
                prompt_template,
                prompt_args,
                label["chosen_label"],
                full_assistant_response=True,
            )
        )
        other_features = {
            "layer": int(layer),
            "feature_idx": feature_idx,
            "label": label,
            "all_labels": label["all_labels"],
            "all_label_scores": label["all_label_scores"],
            "chosen_label_score": (
                label["all_label_scores"][label["chosen_label"]]
                if label["chosen_label"] in label["all_label_scores"]
                else None
            ),
            "full_label": label,
        }
        if "context" in label:
            other_features["context"] = label["context"]

        prompt_continuous_tokens = feature_continuous_token.unsqueeze(0)
        prompt_without_answer_continuous_tokens = prompt_continuous_tokens.clone()

        return ExampleLabels(
            prompt=prompt_with_answer,
            prompt_without_answer=prompt_without_answer,
            expected_label=expected_label,
            evaluation_type=self.question_config["evaluation_type"],
            task="token_comparison",
            question_type=question_type,
            dataset_name=self.example_to_dataset_name[example],
            extra_args=other_features,
            prompt_continuous_tokens=prompt_continuous_tokens,
            prompt_without_answer_continuous_tokens=prompt_without_answer_continuous_tokens,
        )

    def __iter__(self):
        """Iterate through all examples and question types."""
        for example_idx in range(len(self.examples_list)):
            for repeat_idx in range(self.num_repeats_per_example):
                if self.question_types:
                    question_type = self.question_types[repeat_idx]
                else:
                    question_type = None

                example = self.examples_list[example_idx]
                label = self.labels_list[example_idx]

                # Get the formatted example using existing logic
                example_labels = self.get_labels(
                    example,
                    label=label,
                    question_type=question_type,
                    idx=example_idx * self.num_repeats_per_example + repeat_idx,
                )

                yield {
                    "example": example,
                    "prompt": example_labels.prompt,
                    "prompt_without_answer": example_labels.prompt_without_answer,
                    "expected_label": example_labels.expected_label,
                    "evaluation_type": example_labels.evaluation_type,
                    "task": example_labels.task,
                    "dataset_name": example_labels.dataset_name,
                    "question_type": example_labels.question_type,
                    "extra_args": example_labels.extra_args,
                    "prompt_continuous_tokens": example_labels.prompt_continuous_tokens,
                    "prompt_without_answer_continuous_tokens": example_labels.prompt_without_answer_continuous_tokens,
                    "label_continuous_tokens": example_labels.label_continuous_tokens,
                }
