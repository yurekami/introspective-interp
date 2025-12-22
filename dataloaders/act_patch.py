import copy
import random
import warnings
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import torch
from dataloaders.base_dataset import ExampleLabels
from dataloaders.causal import CausalBaseDataset
from tqdm import tqdm
from datasets import load_dataset

warnings.filterwarnings("ignore")


class ActPatchDataset(CausalBaseDataset):
    def load_data_files_for_dataset(
        self, question_config: Dict[str, Any], data_split: str
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str]]:
        """Load intervention data from Hugging Face dataset"""
        dataset_name = question_config["intervention_path"]
        print(f"Loading from Hugging Face dataset: {dataset_name}, split: {data_split}")

        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(
                dataset_name,
                split=data_split,
                cache_dir=question_config.get("hf_data_cache_dir", None),
            )

            df = dataset.to_pandas()

            print(f"Loaded {len(df)} rows from Hugging Face dataset")

            # Verify expected columns exist
            required_cols = [
                "layer",
                "input_tokens",
                "original_continuation",
                "ablated_continuation",
                "is_different",
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Dataset missing required columns: {missing_cols}")

            return self._process_intervention_dataframe(df)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace dataset {dataset_name} for split {data_split}: {e}"
            )

    def _process_intervention_dataframe(
        self, combined_df
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str]]:
        """Process the combined intervention dataframe into the format expected by BaseDataset"""
        if combined_df.empty:
            return [], [], {}

        # Create hashable column for unique counting
        combined_df["input_tokens_hashable"] = combined_df["input_tokens"].apply(
            lambda x: tuple(x) if isinstance(x, (list, np.ndarray)) else x
        )
        num_unique_samples = len(combined_df["input_tokens_hashable"].unique())
        print(
            f"Loaded {len(combined_df)} intervention examples with {num_unique_samples} unique exemplars"
        )

        # Convert to the format expected by BaseDataset
        examples_list = []
        labels_list = []
        example_to_dataset_name = {}

        # Process intervention data
        for i, row in tqdm(
            combined_df.iterrows(),
            total=len(combined_df),
            desc="Processing intervention data",
        ):
            # Create a unique example identifier
            example_id = {
                "layer": (
                    row["layer"].tolist()
                    if isinstance(row["layer"], np.ndarray)
                    else row["layer"]
                ),
                "text": "".join(row["input_tokens"]),
            }
            if "ablated_token_positions" in row:
                example_id["ablated_token_positions"] = row[
                    "ablated_token_positions"
                ].tolist()
            if "feature_idx" in row:
                example_id["feature_idx"] = row["feature_idx"]
            example_id = json.dumps(example_id)

            if (
                self.predictor_tokenizer.eos_token in row["original_continuation"]
                or self.predictor_tokenizer.eos_token in row["ablated_continuation"]
            ):
                continue

            # Store the intervention data as the label
            label = {
                "input_tokens": row["input_tokens"],
                "original_continuation": row["original_continuation"],
                "ablated_continuation": row["ablated_continuation"],
                "chosen_label": self._format_causal_change(
                    row["original_continuation"],
                    row["ablated_continuation"],
                )[2],
                "all_labels": [
                    self._format_causal_change(
                        row["original_continuation"],
                        row["ablated_continuation"],
                    )[2]
                ],
                "is_different": row["is_different"],
                "all_label_scores": {},
                "dataset": "causal_act_patch",
            }
            label["patch_position"] = row["patch_position"]
            label["counterfactual_text"] = row["counterfactual_text"]
            label["gt_original_target"] = row["gt_original_target"]
            label["gt_counterfactual_target"] = row["gt_counterfactual_target"]
            if "ablated_token_positions" in row:
                label["ablated_token_positions"] = row["ablated_token_positions"]
            if "ablated_tokens" in row:
                label["ablated_tokens"] = row["ablated_tokens"]
            if "description" in row:
                label["description"] = row["description"]
            if "activating_positions" in row:
                label["activating_positions"] = row["activating_positions"]
            if "activation_values" in row:
                label["activation_values"] = row["activation_values"]
            if "token_position_weights" in row:
                label["token_position_weights"] = row["token_position_weights"]

            examples_list.append(example_id)
            labels_list.append(label)
            example_to_dataset_name[example_id] = "causal_act_patch"

            if self.debug and len(examples_list) >= 1000:
                break

        print(f"Prepared {len(examples_list)} examples")
        return (
            examples_list,
            labels_list,
            example_to_dataset_name,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        """Iterate through data and convert to training examples."""
        samples_yielded = 0
        samples_skipped = 0
        pbar = tqdm(
            enumerate(self.examples_list),
            total=len(self.examples_list),
            desc="Generating samples",
        )
        # Generate samples based on weights
        # For activation prediction: generate one sample per activation
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

                # example = (sample, question_types)
                example_label = self.labels_list[s]

                training_example = self.get_labels_intervention(
                    example=example,
                    label=example_label,
                    question_type=question_type,
                )
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
                    "prompt_continuous_tokens": training_example.prompt_continuous_tokens,
                    "prompt_without_answer_continuous_tokens": training_example.prompt_without_answer_continuous_tokens,
                    "label_continuous_tokens": training_example.label_continuous_tokens,
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

    def recompute_labels_intervention(
        self,
        input_text: str,
        label: Optional[Dict[str, Any]] = None,
        layer: Optional[List[int]] = None,
        feature_continuous_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        tokens = self.target_tokenizer(input_text, return_tensors="pt")
        special_tokens = [
            self.target_tokenizer.bos_token_id,
            self.target_tokenizer.eos_token_id,
            self.target_tokenizer.pad_token_id,
        ]
        start_token_pos = 0
        while (
            start_token_pos < len(tokens["input_ids"][0])
            and tokens["input_ids"][0][start_token_pos] in special_tokens
        ):
            start_token_pos += 1
        self.target_model.eval()
        with torch.no_grad():
            tokens = {k: v.to(self.target_model.device) for k, v in tokens.items()}

            # 1. get original continuation
            original_next_token = self.target_model(**tokens)
            original_next_token = original_next_token.logits[0, -1, :]
            original_next_token = torch.argmax(original_next_token, dim=-1)
            original_continuation = [
                self.target_tokenizer.convert_tokens_to_string(
                    self.target_tokenizer.convert_ids_to_tokens([original_next_token])
                )
            ]

            # 2. get ablated continuation
            with self.nnsight_model.trace(tokens) as tracer:
                feature_continuous_token = torch.tensor(
                    feature_continuous_token,
                    device=self.nnsight_model.device,
                    dtype=torch.bfloat16,
                )
                assert (
                    self.target_tokenizer.convert_ids_to_tokens(
                        tokens["input_ids"][:, label["patch_position"]["orig_pos"]]
                    )[0]
                    == label["patch_position"]["orig_text_token"]
                )
                for l_idx in layer:
                    self.nnsight_model.model.layers[l_idx].output[
                        0, label["patch_position"]["orig_pos"]
                    ] = feature_continuous_token
                ablated_logits = self.nnsight_model.lm_head.output[0, -1, :].save()

            ablated_next_token = torch.argmax(ablated_logits, dim=-1)
            ablated_continuation = [
                self.target_tokenizer.convert_tokens_to_string(
                    self.target_tokenizer.convert_ids_to_tokens([ablated_next_token])
                )
            ]
            label["original_continuation"] = original_continuation
            label["ablated_continuation"] = ablated_continuation

        return label

    def get_labels_intervention(
        self,
        example: str,
        label: Optional[Dict[str, Any]] = None,
        question_type: Optional[str] = None,
        idx: Optional[int] = None,
    ) -> ExampleLabels:
        """
        example: (layer, feature_idx)
        label: dict[str, Any] | None
        question_type: str

        Returns:
            ExampleLabels
        """
        example_dict = json.loads(example)
        layer = example_dict["layer"]
        question_config = self.question_config["question_types"][question_type]

        chosen_prompt_template = copy.deepcopy(
            random.choice(question_config["prompts"])
        )
        input_text = self.predictor_tokenizer.convert_tokens_to_string(
            label["input_tokens"]
        )
        if isinstance(chosen_prompt_template, str):
            prompt_template = chosen_prompt_template
        else:
            prompt_template = chosen_prompt_template["messages"]
        prompt_args = {}
        prompt_args["begin_continuous"] = self.special_tokens["begin_continuous"]
        prompt_args["end_continuous"] = self.special_tokens["end_continuous"]
        prompt_args["feature"] = self.special_tokens["continuous_rep"]
        layer_str = ", ".join(str(layer) for layer in layer)
        text_token = self.target_tokenizer.convert_tokens_to_string(
            [label["patch_position"]["orig_text_token"]]
        )
        ablated_tokens = f"<{text_token}>"
        prompt_args.update({"tokens": ablated_tokens})
        prompt_args.update(
            {
                "layer": layer_str,
                "input": input_text,
            }
        )
        feature_continuous_token = label["patch_position"]["intervention_vector"]

        if self.self_consistency:
            label = self.recompute_labels_intervention(
                input_text=input_text,
                label=label,
                layer=layer,
                feature_continuous_token=feature_continuous_token,
            )

        if question_type == "generative_explanation":
            original_response, new_response, response = self._format_causal_change(
                label["original_continuation"],
                label["ablated_continuation"],
            )
        elif question_type == "orig_match":
            original_response = self.predictor_tokenizer.convert_tokens_to_string(
                label["original_continuation"]
            )
            new_response = self.predictor_tokenizer.convert_tokens_to_string(
                label["ablated_continuation"]
            )
            response = original_response
        elif question_type == "change_match":
            original_response = self.predictor_tokenizer.convert_tokens_to_string(
                label["original_continuation"]
            )
            new_response = self.predictor_tokenizer.convert_tokens_to_string(
                label["ablated_continuation"]
            )
            response = new_response
        else:
            raise ValueError(f"Invalid question type: {question_type}")

        prompt_with_answer, prompt_without_answer, expected_label = (
            self.fill_in_prompt_args(
                self.predictor_tokenizer,
                prompt_template,
                prompt_args,
                response,
                full_assistant_response=True,
            )
        )
        other_features = {
            "layer": layer,
            "input": input_text,
            "original_response": original_response,
            "ablated_response": new_response,
            "input_tokens": label["input_tokens"].tolist(),
            "is_different": label["is_different"],
            "question_type_weight": question_config.get("weight", 1.0),
        }
        other_features["patch_position"] = {
            "orig_pos": label["patch_position"]["orig_pos"],
            "counterfact_pos": label["patch_position"]["counterfact_pos"],
            "orig_text_token": label["patch_position"]["orig_text_token"],
            "counterfact_text_token": label["patch_position"]["counterfact_text_token"],
        }
        other_features["counterfactual_text"] = label["counterfactual_text"]
        other_features["gt_original_target"] = label["gt_original_target"]
        other_features["gt_counterfactual_target"] = label["gt_counterfactual_target"]

        prompt_continuous_tokens = torch.tensor(feature_continuous_token).unsqueeze(0)
        prompt_without_answer_continuous_tokens = prompt_continuous_tokens.clone()
        return ExampleLabels(
            prompt=prompt_with_answer,
            prompt_without_answer=prompt_without_answer,
            expected_label=expected_label,
            evaluation_type=self.question_config["evaluation_type"],
            task="causal_act_patch",
            question_type=question_type,
            dataset_name=self.example_to_dataset_name[example],
            extra_args=other_features,
            prompt_continuous_tokens=prompt_continuous_tokens,
            prompt_without_answer_continuous_tokens=prompt_without_answer_continuous_tokens,
        )
