import json
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainerCallback, PreTrainedTokenizer


class BaseEvaluator:
    def __init__(
        self,
        config: Dict[str, Any],
        model: Module,
        tokenizer: PreTrainedTokenizer,
        test_dataloader: DataLoader,
        fs_dataloader: Optional[DataLoader] = None,
        cap_at_100: bool = False,
        **kwargs,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.test_dataloader = test_dataloader
        self.fs_dataloader = fs_dataloader
        self.generate_limit = config.get("generate_limit", 20)
        self.main_metric = None  # Set in subclasses
        self.greater_is_better = None  # Set in subclasses
        self.generation_evaluation = True
        self.cap_at_100 = cap_at_100
        self.data_split = self.test_dataloader.dataset.data_split

    def get_prediction(self, outputs: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[str, Any]:
        if (
            inputs["task"] == "intervention"
            and inputs["question_type"] == "new_prediction"
        ):
            # only compare first token
            return (
                self.tokenizer.decode(
                    outputs["sequences"][
                        len(inputs["input_ids_without_answer"]) : len(
                            inputs["input_ids_without_answer"]
                        )
                        + 1
                    ],
                    skip_special_tokens=False,
                ),
                None,
            )
        return (
            self.tokenizer.decode(
                outputs["sequences"][len(inputs["input_ids_without_answer"]) :],
                skip_special_tokens=True,
            ).strip(),
            None,
        )

    def evaluate(self, fewshot: bool = False, save_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model on the test set.
        """
        predictions = []
        self.reset_metrics()
        self.model.eval()
        if fewshot:
            pbar = tqdm(self.fs_dataloader, desc="Evaluating")
        else:
            pbar = tqdm(self.test_dataloader, desc="Evaluating")

        for batch in pbar:
            if self.cap_at_100 and len(predictions) > 100:
                break
            # Check if the model is a MajorityClassModel which needs task and question_type
            if hasattr(self.model, "generate"):
                with torch.no_grad():
                    model_output_dict = self.model(
                        input_ids=batch["input_ids"].to(self.model.device),
                        attention_mask=batch["attention_mask"].to(self.model.device),
                        labels=batch["labels"].to(self.model.device),
                        inputs_continuous_tokens=(
                            batch["inputs_continuous_tokens"]
                            if "inputs_continuous_tokens" in batch
                            else None
                        ),
                        labels_continuous_tokens=(
                            batch["labels_continuous_tokens"]
                            if "labels_continuous_tokens" in batch
                            else None
                        ),
                        extra_args=batch.get("extra_args", []),
                    )
                    if "ContinuousQwen3ForCausalLM" in self.model.config.architectures:
                        stop_strings = None
                    elif "sae_steering" in batch["task"]:
                        stop_strings = ["\n", ">>>."]
                    elif "decode_layer_token" in batch["task"]:
                        stop_strings = ["."]
                    else:
                        # For causal tasks, use [END] as stop string instead of eos_token
                        stop_strings = [" [END]", "\n", ">>>."]
                    generation_outputs = self.model.generate(
                        input_ids=batch["input_ids_without_answer"].to(
                            self.model.device
                        ),
                        attention_mask=batch["attention_mask_without_answer"].to(
                            self.model.device
                        ),
                        max_new_tokens=self.generate_limit,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=(
                            None
                            if "causal" in batch["task"]
                            else self.tokenizer.eos_token_id
                        ),
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        inputs_continuous_tokens=(
                            batch["inputs_without_answer_continuous_tokens"]
                            if "inputs_without_answer_continuous_tokens" in batch
                            else None
                        ),
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        stop_strings=stop_strings,
                        tokenizer=self.tokenizer,
                        extra_args=batch.get("extra_args", []),
                    )

                    if generation_outputs is not None:
                        generation_outputs["loss"] = model_output_dict["loss"]
            # Track indices for generation outputs
            generation_idx = 0
            metadata = None

            for i in range(len(batch["expected_label"])):
                item_features = {key: batch[key][i] for key in batch.keys()}

                # Check if this is an activation prediction example
                if hasattr(self.model, "generate") and generation_outputs is not None:
                    # For explanation examples: use generation outputs
                    output_features = {}
                    for key in generation_outputs.keys():
                        if key == "hidden_states" or key == "loss":
                            continue
                        if key == "past_key_values":
                            output_features[key] = [
                                [
                                    output_layer[generation_idx]
                                    for output_layer in output_per_input
                                ]
                                for output_per_input in generation_outputs[key]
                            ]
                        else:
                            output_features[key] = generation_outputs[key][
                                generation_idx
                            ]
                    generation_idx += 1  # Increment for next explanation example
                    predicted_label, metadata = self.get_prediction(
                        output_features, item_features
                    )
                    if predicted_label is None:
                        continue
                    metrics = self.evaluate_item(
                        predicted_label, batch["expected_label"][i], item_features
                    )
                elif hasattr(self.model, "get_prediction"):
                    predicted_label, metadata = self.model.get_prediction(item_features)
                    metrics = self.evaluate_item(
                        predicted_label, batch["expected_label"][i], item_features
                    )
                else:
                    # Handle cases where model doesn't have generate or get_prediction methods
                    print(
                        "Warning: Model does not have generate() or get_prediction() methods"
                    )
                    predicted_label = None
                    metrics = None
                prediction = {}
                prediction["predicted_label"] = predicted_label
                prediction["expected_label"] = batch["expected_label"][i]
                if metadata is not None:
                    prediction["metadata"] = metadata

                def filter_tensors(obj):
                    """Recursively filter out torch.Tensor objects from nested structures."""
                    if isinstance(obj, torch.Tensor):
                        return None
                    elif isinstance(obj, dict):
                        return {
                            k: v
                            for k, v in ((k, filter_tensors(v)) for k, v in obj.items())
                            if v is not None
                        }
                    elif isinstance(obj, (list, tuple)):
                        filtered = [filter_tensors(item) for item in obj]
                        return [item for item in filtered if item is not None]
                    else:
                        return obj

                filtered_features = filter_tensors(item_features)
                prediction.update(filtered_features)
                if metrics is not None:
                    prediction.update(metrics)
                predictions.append(prediction)
                if getattr(self, "output_type", "text") == "text" and save_file:
                    with open(save_file, "a") as f:
                        f.write(json.dumps(predictions[-1]) + "\n")

            self.pbar_update(pbar)
        self.finalize_metrics()
        # Save activations if available
        if getattr(self, "output_type", "text") != "text" and save_file:
            predictions = [p["predicted_label"] for p in predictions]
            try:
                if (
                    predictions
                    and "extra_args" in predictions[0]
                    and "activations" in predictions[0].get("extra_args", {})
                ):
                    activations = [
                        p["extra_args"]["activations"]
                        for p in predictions
                        if "extra_args" in p and "activations" in p["extra_args"]
                    ]
                    if activations:
                        with open(
                            save_file.replace(".json", "_activations.pt"), "wb"
                        ) as f:
                            torch.save(activations, f)
            except Exception as e:
                print(f"Warning: Could not save additional evaluation data: {e}")
        self.model.train()
        return predictions

    def parse_predicted_label(self, predicted_label: str) -> Any:
        pass

    def pbar_update(self, pbar: tqdm):
        pass

    def log_metrics(self, prefix: str = "", step: Optional[int] = None):
        pass

    def print_metrics(self, prefix: str = ""):
        pass

    def reset_metrics(self):
        """
        Reset metrics to 0
        """

    def finalize_metrics(self):
        """
        Normalize metrics on evaluation end
        """

    def evaluate_item(self, predicted_label: str, expected_label: str, full_item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update metrics for a single item.
        """


class EvaluationCallback(TrainerCallback):
    """Callback to evaluate the model at the end of each epoch."""

    def __init__(
        self,
        evaluators: Dict[str, BaseEvaluator],
        eval_strategy: str = "epoch",
        eval_steps: int = 500,
        debug_mode: bool = False,
    ):
        self.evaluators = evaluators
        self.eval_strategy = eval_strategy
        self.eval_steps = eval_steps
        self.debug_mode = debug_mode

    def on_step_end(self, args, state, control, **kwargs):
        if self.eval_strategy == "steps" and state.global_step % self.eval_steps == 0:
            for evaluator_name, evaluator in self.evaluators.items():
                evaluator.evaluate(
                    fewshot=False,
                    save_file=f"{evaluator_name}_step_{state.global_step}_predictions.json",
                )
                if not self.debug_mode:
                    evaluator.log_metrics(
                        prefix=f"per_step_{evaluator_name}_",
                        step=int(state.global_step),
                    )
                evaluator.print_metrics(
                    prefix=f"Step {state.global_step} - {evaluator_name}: "
                )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.eval_strategy == "epoch":
            for evaluator_name, evaluator in self.evaluators.items():
                evaluator.evaluate(
                    fewshot=False,
                    save_file=f"{evaluator_name}_epoch_{state.epoch}_predictions.json",
                )
                if not self.debug_mode:
                    evaluator.log_metrics(
                        prefix=f"per_epoch_{evaluator_name}_",
                        step=int(state.global_step),
                    )
                evaluator.print_metrics(
                    prefix=f"Epoch {state.epoch} - {evaluator_name}: "
                )

        return control
