from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


def merge_config_with_parent(
    parent_config: Dict[str, Any], task_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge task config with parent config, with task config taking precedence.
    Handles nested dictionaries properly.
    """
    merged = parent_config.copy()

    for key, value in task_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_config_with_parent(merged[key], value)
        else:
            # Task config overrides parent config
            merged[key] = value

    return merged


def get_first_available_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    devices = list(range(torch.cuda.device_count()))
    min_used: Optional[int] = None
    best_device: Optional[int] = None
    for d in devices:
        # stats = torch.cuda.memory_stats(d)
        free, total = torch.cuda.mem_get_info(d)
        used = total - free
        if min_used is None or used < min_used:
            min_used = used
            best_device = d
    return torch.device(best_device)


def save_reserved_token_embeddings(model: Any, special_tokens_ids: dict[str, int], save_dir: str) -> None:
    """
    Save the initialized embeddings for reserved special tokens.

    Args:
        model: The model with initialized embeddings
        special_tokens_ids: Dictionary mapping token names to token IDs
        save_dir: Directory to save the embeddings
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save embedding layer weights
    embedding_weights = {}
    for token_name, token_id in special_tokens_ids.items():
        embedding_weights[token_name] = {
            "token_id": token_id,
            "embedding": model.model.embed_tokens.weight[token_id].detach().cpu(),
        }

    # Save lm_head weights if available
    lm_head_weights = {}
    if hasattr(model, "lm_head"):
        for token_name, token_id in special_tokens_ids.items():
            lm_head_weights[token_name] = {
                "token_id": token_id,
                "weight": model.lm_head.weight[token_id].detach().cpu(),
            }

    # Save to files
    torch.save(
        embedding_weights, os.path.join(save_dir, "reserved_token_embeddings.pt")
    )
    if lm_head_weights:
        torch.save(lm_head_weights, os.path.join(save_dir, "reserved_token_lm_head.pt"))

    # Save metadata
    metadata = {
        "special_tokens_ids": special_tokens_ids,
        "embedding_dim": model.model.embed_tokens.embedding_dim,
        "vocab_size": model.model.embed_tokens.num_embeddings,
    }
    with open(os.path.join(save_dir, "reserved_token_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved reserved token embeddings to {save_dir}")


def load_reserved_token_embeddings(model: Any, load_dir: str) -> Optional[dict[str, int]]:
    """
    Load and apply the saved embeddings for reserved special tokens.

    Args:
        model: The model to apply embeddings to
        load_dir: Directory containing the saved embeddings

    Returns:
        Dictionary mapping token names to token IDs
    """
    # Load metadata
    metadata_path = os.path.join(load_dir, "reserved_token_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Warning: No metadata found at {metadata_path}")
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    special_tokens_ids = metadata["special_tokens_ids"]

    # Load embedding weights
    embedding_path = os.path.join(load_dir, "reserved_token_embeddings.pt")
    if os.path.exists(embedding_path):
        embedding_weights = torch.load(embedding_path, map_location="cpu")

        with torch.no_grad():
            for token_name, data in embedding_weights.items():
                token_id = data["token_id"]
                embedding = data["embedding"]

                # Apply to model
                model.model.embed_tokens.weight[token_id] = embedding.to(
                    model.model.embed_tokens.weight.device
                )
                print(f"Loaded embedding for {token_name} (ID: {token_id})")

    # Load lm_head weights if available
    lm_head_path = os.path.join(load_dir, "reserved_token_lm_head.pt")
    if os.path.exists(lm_head_path) and hasattr(model, "lm_head"):
        lm_head_weights = torch.load(lm_head_path, map_location="cpu")

        with torch.no_grad():
            for token_name, data in lm_head_weights.items():
                token_id = data["token_id"]
                weight = data["weight"]

                # Apply to model
                model.lm_head.weight[token_id] = weight.to(model.lm_head.weight.device)
                print(f"Loaded lm_head weight for {token_name} (ID: {token_id})")

    print(f"Loaded reserved token embeddings from {load_dir}")

    return special_tokens_ids


def initialize_reserved_token_embeddings(model: Any, special_tokens_ids: dict[str, int], save_dir: Optional[str] = None) -> dict[str, int]:
    """
    Initialize reserved special token embeddings with mean + variance and optionally save them.

    Args:
        model: The model to initialize embeddings for
        special_tokens_ids: Dictionary mapping token names to token IDs
        save_dir: Optional directory to save the initialized embeddings

    Returns:
        Dictionary mapping token names to token IDs
    """
    # Get the embedding layer
    embedding_layer = model.model.embed_tokens
    vocab_size = embedding_layer.num_embeddings
    embedding_dim = embedding_layer.embedding_dim

    print(
        f"Initializing {len(special_tokens_ids)} reserved special token embeddings..."
    )

    # Calculate mean and std of existing embeddings and lm_head weights
    with torch.no_grad():
        embed_mean = embedding_layer.weight.mean(0)
        embed_var = embedding_layer.weight.var(0)
        embed_std = torch.sqrt(embed_var)

        lm_head_mean = model.lm_head.weight.mean(0)
        lm_head_var = model.lm_head.weight.var(0)
        lm_head_std = torch.sqrt(lm_head_var)

        print(
            f"  Input embeddings - Mean: {embed_mean.mean().item():.4f}, Std: {embed_std.mean().item():.4f}"
        )
        print(
            f"  LM head weights - Mean: {lm_head_mean.mean().item():.4f}, Std: {lm_head_std.mean().item():.4f}"
        )

    # Initialize each token
    for token_name, token_id in special_tokens_ids.items():
        if token_id < vocab_size:
            with torch.no_grad():
                # Initialize input embeddings
                new_embedding = embed_mean + embed_std * torch.randn(
                    embedding_dim, device=embedding_layer.weight.device
                )
                embedding_layer.weight[token_id] = new_embedding

                # Initialize output layer (lm_head)
                if hasattr(model, "lm_head"):
                    new_lm_head = lm_head_mean + lm_head_std * torch.randn(
                        embedding_dim, device=model.lm_head.weight.device
                    )
                    model.lm_head.weight[token_id] = new_lm_head

            print(
                f"  Initialized {token_name} (ID: {token_id}) with mean + variance embeddings"
            )
        else:
            print(
                f"  Warning: Token ID {token_id} for {token_name} exceeds vocabulary size {vocab_size}"
            )

    # Save embeddings if save_dir is provided
    if save_dir is not None:
        save_reserved_token_embeddings(model, special_tokens_ids, save_dir)

    return special_tokens_ids


class LossLoggingCallback(TrainerCallback):
    """Custom callback to log detailed loss components."""

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: Any, logs: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        """Called when logging occurs - extract and log detailed losses."""
        if (
            logs is not None
            and hasattr(model, "last_outputs")
            and model.last_outputs is not None
        ):
            outputs = model.last_outputs

            # Extract individual loss components
            if hasattr(outputs, "text_loss") and outputs.text_loss is not None:
                logs["train/text_loss"] = round(outputs.text_loss.item(), 4)
                # print(f"Text Loss: {outputs.text_loss.item():.4f}")

            if (
                hasattr(outputs, "activation_loss")
                and outputs.activation_loss is not None
            ):
                logs["train/activation_loss"] = round(outputs.activation_loss.item(), 4)
                # print(f"Activation Loss: {outputs.activation_loss.item():.4f}")

            if (
                hasattr(outputs, "continuous_loss")
                and outputs.continuous_loss is not None
            ):
                if isinstance(outputs.continuous_loss, list):
                    avg_cont_loss = sum(outputs.continuous_loss) / len(
                        outputs.continuous_loss
                    )
                    logs["train/continuous_loss"] = round(avg_cont_loss, 4)
                else:
                    logs["train/continuous_loss"] = round(
                        outputs.continuous_loss.item(), 4
                    )


class EarlyStoppingCallback(TrainerCallback):
    def __init__(
        self, patience: int = 5, metric_for_best_model: str = "eval_loss", greater_is_better: bool = False
    ) -> None:
        self.patience = patience
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.best_metric: Optional[float] = None
        self.patience_counter: int = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: Optional[Any] = None, logs: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        if logs is None:
            return

        # Look for the metric in logs - try different possible formats
        current_metric: Optional[float] = None
        metric_keys = [
            self.metric_for_best_model,
            f"eval_{self.metric_for_best_model}",
            f"test_{self.metric_for_best_model}",
        ]

        for key in metric_keys:
            if key in logs:
                current_metric = logs[key]
                break

        # If we can't find any eval metric, try to use any eval metric available
        if current_metric is None:
            eval_metrics = {
                k: v
                for k, v in logs.items()
                if k.startswith("eval_")
                or k.startswith("test_")
                or k.startswith("per_epoch_")
                or k.startswith("per_step_")
            }
            if eval_metrics:
                # Use the first available eval metric
                metric_key = list(eval_metrics.keys())[0]
                current_metric = eval_metrics[metric_key]
                if self.best_metric is None:  # First time, update the metric name
                    print(f"Early stopping using metric: {metric_key}")
                    self.metric_for_best_model = metric_key

        if current_metric is None:
            return

        # Check if this is the best metric we've seen
        is_better: bool = False
        if self.best_metric is None:
            is_better = True
        elif self.greater_is_better:
            is_better = current_metric > self.best_metric
        else:
            is_better = current_metric < self.best_metric

        if is_better:
            self.best_metric = current_metric
            self.patience_counter = 0
            print(f"New best {self.metric_for_best_model}: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            print(
                f"No improvement in {self.metric_for_best_model} for {self.patience_counter}/{self.patience} evaluations"
            )

        # Stop training if patience exceeded
        if self.patience_counter >= self.patience:
            print(
                f"Early stopping triggered after {self.patience} evaluations without improvement"
            )
            control.should_training_stop = True
