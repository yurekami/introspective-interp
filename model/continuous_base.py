import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class ContinuousCausalLMOutputWithPast(CausalLMOutputWithPast):
    activation_loss: Optional[torch.Tensor] = None
    binary_loss: Optional[torch.Tensor] = None
    magnitude_loss: Optional[torch.Tensor] = None
    text_loss: Optional[torch.Tensor] = None
    total_loss: Optional[torch.Tensor] = None


def fixed_cross_entropy_batchwise(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction="none"
    )
    # if num_items_in_batch is not None:
    #     loss = loss / num_items_in_batch
    return loss.mean(-1)


def ForCausalLMLossBatchwise(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    shift_labels = shift_labels.to(logits.device)
    logits = logits.permute(0, 2, 1)
    loss = fixed_cross_entropy_batchwise(
        logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    if num_items_in_batch is not None:
        num_items_in_batch = num_items_in_batch.to(logits.device)
    loss = fixed_cross_entropy(
        logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss


class ContinuousCausalLMBase:
    """
    Base class for continuous causal language models that provides shared functionality
    for continuous token processing, generation, and loss computation.
    """

    def __init__(
        self,
        batch_size: int = 24,
        special_tokens_ids: Dict[str, int] | None = None,
        subject_embed_dim: int | None = None,
        use_embed_proj: bool = True,
        target_model_layers: int | None = None,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.debug = False
        self.special_tokens_ids = special_tokens_ids

        # Embed projection for different subject dimensions
        if use_embed_proj or (
            subject_embed_dim is not None
            and subject_embed_dim != self.config.hidden_size
        ):
            if target_model_layers is None:
                target_model_layers = self.config.num_hidden_layers
            # W_subject -> W_predictor
            self.embed_projs = nn.ModuleList(
                [
                    nn.Linear(
                        subject_embed_dim,
                        self.config.hidden_size,
                        bias=False,
                    )
                    for _ in range(target_model_layers)
                ]
            )
            for embed_proj in self.embed_projs:
                embed_proj.requires_grad_(True)  # type: ignore
        else:
            self.embed_projs = None

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Prepare inputs for generation, handling continuous input tokens."""
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Simply pass through continuous input tokens if provided
        if "inputs_continuous_tokens" in kwargs:
            model_inputs["inputs_continuous_tokens"] = kwargs[
                "inputs_continuous_tokens"
            ]

        return model_inputs

    def train(self, is_training: bool = True):
        super().train(is_training)
        if is_training:
            self._loss_function = ForCausalLMLoss
            self.is_training = True
        else:
            self._loss_function = ForCausalLMLossBatchwise
            self.is_training = False
        return self

    def eval(self):
        super().eval()
        self._loss_function = ForCausalLMLossBatchwise
        return self

    def _process_continuous_tokens(
        self,
        input_ids: torch.LongTensor,
        inputs_continuous_tokens: Optional[List[torch.LongTensor]],
        layers: List[int],
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, bool, Optional[torch.Tensor]]:
        """
        Process continuous tokens in inputs and labels.
        Returns (inputs_embeds, labels_have_continuous_tokens, labels_continuous_token_mask)
        """
        labels_have_continuous_tokens = False
        labels_continuous_token_mask = None

        # Always get base embeddings first
        inputs_embeds = self.model.embed_tokens(input_ids)

        # Process continuous tokens using masked operations to avoid data-dependent branching
        if (
            (inputs_continuous_tokens is not None)
            and any([inputs_continuous_tokens is not None])
            and self.special_tokens_ids is not None
            and self.special_tokens_ids.get("continuous_rep") is not None
        ):
            continuous_input_tokens_mask = (
                input_ids == self.special_tokens_ids["continuous_rep"]
            )
            # Clone embeddings - this happens regardless of mask content
            new_inputs_embeds = inputs_embeds.clone()

            for i in range(len(inputs_continuous_tokens)):
                if (
                    inputs_continuous_tokens[i] is not None
                    and inputs_continuous_tokens[i].shape[0] > 0
                ):
                    # Use masked operations instead of conditional branching
                    mask_i = continuous_input_tokens_mask[i]
                    num_continuous = mask_i.sum()

                    # Create a condition tensor for whether to update (avoids .any())
                    should_update = num_continuous > 0

                    if (
                        should_update
                    ):  # This is still data-dependent, but more Dynamo-friendly
                        if (
                            new_inputs_embeds.shape[-1]
                            != inputs_continuous_tokens[i].shape[-1]
                        ):
                            assert self.embed_projs is not None
                            # Project continuous tokens to embedding space
                            projected_tokens = self.embed_projs[layers[i]](
                                inputs_continuous_tokens[i][-num_continuous:].to(
                                    self.embed_projs[layers[i]].weight.device,
                                    dtype=self.embed_projs[layers[i]].weight.dtype,
                                )
                            ).to(new_inputs_embeds.device)

                            # Use masked assignment
                            new_inputs_embeds[i, mask_i, :] = projected_tokens
                        else:
                            # Direct replacement without projection
                            continuous_tokens = inputs_continuous_tokens[i][
                                -num_continuous:
                            ].to(
                                device=new_inputs_embeds.device,
                                dtype=new_inputs_embeds.dtype,
                            )
                            new_inputs_embeds[i, mask_i, :] = continuous_tokens
                    else:
                        assert num_continuous == 0

            inputs_embeds = new_inputs_embeds

        # Process labels using masked operations to avoid data-dependent branching
        if labels is not None and self.special_tokens_ids is not None:
            labels_continuous_token_mask = (
                labels == self.special_tokens_ids["continuous_rep"]
            )
            # Replace continuous tokens with ignore index using torch.where
            labels = torch.where(
                labels_continuous_token_mask,
                torch.tensor(-100, device=labels.device, dtype=labels.dtype),
                labels,
            )
            # Avoid .any() which causes data-dependent branching
            # Use sum() > 0 which is more Dynamo-friendly
            labels_have_continuous_tokens = labels_continuous_token_mask.any()

        return (
            inputs_embeds,
            labels_have_continuous_tokens,
            labels_continuous_token_mask,
        )

    def _register_intervention_hooks(
        self,
        intervention_mask,
        intervention_layers,
        intervention_positions,
        intervention_vectors,
    ):
        """
        Register forward hooks to apply causal interventions during model forward pass.

        Args:
            intervention_mask: Boolean tensor indicating which batch items need interventions
            intervention_layers: List of layer indices for each batch item
            intervention_positions: List of token positions for each batch item
            intervention_vectors: List of intervention vector info for each batch item

        Returns:
            List of hook handles that can be removed later
        """
        hooks = []

        # Group interventions by layer for efficiency
        layer_to_interventions = {}
        for batch_idx in range(len(intervention_mask)):
            if (
                intervention_mask[batch_idx]
                and intervention_vectors[batch_idx] is not None
            ):
                layer = intervention_layers[batch_idx]
                if layer not in layer_to_interventions:
                    layer_to_interventions[layer] = []
                layer_to_interventions[layer].append(
                    {
                        "batch_idx": batch_idx,
                        "positions": intervention_positions[batch_idx],
                        "vector_info": intervention_vectors[batch_idx],
                    }
                )

        # Register hooks for each layer that needs interventions
        for layer_idx, interventions in layer_to_interventions.items():

            def make_hook(layer_idx, interventions):
                def intervention_hook(module, input, output):
                    # output is the hidden states tensor [batch_size, seq_len, hidden_size]
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    # Apply interventions for each batch item
                    for intervention in interventions:
                        batch_idx = intervention["batch_idx"]
                        positions = intervention["positions"]
                        vector_info = intervention["vector_info"]

                        # Apply intervention using the helper method
                        intervention_info = vector_info.copy()
                        intervention_info["positions"] = positions

                        # Apply intervention to this batch item's hidden states
                        hidden_states[batch_idx : batch_idx + 1] = (
                            self._apply_causal_intervention(
                                hidden_states[batch_idx : batch_idx + 1],
                                intervention_info,
                                layer_idx,
                            )
                        )

                    # Return modified output in the same format
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states

                return intervention_hook

            # Register the hook on the appropriate layer
            layer_module = self.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(
                make_hook(layer_idx, interventions)
            )
            hooks.append(hook)

        return hooks

    def _apply_causal_intervention(
        self, hidden_states: torch.Tensor, intervention_info: dict, layer_idx: int
    ) -> torch.Tensor:
        """
        Apply causal intervention to hidden states at specified layer and positions.

        Args:
            hidden_states: Layer hidden states [batch_size, seq_len, hidden_size]
            intervention_info: Dict containing intervention details
            layer_idx: Current layer index

        Returns:
            Modified hidden states with intervention applied
        """
        if intervention_info is None or intervention_info.get("layer") != layer_idx:
            return hidden_states

        # Create a copy to avoid modifying original
        modified_hidden_states = hidden_states.clone()
        intervention_vector = intervention_info["vector"]

        # Project if needed
        if self.embed_projs is not None:
            intervention_vector = self.embed_projs[layer_idx](intervention_vector)

        # Apply to specified positions
        positions = intervention_info.get("positions", [])
        if positions:
            for pos in positions:
                if isinstance(pos, (list, np.ndarray)):
                    pos = pos.tolist() if hasattr(pos, "tolist") else pos
                if isinstance(pos, list):
                    for p in pos:
                        if 0 <= p < modified_hidden_states.shape[1]:
                            modified_hidden_states[:, p, :] += intervention_vector.to(
                                modified_hidden_states.device
                            )
                else:
                    if 0 <= pos < modified_hidden_states.shape[1]:
                        modified_hidden_states[:, pos, :] += intervention_vector.to(
                            modified_hidden_states.device
                        )

        return modified_hidden_states

    @property
    def loss_function(self):
        """Get the appropriate loss function based on training mode."""
        return getattr(self, "_loss_function", ForCausalLMLoss)

    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        model = super().from_pretrained(model_path, *args, **kwargs)
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """Override save_pretrained to include all custom components."""
        # First save the base model
        super().save_pretrained(save_directory, **kwargs)
        print(f"Saved model to {save_directory}")

    def _get_model_hidden_states(self, outputs):
        """Get hidden states from model outputs - handles different model types."""
        # Llama uses outputs[0], while Gemma/Qwen use outputs.last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        else:
            return outputs[0]

    def _apply_logit_postprocessing(self, logits):
        """Apply model-specific logit postprocessing (e.g., Gemma's softcapping)."""
        # Gemma-specific softcapping
        if (
            hasattr(self.config, "final_logit_softcapping")
            and self.config.final_logit_softcapping is not None
        ):
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        return logits

    def shared_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        num_logits_to_keep=0,  # Llama parameter name
        logits_to_keep=None,  # Gemma/Qwen parameter name
        inputs_continuous_tokens=None,
        labels_continuous_tokens=None,
        debug=False,
        extra_args=None,
        # Causal intervention parameters
        intervention_mask=None,
        intervention_layers=None,
        intervention_positions=None,
        intervention_vectors=None,
        **kwargs,
    ):
        """Shared forward method implementation for all continuous models."""
        # Normalize parameter names - use logits_to_keep consistently
        if logits_to_keep is None:
            logits_to_keep = num_logits_to_keep

        if "num_items_in_batch" in kwargs:
            del kwargs["num_items_in_batch"]

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Process continuous tokens using base class method
        # try:
        inputs_embeds, labels_have_continuous_tokens, _ = (
            self._process_continuous_tokens(
                input_ids,
                inputs_continuous_tokens,
                layers=(
                    [item.get("layer", None) for item in extra_args]
                    if extra_args is not None
                    else []
                ),
                labels=labels,
            )
        )

        # Check if we need to apply causal interventions
        has_interventions = (
            intervention_mask is not None
            and intervention_mask.any()
            and intervention_layers is not None
            and intervention_vectors is not None
        )

        # Apply intervention hooks if needed
        intervention_hooks = []
        if has_interventions:
            intervention_hooks = self._register_intervention_hooks(
                intervention_mask,
                intervention_layers,
                intervention_positions,
                intervention_vectors,
            )

        # Run the model
        outputs = self.model(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        # Always remove hooks to prevent memory leaks
        for hook in intervention_hooks:
            hook.remove()

        # Get hidden states (handles different model types)
        last_hidden_states = self._get_model_hidden_states(outputs)

        # Compute logits with proper slicing
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(last_hidden_states[:, slice_indices, :])

        # Apply model-specific postprocessing (e.g., Gemma softcapping)
        logits = self._apply_logit_postprocessing(logits)

        # Extract question type weights from batch for loss weighting
        question_type_weights = kwargs.get("question_type_weights", None)

        loss = None
        text_loss = None
        activation_loss = None
        binary_loss = None
        magnitude_loss = None
        if labels is not None:
            # Use batchwise loss to get per-sample losses for weighting
            losses_per_sample = ForCausalLMLossBatchwise(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

            # Apply question type weights if available
            if question_type_weights is not None:
                weights = question_type_weights.to(losses_per_sample.device)
                weighted_losses = losses_per_sample * weights
                loss = weighted_losses.mean()
            else:
                loss = losses_per_sample.mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # Create the output object
        output_obj = ContinuousCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            text_loss=text_loss,
            activation_loss=activation_loss,
            binary_loss=binary_loss,
            magnitude_loss=magnitude_loss,
        )

        # Save outputs for logging callback
        self.last_outputs = output_obj

        return output_obj
