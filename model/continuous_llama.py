from typing import Optional, Tuple, Union

import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache

from .continuous_base import ContinuousCausalLMBase, ContinuousCausalLMOutputWithPast


class ContinuousLlama(ContinuousCausalLMBase, LlamaForCausalLM):
    """
    Extension of LlamaForCausalLM that allows for intervention in the hidden states
    and attention head outputs during the forward pass.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 24,
        special_tokens_ids: dict[str, int] | None = None,
        subject_embed_dim: int | None = None,
        use_embed_proj: bool = False,
        target_model_layers: int | None = None,
        **kwargs,
    ):
        LlamaForCausalLM.__init__(self, *args, **kwargs)
        ContinuousCausalLMBase.__init__(
            self,
            batch_size=batch_size,
            special_tokens_ids=special_tokens_ids,
            subject_embed_dim=subject_embed_dim,
            use_embed_proj=use_embed_proj,
            target_model_layers=target_model_layers,
            **kwargs,
        )
        # State for hidden state interventions (existing)
        self._intervention_hidden_states = None
        self._intervention_active = False
        self._intervention_is_batched = False
        # State for attention head interventions (updated)
        self._head_intervention_layer = None
        self._head_intervention_indices_positions = None  # Can be list for batching
        self._head_intervention_active = False
        self._head_intervention_is_batched = False
        self._head_patching_handles = []  # To store hook handles
        # State for attribution patching
        self._attribution_active = False
        self._attribution_hooks = []
        self._attribution_gradients = {}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        inputs_continuous_tokens: Optional[list[torch.LongTensor]] = None,
        labels_continuous_tokens: Optional[list[torch.LongTensor]] = None,
        debug: bool = False,
        extra_args: Optional[list[dict]] = None,
        **kwargs,
    ) -> Union[Tuple, ContinuousCausalLMOutputWithPast]:
        """
        inputs_continuous_tokens: List[torch.LongTensor] of dim batch_size x (seq_len)
        labels_continuous_tokens: List[torch.LongTensor] of dim batch_size x (seq_len)
        """
        return self.shared_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            inputs_continuous_tokens=inputs_continuous_tokens,
            labels_continuous_tokens=labels_continuous_tokens,
            debug=debug,
            extra_args=extra_args,
            **kwargs,
        )
