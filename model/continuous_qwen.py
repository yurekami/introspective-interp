from typing import Optional, Tuple, Union

import torch
from transformers import Qwen3ForCausalLM
from transformers.cache_utils import Cache

from .continuous_base import ContinuousCausalLMBase, ContinuousCausalLMOutputWithPast


class ContinuousQwen3ForCausalLM(ContinuousCausalLMBase, Qwen3ForCausalLM):
    """
    Continuous Qwen3 model for causal language modeling with continuous token support.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config,
        batch_size: int = 24,
        special_tokens_ids: dict[str, int] | None = None,
        subject_embed_dim: int | None = None,
        use_embed_proj: bool = True,
        target_model_layers: int | None = None,
        **kwargs,
    ):
        Qwen3ForCausalLM.__init__(self, config)
        ContinuousCausalLMBase.__init__(
            self,
            batch_size=batch_size,
            special_tokens_ids=special_tokens_ids,
            subject_embed_dim=subject_embed_dim,
            use_embed_proj=use_embed_proj,
            target_model_layers=target_model_layers,
            **kwargs,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        inputs_continuous_tokens: Optional[list[torch.LongTensor]] = None,
        labels_continuous_tokens: Optional[list[torch.LongTensor]] = None,
        debug: bool = False,
        extra_args: Optional[list[dict]] = None,
        **kwargs,
    ) -> Union[Tuple, ContinuousCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        return self.shared_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            inputs_continuous_tokens=inputs_continuous_tokens,
            labels_continuous_tokens=labels_continuous_tokens,
            debug=debug,
            extra_args=extra_args,
            **kwargs,
        )
