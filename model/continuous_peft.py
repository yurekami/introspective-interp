from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from peft import PeftModelForCausalLM

# from transformers.models.llama.modeling_llama import KwargsForCausalLM


class ContinuousPeft(PeftModelForCausalLM):
    """
    Peft wrapper around a continuous model.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 24,
        special_tokens_ids: Dict[str, int] | None = None,
        subject_embed_dim: int | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        self.batch_size = batch_size
        self.special_tokens_ids = special_tokens_ids
        self.subject_embed_dim = subject_embed_dim

    def forward_with_hidden_states(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        layer: int = 0,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return self.base_model.forward_with_hidden_states(
            input_ids,
            attention_mask,
            hidden_states,
            layer,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            **kwargs,
        )

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        return self.base_model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

    def train(self, is_training: bool = True):
        super().train(is_training)
        return self.base_model.train(is_training)

    def eval(self):
        super().eval()
        return self.base_model.eval()

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[List[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        **kwargs: Any,
    ) -> None:
        """Override to save embed_proj weights alongside adapter weights."""
        # Call parent's save_pretrained
        super().save_pretrained(
            save_directory=save_directory,
            safe_serialization=safe_serialization,
            selected_adapters=selected_adapters,
            save_embedding_layers=True,
            is_main_process=is_main_process,
            **kwargs,
        )

        # Additionally save embed_proj if it exists (using base model's method)
        if is_main_process and hasattr(self.base_model, "save_embed_proj"):
            self.base_model.save_embed_proj(save_directory, safe_serialization)

    @classmethod
    def from_pretrained(
        cls,
        base_model,
        model_path,
        peft_config,
        adapter_name="default",
        autocast_adapter_dtype=True,
        low_cpu_mem_usage=False,
        special_tokens=None,
        **kwargs,
    ):
        # Call parent class's from_pretrained
        model = super().from_pretrained(base_model, model_path)

        # Convert to PeftLlamaWithIntervention
        new_model = cls(
            model.base_model,
            peft_config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            special_tokens=special_tokens,
            **kwargs,
        )
        new_model.base_model = model.base_model
        return new_model
