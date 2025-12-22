from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any


class SelfExplanationsModel:
    """
    A model that uses self-explanations to generate predictions.

    This model patches in the continuous representation to early layers of the model, and returns
    the best explanation for the given input.
    """

    def __init__(
        self,
        # dataset,
        # data_dir=None,
        model_name="meta-llama/Llama-3.1-8B",
        scales=None,
        cache_dir=None,
        **kwargs,
    ):
        """
        Initialize the self-explanations model.

        Args:
            dataset: The dataset to analyze for storing continuous representations
            data_dir: Directory to save/load cached explanations
            model_name: Name of the language model to use
            scales: List of scales to use for patching (default: [1, 5, 10, 25, 50])
        """
        self.device = torch.device("cuda")  # Dummy device for compatibility
        # self.dataset = dataset
        self.with_threshold = False
        self.layer_wise_similarities = True
        # self.data_dir = data_dir
        self.model_name = model_name
        self.scales = scales or [1, 5, 10, 25, 50]
        self.max_new_tokens = 25

        self.cache_dir = cache_dir

        self.batch_size = 64
        self.patch_layer = 0

        # Load or compute explanations
        self.explanations_cache = {}

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the language model for patching."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.requires_grad_(False)

        self.is_chat = self.tokenizer.chat_template is not None
        if self.is_chat:
            self.prompt = [
                {"role": "user", "content": "Briefly define the word 'X'."},
                {"role": "assistant", "content": "The meaning of the word 'X' is "},
            ]
        else:
            self.prompt = (
                "What is the meaning of the word 'X'? The meaning of the word 'X' is "
            )

        # Find positions of 'X' in the prompt
        if self.tokenizer.chat_template is not None:
            prompts = self.tokenizer.apply_chat_template(self.prompt, tokenize=False)
            # remove eos token
            prompt_tokens = self.tokenizer.encode(prompts)[:-1]
            self.positions = [
                i
                for i, a in enumerate(prompt_tokens)
                if self.tokenizer.decode([a]) == "X"
            ]
        else:
            self.positions = [
                i
                for i, a in enumerate(self.tokenizer.encode(self.prompt))
                if self.tokenizer.decode([a]) == "X"
            ]

    def _generate_explanations_for_all_scales(
        self, vector: torch.Tensor, layer: int
    ) -> Dict[float, str] | list[Dict[float, str]]:
        """
        Generate explanations for all scales in batch using patching.

        Args:
            vector: Either a single vector (1D tensor) or a batch of vectors (2D tensor with shape [num_vectors, width])
            layer: The layer to patch into

        Returns:
            If single vector: dict mapping scales to explanations
            If batch of vectors: list of dicts, each mapping scales to explanations
        """
        if not self.model:
            self._initialize_model()

        # Convert single vector to batch of size 1 for unified processing
        if vector.dim() == 1:
            vectors = vector.unsqueeze(0)  # Add batch dimension
            is_single = True
        else:
            vectors = vector
            is_single = False

        # Process all vectors in batch
        batch_results = self._generate_explanations_with_input_embeds(vectors, layer)

        # Return single dict for single vector, list for batch
        return batch_results[0] if is_single else batch_results

    def _generate_explanations_with_input_embeds(
        self, vectors: torch.Tensor, layer: int
    ) -> list[Dict[float, str]]:
        """
        Generate explanations using input_embeds approach with HF generate().

        This method creates a single batch of input_embeds where the 'X' token embeddings
        are replaced with scaled continuous vectors for all vectors × all scales,
        then uses HF's generate() method for efficient batched generation.
        """
        if self.model is None or self.tokenizer is None:
            print("Model or tokenizer not initialized")
            num_vectors = vectors.shape[0]
            return [
                {int(scale): "" for scale in self.scales} for _ in range(num_vectors)
            ]

        num_vectors = vectors.shape[0]
        all_explanations = []

        try:
            # Create base prompt tokens
            if self.is_chat:
                prompt_text = self.tokenizer.apply_chat_template(
                    self.prompt, tokenize=False
                )
                base_tokens = self.tokenizer.encode(prompt_text)[
                    :-1
                ]  # Remove EOS token
            else:
                base_tokens = self.tokenizer.encode(self.prompt)

            # Create base embeddings
            base_embeds = self.model.model.embed_tokens(
                torch.tensor([base_tokens], device=self.model.device)
            )

            # Create batched input_embeds for all vectors × all scales
            total_batch_size = num_vectors * len(self.scales)
            batch_input_embeds = base_embeds.repeat(total_batch_size, 1, 1)

            # Fill in the scaled vectors for each combination
            batch_idx = 0
            for vector_idx in range(num_vectors):
                for scale in self.scales:
                    # Scale the continuous vector
                    scaled_vector = vectors[vector_idx].clone() * scale
                    scaled_vector = scaled_vector.to(batch_input_embeds.device)

                    # Replace 'X' token embeddings in this batch item
                    for position in self.positions:
                        batch_input_embeds[batch_idx, position, :] = scaled_vector
                    batch_idx += 1

            # Generate using batched input_embeds
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=batch_input_embeds,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

            # Decode the generated text and organize by vector
            batch_idx = 0
            for vector_idx in range(num_vectors):
                vector_explanations = {}
                for scale in self.scales:
                    try:
                        # Decode the generated text (skip the prompt part)
                        generated_tokens = outputs.sequences[
                            batch_idx
                        ]  # [len(base_tokens) :]
                        decoded_output = self.tokenizer.decode(
                            generated_tokens, skip_special_tokens=True
                        )
                        vector_explanations[int(scale)] = decoded_output.strip()
                    except Exception as e:
                        print(
                            f"Error decoding output for vector {vector_idx}, scale {scale}: {e}"
                        )
                        vector_explanations[int(scale)] = ""
                    batch_idx += 1
                all_explanations.append(vector_explanations)

        except Exception as e:
            print(f"Batched input embeddings generation failed: {e}")
            # Fallback to individual generation if batch fails
            for vector_idx in range(num_vectors):
                try:
                    vector_explanations = {}
                    for scale in self.scales:
                        try:
                            explanation = self._generate_explanation_for_scale(
                                vectors[vector_idx], layer, scale
                            )
                            vector_explanations[int(scale)] = explanation
                        except Exception as e3:
                            print(
                                f"Error generating explanation for scale {scale}: {e3}"
                            )
                            vector_explanations[int(scale)] = ""
                    all_explanations.append(vector_explanations)
                except Exception as e2:
                    print(
                        f"Error generating explanations for vector {vector_idx}: {e2}"
                    )
                    all_explanations.append({int(scale): "" for scale in self.scales})

        return all_explanations

    def _generate_explanation_for_scale(
        self, vector: torch.Tensor, layer: int, scale: float
    ) -> str:
        """
        Generate explanation for a single vector and scale using input_embeds approach.

        This is a helper method for the fallback case in batch processing.
        """
        if self.model is None or self.tokenizer is None:
            return ""

        try:
            # Create base prompt tokens
            if self.is_chat:
                prompt_text = self.tokenizer.apply_chat_template(
                    self.prompt, tokenize=False
                )
                base_tokens = self.tokenizer.encode(prompt_text)[
                    :-1
                ]  # Remove EOS token
            else:
                base_tokens = self.tokenizer.encode(self.prompt)

            # Create base embeddings
            base_embeds = self.model.model.embed_tokens(
                torch.tensor([base_tokens], device=self.model.device)
            )

            # Scale the continuous vector
            scaled_vector = vector.clone() * scale
            scaled_vector = scaled_vector.to(base_embeds.device)

            # Create input_embeds by replacing 'X' token embeddings
            input_embeds = base_embeds.clone()
            for position in self.positions:
                input_embeds[0, position, :] = scaled_vector

            # Generate using input_embeds
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                )

            # Decode the generated text (skip the prompt part)
            generated_tokens = outputs.sequences[0][len(base_tokens) :]
            decoded_output = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            return decoded_output.strip()

        except Exception as e:
            print(f"Error generating explanation for scale {scale}: {e}")
            return ""

    def to(self, device):
        """Dummy method for compatibility"""
        self.device = device
        return self

    def eval(self):
        """Dummy method for compatibility"""
        return self

    def train(self):
        return self

    def get_prediction(self, inputs: Dict) -> tuple[list[str], Any]:
        """
        Get predictions across all scales for the given input.

        Args:
            inputs: Dictionary containing the input data, including:
                   - continuous representation in inputs["prompt_continuous_tokens"]
                   - layer in inputs["extra_args"]["layer"]
                   - feature_idx in inputs["extra_args"]["feature_idx"]

        Returns:
            List of predictions for each scale, or the best prediction if single_prediction=True
        """
        if (
            "extra_args" not in inputs
            or "prompt_continuous_tokens" not in inputs
            or "layer" not in inputs["extra_args"]
        ):
            return [""] * len(self.scales), self.scales

        layer = inputs["extra_args"]["layer"]
        feature_idx = inputs["extra_args"]["feature_idx"]
        if isinstance(feature_idx, str):
            feature_idx = int(feature_idx[1:])

        # Look up cached explanations
        cache_key = f"{layer}_{feature_idx}"
        if cache_key in self.explanations_cache:
            explanations = self.explanations_cache[cache_key]["explanations"]
            # Return explanations for all scales
            return [explanations.get(int(scale), "") for scale in self.scales]
        else:
            # Fallback: generate on-the-fly if not cached
            vector = inputs["prompt_continuous_tokens"][0].to(torch.float)
            vector = vector / vector.norm()

            try:
                explanations_for_scales = self._generate_explanations_for_all_scales(
                    vector, self.patch_layer
                )
                # Since we're passing a single vector, we know it returns a Dict[float, str]
                if isinstance(explanations_for_scales, dict):
                    predictions = [
                        explanations_for_scales.get(int(scale), "")
                        for scale in self.scales
                    ]
                else:
                    # This shouldn't happen with a single vector, but handle it gracefully
                    predictions = [""] * len(self.scales)
            except Exception as e:
                print(f"Error generating explanations for all scales: {e}")
                explanations_for_scales = {int(scale): "" for scale in self.scales}
                predictions = [""] * len(self.scales)

            # save to cache
            self.explanations_cache[cache_key] = {
                "explanations": explanations_for_scales,
                "expected_label": inputs["expected_label"].strip(),
                "layer": layer,
                "feature_idx": feature_idx,
            }
            return predictions, self.scales
