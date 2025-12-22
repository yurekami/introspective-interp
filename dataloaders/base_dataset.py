import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import nltk
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

# Download NLTK data if not already present
try:
    nltk.data.find("corpora/brown")
except LookupError:
    nltk.download("brown")
    nltk.download("punkt")


def format_messages(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]] | str,
    remove_final_assistant_eot: bool = True,
    add_final_assistant_header: bool = True,
    remove_final_user_eot: bool = False,
) -> str:
    """Format messages for the tokenizer"""
    if tokenizer.chat_template is None or isinstance(messages, str):
        return messages
    chat = tokenizer.apply_chat_template(messages, tokenize=False)
    if "<|start_header_id|>assistant<|end_header_id|>\\n\\n" in tokenizer.chat_template:
        assistant_header = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        eot_id = tokenizer.eos_token
    elif "<start_of_turn>model\n" in tokenizer.chat_template:
        assistant_header = "<start_of_turn>model\n"
        eot_id = "<end_of_turn>\n"
    elif "<|im_start|>assistant" in tokenizer.chat_template:
        assistant_header = "<|im_start|>assistant"
        eot_id = "<|im_end|>\n"
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer.chat_template}")
    if messages[-1]["role"] == "user" and add_final_assistant_header:
        # add assistant header if user
        chat += assistant_header
    if (
        messages[-1]["role"] == "assistant"
        and chat.endswith(eot_id)
        and remove_final_assistant_eot
    ):
        # remove last eot_id if assistant
        chat = chat[: -len(eot_id)]
    if (
        messages[-1]["role"] == "user"
        and chat.endswith(eot_id)
        and remove_final_user_eot
    ):
        # remove last eot_id if use
        chat = chat[: -len(eot_id)]
    return chat


@dataclass
class ExampleLabels:
    prompt: str | torch.Tensor
    prompt_without_answer: str | torch.Tensor
    expected_label: str | torch.Tensor
    evaluation_type: str
    task: str
    dataset_name: str
    extra_args: Dict[str, Any] = field(default_factory=dict)
    question_type: Optional[str] = None
    prompt_continuous_tokens: Optional[torch.LongTensor] = (
        None  # size: (num_continuous_tokens, hidden_size)
    )
    prompt_without_answer_continuous_tokens: Optional[torch.LongTensor] = (
        None  # size: (num_continuous_tokens, hidden_size)
    )
    label_continuous_tokens: Optional[torch.LongTensor] = (
        None  # size: (num_continuous_tokens, hidden_size)
    )


class BaseIterableDataset(IterableDataset):
    """
    Base class for iterable datasets with common functionality.
    """

    def __init__(
        self,
        split: str,
        predictor_model: torch.nn.Module,
        target_model: torch.nn.Module,
        predictor_tokenizer: PreTrainedTokenizer,
        target_tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        num_samples: Optional[int] = None,
        special_tokens: Optional[Dict[str, str]] = None,
        debug: bool = False,
        **kwargs,
    ):
        self.data_split = split
        self.predictor_model = predictor_model
        self.target_model = target_model
        self.predictor_tokenizer = predictor_tokenizer
        self.target_tokenizer = target_tokenizer
        self.question_config = config
        self.debug = debug
        self.special_tokens = special_tokens
        self.example_to_dataset_name = {}

        # Handle num_samples configuration
        if num_samples is None:
            try:
                self.num_samples = config["num_samples"]
            except KeyError:
                self.num_samples = None
        else:
            self.num_samples = num_samples

        # Set additional attributes from kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Load question types configuration
        self.load_question_config()

        # Load datasets if specified
        if "dataset" in config:
            self.load_datasets(config["dataset"])

    def load_question_config(self):
        """Load and configure question types."""
        self.question_types = []
        for qtype, config in self.question_config.get("question_types", {}).items():
            self.question_types.append(qtype)

        self.num_repeats_per_example = (
            len(self.question_config["question_types"])
            if "question_types" in self.question_config
            else 1
        )

    def load_datasets(self, dataset_names: List[str]):
        """Load datasets. Should be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement this method")

    def format_messages(
        self, tokenizer: PreTrainedTokenizer, messages: List[Dict[str, str]]
    ) -> str:
        """Format messages for the tokenizer."""
        return format_messages(tokenizer, messages)

    def fill_in_prompt_args(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_template: List[Dict[str, str]] | str,
        prompt_args: Dict[str, str],
        expected_label: Optional[str] = None,
        full_assistant_response: bool = False,
    ) -> tuple[Optional[List[Dict[str, str]]], List[Dict[str, str]], Optional[str]]:
        """Fill in template with args"""
        if isinstance(prompt_template, str):
            prompt_without_answer = prompt_template.format(**prompt_args)
            if expected_label is None:
                return None, prompt_without_answer, None
            expected_label = expected_label
            prompt_with_answer = prompt_without_answer + expected_label
            return prompt_with_answer, prompt_without_answer, expected_label

        prompt_with_answer = copy.deepcopy(prompt_template)
        last_user_turn_idx = -1

        for i, message in enumerate(prompt_with_answer):
            if message["role"] == "user":
                last_user_turn_idx = i
            format_args = {}
            for arg in prompt_args:
                if f"{{{arg}}}" in message.get("content", ""):
                    format_args[arg] = prompt_args[arg]

            if format_args:
                prompt_with_answer[i]["content"] = prompt_with_answer[i][
                    "content"
                ].format(**format_args)

        prompt_without_answer = copy.deepcopy(prompt_with_answer)

        if expected_label is None:
            return None, prompt_without_answer, None

        if last_user_turn_idx == len(prompt_template) - 1:
            prompt_with_answer.append({"role": "assistant", "content": expected_label})
            if not full_assistant_response:
                prompt_without_answer.append({"role": "assistant", "content": ""})
            else:
                expected_label = tokenizer.apply_chat_template(
                    [{"role": "assistant", "content": expected_label}], tokenize=False
                )
                expected_label_start_idx = expected_label.index(
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                expected_label = expected_label[expected_label_start_idx:]
        else:
            if "content" in prompt_with_answer[-1]:
                prompt_with_answer[-1]["content"] += expected_label
            else:
                prompt_with_answer[-1]["content"] = expected_label

        try:
            assert prompt_with_answer[:-1] == prompt_without_answer[:-1]
        except AssertionError:
            print(prompt_with_answer)
            print(prompt_without_answer)
            print(expected_label)

        return prompt_with_answer, prompt_without_answer, expected_label

    def __len__(self) -> int:
        """Return length if available."""
        if hasattr(self, "examples_list") and hasattr(self, "num_repeats_per_example"):
            return len(self.examples_list) * self.num_repeats_per_example
        return self.num_samples or 0

    def __iter__(self):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclass must implement __iter__")


class BaseDataCollator:
    """
    Data collator that computes token representations and similarity on the fly.
    """

    def __init__(
        self,
        predictor_tokenizer: PreTrainedTokenizer,
        target_tokenizer: PreTrainedTokenizer,
        question_config: Optional[Dict[str, Any]] = None,
    ):
        self.predictor_tokenizer = predictor_tokenizer
        self.target_tokenizer = target_tokenizer
        self.question_config = question_config or {}

    def pad_left_to_max_len(
        self,
        arr: torch.Tensor,
        pad_token_id: int,
        max_len: int,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Pads to the left of the tensor.
        """
        if arr.shape[dim] < max_len:
            pad_len = max_len - arr.shape[dim]
            pad_shape = list(arr.shape)
            pad_shape[dim] = pad_len
            pad_tensor = torch.full(
                pad_shape,
                pad_token_id,
                dtype=arr.dtype,
            )
            return torch.cat([pad_tensor, arr], dim=dim)
        else:
            # Create slice objects for all dimensions
            slices = [slice(None)] * arr.ndim
            # Set the slice for the specified dimension to cut to max_len
            slices[dim] = slice(max_len)
            return arr[tuple(slices)]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        all_example_keys = set()
        for example in features:
            for key in example.keys():
                batch[key] = []
                all_example_keys.add(key)

        # Format prompts for the model
        formatted_str_inputs = []
        formatted_str_inputs_without_answer = []
        formatted_str_labels = []
        formatted_str_idxs = []
        arr_idxs = []
        for i in range(len(features)):
            prompt = features[i]["prompt"]
            prompt_without_answer = features[i]["prompt_without_answer"]

            # Format for the model
            formatted_str_idxs.append(i)
            formatted_input_with_answer = format_messages(
                self.predictor_tokenizer,
                prompt,
                remove_final_assistant_eot=False,
                add_final_assistant_header=False,
            )
            formatted_input_without_answer = format_messages(
                self.predictor_tokenizer,
                prompt_without_answer,
                remove_final_assistant_eot=True,
                add_final_assistant_header=False,
            )
            formatted_str_inputs.append(formatted_input_with_answer)
            formatted_str_inputs_without_answer.append(formatted_input_without_answer)
            formatted_str_labels.append(features[i]["expected_label"])

        batch_idxs_order = formatted_str_idxs + arr_idxs

        max_len = 0
        if formatted_str_inputs:
            # Tokenize inputs
            tokenized_str_inputs = self.predictor_tokenizer(
                formatted_str_inputs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                padding_side="left",
            )
            tokenized_str_inputs_without_answer = self.predictor_tokenizer(
                formatted_str_inputs_without_answer,
                padding=True,
                truncation=True,
                return_tensors="pt",
                padding_side="left",
            )

            # Get input_ids from tokenized inputs
            input_ids = tokenized_str_inputs["input_ids"]
            input_ids_without_answer = tokenized_str_inputs_without_answer["input_ids"]
            max_len = max(max_len, input_ids.shape[1])

            # pad to max len
            input_ids = self.pad_left_to_max_len(
                input_ids, self.predictor_tokenizer.pad_token_id, max_len
            )
            input_ids_without_answer = self.pad_left_to_max_len(
                input_ids_without_answer, self.predictor_tokenizer.pad_token_id, max_len
            )
            attention_mask = self.pad_left_to_max_len(
                tokenized_str_inputs["attention_mask"],
                0,
                max_len,
            )
            attention_mask_without_answer = self.pad_left_to_max_len(
                tokenized_str_inputs_without_answer["attention_mask"],
                0,
                max_len,
            )

            # Create labels for the expected next tokens
            labels = input_ids.clone()

            # Mask out all tokens except the ones we want to predict
            for i in range(len(labels)):
                # Predict only the answer part
                expected_label = formatted_str_labels[i]
                for t in range(1, labels.shape[1]):
                    if self.predictor_tokenizer.decode(
                        labels[i, -t:], skip_special_tokens=True
                    ).strip().replace(" ", "") == expected_label.strip().replace(
                        " ", ""
                    ):
                        labels[i, :-t] = -100
                        break
                if not (
                    self.predictor_tokenizer.decode(labels[i][labels[i] >= 0])
                    .strip()
                    .replace(" ", "")
                    .startswith(expected_label.strip().replace(" ", ""))
                ):
                    for t in range(labels.shape[1]):
                        if "".join(
                            self.predictor_tokenizer.decode(
                                labels[i, :t], skip_special_tokens=True
                            ).split()
                        ) == "".join(
                            self.predictor_tokenizer.decode(
                                input_ids_without_answer[i], skip_special_tokens=True
                            ).split()
                        ):
                            break
                    t = len(self.predictor_tokenizer.tokenize(expected_label))
                    labels[i, :t] = -100
                if (
                    not self.predictor_tokenizer.decode(labels[i][labels[i] >= 0])
                    .strip()
                    .replace(" ", "")
                    .startswith(expected_label.strip().replace(" ", ""))
                ):
                    print(self.predictor_tokenizer.decode(labels[i][labels[i] >= 0]))
                    print("Label mismatch: ", expected_label)
        else:
            input_ids = torch.tensor([], dtype=torch.long)
            input_ids_without_answer = torch.tensor([], dtype=torch.long)
            attention_mask = torch.tensor([], dtype=torch.long)
            attention_mask_without_answer = torch.tensor([], dtype=torch.long)
            labels = torch.tensor([], dtype=torch.long)
            input_ids_without_answer = torch.tensor([], dtype=torch.long)
            attention_mask_without_answer = torch.tensor([], dtype=torch.long)

        batch["input_ids"] = input_ids
        batch["attention_mask"] = attention_mask
        batch["labels"] = labels
        batch["input_ids_without_answer"] = input_ids_without_answer
        batch["attention_mask_without_answer"] = attention_mask_without_answer
        if "prompt_continuous_tokens" in features[0]:
            batch["inputs_continuous_tokens"] = []
            batch["labels_continuous_tokens"] = []
            batch["inputs_without_answer_continuous_tokens"] = []
            for i in batch_idxs_order:
                example = features[i]
                batch["inputs_continuous_tokens"].append(
                    example["prompt_continuous_tokens"]
                )
                batch["labels_continuous_tokens"].append(
                    example["label_continuous_tokens"]
                )
                batch["inputs_without_answer_continuous_tokens"].append(
                    example["prompt_without_answer_continuous_tokens"]
                )

        # add other features
        for i in batch_idxs_order:
            example = features[i]
            for key in all_example_keys:
                if key not in example:
                    batch[key].append(None)
                else:
                    batch[key].append(example[key])

        return batch
