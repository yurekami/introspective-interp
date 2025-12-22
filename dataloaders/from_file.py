import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
from dataloaders.base_dataset import BaseIterableDataset


class FromFileDataset(BaseIterableDataset):
    def __init__(self, split: str, examples: List[Dict[str, Any]], num_samples: Optional[int] = None, **kwargs):
        self.data_split = split
        self.examples = examples
        self.num_samples = num_samples
        if num_samples is not None:
            self.examples = self.examples[:num_samples]

    @classmethod
    def from_file(cls, split: str, file_path: str, task: str, num_samples: Optional[int] = None, **kwargs) -> "FromFileDataset":
        data = []
        file_folder = os.path.dirname(file_path)
        continuous_tokens_path = os.path.join(
            file_folder, f"{task}_continuous_tokens.pt"
        )
        without_answers_continuous_tokens_path = os.path.join(
            file_folder, f"{task}_without_answers_continuous_tokens.pt"
        )
        label_continuous_tokens_path = os.path.join(
            file_folder, f"{task}_label_continuous_tokens.pt"
        )
        continuous_tokens = None
        without_answers_continuous_tokens = None
        label_continuous_tokens = None
        start_time = time.time()
        print("Loading continuous tokens from: ", continuous_tokens_path)
        if os.path.exists(continuous_tokens_path) and os.path.exists(
            without_answers_continuous_tokens_path
        ):
            with open(continuous_tokens_path, "rb") as f:
                continuous_tokens = torch.load(f)
            with open(without_answers_continuous_tokens_path, "rb") as f:
                without_answers_continuous_tokens = torch.load(f)
            with open(label_continuous_tokens_path, "rb") as f:
                label_continuous_tokens = torch.load(f)
        print(
            f"Time taken to load continuous tokens: {time.time() - start_time} seconds"
        )

        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    example = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
                    continue
                if without_answers_continuous_tokens is not None:
                    example["prompt_without_answer_continuous_tokens"] = (
                        without_answers_continuous_tokens[i]
                    )
                if continuous_tokens is not None:
                    example["prompt_continuous_tokens"] = continuous_tokens[i]
                if label_continuous_tokens is not None:
                    example["label_continuous_tokens"] = label_continuous_tokens[i]
                data.append(example)
        return cls(split, data, num_samples, **kwargs)

    def __len__(self) -> int:
        return len(self.examples)

    def get_single_item(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]

    def __iter__(self):
        return iter(self.examples)

    def __next__(self):
        return next(self.examples)
