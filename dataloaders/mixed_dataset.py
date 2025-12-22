import math
import random
from typing import Any, Dict, List, Optional
from dataloaders.base_dataset import BaseIterableDataset
from dataloaders.registry import DatasetRegistry
from utils import merge_config_with_parent
from torch.nn import Module
from transformers import PreTrainedTokenizer


class MixedDataset(BaseIterableDataset):
    """
    Dataset that mixes multiple task types together.
    """

    def __init__(
        self,
        split: str,
        predictor_model: Module,
        target_model: Module,
        predictor_tokenizer: PreTrainedTokenizer,
        target_tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(
            split,
            predictor_model,
            target_model,
            predictor_tokenizer,
            target_tokenizer,
            config,
            do_load_datasets=False,
            **kwargs,
        )

        # Get task dataset mapping from registry
        self.task_dataset_mapping = DatasetRegistry.get_all()

        # Create datasets for each task type
        self.datasets = {}
        self.task_weights = {}
        total_weight = sum(
            task_config.get("weight", 1.0) for task_config in config["tasks"].values()
        )

        for task_name, task_config in config["tasks"].items():
            if not task_config.get("enabled", True):
                continue

            # Store weight for sampling
            weight = task_config.get("weight", 1.0) / total_weight
            self.task_weights[task_name] = weight

            # Create dataset for this task
            dataset_class = self.task_dataset_mapping[task_name]
            # Merge task config with parent config
            merged_task_config = merge_config_with_parent(config, task_config)
            self.datasets[task_name] = dataset_class(
                split,
                predictor_model,
                target_model,
                predictor_tokenizer,
                target_tokenizer,
                merged_task_config,
                num_samples=math.ceil(config.get("num_samples") * weight),
                **kwargs,
            )

        # Normalize weights
        for task_name in self.task_weights:
            self.task_weights[task_name] /= total_weight

        # Calculate total number of samples
        self.num_samples = config.get("num_samples", 1000)

        # Pre-determine which task to use for each index and map to sub-dataset indices
        dataset_counters = {
            task: 0 for task in self.datasets.keys()
        }  # temporary variable to keep track of index for allocation purposes
        self.task_assignments, self.idx_mapping = self.allocate_tasks(
            self.num_samples, dataset_counters
        )

    def allocate_tasks(
        self, num_samples: int, dataset_counters: Optional[Dict[str, int]] = None
    ):
        """
        Allocate tasks based on weights and create index mapping.
        Returns:
        1. A list of task names with length num_samples
        2. A mapping from dataset indices to (task, sub_idx) pairs
        """
        task_assignments = []
        idx_mapping = {}
        task_names = list(self.datasets.keys())

        # Initialize counters for each task
        if dataset_counters is None:
            dataset_counters = {task: 0 for task in task_names}

        # Calculate available samples for each task
        task_counts = {
            task: len(self.datasets[task]) - dataset_counters[task]
            for task in task_names
        }

        # Allocate based on weights
        for i in range(num_samples):
            possible_task_names = [task for task in task_names if task_counts[task] > 0]
            if len(possible_task_names) == 0:
                break
            possible_task_probs = [
                self.task_weights[task] for task in possible_task_names
            ]
            task = random.choices(
                possible_task_names, weights=possible_task_probs, k=1
            )[0]

            # Store the task assignment and mapping
            task_assignments.append(task)
            idx_mapping[i] = (task, dataset_counters[task])

            # Update counters
            dataset_counters[task] += 1
            task_counts[task] -= 1

        return task_assignments, idx_mapping

    def __len__(self) -> int:
        return len(self.task_assignments)

    def batch_processing(self, idxs: List[int]):
        # batch up subdataset items
        dataset_to_idxs = {}
        for idx in idxs:
            task, sub_idx = self.idx_mapping[idx]
            dataset = self.datasets[task]
            if task not in dataset_to_idxs:
                dataset_to_idxs[task] = []
            dataset_to_idxs[task].append(sub_idx)

        # run batch processing on sub-datasets if they exist
        for task, data_idxs in dataset_to_idxs.items():
            dataset = self.datasets[task]
            if hasattr(dataset, "batch_processing"):
                dataset.batch_processing(data_idxs)

    def get_single_item(self, idx: int) -> Dict[str, Any]:
        # Get the task and sub-dataset index for this item
        task, sub_idx = self.idx_mapping[idx]
        dataset = self.datasets[task]
        item = dataset[sub_idx // dataset.num_repeats_per_example]

        # Get the actual example using the target task's dataset
        example = item["example"]
        label = item["extra_args"].get("label", item["expected_label"])

        if dataset.question_types:
            question_type = dataset.question_types[
                sub_idx % dataset.num_repeats_per_example
            ]
        else:
            question_type = None

        # Get the formatted example
        example_labels = dataset.get_labels(example, label, question_type, idx=sub_idx)

        # Update the features
        return {
            "example": example,
            "prompt": example_labels.prompt,
            "prompt_without_answer": example_labels.prompt_without_answer,
            "expected_label": example_labels.expected_label,
            "evaluation_type": example_labels.evaluation_type,
            "task": example_labels.task,
            "dataset_name": example_labels.dataset_name,
            "question_type": example_labels.question_type,
            "extra_args": example_labels.extra_args,
            "prompt_continuous_tokens": example_labels.prompt_continuous_tokens,
            "prompt_without_answer_continuous_tokens": example_labels.prompt_without_answer_continuous_tokens,
            "label_continuous_tokens": example_labels.label_continuous_tokens,
        }
