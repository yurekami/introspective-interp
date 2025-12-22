import json
import pickle
from typing import Any, Dict, List, Tuple
import os


def load_features_dataset(
    dataset_name: str,
    question_config: Dict[str, Any],
    data_split: str,
) -> Tuple[str, List[Tuple[int, str]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load feature dataset with simplified two-stage approach:
    1. Load/create splits if they don't exist
    2. Load the specific split data
    """
    extra_args = {}
    prompt = ""
    split_explanations_file = os.path.join(
        question_config["explanation_dir"],
        question_config["split_explanations_data"].format(split=data_split),
    )

    # Generate split file paths
    if dataset_name == "sae_explanations":
        # Stage 1b: Load the specific split
        print(f"Loading {data_split} split...")
        with open(split_explanations_file, "rb") as f:
            explanations = pickle.load(f)

        # extra_args["examples"] = explanations
        examples_list, labels_list = zip(*explanations)
    else:
        print(f"Loading {data_split} split...")
        with open(split_explanations_file) as f:
            examples_list = json.load(f)
        examples_list = [tuple(example) for example in examples_list]
        # Does not have GT labels, so we need to create dummy labels
        labels_list = [
            {
                "chosen_label": "",
                "all_labels": [],
                "all_label_scores": {"": 0.0},
                "context": "",
            }
            for _ in examples_list
        ]

    print(f"Loaded {len(examples_list)} examples for {data_split}")

    return (
        prompt,
        list(examples_list),
        list(labels_list),
        extra_args,
    )
