from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm


class NearestNeighborModel:
    """
    A baseline model that finds the nearest neighbor based on continuous representations.

    This model stores the continuous representations from the training data, organized by layer,
    and during inference, finds the closest example based on cosine similarity of the continuous
    representations within the appropriate layer.
    """

    def __init__(
        self,
        dataset: Any,
        layerwise_similarities: bool = False,
        topk: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the nearest neighbor model by storing continuous representations.

        Args:
            dataset: The dataset to analyze for storing continuous representations
        """
        self.device = torch.device("cuda")  # Dummy device for compatibility
        self.train_dataset = dataset
        self.layer_wise_similarities = layerwise_similarities
        self.topk = topk
        (
            self.layer_continuous_reps,
            self.layer_labels,
            self.layer_feature_idxs,
            self.all_continuous_reps,
            self.all_labels,
            self.all_layer_feature_idxs,
        ) = self._store_continuous_reps(dataset)

    def _store_continuous_reps(self, dataset: Any) -> Tuple[
        Dict[int, List[torch.Tensor]],
        Dict[int, List[str]],
        Dict[int, List[int]],
        Dict[int, Dict[float, torch.Tensor]],
        torch.Tensor,
        List[str],
        List[Tuple[int, int]],
    ]:
        """
        Store continuous representations and their corresponding labels from the dataset,
        organized by layer.

        Args:
            dataset: The dataset to analyze

        Returns:
            Tuple of (dict mapping layer to list of continuous representations,
                     dict mapping layer to list of corresponding labels)
        """
        layer_continuous_reps = {}  # layer -> E x H
        layer_labels = {}  # layer -> [E]
        layer_feature_idxs = {}  # layer -> E
        all_continuous_reps = []  # EL x H
        all_layer_feature_idxs = []  # EL
        all_labels = []  # [EL]

        # Collect all continuous representations and labels by layer
        for item in tqdm(
            dataset,
            desc=f"Storing continuous representations with {'layer-wise' if self.layer_wise_similarities else 'all-layer'} similarities",
        ):
            if (
                "extra_args" in item
                and "prompt_continuous_tokens" in item
                and "layer" in item["extra_args"]
            ):
                layer = item["extra_args"]["layer"]
                feature_idx = item["extra_args"]["feature_idx"]
                if isinstance(feature_idx, str):
                    feature_idx = int(feature_idx[1:])
                if layer not in layer_continuous_reps:
                    layer_continuous_reps[layer] = []
                    layer_labels[layer] = []
                    layer_feature_idxs[layer] = []
                feature_rep = item["prompt_continuous_tokens"][0]
                feature_description = item[
                    "expected_label"
                ].strip()  # strip("<|end_of_text|>")
                if feature_description.endswith("<|end_of_text|>"):
                    feature_description = feature_description[: -len("<|end_of_text|>")]
                layer_continuous_reps[layer].append(feature_rep)
                layer_labels[layer].append(feature_description)
                all_continuous_reps.append(feature_rep)
                all_labels.append(feature_description)
                layer_feature_idxs[layer].append((layer, feature_idx))
                all_layer_feature_idxs.append((layer, feature_idx))

        for layer in layer_continuous_reps.keys():
            # E x H
            layer_continuous_reps[layer] = torch.stack(layer_continuous_reps[layer]).to(
                torch.float
            )
        all_continuous_reps = torch.stack(all_continuous_reps).to(torch.float)

        return (
            layer_continuous_reps,
            layer_labels,
            layer_feature_idxs,
            all_continuous_reps,
            all_labels,
            all_layer_feature_idxs,
        )

    def to(self, device):
        """Dummy method for compatibility"""
        self.device = device
        return self

    def eval(self):
        """Dummy method for compatibility"""
        return self

    def cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Calculate cosine similarity between two vectors."""
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

    def train(self):
        return self

    def get_prediction(self, inputs):
        """
        Find the nearest neighbor based on continuous representation similarity within the appropriate layer.

        Args:
            inputs: Dictionary containing the input data, including:
                   - continuous representation in inputs["extra_args"]["feature"]
                   - layer in inputs["extra_args"]["layer"]

        Returns:
            The label of the nearest neighbor from the same layer
        """
        if (
            "extra_args" not in inputs
            # or "feature_idx" not in inputs["extra_args"]
            or "prompt_continuous_tokens" not in inputs
            or "layer" not in inputs["extra_args"]
        ):
            return "", None

        # H
        query_rep = inputs["prompt_continuous_tokens"][0].to(torch.float)
        layer = inputs["extra_args"]["layer"]

        if self.layer_wise_similarities:
            # Check if we have any examples for this layer
            if layer not in self.layer_continuous_reps:
                print("No examples for this layer!")
                return "", None
            reps_to_use = self.layer_continuous_reps[layer]
            labels_to_use = self.layer_labels[layer]
            feature_idxs_to_use = self.layer_feature_idxs[layer]
        else:
            reps_to_use = self.all_continuous_reps
            labels_to_use = self.all_labels
            feature_idxs_to_use = self.all_layer_feature_idxs

        # Calculate similarities with all stored representations in this layer
        similarities = query_rep.matmul(reps_to_use.T)  # E

        # Get top k highest similarities
        top_k_indices = similarities.topk(self.topk).indices
        top_k_labels = [labels_to_use[idx] for idx in top_k_indices]
        top_k_features = [list(feature_idxs_to_use[idx]) for idx in top_k_indices]
        return top_k_labels, {"nearest_feature": top_k_features}
