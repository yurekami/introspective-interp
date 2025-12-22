from .act_patch import ActPatchDataset
from .base_dataset import BaseDataCollator, BaseIterableDataset
from .causal import CausalBaseDataset, CausalDataCollator
from .features_explain import FeaturesExplainDataset
from .registry import DatasetRegistry
from .hint_attribution import HintAttributionDataset

__all__ = [
    "BaseIterableDataset",
    "BaseDataCollator",
    "DatasetRegistry",
    "FeaturesExplainDataset",
    "CausalBaseDataset",
    "CausalDataCollator",
    "ActPatchDataset",
    "HintAttributionDataset",
]


# Update registry with actual classes
DatasetRegistry.register("features_explain", FeaturesExplainDataset)
DatasetRegistry.register("act_patch", ActPatchDataset)
DatasetRegistry.register("hint_attribution", HintAttributionDataset)

# Create datasets mapping from registry
TASK_DATASET_MAPPING = DatasetRegistry.get_all()
TASK_DATALOADER_MAPPING = {"causal": CausalDataCollator}
