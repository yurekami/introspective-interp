"""
Registry for dataset classes to avoid circular imports.
"""
from typing import Dict, Type, Optional


class DatasetRegistry:
    """Registry for dataset classes to avoid circular imports."""

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, dataset_class: Type) -> Type:
        """Register a dataset class with a name."""
        cls._registry[name] = dataset_class
        return dataset_class

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """Get a dataset class by name."""
        return cls._registry.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, Type]:
        """Get all registered dataset classes."""
        return cls._registry
