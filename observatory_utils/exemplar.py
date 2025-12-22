from enum import Enum
from pydantic import BaseModel
from observatory_utils.general import (
    Subject,
    get_subject_config,
    ActivationRecord,
    calculate_max_activation,
    calculate_min_activation,
    NDFloatArray,
    NDIntArray,
    HFDatasetWrapper,
)
import os
from typing import (
    Optional,
    Sequence,
    Tuple,
    Dict,
    List,
    Literal,
    Set,
)
from collections import defaultdict
import numpy as np

QUANTILE_KEYS = (
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1 - 1e-4,
    1 - 1e-5,
    1 - 1e-6,
    1 - 1e-7,
    1 - 1e-8,
)


class ExemplarSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    RANDOM_TRAIN = "random_train"
    RANDOM_VALID = "random_valid"
    RANDOM_TEST = "random_test"


class ExemplarType(str, Enum):
    MAX = "max"
    MIN = "min"


class HFDatasetWrapperConfig(BaseModel):
    hf_dataset_id: str
    dataset_config_name: Optional[str] = None
    hf_split: str = "train"
    seed: int = 54
    cache_dir: Optional[str] = None


fineweb_dset_config = HFDatasetWrapperConfig(
    hf_dataset_id="HuggingFaceFW/fineweb",
    dataset_config_name="sample-10BT",
    hf_split="train",
    seed=54,
    cache_dir="/data/artifacts/bzl/datasets/",
)

lmsys_dset_config = HFDatasetWrapperConfig(
    hf_dataset_id="lmsys/lmsys-chat-1m",
    hf_split="train",
    seed=54,
    cache_dir="/data/artifacts/bzl/datasets/",
)


class ExemplarConfig(BaseModel):
    hf_model_id: str
    hf_dataset_configs: Tuple[HFDatasetWrapperConfig, ...] = (
        fineweb_dset_config,
        lmsys_dset_config,
    )
    sampling_ratios: Optional[List[float]] = None
    num_seqs: int = 1_000_000
    seq_len: int = 64
    k: int = 100
    num_top_acts_to_save: int = 10_000
    batch_size: int = 512
    rand_seqs: int = 10
    seed: int = 64
    activation_type: Literal["MLP", "RES", "SAE_RES", "custom"] = "MLP"
    model_path: str | None = None
    indices_file: str | None = None
    sae_release: str | None = None
    sae_expansion: str | None = None


class NeuronExemplars:
    """Exemplars for a neuron, for a specific split (one of train, valid, test)"""

    def __init__(
        self,
        activation_records: dict[
            ExemplarSplit, dict[ExemplarType, List[ActivationRecord]]
        ],
        activation_percentiles: dict[float, float],
        dataset_names: dict[ExemplarSplit, dict[ExemplarType, List[str]]],
    ):
        extrema: Dict[ExemplarType, float] = {}
        for extype in ExemplarType:
            extrema[extype] = (
                -float("inf") if extype == ExemplarType.MAX else float("inf")
            )
            for split in activation_records:
                act_recs = activation_records[split][extype]
                if extype == ExemplarType.MAX:
                    extremum = calculate_max_activation(act_recs)
                    extrema[extype] = max(extremum, extrema[extype])
                else:  # extype == ExemplarType.MIN:
                    extremum = calculate_min_activation(act_recs)
                    extrema[extype] = min(extremum, extrema[extype])

        self.activation_records = activation_records
        self.activation_percentiles = activation_percentiles
        self.dataset_names = dataset_names
        self.extrema = extrema

    def get_normalized_act_records(
        self, exemplar_split: ExemplarSplit, mask_opposite_sign: bool = False
    ) -> Dict[ExemplarType, List[ActivationRecord]]:
        normalized_act_records: Dict[ExemplarType, List[ActivationRecord]] = {}
        for extype in self.activation_records[exemplar_split]:
            extremum = self.extrema[extype]
            normalized_act_records[extype] = []
            for act_rec in self.activation_records[exemplar_split][extype]:
                norm_acts: List[float] = []
                for act in act_rec.activations:
                    if extype == ExemplarType.MAX and extremum <= 0:
                        norm_acts.append(0)
                    elif extype == ExemplarType.MIN and extremum >= 0:
                        norm_acts.append(0)
                    else:
                        norm_act = act / extremum
                        norm_acts.append(
                            max(0, norm_act) if mask_opposite_sign else norm_act
                        )
                normalized_act_records[extype].append(
                    ActivationRecord(
                        tokens=act_rec.tokens,
                        token_ids=act_rec.token_ids,
                        activations=norm_acts,
                    )
                )
        return normalized_act_records


def strip_padding(token_ids: Sequence[int], pad_token_id: int) -> Sequence[int]:
    first_non_pad_idx = 0
    while first_non_pad_idx < len(token_ids):
        if token_ids[first_non_pad_idx] != pad_token_id:
            break
        first_non_pad_idx += 1
    return token_ids[first_non_pad_idx:]


def approximate_quantile(
    q: float,
    N: int,
    k: int,
    bottom_k_values: NDFloatArray,
    top_k_values: NDFloatArray,
) -> NDFloatArray:
    """
    Approximate the q-quantile for each batch, given the bottom k and top k values.

    Parameters:
    - q: The desired quantile (cumulative probability).
    - N: The total number of data points.
    - k: The number of known bottom and top values.
    - bottom_k_values: Array of shape (batch_size, k) containing bottom k values.
    - top_k_values: Array of shape (batch_size, k) containing top k values.

    Returns:
    - approx_values: Array of shape (batch_size,) with the approximated quantile values.
    """
    batch_size = bottom_k_values.shape[0]
    approx_values = np.empty(batch_size, dtype=np.float64)

    # Known cumulative probabilities for bottom_k_values and top_k_values
    bottom_p = np.arange(1, k + 1) / N  # Shape: (k,)
    top_p = (N - k + np.arange(1, k + 1)) / N  # Shape: (k,)

    # Determine if q is in lower or upper quantile range
    if (1 / N) <= q <= (k / N):
        # Lower quantiles
        p = bottom_p
        values = bottom_k_values
    elif ((N - k + 1) / N) <= q <= 1:
        # Upper quantiles
        p = top_p
        values = top_k_values
    else:
        raise ValueError(
            f"q={q} is out of the known quantile ranges based on k={k} and N={N}."
        )

    # Find the indices for interpolation
    indices = np.searchsorted(p, q, side="right") - 1
    indices = np.clip(indices, 0, k - 2)  # Ensure indices are within valid range

    # Get the cumulative probabilities and values for interpolation
    p_lower = p[indices]  # Shape: (batch_size,)
    p_upper = p[indices + 1]  # Shape: (batch_size,)
    v_lower = values[:, indices]  # Shape: (batch_size,)
    v_upper = values[:, indices + 1]  # Shape: (batch_size,)

    # Compute the fraction for interpolation
    fraction = (v_upper - v_lower) / (p_upper - p_lower)

    # Handle cases where p_upper == p_lower to avoid division by zero
    zero_denominator = p_upper == p_lower
    approx_values[zero_denominator] = v_lower[zero_denominator]
    approx_values[~zero_denominator] = v_lower[~zero_denominator] + fraction * (
        q - p_lower[~zero_denominator]
    )

    return approx_values


class ExemplarsWrapper:
    def __init__(
        self,
        data_dir: str,
        config: ExemplarConfig,
        subject: Subject,
    ):
        # Check whether hf_model_id matches subject.
        if config.hf_model_id == subject.lm_config.hf_model_id:
            print(f"Model {config.hf_model_id} matches subject.lm_config.hf_model_id")
        else:
            print(
                f"Model {config.hf_model_id} does not match subject.lm_config.hf_model_id {subject.lm_config.hf_model_id}"
            )

        hf_datasets: Dict[str, HFDatasetWrapper] = {}
        for hf_dataset_config in config.hf_dataset_configs:
            hf_dataset = HFDatasetWrapper(config=hf_dataset_config, subject=subject)
            dataset_name = hf_dataset_config.hf_dataset_id.split("/")[-1]
            hf_datasets[dataset_name] = hf_dataset
        dataset_names: List[str] = sorted(hf_datasets.keys())

        save_path = data_dir
        os.makedirs(save_path, exist_ok=True)

        # Define cache. This is useful if we want to load exemplars for many individual neurons
        # since exemplar information is saved by layer.
        self.layers_cache: Dict[
            int,
            Tuple[
                Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]],
                Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
                Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
                Dict[float, NDFloatArray],
            ],
        ] = {}

        self.config: ExemplarConfig = config
        self.subject: Subject = subject
        self.hf_datasets: List[HFDatasetWrapper] = [
            hf_datasets[name] for name in dataset_names
        ]
        self.dataset_names: List[str] = dataset_names
        self.save_path: str = save_path

    def get_layer_dir(
        self,
        layer: int,
        split: ExemplarSplit,
    ) -> str:
        save_path = self.save_path
        return os.path.join(
            save_path,
            split.value,
            f"layer{layer}",
        )

    def get_layer_act_percs(
        self,
        layer: int,
    ) -> Dict[float, NDFloatArray]:
        """
        Computes the activation quantile information for a layer using top-activating sequences
        from the train, valid, and test splits.
        """
        act_percs_path = os.path.join(self.save_path, f"layer{layer}_act_percs.npy")
        if os.path.exists(act_percs_path):
            return np.load(act_percs_path, allow_pickle=True).item()

        layer_acts: Dict[ExemplarType, NDFloatArray] = {}
        splits: Set[ExemplarSplit] = set()
        for extype in ExemplarType:
            acts: Dict[ExemplarSplit, NDFloatArray] = {}
            for split in [ExemplarSplit.TRAIN, ExemplarSplit.VALID, ExemplarSplit.TEST]:
                layer_dir = self.get_layer_dir(layer, split)
                try:
                    acts[split] = np.load(
                        os.path.join(layer_dir, f"{extype.value}_acts.npy"),
                        mmap_mode="r",
                    )
                except:
                    continue
                splits.add(split)
            concatenated_acts = np.concatenate(list(acts.values()), axis=1)
            sorted_acts = np.sort(concatenated_acts, axis=1)
            layer_acts[extype] = sorted_acts

        num_tokens_seen = 0
        for split in splits:
            layer_dir = self.get_layer_dir(layer, split)
            with open(os.path.join(layer_dir, "num_tokens_seen.txt"), "r") as f:
                num_tokens_seen += int(f.read())

        act_percs: Dict[float, NDFloatArray] = {}
        for q in QUANTILE_KEYS:
            try:
                act_percs[q] = approximate_quantile(
                    q=q,
                    N=num_tokens_seen,
                    k=layer_acts[ExemplarType.MAX].shape[1],
                    bottom_k_values=layer_acts[ExemplarType.MIN],
                    top_k_values=layer_acts[ExemplarType.MAX],
                )
            except:
                continue
        np.save(act_percs_path, act_percs)  # type: ignore
        return act_percs

    def get_layer_data(
        self,
        layer: int,
        # custom_vectors_save_name: str | None = None,
    ) -> Tuple[
        Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]],
        Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
        Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]],
        Dict[float, NDFloatArray],
    ]:
        """
        Loads exemplar data for a layer.
        """
        if layer in self.layers_cache:
            return self.layers_cache[layer]

        act_percs = self.get_layer_act_percs(layer)

        seq_acts: Dict[ExemplarSplit, Dict[ExemplarType, NDFloatArray]] = defaultdict(
            dict
        )
        token_ids: Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]] = defaultdict(
            dict
        )
        dataset_ids: Dict[ExemplarSplit, Dict[ExemplarType, NDIntArray]] = defaultdict(
            dict
        )
        for split in ExemplarSplit:
            layer_dir = self.get_layer_dir(layer, split)

            try:
                for extype in ExemplarType:
                    layer_seq_acts = np.load(
                        os.path.join(layer_dir, f"{extype.value}_seq_acts.npy"),
                        mmap_mode="r",
                    )
                    layer_seq_ids = np.load(
                        os.path.join(layer_dir, f"{extype.value}_seq_ids.npy"),
                        mmap_mode="r",
                    )
                    layer_dataset_ids = np.load(
                        os.path.join(layer_dir, f"{extype.value}_dataset_ids.npy")
                    )
                    seq_acts[split][extype] = layer_seq_acts  # [:num_neurons_per_layer]
                    token_ids[split][extype] = layer_seq_ids  # [:num_neurons_per_layer]
                    dataset_ids[split][
                        extype
                    ] = layer_dataset_ids  # [:num_neurons_per_layer]
            except:
                continue

        # Save data in cache.
        self.layers_cache[layer] = (seq_acts, token_ids, dataset_ids, act_percs)
        return self.layers_cache[layer]

    def get_neuron_exemplars(
        self,
        layer: int,
        neuron_idx: int,
    ) -> NeuronExemplars:
        """Returns NeuronExemplars for a neuron, given a split."""
        (
            layer_acts,
            layer_token_ids,
            layer_dataset_ids,
            layer_act_percs,
        ) = self.get_layer_data(
            layer,
        )

        pad_id = self.subject.tokenizer.pad_token_id
        assert pad_id is not None and isinstance(pad_id, int)

        act_records: Dict[ExemplarSplit, Dict[ExemplarType, List[ActivationRecord]]] = (
            {}
        )
        dset_names: Dict[ExemplarSplit, Dict[ExemplarType, List[str]]] = {}
        for split in layer_acts.keys():
            act_records[split] = {}
            dset_names[split] = {}
            for extype in ExemplarType:
                act_records[split][extype] = []
                neuron_acts = layer_acts[split][extype][neuron_idx]
                neuron_token_ids = layer_token_ids[split][extype][neuron_idx]

                for acts, ids in zip(neuron_acts, neuron_token_ids):
                    ids = strip_padding(ids, pad_id)
                    acts = acts[-len(ids) :]

                    tokens: List[str] = [self.subject.decode(id) for id in ids]
                    act_records[split][extype].append(
                        ActivationRecord(
                            tokens=tokens,
                            token_ids=list(ids),
                            activations=acts.tolist(),
                        )
                    )

                dset_names[split][extype] = []
                for dataset_id in layer_dataset_ids[split][extype][neuron_idx]:
                    dataset_name = self.dataset_names[int(dataset_id)]
                    dset_names[split][extype].append(dataset_name)

        act_percs: Dict[float, float] = {
            q: perc[neuron_idx] for q, perc in layer_act_percs.items()
        }

        return NeuronExemplars(
            activation_records=act_records,
            activation_percentiles=act_percs,
            dataset_names=dset_names,
        )
