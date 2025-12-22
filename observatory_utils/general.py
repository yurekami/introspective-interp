from __future__ import annotations

import os
import torch
from queue import Queue
from typing import (
    Any,
    Callable,
    Literal,
    cast,
    List,
    Tuple,
    Optional,
    NotRequired,
    TypedDict,
    Iterator,
    Dict,
)
from enum import Enum
from torch.utils.data import Dataset
from datasets import load_dataset
from nnsight import LanguageModel
from nnsight.util import fetch_attr
import torch.nn.functional as F
from transformers.generation.streamers import BaseStreamer
from transformers.models.llama import LlamaConfig, LlamaForCausalLM  # type: ignore
from transformers.models.llama.modeling_llama import LlamaDecoderLayer  # type: ignore
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import logging


logger = logging.getLogger(__name__)


NDFloatArray = NDArray[np.floating[Any]]
NDIntArray = NDArray[np.integer[Any]]
NDBoolArray = NDArray[np.bool_]


class ToolCallView(TypedDict):
    title: str
    format: str
    content: str


class ToolCall(TypedDict):
    id: str
    function: str
    arguments: dict[str, str]
    view: NotRequired[ToolCallView | None]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


def is_project_root(directory: Path):
    return (directory / ".root").exists()


def find_dotenv():
    """
    Find the .env file in the project directory. Stops ascending at the project root.
    Raises an error with the list of paths explored if no .env file is found.
    """
    current_dir = Path(__file__).parent.resolve()
    paths_explored: list[str] = []

    while True:
        paths_explored.append(str(current_dir))
        env_file = current_dir / ".env"
        if env_file.is_file():
            return str(env_file)
        if is_project_root(current_dir):
            break
        if current_dir == current_dir.parent:
            break
        current_dir = current_dir.parent

    raise FileNotFoundError(
        f"No .env file found. Paths explored: {', '.join(paths_explored)}"
    )


class EnvironmentVariables(BaseModel):
    OPENAI_API_KEY: str | None
    ANTHROPIC_API_KEY: str | None
    HF_TOKEN: str | None

    TOGETHER_API_KEY: str | None
    PERPLEXITY_API_KEY: str | None

    MORPH_API_KEY: str | None

    # Monitor database
    PG_USER: str | None
    PG_PASSWORD: str | None
    PG_HOST: str | None
    PG_PORT: str | None
    PG_DATABASE: str | None

    # Docent database
    DOCENT_PG_USER: str | None
    DOCENT_PG_PASSWORD: str | None
    DOCENT_PG_HOST: str | None
    DOCENT_PG_PORT: str | None
    DOCENT_PG_DATABASE: str | None

    LLM_CACHE_PATH: str | None
    INSPECT_EXPERIMENT_CACHE_PATH: str | None

    EVAL_LOGS_DIR: str | None
    ENV_TYPE: (
        Literal["dev", "prod", "staging"] | str | None
    )  # Extra str is for custom deployments for other people

    @classmethod
    def load_from_env(cls):
        env_file = find_dotenv()
        load_dotenv(env_file)

        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        hf_token = os.getenv("HF_TOKEN")

        together_api_key = os.getenv("TOGETHER_API_KEY")
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

        morph_api_key = os.getenv("MORPH_API_KEY")

        pg_user = os.getenv("PG_USER")
        pg_password = os.getenv("PG_PASSWORD")
        pg_host = os.getenv("PG_HOST")
        pg_port = os.getenv("PG_PORT")
        pg_database = os.getenv("PG_DATABASE")

        docent_pg_user = os.getenv("DOCENT_PG_USER")
        docent_pg_password = os.getenv("DOCENT_PG_PASSWORD")
        docent_pg_host = os.getenv("DOCENT_PG_HOST")
        docent_pg_port = os.getenv("DOCENT_PG_PORT")
        docent_pg_database = os.getenv("DOCENT_PG_DATABASE")

        llm_cache_path = os.getenv("LLM_CACHE_PATH")
        inspect_experiment_cache_path = os.getenv("INSPECT_EXPERIMENT_CACHE_PATH")

        eval_logs_dir = os.getenv("EVAL_LOGS_DIR")
        env_type = os.getenv("ENV_TYPE")

        logger.info(f"ENV_TYPE: {env_type}")

        return cls(
            OPENAI_API_KEY=openai_api_key,
            ANTHROPIC_API_KEY=anthropic_api_key,
            HF_TOKEN=hf_token,
            TOGETHER_API_KEY=together_api_key,
            PERPLEXITY_API_KEY=perplexity_api_key,
            MORPH_API_KEY=morph_api_key,
            PG_USER=pg_user,
            PG_PASSWORD=pg_password,
            PG_HOST=pg_host,
            PG_PORT=pg_port,
            PG_DATABASE=pg_database,
            DOCENT_PG_USER=docent_pg_user,
            DOCENT_PG_PASSWORD=docent_pg_password,
            DOCENT_PG_HOST=docent_pg_host,
            DOCENT_PG_PORT=docent_pg_port,
            DOCENT_PG_DATABASE=docent_pg_database,
            LLM_CACHE_PATH=llm_cache_path,
            INSPECT_EXPERIMENT_CACHE_PATH=inspect_experiment_cache_path,
            EVAL_LOGS_DIR=eval_logs_dir,
            ENV_TYPE=env_type,
        )


ENV = EnvironmentVariables.load_from_env()


def _ct(x: Any) -> torch.Tensor:
    return cast(torch.Tensor, x)


class TokenIdStreamer(BaseStreamer):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        verbose: bool = False,
        timeout: float | None = None,
    ):
        self.tokenizer = tokenizer
        self.token_id_queue: Queue[int | None] = Queue()

        self.verbose, self.timeout = verbose, timeout
        self.stop_signal = None

    def put(self, value: torch.Tensor):
        """
        Receives token IDs and puts them in the queue.
        """

        assert isinstance(
            value, torch.Tensor
        ), "TokenIdStreamer only expects streaming torch.Tensors"

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TokenIdStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]  # Discard future batches

        tokens: list[int] = value.tolist()  # type: ignore
        for token_id in tokens:
            self.token_id_queue.put(token_id, timeout=self.timeout)
            if self.verbose:
                print(self.tokenizer.decode(token_id), end="", flush=True)  # type: ignore

    def end(self):
        """Signals the end of token ID generation by putting a stop signal in the queue."""
        self.token_id_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_id_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class LMConfig(BaseModel):
    """
    Configuration class for Language Models.

    This class defines the structure and properties of a language model,
    including its architecture, module paths, and dimensional information.

    The layernorm_fn should have the following signature:
        (tensor to normalize, tensor to compute statistics with, norm weight, norm_eps) -> normalized tensor
    """

    hf_model_id: str
    is_chat_model: bool

    unembed_module_str: str
    w_in_module_template: str
    w_gate_module_template: str
    w_out_module_template: str
    layer_module_template: str
    mlp_module_template: str
    attn_module_template: str
    v_proj_module_template: str
    o_proj_module_template: str
    input_norm_module_template: str
    unembed_norm_module_str: str

    # Metadata about model dims
    I_name: str  # Intermediate size (# of neurons)
    D_name: str  # Residual stream size
    V_name: str  # Vocab size
    L_name: str  # Num layers
    Q_name: str  # Num query attention heads
    K_name: str  # Num key/value attention heads (might not equal Q in grouped K/V attention)

    # Layernorm impl with signature
    #   (tensor to normalize, tensor to compute statistics with, norm weight, norm_eps)
    #       -> normalized tensor
    layernorm_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor
    ]


###########
# Llama 3 #
###########


def _llama3_layernorm_fn(
    x_X1X2D: torch.Tensor,
    estimator_X1D: torch.Tensor,
    norm_w_D: torch.Tensor,
    eps: float,
):
    """
    Normalizes x along the X1/X2 dimensions by computing RMS statistics across the D dimension of estimator_X1D,
    then applying the same normalization to constant to X2D for all X1.
    """

    # Put everything on the device that input is on
    device = x_X1X2D.device

    # Compute
    return (
        norm_w_D[None, None, :].to(device)
        * x_X1X2D
        * torch.rsqrt(estimator_X1D.to(device).pow(2).mean(dim=1) + eps)[:, None, None]
    )


class Llama3Config(LMConfig):
    unembed_module_str: str = "lm_head"
    unembed_norm_module_str: str = "model.norm"
    w_in_module_template: str = "model.layers.{layer}.mlp.up_proj"
    w_gate_module_template: str = "model.layers.{layer}.mlp.gate_proj"
    w_out_module_template: str = "model.layers.{layer}.mlp.down_proj"
    layer_module_template: str = "model.layers.{layer}"
    mlp_module_template: str = "model.layers.{layer}.mlp"
    attn_module_template: str = "model.layers.{layer}.self_attn"
    v_proj_module_template: str = "model.layers.{layer}.self_attn.v_proj"
    o_proj_module_template: str = "model.layers.{layer}.self_attn.o_proj"
    input_norm_module_template: str = "model.layers.{layer}.input_layernorm"

    I_name: str = "intermediate_size"
    D_name: str = "hidden_size"
    V_name: str = "vocab_size"
    L_name: str = "num_hidden_layers"
    Q_name: str = "num_attention_heads"
    K_name: str = "num_key_value_heads"

    layernorm_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor
    ] = _llama3_layernorm_fn


class Qwen3Config(LMConfig):
    """
        Qwen3ForCausalLM(
      (model): Qwen3Model(
        (embed_tokens): Embedding(151936, 4096)
        (layers): ModuleList(
          (0-35): 36 x Qwen3DecoderLayer(
            (self_attn): Qwen3Attention(
              (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
              (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
              (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
              (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
              (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
              (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
            )
            (mlp): Qwen3MLP(
              (gate_proj): Linear(in_features=4096, out_features=12288, bias=False)
              (up_proj): Linear(in_features=4096, out_features=12288, bias=False)
              (down_proj): Linear(in_features=12288, out_features=4096, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen3RMSNorm((4096,), eps=1e-06)
            (post_attention_layernorm): Qwen3RMSNorm((4096,), eps=1e-06)
          )
        )
        (norm): Qwen3RMSNorm((4096,), eps=1e-06)
        (rotary_emb): Qwen3RotaryEmbedding()
      )
      (lm_head): Linear(in_features=4096, out_features=151936, bias=False)
    )
    """

    unembed_module_str: str = "lm_head"
    unembed_norm_module_str: str = "model.norm"
    w_in_module_template: str = "model.layers.{layer}.mlp.up_proj"
    w_gate_module_template: str = "model.layers.{layer}.mlp.gate_proj"
    w_out_module_template: str = "model.layers.{layer}.mlp.down_proj"
    layer_module_template: str = "model.layers.{layer}"
    mlp_module_template: str = "model.layers.{layer}.mlp"
    attn_module_template: str = "model.layers.{layer}.self_attn"
    v_proj_module_template: str = "model.layers.{layer}.self_attn.v_proj"
    o_proj_module_template: str = "model.layers.{layer}.self_attn.o_proj"
    input_norm_module_template: str = "model.layers.{layer}.input_layernorm"

    I_name: str = "intermediate_size"
    D_name: str = "hidden_size"
    V_name: str = "vocab_size"
    L_name: str = "num_hidden_layers"
    Q_name: str = "num_attention_heads"
    K_name: str = "num_key_value_heads"

    layernorm_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor
    ] = None


llama3_8B_config = Llama3Config(
    hf_model_id="meta-llama/Meta-Llama-3-8B",
    is_chat_model=False,
)

llama31_8B_config = Llama3Config(
    hf_model_id="meta-llama/Llama-3.1-8B",
    is_chat_model=False,
)

llama31_8B_instruct_config = Llama3Config(
    hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
    is_chat_model=True,
)

llama31_70B_instruct_config = Llama3Config(
    hf_model_id="meta-llama/Llama-3.1-70B-Instruct",
    is_chat_model=True,
)

llama31_70B_config = Llama3Config(
    hf_model_id="meta-llama/Llama-3.1-70B",
    is_chat_model=False,
)

qwen3_8B_config = Qwen3Config(
    hf_model_id="Qwen/Qwen3-8B",
    is_chat_model=True,
)


def get_subject_config(hf_model_id: str):
    if (
        hf_model_id == "meta-llama/Llama-3-8B"
        or hf_model_id == "meta-llama/Meta-Llama-3-8B"
    ):
        return llama3_8B_config
    elif hf_model_id == "meta-llama/Llama-3.1-8B":
        return llama31_8B_config
    elif hf_model_id == "meta-llama/Llama-3.1-8B-Instruct":
        return llama31_8B_instruct_config
    elif hf_model_id == "meta-llama/Llama-3.1-70B-Instruct":
        return llama31_70B_instruct_config
    elif hf_model_id == "meta-llama/Llama-3.1-70B":
        return llama31_70B_config
    elif hf_model_id == "Qwen/Qwen3-8B":
        return qwen3_8B_config
    else:
        raise ValueError(f"Unsupported hf_model_id={hf_model_id}")


class Subject:
    """
    This class encapsulates a language model along with its configuration,
    tokenizer, and various components. It provides easy access to model
    metadata, layers, and specific modules like embeddings, attention,
    and MLPs.
    """

    def __init__(
        self,
        config: LMConfig,
        output_attentions: bool = False,
        cast_to_hf_config_dtype: bool = True,
        nnsight_lm_kwargs: dict[str, Any] = {},
        device: str = "auto",
        cache_dir: str | None = None,
        tokenizer: AutoTokenizer | None = None,
        model: LanguageModel | None = None,
    ):
        hf_config = AutoConfig.from_pretrained(  # type: ignore
            config.hf_model_id, output_attentions=output_attentions, cache_dir=cache_dir
        )

        # Load model + tokenizer
        kwargs = {
            "dispatch": False,
            "device_map": device,
            "token": ENV.HF_TOKEN,
            "cache_dir": cache_dir,
        }
        kwargs.update({"torch_dtype": hf_config.torch_dtype} if cast_to_hf_config_dtype else {})  # type: ignore
        kwargs.update(nnsight_lm_kwargs)
        kwargs.update({"tokenizer": tokenizer} if tokenizer is not None else {})
        if model is not None:
            self.model = model
        else:
            self.model = LanguageModel(config.hf_model_id, **kwargs)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self.model.tokenizer
        self.hf_config, self.lm_config = hf_config, config  # type: ignore
        self.is_chat_model = config.is_chat_model
        self.model_name = config.hf_model_id.split("/")[-1].lower().replace("-", "_")

        # Padding always on the left
        self.tokenizer.padding_side = "left"

        # Metadata about the model
        self.I: int = int(hf_config.__dict__[config.I_name])  # type: ignore
        self.D: int = int(hf_config.__dict__[config.D_name])  # type: ignore
        self.V: int = int(hf_config.__dict__[config.V_name])  # type: ignore
        self.L: int = int(hf_config.__dict__[config.L_name])  # type: ignore
        self.Q: int = int(hf_config.__dict__[config.Q_name])  # type: ignore
        self.K: int = int(hf_config.__dict__[config.K_name])  # type: ignore

        # Model components
        self.unembed = fetch_attr(self.model, config.unembed_module_str)
        self.unembed_norm = fetch_attr(self.model, config.unembed_norm_module_str)
        self.w_ins = {
            layer: fetch_attr(
                self.model, config.w_in_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }
        self.w_gates = {
            layer: fetch_attr(
                self.model, config.w_gate_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }
        self.w_outs = {
            layer: fetch_attr(
                self.model, config.w_out_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }
        self.layers = {
            layer: fetch_attr(
                self.model, config.layer_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }
        self.mlps = {
            layer: fetch_attr(
                self.model, config.mlp_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }
        self.attns = {
            layer: fetch_attr(
                self.model, config.attn_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }
        self.attn_vs = {
            layer: fetch_attr(
                self.model, config.v_proj_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }
        self.attn_os = {
            layer: fetch_attr(
                self.model, config.o_proj_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }
        self.input_norms = {
            layer: fetch_attr(
                self.model, config.input_norm_module_template.format(layer=layer)
            )
            for layer in range(self.L)
        }

        # Layernorm implementation
        self.layernorm_fn = config.layernorm_fn

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype  # type: ignore

    ################
    # Tokenization #
    ################

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text)["input_ids"]  # type: ignore

    def decode(self, token_ids: int | list[int] | torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids)  # type: ignore

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id  # type: ignore


class ActivationRecord(BaseModel):
    """A sequence of tokens and their corresponding activations for a single feature."""

    tokens: list[str]
    """Tokens in a sequence."""
    activations: list[float]
    """Raw activation values for the feature corresponding to each token in the sequence."""
    token_ids: list[int] | None = None
    """Token IDs for the tokens in the sequence."""
    conversation: list[dict[str, str]] | None = None
    """Conversation corresponding to the token sequence."""

    def all_positive(self) -> bool:
        return all(act > 0 for act in self.activations)

    def any_positive(self) -> bool:
        return any(act > 0 for act in self.activations)

    def all_negative(self) -> bool:
        return all(act < 0 for act in self.activations)

    def any_negative(self) -> bool:
        return any(act < 0 for act in self.activations)


class ActivationSign(str, Enum):
    POS = "positive"
    NEG = "negative"

    def flip(self) -> "ActivationSign":
        return ActivationSign.POS if self == ActivationSign.NEG else ActivationSign.NEG

    def abbr(self) -> str:
        return "+" if self == ActivationSign.POS else "-"


def calculate_max_activation(activation_records: list[ActivationRecord]) -> float:
    return np.nanmax([np.nanmax(rec.activations) for rec in activation_records])


def calculate_min_activation(activation_records: list[ActivationRecord]) -> float:
    return np.nanmin([np.nanmin(rec.activations) for rec in activation_records])


class HFDatasetWrapperConfig(BaseModel):
    hf_dataset_id: str
    dataset_config_name: Optional[str] = None
    hf_split: str = "train"
    seed: int = 54
    # dataset_local_path: Optional[str] = None
    cache_dir: Optional[str] = None


class HFSplitDatasetWrapper(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any], is_chat_format: bool, subject: Subject):
        self.dataset = dataset
        self.is_chat_format = is_chat_format
        self.subject = subject

    def __len__(self) -> int:
        assert hasattr(self.dataset, "__len__")
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx: int) -> List[int]:
        example = self.dataset[idx]["data_column"]
        if self.is_chat_format:
            ids = self.subject.tokenizer.apply_chat_template(  # type: ignore
                example, add_generation_prompt=False, tokenize=True  # type: ignore
            )
        else:
            ids = self.subject.tokenizer.encode(example, add_special_tokens=True)  # type: ignore
        ids: List[int] = list(ids)
        return ids

    def __iter__(self) -> Iterator[List[int]]:
        for idx in range(len(self)):
            yield self[idx]


class HFDatasetWrapper:
    def __init__(
        self,
        config: HFDatasetWrapperConfig,
        subject: Subject,
        num_proc: int = 16,
    ):
        if config.hf_dataset_id in [
            "HuggingFaceFW/fineweb",
            "HuggingFaceFW/fineweb-edu",
        ]:
            assert config.hf_split == "train"
            dset_kwargs = {
                "path": config.hf_dataset_id,
                "name": config.dataset_config_name,
                "split": "train",
            }
            if config.cache_dir is not None:
                dset_kwargs["cache_dir"] = config.cache_dir
            column_name = "text"
            is_chat_format = False
        elif config.hf_dataset_id == "lmsys/lmsys-chat-1m":
            assert config.hf_split == "train"
            dset_kwargs = {
                "path": config.hf_dataset_id,
                "split": "train",
            }
            if config.cache_dir is not None:
                dset_kwargs["cache_dir"] = config.cache_dir
            column_name = "conversation"
            is_chat_format = True
        elif config.hf_dataset_id == "HuggingFaceH4/ultrachat_200k":
            assert config.hf_split == "train_sft"
            dset_kwargs = {
                "path": config.hf_dataset_id,
                "split": "train_sft",
            }
            if config.cache_dir is not None:
                dset_kwargs["cache_dir"] = config.cache_dir
            column_name = "messages"
            is_chat_format = True
        else:
            raise ValueError(f'Unrecognized dataset name "{config.hf_dataset_id}"!')

        # If the dataset is chat data, but the subject is not a chat model, throw error.
        if is_chat_format and not subject.is_chat_model:
            raise ValueError("Dataset is chat data, but subject is not a chat model!")

        dataset = load_dataset(num_proc=num_proc, **dset_kwargs)  # type: ignore
        dataset = dataset.rename_column(column_name, "data_column")

        # Shuffle the dataset and split into train, valid, test.
        dataset = dataset.shuffle(seed=config.seed)  # type: ignore
        # This operation takes 1-2 minutes, but only needs to be done once since the results get
        # cached.
        dataset = dataset.flatten_indices(num_proc=num_proc)  # type: ignore
        split_datasets: Dict[str, Dataset[Any]] = {
            "train": dataset.select(range(len(dataset) // 3)),  # type: ignore
            "valid": dataset.select(range(len(dataset) // 3, 2 * len(dataset) // 3)),  # type: ignore
            "test": dataset.select(range(2 * len(dataset) // 3, len(dataset))),  # type: ignore
        }
        self.split_datasets = split_datasets
        self.subject = subject
        self.is_chat_format = (
            is_chat_format  # Whether the dataset is inherently in chat format.
        )

    def get_nonchat_prefix(self):
        if self.subject.lm_config.hf_model_id in [
            "meta-llama/Llama-3.1-8B",
            "meta-llama/Llama-3.1-70B",
        ]:
            # prefix_ids = (128000,)
            prefix_ids = tuple([self.subject.tokenizer.bos_token_id])
        else:
            raise ValueError(
                f"Unsupported model_name {self.subject.lm_config.hf_model_id}!"
            )
        return prefix_ids

    def get_chat_prefix(self) -> Tuple[int, ...]:
        if self.subject.lm_config.hf_model_id in [
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
        ]:
            prefix_ids = (
                128000,
                128006,
                9125,
                128007,
                271,
                38766,
                1303,
                33025,
                2696,
                25,
                6790,
                220,
                2366,
                18,
                198,
                15724,
                2696,
                25,
                220,
                1627,
                10263,
                220,
                2366,
                19,
                271,
                128009,
                128006,
                882,
                128007,
                271,
            )
        elif self.subject.lm_config.hf_model_id in [
            "Qwen/Qwen3-8B",
        ]:
            prefix_ids = tuple(self.subject.tokenizer.encode("<|im_start|>user\n"))
        else:
            raise ValueError(
                f"Unsupported model_name {self.subject.lm_config.hf_model_id}!"
            )
        return prefix_ids

    def get_dataset_for_split(
        self, split: Literal["train", "valid", "test"]
    ) -> HFSplitDatasetWrapper:
        return HFSplitDatasetWrapper(
            self.split_datasets[split], self.is_chat_format, self.subject
        )


SPECIAL_TOKENS = [
    ("<|begin_of_text|>", "<||begin_of_text||>"),
    ("<|start_header_id|>", "<||start_header_id||>"),
    ("<|end_header_id|>", "<||end_header_id||>"),
    ("<|eot_id|>", "<||eot_id||>"),
]


class Llama3TokenizerWrapper:
    def __init__(self, model_path: str, add_special_tokens: bool = False):
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")  # type: ignore
        tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
        if add_special_tokens:
            tokenizer.add_tokens(
                [
                    "<||begin_of_text||>",
                    "<||start_header_id||>",
                    "<||end_header_id||>",
                    "<||eot_id||>",
                ]
            )

        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def update_input(self, seq: str) -> str:
        """Convert special tokens to their updated versions."""
        if not self.add_special_tokens:
            return seq
        for tok, upd_tok in SPECIAL_TOKENS:
            seq = seq.replace(tok, upd_tok)
        return seq

    def update_output(self, seq: str) -> str:
        """Convert updated special tokens to their original versions."""
        if not self.add_special_tokens:
            return seq
        for tok, upd_tok in SPECIAL_TOKENS:
            seq = seq.replace(upd_tok, tok)
        return seq

    def apply_chat_template(
        self, messages: List[ChatMessage], *args: Any, **kwargs: Any
    ):
        upd_messages: List[ChatMessage] = []
        for msg in messages:
            seq = msg["content"]
            upd_seq = self.update_input(seq)
            upd_messages.append(ChatMessage(role=msg["role"], content=upd_seq))
        return self.tokenizer.apply_chat_template(  # type: ignore
            cast(List[Dict[str, str]], upd_messages), *args, **kwargs
        )

    def __call__(self, seq: str | List[str], *args: Any, **kwargs: Any):
        if isinstance(seq, str):
            upd_seq = self.update_input(seq)
        else:
            upd_seq = [self.update_input(s) for s in seq]
        return self.tokenizer(upd_seq, *args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any):
        return self.tokenizer.decode(*args, **kwargs)  # type: ignore


def get_tokenizer(model_path: str, add_special_tokens: bool = False):
    return Llama3TokenizerWrapper(model_path, add_special_tokens)


def is_llamadecoder_layer(module: torch.nn.Module) -> bool:
    return isinstance(module, LlamaDecoderLayer)


def param_init_fn(module: torch.nn.Module) -> None:
    assert hasattr(module, "reset_parameters") and isinstance(
        module.reset_parameters, Callable
    )
    module.reset_parameters()  # type: ignore


class Llama3Model(torch.nn.Module):
    def __init__(
        self,
        model_path: str = "/home/ubuntu/llama3.1_8b_instruct_hf/",
        add_special_tokens: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()  # type: ignore

        # use_cache must be off training
        hf_config = LlamaConfig.from_pretrained(model_path, use_cache=False)  # type: ignore

        self.model = LlamaForCausalLM.from_pretrained(model_path, config=hf_config, cache_dir=cache_dir)  # type: ignore

        original_vocab_size = 128256

        # Resize the token embeddings because we had to fix the tokenizer padding
        additional_tokens = 1
        if add_special_tokens:
            additional_tokens += 4
        self.model.resize_token_embeddings(original_vocab_size + additional_tokens)
        self.model.config.pad_token_id = original_vocab_size  # type: ignore

        self.model.fsdp_wrap_fn = is_llamadecoder_layer  # type: ignore

        self.model.activation_checkpointing_fn = is_llamadecoder_layer  # type: ignore

        self.model.fsdp_param_init_fn = param_init_fn  # type: ignore

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]

        attention_mask = (
            batch["attention_mask"].bool() if "attention_mask" in batch else None
        )
        # Account for padding by updating the position_ids based on attention_mask
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits
        # We want to make sure this is in full precision
        return logits.float()

    def loss(
        self, outputs: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        targets = batch["labels"].view(-1)
        loss_mask = batch["loss_mask"].view(-1)
        unreduced_loss = F.cross_entropy(
            outputs.float().view(-1, outputs.size(-1)),
            targets,
            reduction="none",
        )
        return (unreduced_loss * loss_mask).sum() / loss_mask.sum()
