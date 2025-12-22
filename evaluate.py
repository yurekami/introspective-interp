from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import yaml
from dataloaders import BaseDataCollator, TASK_DATALOADER_MAPPING
from evaluator import EVALUATOR_MAPPING
from model.utils import (
    MODEL_TYPE_TO_VANILLA_MODEL_MAPPING,
    load_models,
    load_target_model,
)
from torch.utils.data import DataLoader
from train import TASK_DATASET_MAPPING
from transformers import AutoTokenizer
from utils import (
    merge_config_with_parent,
)

SAMPLES_MAP: Dict[str, int] = {
    "3200": 3200,
    "16k": 16000,
    "32k": 32000,
    "320k": 320000,
    # "all": None,
}


def load_model(
    args: argparse.Namespace,
    config: Dict[str, Any],
    train_config: Dict[str, Any],
    tokenizer: Any,
    special_token_ids: Optional[Dict[str, int]],
    target_tokenizer: Any,
) -> Tuple[Any, Any]:
    if args.model_path not in [
        "self_explanations",
    ]:
        if args.model_path == "nearest_neighbor":
            model_path = config["model_path"]
        elif args.model_path is not None:
            model_path = args.model_path
        model, target_model, _ = load_models(
            predictor_model_path=model_path,
            target_model_path=args.target_model_path,
            special_tokens_ids=special_token_ids,
            cache_dir=config.get("cache_dir", None),
            train_self=False,
            use_bf16=config["train"].get("bf16", True),
            batch_size=config["test"]["batch_size"],
            ckpt_dir=os.path.dirname(args.model_path.rstrip("/")),
            use_embed_proj=config.get("use_embed_proj", False),
            embed_proj_path=config.get("embed_proj_path", None),
            predictor_model_type=config["model_path"],
        )
        if args.model_path == "nearest_neighbor":
            if args.num_samples is not None:
                train_config["split_keys"] = train_config["split_keys"].replace(
                    ".pkl", f"_{args.num_samples}.pkl"
                )
            train_dataset = TASK_DATASET_MAPPING[args.task](
                "train",
                model,
                target_model,
                tokenizer,
                target_tokenizer,
                config=train_config,
                num_samples=(
                    train_config["num_samples"]
                    if SAMPLES_MAP.get(args.num_samples) is None
                    else SAMPLES_MAP.get(args.num_samples)
                ),
                special_tokens=config.get("continuous_tokens"),
                model_name=config["model_path"],
                model_cache_dir=config.get("cache_dir", None),
                subject_embed_dim=(
                    None if target_model is None else target_model.config.hidden_size
                ),
            )
            # run nearest neighbor baseline
            model = MODEL_TYPE_TO_VANILLA_MODEL_MAPPING[args.model_path](
                train_dataset,
                layerwise_similarities=args.layerwise_similarities,
                topk=args.topk,
            )
    else:
        target_model = load_target_model(
            target_model_path=args.target_model_path,
            cache_dir=config.get("cache_dir", None),
            use_bf16=config["train"].get("bf16", True),
        )
        model = MODEL_TYPE_TO_VANILLA_MODEL_MAPPING[args.model_path](
            # dataset,
            # data_dir=config["test"]["tasks"][args.task]["explanation_dir"],
            model_name=args.target_model_path,
            scales=args.scales,
            cache_dir=config.get("cache_dir", None),
        )
    return model, target_model


def main(args: argparse.Namespace) -> None:
    # Load the config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.target_tokenizer_path:
        target_tokenizer_path = args.target_model_path
    else:
        target_tokenizer_path = args.target_tokenizer_path
    # Load the model
    target_tokenizer = AutoTokenizer.from_pretrained(
        target_tokenizer_path,
        padding_side="left",
        cache_dir=config.get("cache_dir", None),
    )
    target_tokenizer.pad_token = target_tokenizer.eos_token
    target_tokenizer.pad_token_id = target_tokenizer.eos_token_id
    model_tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        padding_side="left",
        cache_dir=config.get("cache_dir", None),
    )
    model_tokenizer.pad_token = model_tokenizer.eos_token
    model_tokenizer.pad_token_id = model_tokenizer.eos_token_id
    if "continuous_tokens" in config:
        special_token_ids = {
            cont_token: model_tokenizer.convert_tokens_to_ids(
                config["continuous_tokens"][cont_token]
            )
            for cont_token in config.get("continuous_tokens", [])
        }
    else:
        special_token_ids = None

    train_config = config["train"]
    train_config = merge_config_with_parent(
        config["train"], train_config["tasks"][args.task]
    )
    test_config = merge_config_with_parent(
        merge_config_with_parent(train_config, config["test"]),
        config["test"]["tasks"][args.task],
    )
    model, target_model = load_model(
        args,
        config,
        train_config,
        model_tokenizer,
        special_token_ids,
        target_tokenizer,
    )
    test_dataset = TASK_DATASET_MAPPING[args.task](
        "test",
        model,
        target_model,
        model_tokenizer,
        target_tokenizer,
        config=test_config,
        special_tokens=config.get("continuous_tokens"),
        debug=args.debug,
        model_name=config["model_path"],
        model_cache_dir=config.get("cache_dir", None),
    )
    data_collator = TASK_DATALOADER_MAPPING.get(args.task, BaseDataCollator)(
        predictor_tokenizer=model_tokenizer,
        target_tokenizer=target_tokenizer,
        question_config=test_config,  # ["tasks"][args.task],
    )
    dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, collate_fn=data_collator
    )

    # Load the evaluator
    evaluator = EVALUATOR_MAPPING[test_config["tasks"][args.task]["evaluation_type"]](
        test_config,  # ["tasks"][args.task],
        model,
        model_tokenizer,
        dataloader,
        fast_eval=args.fast_eval,
        target_model=target_model,
        target_tokenizer=target_tokenizer,
    )

    save_file = os.path.join(args.output_dir, f"{args.task}_predictions.json")
    print("Saving predictions to", save_file)
    open(save_file, "w").close()

    # Evaluate the model + Stream predictions to file
    evaluator.evaluate(save_file=save_file)

    # print metrics
    evaluator.print_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--target_tokenizer_path", type=str, required=False, default=None
    )
    parser.add_argument("--target_model_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fast_eval", action="store_true")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--num_samples", type=str, default=None)
    parser.add_argument(
        "--layerwise_similarities",
        action="store_true",
        help="Use layerwise similarities for nearest neighbor",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        help="List of scales to use for evaluation",
        default=[1.0, 5.0, 10.0, 25.0, 50.0],
    )
    args = parser.parse_args()

    main(args)
