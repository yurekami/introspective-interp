from __future__ import annotations

import argparse
import os
import shutil
from typing import Any, Dict

import torch
import wandb
import yaml
from dataloaders import (
    BaseDataCollator,
    TASK_DATASET_MAPPING,
    TASK_DATALOADER_MAPPING,
)
from evaluator import (
    EVALUATOR_MAPPING,
    EvaluationCallback,
    SimulatorCorrelationEvaluator,
)
from model.utils import make_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments
from utils import (
    EarlyStoppingCallback,
    LossLoggingCallback,
    merge_config_with_parent,
)


def train(config: Dict[str, Any], args: argparse.Namespace) -> Any:
    """
    Step 1: Initialize models + tokenizer
    """
    if args.debug:
        config["train"]["num_samples"] = 100
        config["train"]["batch_size"] = 5
        config["test"]["num_samples"] = 100
        config["test"]["batch_size"] = 5
        config["train"]["eval_steps"] = 20
        for task in config["test"]["tasks"]:
            config["test"]["tasks"][task]["num_samples"] = 100
        config["num_epochs"] = 1
    else:
        shutil.rmtree(config["output_dir"], ignore_errors=True)

    predictor_tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_path", config["model_path"]),
        padding_side="left",
        cache_dir=config.get("cache_dir", None),
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        config.get("target_tokenizer_path", config["target_model_path"]),
        padding_side="left",
        cache_dir=config.get("cache_dir", None),
    )
    predictor_tokenizer.pad_token = predictor_tokenizer.eos_token
    target_tokenizer.pad_token = target_tokenizer.eos_token
    if "continuous_tokens" in config:
        model_special_tokens_ids = {
            cont_token: predictor_tokenizer.convert_tokens_to_ids(
                config["continuous_tokens"][cont_token]
            )
            for cont_token in config["continuous_tokens"]
        }
        print(f"Special token IDs: {model_special_tokens_ids}")
    else:
        model_special_tokens_ids = None

    # Initialize models using the refactored make_model function
    wrapped_model, target_model = make_model(
        config=config,
        model_special_tokens_ids=model_special_tokens_ids,
        output_dir=config["output_dir"],
        train_self=args.train_self,
        use_embed_proj=config.get("use_embed_proj", False),
    )
    if args.train_self:
        assert wrapped_model == target_model

    # wrapped_model.print_trainable_parameters()

    """
    Step 2: Initialize datasets + data collator
    """
    assert len(config["train"]["tasks"]) == 1
    #     task_type = "mixed"
    #     train_config = config["train"]
    #     num_test_samples = None
    # else:
    task_type = list(config["train"]["tasks"].keys())[0]
    train_config = config["train"]["tasks"][task_type]
    train_config = merge_config_with_parent(config["train"], train_config)
    num_test_samples = config["test"]["tasks"][task_type]["num_samples"]
    # Create data collator
    data_collator = TASK_DATALOADER_MAPPING.get(task_type, BaseDataCollator)(
        predictor_tokenizer=predictor_tokenizer,
        target_tokenizer=target_tokenizer,
        question_config=train_config,
    )
    train_dataset = TASK_DATASET_MAPPING[task_type](
        "train",
        wrapped_model,
        target_model,
        predictor_tokenizer,
        target_tokenizer,
        config=train_config,
        special_tokens=config.get("continuous_tokens"),
        debug=args.debug,
        self_train=args.train_self,
        model_name=config["model_path"],
        model_cache_dir=config.get("cache_dir", None),
        num_test_samples=num_test_samples,
    )
    # Test config should inherit from train config, then apply test overrides, then task-specific overrides
    test_datasets = {}
    test_configs = {}
    for task in config["test"]["tasks"]:
        all_features = getattr(train_dataset, "all_features", None)
        # 1. Start with train config as base
        # 2. Apply test-level overrides
        # 3. Apply task-specific overrides
        task_test_config = merge_config_with_parent(
            merge_config_with_parent(train_config, config["test"]),
            config["test"]["tasks"][task],
        )
        test_configs[task] = task_test_config

        test_datasets[task] = TASK_DATASET_MAPPING[task](
            "test",
            wrapped_model,
            target_model,
            predictor_tokenizer,
            target_tokenizer,
            config=task_test_config,
            special_tokens=config.get("continuous_tokens"),
            debug=args.debug,
            self_train=args.train_self,
            all_features=all_features,
            model_name=config["model_path"],
            model_cache_dir=config.get("cache_dir", None),
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        collate_fn=data_collator,
    )
    test_dataloaders = {
        task: DataLoader(
            test_datasets[task],
            batch_size=config["test"]["batch_size"],
            collate_fn=data_collator,
        )
        for task in config["test"]["tasks"]
    }

    test_evaluators = {
        task: EVALUATOR_MAPPING[test_configs[task]["evaluation_type"]](
            test_configs[task],
            wrapped_model,
            predictor_tokenizer,
            test_dataloaders[task],
            cap_at_100=True,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
        )
        for task in config["test"]["tasks"]
    }

    """
    Step 3: Initialize evaluators
    """
    simulator = None
    exemplar_wrapper = None
    for evaluator in test_evaluators.values():
        if isinstance(evaluator, SimulatorCorrelationEvaluator):
            simulator = evaluator.simulator
            exemplar_wrapper = evaluator.exemplar_wrapper
    # For single task, use the merged config
    train_evaluators = {
        task_type: EVALUATOR_MAPPING[train_config["evaluation_type"]](
            train_config,
            wrapped_model,
            predictor_tokenizer,
            train_dataloader,
            cap_at_100=True,
            simulator=simulator,
            exemplar_wrapper=exemplar_wrapper,
            target_model=target_model,
            target_tokenizer=target_tokenizer,
        )
    }
    split_evaluators = {
        "train": train_evaluators,
        "test": test_evaluators,
    }

    eval_callback = EvaluationCallback(
        {
            f"{split}_{task}": split_evaluators[split][task]
            for split in ["train", "test"]
            for task in config[split]["tasks"]
        },
        eval_strategy=config["train"].get("eval_strategy", "epoch"),
        eval_steps=config["train"].get("eval_steps", 500),
        debug_mode=args.debug,
    )

    output_dir = config["output_dir"]
    if not args.debug:
        wandb.init(
            project="introspective_autointerp",
            name=output_dir,
            config={
                "target_model": config.get("target_model_path", config["model_path"]),
                "model": config["model_path"],
                "num_samples": config["train"]["num_samples"],
                "batch_size": config["train"]["batch_size"],
                "learning_rate": config["train"].get("learning_rate", 5e-5),
                "num_epochs": config["train"].get("num_epochs", 100),
            },
        )

    # save test data
    os.makedirs(config["output_dir"], exist_ok=True)
    if args.train_self and "task" in config["test"]["tasks"]:
        test_evaluators["task"].evaluate()
        if not args.debug:
            test_evaluators["task"].log_metrics(prefix="task/", step=0)
        test_evaluators["task"].print_metrics(
            prefix="Before training orig task eval - "
        )
        eval_callback = EvaluationCallback(
            test_evaluators,
            eval_strategy=config["train"].get("eval_strategy", "epoch"),
            eval_steps=config["train"].get("eval_steps", 500),
            debug_mode=args.debug,
        )

    """
    Step 4: Initialize trainer
    """
    print("Using bf16:", config["train"].get("bf16", True))
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["train"].get("num_epochs", 100),
        per_device_train_batch_size=config["train"]["batch_size"],
        per_device_eval_batch_size=config["test"]["batch_size"],
        learning_rate=config["train"].get("learning_rate", 5e-5),
        lr_scheduler_type=config["train"].get("lr_scheduler_type", "constant"),
        weight_decay=config["train"].get("weight_decay", 0.01),
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=config["train"].get("logging_steps", 10),
        save_strategy=config["train"].get("save_strategy", "epoch"),
        save_steps=config["train"].get("save_steps", None),
        save_total_limit=config["train"].get("save_total_limit", None),
        remove_unused_columns=config["train"].get("remove_unused_columns", False),
        bf16=config["train"].get("bf16", True),  # Enable bfloat16 precision
        report_to="wandb" if not args.debug else "none",
        ddp_find_unused_parameters=config["train"].get(
            "ddp_find_unused_parameters", False
        ),
        max_steps=config["train"].get("max_steps", -1),
        accelerator_config={"dispatch_batches": False},
    )

    # Create callbacks list
    trainer_callbacks = [eval_callback]

    # Add early stopping callback if patience is specified in config
    patience = config["train"].get("early_stopping_patience", None)
    if patience is not None and patience > 0:
        # Try to determine the main metric from evaluators
        main_metric = None
        greater_is_better = False

        # Look for the main metric in test evaluators
        for evaluator in test_evaluators.values():
            if hasattr(evaluator, "main_metric") and evaluator.main_metric:
                main_metric = evaluator.main_metric
                greater_is_better = getattr(evaluator, "greater_is_better", False)
                break

        early_stopping_callback = EarlyStoppingCallback(
            patience=patience,
            metric_for_best_model=main_metric or "eval_loss",
            greater_is_better=greater_is_better,
        )
        trainer_callbacks.append(early_stopping_callback)
        print(
            f"Early stopping enabled with patience={patience}, metric={main_metric or 'eval_loss'}, greater_is_better={greater_is_better}"
        )

    # Add loss logging callback
    loss_logging_callback = LossLoggingCallback()
    trainer_callbacks.append(loss_logging_callback)

    # Create trainer
    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=trainer_callbacks,
    )

    torch.cuda.empty_cache()
    # Train the model
    trainer.train()

    # Save the model
    all_tasks = list(test_evaluators.keys()) + list(train_evaluators.keys())
    model_save_path = os.path.join(config["output_dir"], f"{'_'.join(all_tasks)}_model")

    wrapped_model.save_pretrained(model_save_path)
    predictor_tokenizer.save_pretrained(model_save_path)

    for task in test_evaluators:
        test_evaluators[task].evaluate()
        if not args.debug:
            test_evaluators[task].log_metrics(prefix=f"final_test_{task}_")
        test_evaluators[task].print_metrics(prefix=f"Final evaluation (test) - {task}")

    wandb.finish()

    return wrapped_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_self", action="store_true", help="Train model to predict self or other"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/intervention_questions.yaml",
        help="Path to the question configuration file",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    # Load the config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    # Train the token attention model
    train(
        config=config,
        args=args,
    )
