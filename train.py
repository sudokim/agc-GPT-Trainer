import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

import torch
import wandb
from rich.logging import RichHandler
from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
)
from wonderwords import RandomWord

from src.datamodule import GPTDataset


def _parse_args() -> Namespace:
    parser = ArgumentParser()

    paths = parser.add_argument_group("paths", "Paths to data and model")
    # Multiple paths accepted
    paths.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to dataset directory",
    )
    paths.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Huggingface model path. Can be a directory (/path/to/model/dir), or Huggingface model name (t5-base)",
    )
    paths.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Huggingface tokenizer path. Can be a directory (/path/to/tokenizer/dir), "
        "or Huggingface model name (t5-base)",
    )
    paths.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="Base path to save model",
    )

    seeds = parser.add_argument_group("seeds", "Seeds for reproducibility")
    seeds.add_argument("--seed", type=int, default=42, help="Seed for random number generators")

    model = parser.add_argument_group("model", "Model arguments")
    model.add_argument("--model_max_length", type=int, default=2048, help="Maximum length of the model input")

    trainer = parser.add_argument_group("trainer", "Trainer arguments")
    trainer.add_argument("--batch_size", type=int, default=16, help="Batch size")
    trainer.add_argument("--max_steps", type=int, default=-1, help="Max number of steps. -1 for no limit")
    trainer.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer (PreTrainedTokenizerFast) or Python tokenizer (PreTrainedTokenizer)",
    )
    trainer.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    trainer.add_argument("--num_workers", type=int, default=4, help="Number of processes for dataloader")
    trainer.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        help="Accelerator for training (cpu, cuda, ...)",
    )
    trainer.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Strategy for training (auto, ddp, ...)",
    )
    trainer.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to train on (1 for single GPU)",
    )
    trainer.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "tf32", "bf16", "fp16"],
        help="Floating point precision (32, tf32, bf16, fp16)",
    )
    trainer.add_argument(
        "--eval_steps",
        type=eval,
        default=1.0,
        help="Validation check interval. If int, check every n steps. If float, check every n percent of each epoch.",
    )
    trainer.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients on before performing optimizer step",
    )
    trainer.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Early stopping patience (epochs)",
    )
    trainer.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="Learning rate scheduler type",
    )
    trainer.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of steps for warmup",
    )

    logger = parser.add_argument_group("logger", "Logger arguments")
    logger.add_argument("--project_name", type=str, default="agc-GPT-Trainer", help="Project name. Used for logging")
    logger.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    logger.add_argument("--wandb_entity", type=str, default="kocohub", help="WandB entity name")
    logger.add_argument("--wandb_tags", type=str, nargs="+", default=None, help="WandB tags")

    parsed = parser.parse_args()

    # Check arguments
    if parsed.tokenizer_path is None:
        parsed.tokenizer_path = parsed.model_path

    return parsed


# noinspection PyUnusedLocal
def train(
    dataset_paths: list[str],
    model_path: str,
    tokenizer_path: str,
    seed: int,
    accelerator: Literal["cpu", "cuda"],
    early_stopping_patience: int | None,
    strategy: None | Literal["ddp"],
    batch_size: int,
    num_workers: int,
    max_steps: int,
    devices: int,
    lr: float,
    model_max_length: int,
    eval_steps: int | float,
    gradient_accumulation_steps: int,
    precision: str,
    lr_scheduler_type: str,
    warmup_steps: int,
    *args,
    output_path: str = "output",
    use_fast_tokenizer: bool = False,
    project_name: str | None = "agc-GPT-finetuning",
    use_wandb: bool = False,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
    python_logger: logging.Logger | None = None,
    **kwargs,
):
    """
    GPT fine-tuning trainer

    Args:
        dataset_paths: Paths to dataset directories
        model_path: Path to Huggingface model
        tokenizer_path: Path to Huggingface tokenizer
        seed: Seed for random number generators
        accelerator: Accelerator for training (cpu, cuda, ...)
        early_stopping_patience: Early stopping patience (epochs) (None for no early stopping)
        strategy: Strategy for training (auto, ddp, ...)
        batch_size: Batch size
        num_workers: Number of processes for dataloader
        max_steps: Max number of steps. -1 for no limit
        devices: Number of devices to train on (1 for single GPU)
        lr: Learning rate
        model_max_length: Maximum length of the model input
        eval_steps: Validation check interval. If int, check every n steps.
            If float, check every n percent of each epoch.
        gradient_accumulation_steps: Number of steps to accumulate gradients on before performing optimizer step
        precision: Floating point precision (32, bf16, fp16)
        lr_scheduler_type: Learning rate scheduler type
        output_path: Base path to save model
        use_fast_tokenizer: Whether to use fast tokenizer (T5TokenizerFast) or Python tokenizer (T5Tokenizer)
        project_name: Project name. Used for logging
        use_wandb: Whether to use wandb
        wandb_entity: WandB entity name
        wandb_tags: WandB tags
        python_logger: Logger to use
        *args: Additional args
        **kwargs: Additional kwargs

    Returns:
        None

    """
    torch.set_float32_matmul_precision("high")

    if python_logger is None:
        python_logger = logging.getLogger(__name__)

    python_logger.info("=== Starting training ===")

    # Create a unique save path
    random_word_generator = RandomWord()
    while True:
        # Repeat until there is no directory with the same name
        random_word = random_word_generator.random_words(include_parts_of_speech=["nouns"])[0]
        output_dir = Path(output_path) / random_word
        if not output_dir.exists():
            break
    output_dir.mkdir(parents=True)
    python_logger.info(f"Model will be saved to {output_dir.absolute()}")

    # Load tokenizer
    python_logger.info("Loading tokenizer")
    # tokenizer_cls = AutoTokenizer
    tokenizer_cls = PreTrainedTokenizerFast
    tokenizer = tokenizer_cls.from_pretrained(
        tokenizer_path,
        model_max_length=model_max_length,
    )
    python_logger.info(f"Using tokenizer class {tokenizer.__class__.__name__}")

    # Load model and datamodule
    python_logger.info(f"Loading {len(dataset_paths)} datasets from {dataset_paths}")
    dataset = GPTDataset(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    python_logger.info(f"Loading model from {model_path}")
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    python_logger.info(f"Using optimizer {optimizer.__class__.__name__}")

    python_logger.info("Loading TrainingArguments and Trainer")
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        max_steps=max_steps,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        seed=seed,
        data_seed=seed,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        tf32=(precision == "tf32"),
        load_best_model_at_end=True,
        dataloader_num_workers=num_workers,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
    )
    python_logger.info(f"Using TrainingArguments {training_arguments}")

    if not use_wandb:
        python_logger.info("WandB is disabled")
        os.environ["WANDB_MODE"] = "disabled"
    else:
        python_logger.info("WandB is enabled")
    wandb.init(
        dir=output_dir,
        project=project_name,
        entity=wandb_entity,
        tags=wandb_tags,
        config={
            "dataset_paths": dataset_paths,
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "save_pretrained_path": output_dir.absolute(),
            "accelerator": accelerator,
            "strategy": strategy,
            "devices": devices,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_steps": max_steps,
            "lr": lr,
            "eval_steps": eval_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "use_fast_tokenizer": use_fast_tokenizer,
        },
    )

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        data_collator=dataset.collator_with_labels,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.dev_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    python_logger.info("Training started")
    trainer.train()

    python_logger.info("Testing started")
    trainer.evaluate(eval_dataset=dataset.test_dataset)

    # Save model
    python_logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _main():
    python_logger = logging.getLogger("Trainer")
    python_logger.setLevel(logging.INFO)
    python_logger.addHandler(RichHandler())

    args = _parse_args()
    train(python_logger=python_logger, **vars(args))


if __name__ == "__main__":
    _main()
