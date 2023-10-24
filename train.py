import logging
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from rich.logging import RichHandler
from transformers import T5Tokenizer, T5TokenizerFast
from wonderwords import RandomWord

from src.datamodule import GPTFineTuningDataModule
from src.module import GPTModule


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

    seeds = parser.add_argument_group("seeds", "Seeds for reproducibility")
    seeds.add_argument("--seed", type=int, default=42, help="Seed for random number generators")

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
        choices=["64", "32", "16-mixed", "bf16-mixed"],
        help="Floating point precision (64, 32, 16, bf16)",
    )
    trainer.add_argument(
        "--val_check_interval",
        type=eval,
        default=1.0,
        help="Validation check interval. If int, check every n steps. If float, check every n percent of each epoch.",
    )
    trainer.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients on before performing optimizer step",
    )
    trainer.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Early stopping patience (epochs)",
    )

    logger = parser.add_argument_group("logger", "Logger arguments")
    logger.add_argument("--project_name", type=str, default="kodocT5query", help="Project name. Used for logging")
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
    val_check_interval: int | float,
    accumulate_grad_batches: int,
    *args,
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
        val_check_interval: Validation check interval. If int, check every n steps.
            If float, check every n percent of each epoch.
        accumulate_grad_batches: Number of steps to accumulate gradients on before performing optimizer step
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
    torch.set_float32_matmul_precision("medium")

    if python_logger is None:
        python_logger = logging.getLogger(__name__)

    python_logger.info("Starting training")
    seed_everything(seed)

    # Create a unique save path
    random_word_generator = RandomWord()
    while True:
        # Repeat until there is no directory with the same name
        random_word = random_word_generator.random_words(include_parts_of_speech=["nouns"])[0]
        save_pretrained_path = Path("output") / random_word
        if not save_pretrained_path.exists():
            break
    save_pretrained_path.mkdir(parents=True)
    python_logger.info(f"Model will be saved to {save_pretrained_path.absolute()}")

    # Load tokenizer
    if use_fast_tokenizer:
        python_logger.info("Using fast tokenizer")
        tokenizer_cls = T5TokenizerFast
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    else:
        python_logger.info("Using Python tokenizer")
        tokenizer_cls = T5Tokenizer
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_path, model_max_length=512)
    python_logger.info(f"Using tokenizer {tokenizer_cls.__name__}")

    # Load model and datamodule
    python_logger.info(f"Loading model from {model_path}")
    module = GPTModule(
        model_path=model_path,
        lr=lr,
    )
    python_logger.info(f"Loading {len(dataset_paths)} datasets from {dataset_paths}")
    datamodule = GPTFineTuningDataModule(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Setup trainer
    python_logger.info("Loading callbacks")
    callbacks = [
        ModelCheckpoint(
            dirpath=save_pretrained_path,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="{epoch}-{val_loss:.2f}",
        ),
        RichModelSummary(max_depth=2),
        RichProgressBar(),
    ]
    if early_stopping_patience is not None:
        python_logger.info(f"Using early stopping with patience {early_stopping_patience}")
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=5,
            ),
        )

    python_logger.info("Loading loggers")
    loggers = [CSVLogger(save_dir="logs", name=project_name)]
    if use_wandb:
        python_logger.info("Using wandb")
        loggers.append(
            WandbLogger(
                project=project_name,
                entity=wandb_entity,
                name=f"{random_word}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                tags=wandb_tags,
            ),
        )

    trainer = Trainer(
        enable_model_summary=False,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        # Step-based training
        max_epochs=-1,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        # Gradient accumulation
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=loggers,
    )

    for pl_logger in loggers:
        pl_logger.log_hyperparams(
            {
                "dataset_paths": dataset_paths,
                "model_path": model_path,
                "tokenizer_path": tokenizer_path,
                "save_pretrained_path": save_pretrained_path.absolute(),
                "accelerator": accelerator,
                "strategy": strategy,
                "devices": devices,
                "seed": seed,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_steps": max_steps,
                "lr": lr,
                "val_check_interval": val_check_interval,
                "accumulate_grad_batches": accumulate_grad_batches,
                "use_fast_tokenizer": use_fast_tokenizer,
            },
        )

    python_logger.info("Training started")
    trainer.fit(module, datamodule=datamodule)

    python_logger.info("Testing started")
    try:
        trainer.test(ckpt_path="best", datamodule=datamodule)
    except ValueError:
        python_logger.warning("No best checkpoint found. Using current model.")
        trainer.test(module, datamodule=datamodule)

    # noinspection PyUnresolvedReferences
    python_logger.info(
        f"Training finished.\n"
        f"Best path: {trainer.checkpoint_callback.best_model_path}\n"
        f"Best score: {trainer.checkpoint_callback.best_model_score:.4f}\n"
    )

    # Save model
    python_logger.info(f"Saving model to {save_pretrained_path}")
    module.model.save_pretrained(save_pretrained_path)
    tokenizer.save_pretrained(save_pretrained_path)


def _main():
    python_logger = logging.getLogger("Trainer")
    python_logger.setLevel(logging.INFO)
    python_logger.addHandler(RichHandler())

    args = _parse_args()
    train(python_logger=python_logger, **vars(args))


if __name__ == "__main__":
    _main()
