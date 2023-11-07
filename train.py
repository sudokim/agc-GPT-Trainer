import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

import torch
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from rich.logging import RichHandler
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer,
)
from wonderwords import RandomWord

from src.datamodule import GPTDataset
from src.prompt_template import TEMPLATE_MAP


def _parse_args() -> Namespace:
    parser = ArgumentParser()

    paths = parser.add_argument_group("paths", "Paths to data and model")
    # Multiple paths accepted
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
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Path to adapter weights. If provided, adapter will be loaded before training",
    )
    paths.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="Base path to save model",
    )
    paths.add_argument(
        "--random_word",
        type=str,
        default=None,
        help="Random word to use for saving model. If None, a random word will be generated",
    )

    dataset = parser.add_argument_group("dataset", "Dataset arguments")
    dataset.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to dataset directory",
    )
    dataset.add_argument(
        "--template",
        type=str,
        required=True,
        help="Template to use for prompt generation. If None, template is auto-selected based on model name",
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
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model (auto, cpu, cuda:0, cuda:1, ...)",
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
        default=2,
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
    trainer.add_argument(
        "--test_with_small_model",
        action="store_true",
        help="Whether to test with a small model (skt/kogpt2-base-v2). "
        "If True, model_path and tokenizer_path will be ignored",
    )
    trainer.add_argument(
        "--compile_model",
        action="store_true",
        help="Whether to compile model",
    )

    peft = parser.add_argument_group("peft", "Parameter-efficient fine-tuning arguments")
    peft.add_argument("--load_in_8bit", action="store_true", help="Whether to load model in 8bit training mode")
    peft.add_argument("--lora", action="store_true", help="Whether to use LoRA")
    peft.add_argument("--lora_r", type=int, default=32, help="LoRA attention dimension")
    peft.add_argument("--lora_alpha", type=int, default=64, help="Alpha for LoRA scaling")
    peft.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["query_key_value"],
        help="Target modules for LoRA",
    )
    peft.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout for LoRA")

    logger = parser.add_argument_group("logger", "Logger arguments")
    logger.add_argument("--project_name", type=str, default="agc-GPT-Trainer", help="Project name. Used for logging")
    logger.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    logger.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")
    logger.add_argument("--wandb_tags", type=str, nargs="+", default=None, help="WandB tags")

    parsed = parser.parse_args()

    # Check arguments
    if parsed.tokenizer_path is None:
        parsed.tokenizer_path = parsed.model_path

    return parsed


def train(
    dataset_paths: list[str],
    model_path: str,
    tokenizer_path: str,
    lora_adapter_path: str | None,
    template: str,
    seed: int,
    early_stopping_patience: int | None,
    strategy: None | Literal["ddp"],
    batch_size: int,
    num_workers: int,
    max_steps: int,
    device_map: str,
    lr: float,
    model_max_length: int,
    eval_steps: int | float,
    gradient_accumulation_steps: int,
    precision: str,
    lr_scheduler_type: str,
    warmup_steps: int,
    load_in_8bit: bool,
    lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_target_modules: list[str],
    lora_dropout: float,
    *args,
    output_path: str = "output",
    use_fast_tokenizer: bool = False,
    project_name: str | None = "agc-GPT-finetuning",
    use_wandb: bool = False,
    wandb_entity: str | None = None,
    wandb_tags: list[str] | None = None,
    python_logger: logging.Logger | None = None,
    test_with_small_model: bool = False,
    compile_model: bool = False,
    random_word: str | None = None,
    **kwargs,
):
    """
    GPT fine-tuning trainer

    Args:
        dataset_paths: Paths to dataset directories
        model_path: Path to Huggingface model
        tokenizer_path: Path to Huggingface tokenizer
        lora_adapter_path: Path to adapter weights. If provided, adapter will be loaded before training
        template: Template to use for prompt generation. If None, template is auto-selected based on model name
        seed: Seed for random number generators
        early_stopping_patience: Early stopping patience (epochs) (None for no early stopping)
        strategy: Strategy for training (auto, ddp, ...)
        batch_size: Batch size
        num_workers: Number of processes for dataloader
        max_steps: Max number of steps. -1 for no limit
        device_map: Device map for model
        lr: Learning rate
        model_max_length: Maximum length of the model input
        eval_steps: Validation check interval. If int, check every n steps.
            If float, check every n percent of each epoch.
        gradient_accumulation_steps: Number of steps to accumulate gradients on before performing optimizer step
        precision: Floating point precision (32, bf16, fp16)
        lr_scheduler_type: Learning rate scheduler type
        warmup_steps: Number of steps for warmup
        load_in_8bit: Whether to load model in 8bit training mode
        lora: Whether to use LoRA
        lora_r: LoRA attention dimension
        lora_alpha: Alpha for LoRA scaling
        lora_target_modules: Target modules for LoRA
        lora_dropout: Dropout for LoRA
        output_path: Base path to save model
        use_fast_tokenizer: Whether to use fast tokenizer (T5TokenizerFast) or Python tokenizer (T5Tokenizer)
        project_name: Project name. Used for logging
        use_wandb: Whether to use wandb
        wandb_entity: WandB entity name
        wandb_tags: WandB tags
        python_logger: Logger to use
        test_with_small_model: Whether to test with a small model (skt/kogpt2-base-v2)
        compile_model: Whether to compile model
        random_word: Random word to use for saving model. If None, a random word will be generated
        *args: Additional args
        **kwargs: Additional kwargs

    Returns:
        None

    """
    torch.set_float32_matmul_precision("high")

    if python_logger is None:
        python_logger = logging.getLogger(__name__)

    python_logger.info("=== Starting training ===")
    if test_with_small_model:
        python_logger.warning("Testing with a small model (skt/kogpt2-base-v2)")

    # Create a unique save path
    if random_word is None:
        random_word_generator = RandomWord()
        while True:
            # Repeat until there is no directory with the same name
            random_word = random_word_generator.random_words(include_parts_of_speech=["nouns"])[0]
            output_dir = Path(output_path) / random_word
            if not output_dir.exists():
                break
    else:
        python_logger.info(f"Using argument-provided random word: {random_word}")
        output_dir = Path(output_path) / random_word
    output_dir.mkdir(parents=True)
    python_logger.info(f"Model will be saved to {output_dir.absolute()}")

    python_logger.info("Loading WandB")
    if use_wandb:
        python_logger.info("WandB is enabled")
    else:
        python_logger.info("WandB is disabled")
        os.environ["WANDB_MODE"] = "disabled"
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
            "strategy": strategy,
            "device_map": device_map,
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

    # Load tokenizer
    python_logger.info("Loading tokenizer")
    # tokenizer_cls = AutoTokenizer
    tokenizer_cls = AutoTokenizer
    if test_with_small_model:
        tokenizer = tokenizer_cls.from_pretrained(
            "skt/kogpt2-base-v2",
            bos_token="</s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
    else:
        tokenizer = tokenizer_cls.from_pretrained(
            tokenizer_path,
            model_max_length=model_max_length,
        )
    python_logger.info(f"Using tokenizer class {tokenizer.__class__.__name__}")

    # Load model and datamodule
    python_logger.info(f"Loading {len(dataset_paths)} datasets from {dataset_paths}")
    try:
        template = TEMPLATE_MAP[template]
    except KeyError:
        raise ValueError(f"Invalid template: {template}")
    dataset = GPTDataset(
        dataset_paths=dataset_paths,
        template=template,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if test_with_small_model:
        model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
    else:
        python_logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, load_in_8bit=load_in_8bit)
        if load_in_8bit:
            python_logger.info("Loading model in 8bit training mode")
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if lora:
        python_logger.info("Loading model to train with LoRA")
        if lora_adapter_path is None:
            python_logger.info("Loading LoRA adapter from scratch")
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            model.config.use_cache = False
        else:
            python_logger.info(f"Loading LoRA adapter from {lora_adapter_path}")
            model = PeftModel.from_pretrained(model=model, model_id=lora_adapter_path, is_trainable=True)
            model.config.use_cache = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    python_logger.info(
        f"Trainable parameters: {trainable_params:,} | All parameters: {all_params:,} | "
        f"Ratio: {trainable_params / all_params * 100}%"
    )
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
        torch_compile=compile_model,
    )
    python_logger.info(f"Using TrainingArguments {training_arguments}")

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
