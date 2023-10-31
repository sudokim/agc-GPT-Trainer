import json
import logging
from argparse import ArgumentParser, Namespace
from typing import Literal

import torch
from rich.logging import RichHandler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

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

    model = parser.add_argument_group("model", "Model arguments")
    model.add_argument("--model_max_length", type=int, default=2048, help="Maximum length of the model input")

    trainer = parser.add_argument_group("trainer", "Trainer arguments")
    trainer.add_argument("--batch_size", type=int, default=16, help="Batch size")
    trainer.add_argument("--num_workers", type=int, default=4, help="Number of processes for dataloader")
    trainer.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        help="Accelerator for training (cpu, cuda, ...)",
    )
    trainer.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["32", "tf32", "bf16", "fp16"],
        help="Floating point precision (32, tf32, bf16, fp16)",
    )
    trainer.add_argument(
        "--test_with_small_model",
        action="store_true",
        help="Whether to test with a small model (skt/kogpt2-base-v2). "
        "If True, model_path and tokenizer_path will be ignored",
    )

    parsed = parser.parse_args()

    # Check arguments
    if parsed.tokenizer_path is None:
        parsed.tokenizer_path = parsed.model_path

    return parsed


def generate(model, tokenizer, batch, skip_special_tokens: bool = True, **kwargs) -> list[str]:
    # tokenized = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
    # del tokenized["token_type_ids"]
    #
    # Generate
    batch = batch.to(model.device)
    output = model.generate(
        **batch,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode
    decoded = tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)

    return decoded


# noinspection PyUnusedLocal
@torch.no_grad()
def train(
    dataset_paths: list[str],
    model_path: str,
    tokenizer_path: str,
    accelerator: Literal["cpu", "cuda"],
    batch_size: int,
    num_workers: int,
    model_max_length: int,
    *args,
    python_logger: logging.Logger | None = None,
    test_with_small_model: bool = False,
    **kwargs,
):
    torch.set_float32_matmul_precision("high")

    if python_logger is None:
        python_logger = logging.getLogger(__name__)

    python_logger.info("=== Starting inference ===")
    if test_with_small_model:
        python_logger.warning("Testing with a small model (skt/kogpt2-base-v2)")

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
            padding_side="left",
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

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if test_with_small_model:
        model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
    else:
        python_logger.info(f"Loading model from {model_path}")
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:1")
        
    fp = open("result.json", "w")

    for batch in tqdm(dataset.predict_dataloader()):
        prompt = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)[0]
        generated = generate(model, tokenizer, batch, skip_special_tokens=False)[0]
        generated = generated[len(prompt):]

        fp.write(json.dumps({"prompt": prompt, "generated": generated}, ensure_ascii=False) + "\n")
        fp.flush()

    fp.close()


def _main():
    python_logger = logging.getLogger("Trainer")
    python_logger.setLevel(logging.INFO)
    python_logger.addHandler(RichHandler())

    args = _parse_args()
    train(python_logger=python_logger, **vars(args))


if __name__ == "__main__":
    _main()
