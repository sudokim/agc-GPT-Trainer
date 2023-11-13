import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

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
        "--prediction_path",
        type=str,
        required=True,
        help="Path to prediction file",
    )
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
        "--adapter_weight_path",
        type=str,
        nargs="+",
        default=None,
        help="Path to adapter weights",
    )

    model = parser.add_argument_group("model", "Model arguments")
    model.add_argument("--model_max_length", type=int, default=2048, help="Maximum length of the model input")

    trainer = parser.add_argument_group("trainer", "Trainer arguments")
    trainer.add_argument("--batch_size", type=int, default=16, help="Batch size")
    trainer.add_argument("--num_workers", type=int, default=4, help="Number of processes for dataloader")
    trainer.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model (auto, cpu, cuda:0, cuda:1, ...)",
    )
    trainer.add_argument(
        "--test_with_small_model",
        action="store_true",
        help="Whether to test with a small model (skt/kogpt2-base-v2). "
        "If True, model_path and tokenizer_path will be ignored",
    )
    trainer.add_argument(
        "--llama_length",
        action="store_true",
        help="Allow more tokens for LLAMA",
    )
    trainer.add_argument(
        "--kullm_template",
        action="store_true",
        help="Use KULLM template",
    )

    peft = parser.add_argument_group("peft", "Parameter-efficient fine-tuning arguments")
    peft.add_argument("--load_in_8bit", action="store_true", help="Whether to load model in 8bit training mode")
    peft.add_argument("--lora", action="store_true", help="Whether to use LoRA")

    parsed = parser.parse_args()

    # Check arguments
    if parsed.tokenizer_path is None:
        parsed.tokenizer_path = parsed.model_path

    return parsed


def generate(model, tokenizer, batch, skip_special_tokens: bool = True, **kwargs) -> list[str]:
    # Generate
    batch = batch.to(model.device)

    generate_params = (
        dict(
            do_sample=True,
            max_new_tokens=512,
            temperature=0.5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        | kwargs
    )

    output = model.generate(
        **batch,
        **generate_params,
    )

    # Decode
    decoded = tokenizer.batch_decode(output, skip_special_tokens=skip_special_tokens)

    return decoded


# noinspection PyUnusedLocal
@torch.no_grad()
def predict(
    prediction_path: str,
    dataset_paths: list[str],
    model_path: str,  # HuggingFace model path
    tokenizer_path: str,  # HuggingFace tokenizer path
    lora: bool,
    adapter_weight_path: str | list[str] | None,  # Path to adapter weights
    device_map: str,
    batch_size: int,
    num_workers: int,
    model_max_length: int,
    load_in_8bit: bool,
    *args,
    python_logger: logging.Logger | None = None,
    test_with_small_model: bool = False,
    llama_length: bool = False,  # Allow more tokens for LLAMA
    kullm_template: bool = False,
    **kwargs,
):
    if "{model}" not in prediction_path or "{index}" not in prediction_path:
        raise ValueError("prediction_path must contain {model} and {index}")

    if lora and adapter_weight_path is None:
        raise ValueError("adapter_weight_path must be specified when using LoRA")
    if not lora and adapter_weight_path is not None:
        raise ValueError("adapter_weight_path is specified, but not using LoRA. Specify --lora to use LoRA.")

    torch.set_float32_matmul_precision("high")

    if python_logger is None:
        python_logger = logging.getLogger(__name__)
        # Set logger for all modules

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

    if llama_length:
        length_args = {
            "document_max_length": 1024 * 3,
            "query_max_length": 1024,
        }
    else:
        length_args = {}

    dataset = GPTDataset(
        dataset_paths=dataset_paths,
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_workers=num_workers,
        kullm_template=kullm_template,
        **length_args,
    )

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    if test_with_small_model:
        model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
    else:
        python_logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, torch_dtype=torch.bfloat16)

    if isinstance(adapter_weight_path, str):
        adapter_weight_path = [adapter_weight_path]
    adapter_weight_path = list(map(Path, adapter_weight_path))

    for do_sample in (True, FAlse):)

    params = [
        dict(
            temperature=0.1,
            do_sample=True,
        ),
        dict(
            temperature=0.3,
            do_sample=True,
        ),
        dict(
            temperature=0.5,
            do_sample=True,
        ),
        dict(
            temperature=0.7,
            do_sample=True,
        ),
        dict(
            temperature=0.9,
            do_sample=True,
        ),
        dict(
            temperature=1.1,
            do_sample=True,
        ),
        dict(
            top_p=0.8,
            do_sample=True,
        ),
        dict(
            top_p=0.9,
            do_sample=True,
        ),
        dict(
            top_p=0.95,
            do_sample=True,
        ),
        dict(
            top_k=5,
            do_sample=True,
        ),
        dict(
            top_k=10,
            do_sample=True,
        ),
        dict(
            top_k=20,
            do_sample=True,
        ),
        dict(
            top_k=40,
            do_sample=True,
        ),
        dict(
            top_k=60,
            do_sample=True,
        ),
        dict(
            penalty_alpha=0.1,
            top_k=5,
        ),
        dict(
            penalty_alpha=0.2,
            top_k=5,
        ),
        dict(
            penalty_alpha=0.4,
            top_k=5,
        ),
        dict(
            penalty_alpha=0.6,
            top_k=5,
        ),
        dict(
            penalty_alpha=0.8,
            top_k=5,
        ),
        # dict(
        #     num_beams=3,
        #     do_sample=False,
        # ),
        # dict(
        #     num_beams=5,
        #     do_sample=False,
        # ),
        # dict(
        #     num_beams=10,
        #     do_sample=False,
        # ),
    ]

    json.dump(
        params,
        open(prediction_path.format(model="generation", index="params"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )

    times = defaultdict(dict)

    for i, adapter_weight in enumerate(adapter_weight_path):
        print(f"Loading adapter weights from {adapter_weight} ({i+1}/{len(adapter_weight_path)})")
        model.load_adapter(peft_model_id=adapter_weight, adapter_name=adapter_weight.name)

        model.set_adapter(adapter_weight.name)
        print(f"Active adapter: {model.active_adapter}")
        model.eval()

        for j, param in enumerate(params):
            fp = open(prediction_path.format(model=adapter_weight.name, index=f"{j}"), "w", encoding="utf-8")

            start_time = time.time()
            for batch in tqdm(dataset.predict_dataloader()):
                prompt = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)[0]
                generated = generate(model, tokenizer, batch, skip_special_tokens=False, **param)[0]
                generated = generated[len(prompt) :]

                fp.write(json.dumps({"prompt": prompt, "generated": generated}, ensure_ascii=False) + "\n")
                fp.flush()

            fp.close()
            end_time = time.time()
            times[adapter_weight.name][j] = round(end_time - start_time, 2)

    print("Times:", times)


def _main():
    python_logger = logging.getLogger("Trainer")
    python_logger.setLevel(logging.INFO)
    python_logger.addHandler(RichHandler())

    args = _parse_args()
    predict(python_logger=python_logger, **vars(args))


if __name__ == "__main__":
    _main()
