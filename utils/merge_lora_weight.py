from sys import argv

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_path: str, adapter_weight_path: str, output_path: str):
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path)

    print("Loading adapter weights...")
    lora_model = PeftModel.from_pretrained(base_model, adapter_weight_path)

    print("Merging and unloading...")
    merged_model = lora_model.merge_and_unload(progressbar=True)

    print("Saving Model...")
    merged_model.save_pretrained(output_path)

    print("Saving tokenizer...")
    AutoTokenizer.from_pretrained(model_path).save_pretrained(output_path)


if __name__ == "__main__":
    if len(argv) != 4:
        print("Usage: python merge_lora_weight.py <model_path> <adapter_weight_path> <output_path>")
        exit(1)

    main(*argv[1:])
