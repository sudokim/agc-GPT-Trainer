import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

from utils.eval_utils import QualityEvaluator


def main(
    output_path: str | Path,
    ground_truth_file_names: list[str],
    bertscore_model_name: str | None = None,
    perplexity_model_name: str | None = None,
    perplexity_model_revision: str | None = None,
):
    if bertscore_model_name is None:
        bertscore_model_name = QualityEvaluator.BERTSCORE_MODEL_NAME

    if perplexity_model_name is None and perplexity_model_revision is None:
        perplexity_model_name = QualityEvaluator.PERPLEXITY_MODEL_NAME
        perplexity_model_revision = QualityEvaluator.PERPLEXITY_MODEL_REVISION

    print("Loading QualityEvaluator...")
    quality_evaluator = QualityEvaluator(
        device="cuda:0",
        bertscore_model_name=bertscore_model_name,
        perplexity_model_name=perplexity_model_name,
        perplexity_model_revision=perplexity_model_revision,
    )

    if isinstance(output_path, str):
        output_path = Path(output_path)

    if not output_path.is_dir():
        raise ValueError(f"output_path must be a directory, but got {output_path}")

    output = {}
    ground_truth_output = {}

    for output_file in output_path.glob("*.jsonl"):
        if output_file.name in ground_truth_file_names:
            print(f"Loading ground truth {output_file.name}...")
            target_dict = ground_truth_output
        else:
            print(f"Loading output {output_file.name}...")
            target_dict = output
        target_dict[output_file.name] = [
            prediction["generated"].strip().replace("<|endoftext|>", "")
            for prediction in map(json.loads, output_file.open("r", encoding="utf-8").readlines())
        ]

    for output_file in output_path.glob("*.json"):
        if output_file.name in ground_truth_file_names:
            print(f"Loading ground truth {output_file.name}...")
            target_dict = ground_truth_output
        else:
            print(f"Loading output {output_file.name}...")
            target_dict = output
        target_dict[output_file.name] = [
            prediction["generated"].strip().replace("<|endoftext|>", "")
            for prediction in json.load(output_file.open("r", encoding="utf-8"))
        ]

    scores = defaultdict(dict)
    for i, (output_file_name, predictions) in enumerate(output.items()):
        print(f"Evaluating {i+1}/{len(output)}...")
        for ground_truth_file_name, ground_truths in ground_truth_output.items():
            assert len(predictions) == len(ground_truths)
            scores[output_file_name][ground_truth_file_name] = quality_evaluator.evaluate(predictions, ground_truths)

    # Convert into rows
    score_rows = []  # (output_file_name, ground_truth_file_name, metric_name, question_index, score)
    for output_file_name, ground_truth_scores in scores.items():
        for ground_truth_file_name, metric_scores in ground_truth_scores.items():
            for metric_name, score in metric_scores.items():
                for question_index, question_score in enumerate(score):
                    score_rows.append(
                        (
                            output_file_name,
                            ground_truth_file_name,
                            metric_name,
                            question_index,
                            question_score,
                        )
                    )

    # Convert into DataFrame
    output_file_name = f"output_{int(time.time())}.csv"
    score_df = pd.DataFrame(
        score_rows, columns=["output_file_name", "ground_truth_file_name", "metric_name", "question_index", "score"]
    )
    score_df.to_csv(output_file_name, index=False)
    print(f"Saved score DataFrame to {output_file_name}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--ground_truth_file_names", nargs="+", type=str, required=True)

    parser.add_argument("--bertscore_model_name", type=str, default=None)
    parser.add_argument("--perplexity_model_name", type=str, default=None)
    parser.add_argument("--perplexity_model_revision", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
