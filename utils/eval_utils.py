from typing import Optional

import torch
from konlpy.tag import Okt
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM, BertModel, BertTokenizer

from utils.KoBERTScore import BERTScore


class QualityEvaluator:
    BERTSCORE_MODEL_NAME = "beomi/kcbert-base"
    PERPLEXITY_MODEL_NAME = "kakaobrain/kogpt"
    PERPLEXITY_MODEL_REVISION = "KoGPT6B-ryan1.5b"

    def __init__(
        self,
        device: str,
        bertscore_model_name: str,
        perplexity_model_name: str,
        perplexity_model_revision: str | None,
    ) -> None:
        self.bertscore_model_name = bertscore_model_name
        self.perplexity_model_name = perplexity_model_name
        self.perplexity_model_revision = perplexity_model_revision

        print(f"BertScore Model Name: {self.bertscore_model_name}")
        print(f"Perplexity Model Name: {self.perplexity_model_name}")
        print(f"Perplexity Model Revision: {self.perplexity_model_revision}")

        print("Loading BERTScore...")
        bert_model = BertModel.from_pretrained(self.bertscore_model_name).eval()
        bert_tokenizer = BertTokenizer.from_pretrained(self.bertscore_model_name, model_max_length=300)

        self.bert_scorer = BERTScore((bert_tokenizer, bert_model), best_layer=4)
        self.rouge = Rouge()
        self.analyzer = Okt()

        self._device = device
        print("Loading Perplexity Model...")
        self.perplexity_model = AutoModelForCausalLM.from_pretrained(
            self.perplexity_model_name, revision=self.perplexity_model_revision, device_map=self._device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.perplexity_model_name,
            revision=self.perplexity_model_revision,
        )

    def eval_bertscore(self, predictions: list[str], references: list[str], batch_size: int = 4) -> list[float]:
        return self.bert_scorer.score(predictions, references, batch_size=batch_size)

    def eval_rouge(self, predictions: list[str], references: list[str]) -> list[dict]:
        tokenized_predictions = [" ".join(self.analyzer.morphs(text)) for text in predictions]
        tokenized_references = [" ".join(self.analyzer.morphs(text)) for text in references]

        rouge_score = self.rouge.get_scores(tokenized_predictions, tokenized_references, avg=False)

        return rouge_score

    @torch.no_grad()
    def eval_perplexity(self, predictions: list[str], model) -> list[float]:
        perplexity_list = []

        for prediction in predictions:
            input_tokens = self.tokenizer.encode(prediction, return_tensors="pt").to(self._device)
            outputs = model(input_tokens, labels=input_tokens)
            loss = outputs.loss.float().detach().cpu()
            perplexity = torch.exp(loss).item()
            perplexity_list.append(perplexity)

        return perplexity_list

    def evaluate(
        self,
        predictions: list[str],
        references: list[str],
        batch_size: int = 4,
        is_title: Optional[bool] = False,
    ) -> dict:
        if isinstance(predictions, str):
            raise ValueError("predictions must be a list of strings, but got a string")

        if isinstance(references, str):
            raise ValueError("references must be a list of strings, but got a string")

        bertscore = self.eval_bertscore(predictions, references, batch_size=batch_size)
        rouge_score = self.eval_rouge(predictions, references)
        rouge_score_l = [score["rouge-l"]["f"] for score in rouge_score]

        if is_title:
            score = [0.6 * r_s + 0.4 * b_s for r_s, b_s in zip(rouge_score_l, bertscore)]
            score_dict = {"bert_score": bertscore, "rouge_score_l": rouge_score_l, "score": score}
        else:
            perplexity = self.eval_perplexity(predictions, model=self.perplexity_model)
            score_sim = [0.6 * r_s + 0.4 * b_s for r_s, b_s in zip(rouge_score_l, bertscore)]
            score_fluency = list()
            for p in perplexity:
                if p < 4:
                    score_fluency.append(1)
                else:
                    score_fluency.append(4 / p)
            score = [0.7 * s_s + 0.3 * s_f for s_s, s_f in zip(score_sim, score_fluency)]
            score_dict = {
                "bert_score": bertscore,
                "rouge_score_l": rouge_score_l,
                "perplexity": perplexity,
                "score": score,
            }

        return score_dict
