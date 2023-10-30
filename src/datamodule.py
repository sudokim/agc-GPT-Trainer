import json
from logging import getLogger
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.utils import FinetuningCollator

logger = getLogger(__name__)


class GPTFineTuningDataset(Dataset):
    def __init__(
        self,
        docid_to_doc: dict[str, str],  # (doc_id, document)
        data: list[tuple[str, list[str], str]],
        # (question, list of candidate doc ids, answer)
    ):
        """
        Dataset for fine-tuning GPT

        Args:
            docid_to_doc (dict[str, str]): Dict of (doc id, document) pairs
            data (list[tuple[str, list[str], str]]): List of (question, list of candidate doc ids, answer)
        """
        self.docid_to_doc = docid_to_doc
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, list[str], str]:
        """
        Get a single item from the dataset

        Args:
            idx (int): Index of the item to get

        Returns:

        """
        question, doc_ids, answer = self.data[idx]

        # Substitute doc_ids with documents
        if isinstance(doc_ids, str):
            # Document is directly given
            documents = doc_ids
        elif isinstance(doc_ids, list):
            # List of docids is given
            documents = [self.docid_to_doc[doc_id] for doc_id in doc_ids]
        else:
            raise ValueError(f"Invalid type for doc_ids: {type(doc_ids)}")

        return question, documents, answer


class GPTPredictDataset(Dataset):
    def __init__(
        self,
        docid_to_doc: dict[str, str],  # (doc_id, document)
        data: list[tuple[str, list[str]]],
        # (question, list of candidate doc ids, answer)
    ):
        """
        Dataset for fine-tuning GPT

        Args:
            docid_to_doc (dict[str, str]): Dict of (doc id, document) pairs
            data (list[tuple[str, list[str], str]]): List of (question, list of candidate doc ids) pairs
        """
        self.docid_to_doc = docid_to_doc
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, list[str]]:
        """
        Get a single item from the dataset

        Args:
            idx (int): Index of the item to get

        Returns:

        """
        question, doc_ids = self.data[idx]

        # Substitute doc_ids with documents
        if isinstance(doc_ids, str):
            # Document is directly given
            documents = doc_ids
        elif isinstance(doc_ids, list):
            # List of docids is given
            documents = [self.docid_to_doc[doc_id] for doc_id in doc_ids]
        else:
            raise ValueError(f"Invalid type for doc_ids: {type(doc_ids)}")

        return question, documents


class GPTDataset:
    DOCID_PARAGRAPHID_DELIMITER = "|"

    def __init__(
        self,
        dataset_paths: list[str],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        batch_size: int,
        num_workers: int = 4,
        prompt_template_input: str | None = None,
        prompt_template_target: str | None = None,
    ):
        """
        DataModule for the DocT5QueryModule

        Args:
            dataset_paths (list[str]): List of paths to the dataset. Each path contains 3 json files of
                question/document/answer pair for train/dev/test. Refer to data/README.md for more details.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use
            batch_size (int): Batch size
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 4.
            prompt_template_input: Prompt to use for the question. The prompt should contain
                {question} and {document} placeholders. Defaults to None (use the default prompt).
            prompt_template_target: Prompt to use for the answer. The prompt should contain {answer} placeholder.
                Defaults to None (use the default prompt).
        """
        super().__init__()

        self.raw_docs: list[dict[str, Any]] = []  # [{doc_id: document, ...}, ...]
        self.docid_to_doc: dict[str, str] = {}  # {doc_id + paragraph_id: document, ...}
        self.train_dataset = []  # (question, list of candidate doc ids, answer)
        self.dev_dataset = []
        self.test_dataset = []

        for dataset_path in dataset_paths:
            self._process_data(dataset_path)

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

        if prompt_template_input is None:
            prompt_template_input = "주어진 문서의 내용을 참고하여 질문에 답하시오.<|sep|>" "질문: {question}<|sep|>문서: {document}<|sep|>답변:"
        if prompt_template_target is None:
            prompt_template_target = "{answer}<|endoftext|>"

        if prompt_template_input.count("{question}") != 1:
            raise ValueError("prompt_input should contain one {question} placeholder")
        if prompt_template_input.count("{document}") != 1:
            raise ValueError("prompt_input should contain one {document} placeholder")

        if prompt_template_target.count("{answer}") != 1:
            raise ValueError("prompt_target should contain one {answer} placeholder")

        self.prompt_template_input = prompt_template_input
        self.prompt_template_target = prompt_template_target

        self.collator_with_labels = FinetuningCollator(
            tokenizer=self.tokenizer,
            prompt_template_input=self.prompt_template_input,
            prompt_template_output=self.prompt_template_target,
        )

        # Used for prediction dataset
        # self.collator_without_labels = PromptCollator(
        #     tokenizer=self.tokenizer,
        #     return_labels=False,
        #     prompt_input=self.prompt_input,
        #     document_max_length=1024,
        #     query_max_length=128,
        # )

        self.train_dataset = GPTFineTuningDataset(
            docid_to_doc=self.docid_to_doc,
            data=self.train_dataset,
        )
        self.dev_dataset = GPTFineTuningDataset(
            docid_to_doc=self.docid_to_doc,
            data=self.dev_dataset,
        )
        self.test_dataset = GPTFineTuningDataset(
            docid_to_doc=self.docid_to_doc,
            data=self.test_dataset,
        )

    def _process_data(self, dataset_path: str | Path):
        logger.info(f"Processing data from {dataset_path}")

        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)

        # Load documents
        for document_path in (dataset_path / "documents").iterdir():
            if document_path.suffix != ".json":
                logger.warning(f"Skipping {document_path} as it is not a json file")
                continue

            with open(document_path, "r", encoding="utf-8") as f:
                document = json.load(f)

            if not isinstance(document, dict):
                raise ValueError(f"Document {document_path} is not a dict")

            self.raw_docs.append(document)

            docid = document["docid"]
            for paragraph_id, paragraph in document["content"].items():
                final_id = docid + self.DOCID_PARAGRAPHID_DELIMITER + paragraph_id
                assert final_id not in self.docid_to_doc, f"Duplicate doc id {final_id}"

                self.docid_to_doc[final_id] = paragraph["text"]

        # Load train/dev/test data
        for split, dataset in zip(["train", "dev", "test"], [self.train_dataset, self.dev_dataset, self.test_dataset]):
            dataset.extend(self._process_split(dataset_path, split))

    @staticmethod
    def _process_split(dataset_path: Path, split_: str) -> list[tuple[str, list[str] | str, str]]:
        result = []

        with open(dataset_path / f"{split_}.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Data in {dataset_path / f'{split_}.json'} is not a list")

        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"Item {item} in {dataset_path / f'{split_}.json'} is not a dict")

            # assert {"question", "document", "answer"} in keys
            if not {"question", "document", "answer"}.issubset(item.keys()):
                raise ValueError(f"Item {item} in {dataset_path / f'{split_}.json'} does not contain all required keys")

            if not isinstance(item["question"], str):
                raise ValueError(
                    f"Incorrect type for question in {item} in {dataset_path / f'{split_}.json'}; "
                    f"expected str, got {type(item['question'])}"
                )
            if not isinstance(item["answer"], str):
                raise ValueError(
                    f"Incorrect type for answer in {item} in {dataset_path / f'{split_}.json'}; "
                    f"expected str, got {type(item['answer'])}"
                )
            if not isinstance(item["document"], (str, list)):
                raise ValueError(
                    f"Incorrect type for document in {item} in {dataset_path / f'{split_}.json'}; "
                    f"expected str or list[str], got {type(item['document'])}"
                )

            result.append((item["question"], item["document"], item["answer"]))

        return result

    def train_dataloader(self):
        logger.info("Creating train dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collator_with_labels,
        )

    def val_dataloader(self):
        logger.info("Creating dev dataloader")
        return DataLoader(
            self.dev_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator_with_labels,
        )

    def test_dataloader(self):
        logger.info("Creating test dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator_with_labels,
        )
