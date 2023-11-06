import json
import warnings
from logging import getLogger
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.prompt_template import *
from src.utils import FineTuningCollator, PromptCollator

logger = getLogger("Trainer")


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
        template: Template,
        num_workers: int = 4,
        document_max_length: int = 1024 + 512,
        query_max_length: int = 512,
    ):
        """
        DataModule for the DocT5QueryModule

        Args:
            dataset_paths (list[str]): List of paths to the dataset. Each path contains 3 json files of
                question/document/answer pair for train/dev/test. Refer to data/README.md for more details.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use
            batch_size (int): Batch size
            template (Template): Template to use. Defaults to Template.POLYGLOT_QA.
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 4.
            document_max_length (int): Max length of questions + documents
            query_max_length (int): Max length of labels
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = tokenizer
        self._tokenizer_setup_special_tokens()

        self.raw_docs: list[dict[str, Any]] = []  # [{doc_id: document, ...}, ...]
        self.docid_to_doc: dict[str, str] = {}  # {doc_id + paragraph_id: document, ...}
        self.train_dataset = []  # (question, list of candidate doc ids, answer)
        self.dev_dataset = []
        self.test_dataset = []

        for dataset_path in dataset_paths:
            self._process_data(dataset_path)

        # Setup template
        self.prompt_template_input = template.value.input
        self.prompt_template_target = template.value.target
        if isinstance(self.prompt_template_input, list):
            self.prompt_template_input = self.tokenizer.sep_token.join(self.prompt_template_input)
        logger.info(f"Using template: {template.value}")

        self.collator_with_labels = FineTuningCollator(
            tokenizer=self.tokenizer,
            prompt_template_input=self.prompt_template_input,
            prompt_template_output=self.prompt_template_target,
            document_max_length=document_max_length,
            query_max_length=query_max_length,
        )
        self.collator_without_labels = PromptCollator(
            tokenizer=self.tokenizer,
            prompt_template_input=self.prompt_template_input,
            document_max_length=document_max_length,
        )

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

    def _tokenizer_setup_special_tokens(self):
        _tokenizer_vocabs = set(self.tokenizer.get_vocab().keys())
        if "<|sep|>" in _tokenizer_vocabs:
            self.tokenizer.sep_token = "<|sep|>"
        elif self.tokenizer.sep_token is None:
            warnings.warn(f"Tokenizer {self.tokenizer} does not have a sep_token. \\n\\n will be used instead.")
            self.tokenizer.sep_token = "\\n\\n"
        if self.tokenizer.eos_token is None:
            warnings.warn(f"Tokenizer {self.tokenizer} does not have a eos_token. <|endoftext|> will be used instead.")
            self.tokenizer.eos_token = "<|endoftext|>"

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

                self.docid_to_doc[final_id] = paragraph["text"].replace("\uf000", "").replace("\u200b", "").strip()

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

            if isinstance(item["document"], str):
                # Document is directly given
                item["document"] = item["document"].replace("\uf000", "").replace("\u200b", "").strip()
            elif not isinstance(item["document"], list):
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

    def predict_dataloader(self):
        logger.info("Creating predict dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collator_without_labels,
        )
