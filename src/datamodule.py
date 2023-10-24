import json
from logging import getLogger
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.typings import *

logger = getLogger(__name__)


class GPTFineTuningDataset(Dataset):
    def __init__(
        self,
        docid_to_doc: dict[DOCID, DOCUMENT],
        data: list[tuple[QUESTION, list[DOCID], ANSWER]],
        prompt: str,
        document_sep: str = "\n\n",
        return_labels: bool = True,
    ):
        """
        Dataset for fine-tuning GPT

        Args:
            docid_to_doc (dict[DOCID, DOCUMENT]): Mapping from doc id to document
            data (list[tuple[QUESTION, list[DOCID], ANSWER]]): List of (question, list of candidate doc ids, answer) tuples
            prompt (str): Prompt to use for the question. The prompt should contain {question} and {document} placeholders.
            document_sep (str, optional): Separator between documents. Defaults to "\\n\\n".
            return_labels (bool, optional): Whether to return labels (answer). Defaults to True.
        """
        self.docid_to_doc = docid_to_doc
        self.data = data
        self.prompt = prompt
        self.document_sep = document_sep
        self.return_labels = return_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, str] | str:
        """
        Get a single item from the dataset

        Args:
            idx (int): Index of the item to get

        Returns:
            Prompt and label if `return_labels` is True, prompt otherwise
        """
        question, doc_ids, answer = self.data[idx]

        documents = self.document_sep.join([self.docid_to_doc[doc_id] for doc_id in doc_ids])
        prompt = self.prompt.format(question=question, document=documents)

        if self.return_labels:
            return prompt, answer
        else:
            return prompt


class GPTFineTuningDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_paths: list[str],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        batch_size: int,
        num_workers: int = 4,
        prompt: str = "질문: {question}\n\n\n문서: {document}\n\n\n답변:",
    ):
        """
        DataModule for the DocT5QueryModule

        Args:
            dataset_paths (list[str]): List of paths to the dataset. Each path contains 3 json files of
                question/document/answer pair for train/dev/test. Refer to data/README.md for more details.
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use
            batch_size (int): Batch size
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 4.
            prompt: Prompt to use for the question. The prompt should contain {question} and {document} placeholders.
                If None, the default prompt will be used.
        """
        super().__init__()

        self.docid_to_doc: dict[DOCID, DOCUMENT] = {}
        self.train_dataset: list[tuple[QUESTION, list[DOCID], ANSWER]] = []
        self.dev_dataset: list[tuple[QUESTION, list[DOCID], ANSWER]] = []
        self.test_dataset: list[tuple[QUESTION, list[DOCID], ANSWER]] = []

        for dataset_path in dataset_paths:
            self._process_data(dataset_path)

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prompt = prompt

    def _process_data(self, dataset_path: str | Path):
        logger.info(f"Processing data from {dataset_path}")

        if isinstance(dataset_path, str):
            dataset_path = Path(dataset_path)

        # Load documents
        for document_path in (dataset_path / "documents").iterdir():
            if document_path.suffix != ".json":
                logger.warning(f"Skipping {document_path} as it is not a json file")
                continue

            with open(document_path, "r") as f:
                document = json.load(f)

            if not isinstance(document, dict):
                raise ValueError(f"Document {document_path} is not a dict")

            self.docid_to_doc.update(document)

        # Load train/dev/test data
        for split in ["train", "dev", "test"]:
            with open(dataset_path / f"{split}.json", "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Data in {dataset_path / f'{split}.json'} is not a list")

            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"Item {item} in {dataset_path / f'{split}.json'} is not a dict")

                if item.keys() - {"question", "document", "answer"}:
                    raise ValueError(f"Item {item} in {dataset_path / f'{split}.json'} has invalid keys")

                self.train_dataset.append((item["question"], item["document"], item["answer"]))

    def train_dataloader(self):
        logger.info("Creating train dataloader")
        return DataLoader(
            GPTFineTuningDataset(
                docid_to_doc=self.docid_to_doc,
                data=self.train_dataset,
                prompt=self.prompt,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        logger.info("Creating dev dataloader")
        return DataLoader(
            GPTFineTuningDataset(
                docid_to_doc=self.docid_to_doc,
                data=self.dev_dataset,
                prompt=self.prompt,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        logger.info("Creating test dataloader")
        return DataLoader(
            GPTFineTuningDataset(
                docid_to_doc=self.docid_to_doc,
                data=self.test_dataset,
                prompt=self.prompt,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
