import torch
from transformers import BatchEncoding


class FineTuningCollator:
    def __init__(
        self,
        tokenizer,
        prompt_template_input: str,
        prompt_template_output: str,
        document_max_length: int = 1024 + 512,
        query_max_length: int = 512,
        pad_to_multiple_of_8: bool = True,
    ):
        """
        Tokenize and collate a batch of questions, documents, and labels

        Args:
            tokenizer: Tokenizer
            prompt_template_input (str): Prompt to use for the question.
                The prompt should contain {question} and {document} placeholders.
            prompt_template_output (str): Prompt to use for the answer. The prompt should contain {answer} placeholder.
            document_max_length (int): Max length of questions + documents
            query_max_length (int): Max length of labels
        """

        self.tokenizer = tokenizer
        self.prompt_template_input = prompt_template_input
        self.prompt_template_output = prompt_template_output
        self.document_max_length = document_max_length
        self.query_max_length = query_max_length
        self.pad_to_multiple_of_8 = pad_to_multiple_of_8

    def __call__(
        self, batch: list[list[str], list[list[str]], list[str]]  # (questions, documents, labels)
    ) -> BatchEncoding:
        """
        Tokenize and collate a batch of documents and queries

        Args:
            batch (tuple[list[str], list[list[str]], list[str]]): Batch of questions, documents, and labels.

        Returns:
            BatchEncoding: Batch encoding
        """
        questions = []
        docs_raw = []
        labels = []

        for question, doc, label in batch:
            questions.append(question)
            docs_raw.append(doc)
            labels.append(label)

        # !!! Temporarily join documents with \n
        # TODO: Use a better way to join documents
        docs = []
        for doc in docs_raw:
            if isinstance(doc, list):
                doc = "\n".join(doc)
            docs.append(doc)

        prompt_input = [
            (
                self.prompt_template_input.format(question=question, document=doc),
                self.prompt_template_output.format(answer=label),
            )
            for question, doc, label in zip(questions, docs, labels)
        ]

        tokenized = self.tokenizer(
            prompt_input,
            padding=True,
            truncation=True,
            max_length=self.document_max_length,
            return_tensors="pt",
            pad_to_multiple_of=8 if self.pad_to_multiple_of_8 else None,
        )

        # labels_mask: False: label, True: input/padding
        labels_mask = torch.eq(torch.logical_and(tokenized["token_type_ids"], tokenized["attention_mask"]), 0)

        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["labels"][labels_mask] = -100

        del tokenized["token_type_ids"]

        return tokenized


class PromptCollator:
    def __init__(
        self,
        tokenizer,
        prompt_template_input: str,
        document_max_length: int = 1024 + 512,
    ):
        """
        Tokenize and collate a batch of questions, documents, and labels

        Args:
            tokenizer: Tokenizer
            prompt_template_input (str): Prompt to use for the question.
                The prompt should contain {question} and {document} placeholders.
            document_max_length (int): Max length of questions + documents
        """

        self.tokenizer = tokenizer
        self.prompt_template_input = prompt_template_input
        self.document_max_length = document_max_length

    def __call__(self, batch: list[list[str], list[list[str]]]) -> BatchEncoding:  # (questions, documents)
        """
        Tokenize and collate a batch of documents and queries

        Args:
            batch (tuple[list[str], list[list[str]]]): Batch of questions and documents.

        Returns:
            BatchEncoding: Batch encoding
        """
        questions = []
        docs_raw = []

        for question, doc, label in batch:
            questions.append(question)
            docs_raw.append(doc)

        # !!! Temporarily join documents with \n
        # TODO: Use a better way to join documents
        docs = []
        for doc in docs_raw:
            if isinstance(doc, list):
                doc = "\n".join(doc)
            docs.append(doc)

        prompt_input = [
            self.prompt_template_input.format(question=question, document=doc) for question, doc in zip(questions, docs)
        ]

        tokenized = self.tokenizer(
            prompt_input,
            padding=True,
            truncation=True,
            max_length=self.document_max_length,
            return_tensors="pt",
        )

        del tokenized["token_type_ids"]

        return tokenized
