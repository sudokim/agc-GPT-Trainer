from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BatchEncoding


class TokenizeCollate:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        document_max_length: int = 512,
        query_max_length: int = 64,
        return_labels: bool = True,
    ):
        """
        Tokenize and collate a batch of documents and queries

        Args:
            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer
            document_max_length (int): Max length of document
            query_max_length (int): Max length of query
            return_labels (bool, optional): Whether to return labels. Defaults to True.
        """
        self.tokenizer = tokenizer
        self.document_max_length = document_max_length
        self.query_max_length = query_max_length
        self.return_labels = return_labels

    def __call__(self, batch: tuple[list[str], list[str]] | list[str]) -> BatchEncoding:
        """
        Tokenize and collate a batch of documents and queries

        Args:
            batch (tuple[list[str], list[str]] | list[str]): Batch of documents and queries or just documents

        Examples:
            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> collate = TokenizeCollate(tokenizer)
            >>> batch = [
            ...     [
            ...         "This is a document",
            ...         "This is another document",
            ...     ],
            ...     [
            ...         "This is a query",
            ...         "This is another query",
            ...     ],
            ... ]
            >>> collate(batch)
            {'input_ids': tensor(...), 'attention_mask': tensor(...), 'labels': tensor(...)}

        Returns:
            BatchEncoding: Batch encoding

        """
        if self.return_labels:
            docs, queries = zip(*batch)
            docs = list(docs)
            queries = list(queries)
        else:
            docs = batch
            queries = None

        tokenized = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=self.document_max_length,
            return_tensors="pt",
        )

        if not self.return_labels:
            return tokenized

        tokenized_queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )["input_ids"]
        tokenized_queries[tokenized_queries == 0] = -100

        tokenized["labels"] = tokenized_queries

        return tokenized
