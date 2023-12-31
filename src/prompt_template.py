from dataclasses import dataclass
from enum import Enum
from typing import Callable


@dataclass
class PromptTemplate:
    """
    A template for a prompt.

    Attributes
    ----------
    description: str
        A description of the prompt.
    input: str | list[str]
        The input to the prompt.
        Input must have the following fields:
        - {question}
        - {document}
        If a list of str is given, it will be joined with a sep token from the tokenizer.
    target: str
        The target of the prompt.
        Target must have the following field:
        - {answer}
    document_map: Callable[list[str], str] | None
        A function that maps a list of documents to a single document.
        If None, the documents will be joined with "\n".
    """

    description: str
    input: str | list[str]
    target: str
    document_map: Callable[[list[str]], str] | None = None

    def __post_init__(self):
        if isinstance(self.input, list):
            temp_input = " ".join(self.input)
        elif isinstance(self.input, str):
            temp_input = self.input
        else:
            raise TypeError("Input must be str or list[str].")

        if not isinstance(self.target, str):
            raise TypeError("Target must be str.")

        if "{question}" not in temp_input:
            raise ValueError("Input must have {question} field.")
        if "{document}" not in temp_input:
            raise ValueError("Input must have {document} field.")
        if "{answer}" not in self.target:
            raise ValueError("Target must have {answer} field.")


def _join_documents(documents: list[str]) -> str:
    """
    Join documents with "근거 n" where n is the index of the document.
    Args:
        documents (list[str]): A list of documents.

    Returns:
        str: A single document.
    """

    output = [f"근거:\n{document}" for document in documents]

    return "\n\n".join(output)


kullm_zeroshot_qa = PromptTemplate(
    description="Original template from KULLM for QA.",
    input=(
        "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요."
        "\n\n### 명령어:\n주어진 문서의 내용을 참고하여 질문에 답하시오."
        "\n\n### 입력:\n질문: {question}\n\n문서:{document}"
        "\n\n### 응답:\n"
    ),
    target="{answer}",
)

kullm_template_title_generation = PromptTemplate(
    description="Original template from KULLM for title generation.",
    input=(
        "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. 요청을 적절히 완료하는 응답을 작성하세요."
        "\n\n### 명령어:\n주어진 질문과 보고서의 내용을 참고하여 보고서의 제목을 생성하시오."
        "\n\n### 입력:\n질문:\n{question}\n문서:\n{document}"
        "\n\n### 응답:\n"
    ),
    target="{answer}",
)

default_qa = PromptTemplate(
    description="Original template from Polyglot for QA.",
    input=["주어진 문서의 내용을 참고하여 질문에 답하시오.", "질문: {question}", "문서: {document}", "답변:"],
    target=" {answer}",
)


default_qa_separate_documents = PromptTemplate(
    description="Original template from Polyglot for QA, with separate documents.",
    input=["주어진 문서의 내용을 참고하여 질문에 답하시오.", "질문:\n{question}", "문서:\n{document}", "답변:"],
    target="\n{answer}",
    document_map=_join_documents,
)


class Template(Enum):
    """
    An enum class for prompt templates.
    """

    KULLM_ZEROSHOT_QA = kullm_zeroshot_qa
    KULLM_TITLE_GENERATION = kullm_template_title_generation
    DEFAULT_QA = default_qa
    DEFAULT_QA_SEPARATE_DOCUMENTS = default_qa_separate_documents


TEMPLATE_MAP: dict[str, Template] = {
    "kullm_zeroshot_qa": Template.KULLM_ZEROSHOT_QA,
    "kullm_title_generation": Template.KULLM_TITLE_GENERATION,
    "default_qa": Template.DEFAULT_QA,
    "default_qa_separate_documents": Template.DEFAULT_QA_SEPARATE_DOCUMENTS,
}
