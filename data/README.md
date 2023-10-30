# Dataset Structure

## Directory Structure

```Plain Text
dataset
├── documents
│   ├── 0001.json
│   // 추가 문서...
├── train.json
├── dev.json
└── test.json
```

## Training Data Format

```json5
[
    {
        // 필수 attribute
        "question": "...",
        "document": ["docid|paragraphid", "..."],  // List of docid
        // "document": "This is a sample document.",  // Document 직접 입력
        "answer": "...",
        // 추가 attribute
        "document_title": "...",
        // ...
    },
    // ...
]
```

## Document Format

```json5
{
    // 문서 attribute (docid는 필수)
    "docid": "docid",
    "title": "무슨무슨 보고서",
    "year": 2019,
    // ...
    // 문서 내 문단 기록
    "content": {
        "paragraphid": {
            "text": "1.1 소제목에 대한 내용입니다.",
            "title": "1.1 소제목", // Optional
        },
        // ...
    }
}
```
