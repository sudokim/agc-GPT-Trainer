# AGC - GPT Trainer

정책지원 과제의 일환으로 진행된 AGC의 GPT 모델 학습 코드입니다.

## 사용 가능한 모델

|                       모델명                        |  Parameter  |       모델 경로       |            토크나이저 경로            |
|:------------------------------------------------:|:-----------:|:-----------------:|:------------------------------:|
|  [pko-t5](https://github.com/paust-team/pko-t5)  | Base (275M) | paust/pko-t5-base |       paust/pko-t5-base        |
| [koT5](https://github.com/wisenut-research/KoT5) | Base (222M) |    다운로드한 모델 경로    | 다운로드한 모델 경로의 `spiece.model` 파일 |

pko-t5 모델은 별도 다운로드가 필요하지 않으며, Huggingface의 `from_pretrained()` 함수를 통해 사용할 수 있습니다.

koT5 모델은 해당 repository의 `README.md`를 참고하여 다운로드할 수 있습니다.

## 실행 방법

### 1. 가상 환경 생성

```bash
conda create -n doct5query python=3.10.11
conda activate doct5query
conda install pytorch~=2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -r requirements.txt
```

### 2. 실행

#### 2.1. 학습

학습 스크립트는 `train.py`입니다. 다음과 같이 실행할 수 있습니다.

```bash
python train.py \
    --dataset_paths /path/to/data1 /path/to/data2 \
    --model_path /path/to/model \  # Huggingface 모델 경로
    --tokenizer_path /path/to/tokenizer \  # Huggingface 토크나이저 경로 (기본값: model_path)
    --batch_size 32 \
    --max_steps -1 \  # 최대 학습 step (기본값: -1 (무한, EarlyStopping 사용))
    --val_check_interval 5000 \  # 해당 step마다 검증
    --accumulate_grad_batches 4 \  # 그래디언트 누적
    --devices 1 \  # 사용할 GPU 개수
    --lr 1e-4
```

자세한 옵션은 `train.py`의 `parse_args` 함수, 또는 `python train.py --help`를 통해 확인할 수 있습니다.

학습 결과는 `output/{고유 식별자}` 폴더에 저장됩니다. 고유 식별자는 임의의 영단어로 지정됩니다. (e.g., kayak)

모델을 사용하고자 할 때, `huggingface`를 사용하여 모델을 불러오는 방법은 다음과 같습니다.

```python
from transformers import T5ForConditionalGeneration, T5TokenizerFast

model = T5ForConditionalGeneration.from_pretrained('output/{고유 식별자}')
tokenizer = T5TokenizerFast.from_pretrained('output/{고유 식별자}', model_max_length=512)  # koT5 모델인 경우
```

#### 2.2. 추론

추론 스크립트는 `predict.py`입니다. 다음과 같이 실행할 수 있습니다.

```bash
python predict.py --model_path /path/to/model \
    --tokenizer_path /path/to/tokenizer \  # Huggingface 토크나이저 경로 (기본값: model_path)
    --document_path /path/to/collections.tsv \
    --batch_size 100 \
    --output_path /path/to/output
```

추론은 3가지 방법으로 가능합니다.

1. Greedy decoding: 확률이 가장 높은 질의 1개를 생성합니다.
2. Beam search: beam search를 사용하여 여러 개의 질의를 생성합니다.
3. Top-k sampling: 확률이 높은 k개의 토큰을 샘플링하여 질의를 생성합니다.

docT5query의 경우, top-k sampling을 사용하는 것이 가장 좋은 성능을 보입니다.

추론 결과는 `output_path`에 저장됩니다. (기본값은 `document_path`와 동일)
다음 2가지 파일이 생성됩니다.

1. `generated_queries.json`: 각 docid마다 생성된 질의를 저장합니다.
2. `collection_expanded.tsv`: `collections.tsv`에 생성된 질의를 추가하여 저장합니다.

## 데이터 형식

### 1. 문서 (collections)

`collection.tsv`

```text
docid||text
docid||text
...
```

### 2. 질의 (queries)

`questions.tsv`

```text
qid\tquery
qid\tquery
...
```

### 3. Qrel (qrels)

`qrels.tsv`

```text
qid\tdocid
qid\tdocid
...
```

### 4. 생성 결과 (`generated_queries.json`)

```json5
{
    "docid1": [
        "query1",
        "query2",
        // ...
    ],
    "docid2": [
        "query1",
        "query2",
        // ...
    ],
    // ...
}
```
