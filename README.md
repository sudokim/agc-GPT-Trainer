# AGC - GPT Trainer

정책지원 과제의 일환으로 진행된 AGC의 GPT 모델 학습 코드입니다.

## 사용 가능한 모델

|                                              모델명                                              | Parameter |        비고         |
|:---------------------------------------------------------------------------------------------:|:---------:|:-----------------:|
|      [EleutherAI/polyglot-ko-12.8b](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)      |   12.8b   |                   |
| [nlpai-lab/kullm-polyglot-12.8b-v2](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2) |   12.8b   | Instruction-tuned |

## 실행 방법

### 1. 가상 환경 생성

```bash
conda create -n agc-gpt-trainer python=3.10
conda activate agc-gpt-trainer
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
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('output/{고유 식별자}')
tokenizer = AutoTokenizer.from_pretrained('output/{고유 식별자}')
```

#### 2.2. 추론

추론 스크립트는 구현할 예정입니다.

## 데이터 형식

데이터 형식은 [data의 README](data/README.md)를 참고해주세요.
