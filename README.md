# VLM Defect Inspector

> Qwen2.5-VL 7B + QLoRA 파인튜닝 기반 금속 표면 불량 분류 시스템

---

## 핵심 성과

| 지표 | Zero-shot (Before) | QLoRA Fine-tuned (After) | 개선 |
|------|-------------------|--------------------------|------|
| **Type Accuracy** | ~42% | **~87%** | **+45%p** |
| JSON Parse Rate | ~65% | **~97%** | +32%p |
| Severity Accuracy | ~38% | **~81%** | +43%p |
| 학습 파라미터 | 7B (전체) | **~40M (0.5%)** | -99.5% |

> RTX 4080 Super (16GB VRAM) 기준 학습 소요: 약 2~3시간

---

## 시스템 구조

```
이미지 입력 (금속 표면)
        ↓
Qwen2.5-VL 7B (4-bit NF4 양자화, frozen)
        + LoRA Adapter (rank=16, α=32, ~40M params)
        ↓
구조화된 불량 리포트 (JSON)
{"type": "scratches", "type_ko": "스크래치",
 "severity": "high", "description": "..."}
        ↓
FastAPI REST API  →  Docker 배포
```

---

## 왜 QLoRA인가

7B VLM 풀 파인튜닝은 **~56GB VRAM**이 필요하다. 소비자 GPU로는 불가능하다.  
QLoRA는 **4-bit 양자화 + LoRA 어댑터**만 학습해서 **~8GB VRAM**으로 해결한다.  
성능 손실은 풀 파인튜닝 대비 1~3%p 이내다.

```
전체 가중치 (7B) → 4-bit NF4 압축 (frozen)
                           +
              LoRA 어댑터 (q/k/v/o proj, ~40M)  ← 이것만 학습
```

---

## 데이터셋

**NEU Metal Surface Defects** (공개 데이터셋, Northeastern University)

| 클래스 | 한글명 | 이미지 수 | 심각도 |
|--------|--------|-----------|--------|
| crazing | 균열 | 300 | low |
| inclusion | 개재물 | 300 | medium |
| patches | 패치결함 | 300 | low |
| pitted_surface | 피팅 | 300 | high |
| rolled-in_scale | 압연스케일 | 300 | medium |
| scratches | 스크래치 | 300 | high |

- 총 1,800장 → Train/Val/Test = 70/15/15 (stratified)
- VQA 포맷 변환: 3가지 질문 템플릿 × 3가지 설명 변형 → 다양성 확보

---

## 노트북 구성

| 노트북 | 내용 |
|--------|------|
| `01_dataset.ipynb` | NEU 로드 + EDA + VQA 포맷 변환 + Train/Val/Test 분할 |
| `02_baseline.ipynb` | Zero-shot 베이스라인 평가 (Type Acc, JSON Parse, F1) |
| `03_finetune.ipynb` | QLoRA 파인튜닝 (4-bit NF4 + LoRA rank=16 + cosine scheduler) |
| `04_evaluation.ipynb` | Before/After 비교 + 혼동 행렬 + 클래스별 F1 |

---

## 학습 설정

```python
# QLoRA 핵심 설정
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.bfloat16,
                   bnb_4bit_use_double_quant=True)

LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj","k_proj","v_proj","o_proj",
                           "gate_proj","up_proj","down_proj"])

# 학습 설정
optimizer  = AdamW8bit(lr=2e-4, weight_decay=0.01)
scheduler  = cosine_with_warmup(warmup_ratio=0.1)
epochs     = 3
batch_size = 1  # + gradient_accumulation=8 (effective=8)
```

---

## 빠른 시작

### 로컬 실행

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 다운로드 (01_dataset.ipynb 실행)
jupyter notebook notebooks/01_dataset.ipynb

# 3. 순서대로 노트북 실행
#    02 → 03 → 04

# 4. API 서버 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker 실행

```bash
docker-compose up --build
```

### API 사용

```bash
# 헬스체크
curl http://localhost:8000/health

# 이미지 불량 분류
curl -X POST http://localhost:8000/inspect \
  -F "file=@your_image.jpg"
```

**응답 예시:**
```json
{
  "type": "scratches",
  "type_ko": "스크래치",
  "severity": "high",
  "description": "표면에 선형 스크래치 결함이 관찰됩니다. 방향성 있는 긁힘 자국이 선명하게 나타납니다.",
  "confidence": "high",
  "elapsed_ms": 420.3,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct + QLoRA"
}
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 + 모델 정보 |
| POST | `/inspect` | 이미지 파일 업로드 → 불량 분류 |
| POST | `/inspect/base64` | Base64 이미지 → 불량 분류 |

Swagger UI: `http://localhost:8000/docs`

---

## 기술 스택

`Qwen2.5-VL` `QLoRA` `PEFT` `bitsandbytes` `PyTorch` `FastAPI` `Docker`

---

## 관련 레포

- [autonomous-cv-pipeline](https://github.com/MJHolics/autonomous-cv-pipeline) — TensorRT FP16 + QLoRA 자율주행 파이프라인
- [multimodal-rag](https://github.com/MJHolics/multimodal-rag) — BGE-M3 + Qwen2.5-VL 기술문서 RAG
