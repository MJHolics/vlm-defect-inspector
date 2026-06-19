# VLM Defect Inspector

> **Qwen2.5-VL 7B + QLoRA** 기반 금속 표면 불량 자동 분류 시스템  
> NEU Metal Surface Defects 6-class · 소비자 GPU(RTX 4080 Super 16GB)에서 완전 재현 가능

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 핵심 성과 — 3단계 개선

| 단계 | Type Accuracy | Severity Acc | JSON Parse | 비고 |
|------|:---:|:---:|:---:|------|
| Zero-shot (베이스라인) | 33.7% | 27.8% | 100% | 파인튜닝 없음 |
| QLoRA rank=16 | 76.7% | 90.4% | 100% | +42.9%p |
| **Best Combo (rank32+aug+smooth)** | **82.6%** | **90.4%** | **100%** | **+48.9%p** |

- 학습 파라미터: **~80M / 7B (1.1%)** — 나머지 frozen
- Best Combo 학습 시간: **약 64분** (RTX 4080 Super)

![3단계 비교](data/results/three_stage_comparison.png)

---

## 시스템 구조

```
이미지 입력 (금속 표면 200×200 grayscale)
        ↓
Qwen2.5-VL 7B  ← 4-bit NF4 양자화 (frozen)
        + LoRA Adapter rank=32 (~80M params)  ← 학습
        ↓
구조화된 불량 리포트 (JSON)
{
  "type": "scratches",
  "type_ko": "스크래치",
  "severity": "high",
  "description": "표면에 선형 스크래치 결함이..."
}
        ↓
FastAPI REST API  /  Gradio 데모  →  Docker 배포
```

---

## 왜 QLoRA인가

7B VLM 풀 파인튜닝은 **~56GB VRAM** 이 필요하다. 소비자 GPU로는 불가능하다.  
QLoRA는 **4-bit NF4 양자화 + LoRA 어댑터**만 학습해 **~8GB VRAM** 으로 해결한다.

```
전체 가중치 (7B) → 4-bit NF4 압축 (frozen)
                         +
            LoRA 어댑터 (q/k/v/o/gate/up/down proj)  ← 이것만 학습
```

성능 손실은 풀 파인튜닝 대비 1~3%p 이내. 비용·접근성 측면에서 실용적인 선택이다.

---

## 데이터셋

**NEU Metal Surface Defects** (공개, Northeastern University)

| 클래스 | 한글명 | 이미지 수 | 심각도 |
|--------|--------|:---------:|:------:|
| crazing | 균열 | 300 | low |
| inclusion | 개재물 | 300 | medium |
| patches | 패치결함 | 300 | low |
| pitted_surface | 피팅 | 300 | high |
| rolled-in_scale | 압연스케일 | 300 | medium |
| scratches | 스크래치 | 300 | high |

- 총 1,800장 → Train/Val/Test = 70/15/15 (stratified split)
- VQA 포맷 변환: 3가지 질문 템플릿 × 3가지 설명 변형으로 다양성 확보

---

## 노트북 구성

| 노트북 | 내용 |
|--------|------|
| `01_dataset.ipynb` | NEU 로드 · EDA · VQA 포맷 변환 · stratified split |
| `02_baseline.ipynb` | Zero-shot 평가 — Type Acc, JSON Parse, F1, 혼동 행렬 |
| `03_finetune.ipynb` | QLoRA 파인튜닝 (4-bit NF4, rank=16, cosine scheduler) |
| `04_evaluation.ipynb` | Before/After 비교 · 클래스별 F1 · 혼동 행렬 분석 |
| `05_experiments.ipynb` | 복합 실험 A/B/C/D — rank32, 데이터 증강, 레이블 스무딩, 얼리스토핑 |

---

## 실험 설계 (05_experiments)

| ID | 변경점 | Best Val Loss | Type Acc |
|----|--------|:---:|:---:|
| A | LoRA rank 16 → 32 | — | — |
| B | albumentations 증강 | — | — |
| C | 레이블 스무딩 0.1 + 얼리스토핑 | — | — |
| **D (Best)** | **A + B + C 통합** | **0.1046** | **82.6%** |

![학습 곡선](data/results/exp_best_combo_curve.png)

**증강 파이프라인** (Exp B/D): RandomRotate90 · HorizontalFlip · VerticalFlip · RandomBrightnessContrast · GaussNoise · Blur

![증강 예시](data/results/augmentation_preview.png)

---

## 혼동 행렬 (Best Combo)

![혼동 행렬](data/results/exp_best_confusion_matrix.png)

---

## 학습 설정

```python
# 4-bit 양자화
BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Best Combo LoRA
LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

# 학습
optimizer  = AdamW8bit(lr=2e-4, weight_decay=0.01)
scheduler  = cosine_with_warmup(warmup_ratio=0.1)
epochs     = 5  # early stopping patience=2
batch_size = 1 + gradient_accumulation=8  (effective=8)
label_smoothing = 0.1
```

---

## 빠른 시작

### 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데이터 준비 (01_dataset.ipynb 실행)
jupyter notebook notebooks/01_dataset.ipynb

# 3. 순서대로 노트북 실행
#    02_baseline → 03_finetune → 04_evaluation → 05_experiments

# 4. Gradio 데모
python demo.py

# 5. API 서버
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker 실행

```bash
docker-compose up --build
```

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 + 모델 정보 + 검토 임계값 |
| POST | `/inspect` | 이미지 파일 업로드 → 불량 분류 (+ 감사 기록·검토 판정) |
| POST | `/inspect/base64` | Base64 이미지 → 불량 분류 |
| GET | `/review/queue` | 확신도가 낮아 사람 검토가 필요한 건 목록 |
| POST | `/review/{id}/correct` | 검토자가 정답 확정 → 감사 기록 갱신 |
| GET | `/monitor/stats` | 운영 현황: 상태별 분포·평균 confidence·검토건 실측 정확도 |
| GET | `/monitor/drift` | 입력/출력 분포 드리프트 진단 (PSI + confidence 하락) |
| GET | `/registry` | 모델 이력: 현재 운영 버전 + 등록된 버전들(성능·승격 사유) |
| POST | `/registry/rollback` | 직전 보관 모델로 롤백 |

Swagger UI: `http://localhost:8000/docs`

**응답 예시:**
```json
{
  "record_id": 1024,
  "type": "scratches",
  "type_ko": "스크래치",
  "severity": "high",
  "description": "표면에 선형 스크래치 결함이 관찰됩니다.",
  "confidence": 0.913,
  "review_status": "auto_accepted",
  "review_required": false,
  "elapsed_ms": 420.3,
  "model": "Qwen/Qwen2.5-VL-7B-Instruct+QLoRA(best)"
}
```

---

## 검증 가능한 AI 운영 (Production Trust Layer)

정확도 한 숫자만으로는 규제·고위험 현장(제조·의료)에 모델을 내보낼 수 없다.
의료기기 개발(ISO 13485) 경험에서 가져온 "추적·검토·모니터링·수용기준" 관점을 코드로 구현했다.

- **감사 추적 (Audit Trail)** — 모든 추론을 SQLite에 기록(시각·입력 해시·예측·confidence·모델 버전·지연). 원본 이미지는 저장하지 않고 SHA-256 해시만 남겨 추적성과 프라이버시를 함께 확보.
- **사람 검토 루프 (Human-in-the-loop)** — confidence가 임계값(`VLM_CONFIDENCE_THRESHOLD`, 기본 0.80) 미만이거나 파싱 실패면 자동 승인하지 않고 `needs_review`로 분류. `/review/queue`로 큐를 받고 `/review/{id}/correct`로 정답을 확정하면, 그 피드백으로 **검토건 실측 정확도**가 집계된다.
- **드리프트 모니터링** — 최근 N건의 예측 클래스 분포(PSI)와 평균 confidence를 기준 구간과 비교해 `stable / warn / alert` 판정.
- **수용기준 평가 (Acceptance Criteria)** — 비용가중 위험점수로 출고 합격 여부를 판정:

```bash
python scripts/acceptance_eval.py
```

`exp_best`(270건) 실측 결과 — 유형 정확도 82.6%를 "어떤 오류인가"로 분해:

| 지표 | 값 | 의미 |
|------|----|----|
| 위험한 과소평가 (miss) | **3건** (전부 inclusion) | high를 낮게 본 출고 위험 |
| 보수적 과대평가 (false alarm) | 23건 | 불필요 재검토 (안전한 방향) |
| 비용가중 위험점수 | **0.0196** (합격선 ≤ 0.15) | 미검출 ×10 / 오검출 ×1 |
| 최종 판정 | **PASS** | 모델이 안전한 방향으로 치우침, 개선 타깃=inclusion |

---

## 자가개선 루프 (Active Learning + 재학습)

검증 레이어를 한 단계 더 올려, **사람 검토 → 라벨 축적 → 재학습 → 합격 판정 → 모델 교체**가 도는 자가개선 시스템.
설계·결정 근거: [`docs/active_learning_design.md`](docs/active_learning_design.md), [`docs/decisions.md`](docs/decisions.md)

```
추론 → 감사기록 → confidence 낮음 → 검토 큐 → 사람이 정답 확정 → 교정 라벨 축적
   → (라벨 N건 또는 드리프트 alert) → LoRA 재학습 → 고정 평가셋 수용기준 평가
   → 기존보다 안전하면 승격, 아니면 폐기 → 레지스트리 기록 (악화 시 롤백)
```

- **재학습 트리거** (`scripts/retrain_trigger.py`) — 교정 라벨 20건 누적 **OR** 드리프트 alert.
- **라벨 추출** (`scripts/export_labels.py`) — 교정 라벨 + 보관 이미지를 재학습 매니페스트로.
- **승격 안전 게이트** (`app/registry.py`) — 고정 평가셋에서 **위험점수 ≤ 현행 AND 유형정확도 비퇴보**일 때만 교체.
  현행보다 안전하지 않은 후보는 정확도와 무관하게 **거부**된다 — 안전 우선.
- **모델 레지스트리** — `model_version → 학습데이터 ref → 평가결과`를 기록(ISO식 추적성), 직전 버전 보관해 롤백 가능.

> 핵심: 자동화의 편리함이 아니라 **"나쁜 모델이 출고되지 않게 하는 안전장치"** 가 설계의 중심.

### 실주행 결과 — 게이트가 거부도 승격도 한다

루프를 실제로 돌렸다. 운영 모델(`v1-bootstrap`)로 유입 풀(`val`)을 추론해 오답 **32건을 교정**
(inclusion 22 · crazing 4 · scratches 2 · rolled-in_scale 2 · pitted_surface 2)하고, 이를 원본
train에 합쳐 재학습한 후보들을 **고정 평가셋(`test`, 270건)** 으로 평가했다.

| 모델 | 레시피 | 유형정확도 | 위험점수 | 게이트 | 사유 |
|------|--------|:---:|:---:|:---:|------|
| **v1-bootstrap** (기준선) | rank32/α64 + 스무딩 + 증강 + early-stop | 82.6% | 0.0196 | (기준) | 실험 best-of-4로 선정 |
| v2 후보 | rank16, 3 epoch | 72.6% | 0.147 | ⛔ rejected | **과적합** (train loss→0.0002) → 위험·정확도 동시 악화 |
| v3 후보 | rank16, 1 epoch | 74.1% | 0.0996 | ⛔ rejected | **저적합** (미수렴) → pitted_surface 18건 포함 위험 miss 24건 |
| **v4 후보 → 운영** | rank32/α64 + 스무딩 + 증강 + **val early-stop** | **95.9%** | **0.0041** | ✅ **promoted** | 위험·정확도 동시 개선 → 안전 게이트 통과, `active` 교체 |
| v5 후보 | v4 + `inclusion` ×3 오버샘플 | **99.3%** | 0.0074 | ⛔ rejected | **정확도 최고치인데도 거부** — `rolled-in_scale` 2건이 심각도 과소평가로 빠져 위험점수가 현행보다 악화(0.0074 > 0.0041) |

- **3 epoch는 train을 암기**(loss 0.0002), **1 epoch는 미수렴**(loss 0.38) — 약한 재학습 레시피로는
  정상 구간을 못 잡아 두 후보 모두 거부됐다. 진단은 "재학습 레시피가 기준선보다 약하다"였다.
- **v4는 재학습 레시피를 기준선급으로 복원**(rank32/α64 · 라벨 스무딩 0.1 · 증강 · **val 기준
  early-stopping**)하고 교정 32건을 더해 학습했다. 체크포인트는 `val_loss`로 선택(epoch2 best,
  1.5271)했고 **test는 게이트에서 단 한 번만** 평가했다 — test 누수 없이 95.9%/0.0041로 통과해 운영
  모델로 승격됐다.
  - **+14%p 점프의 정체 = 루프가 짚은 블라인드 스팟 교정.** v1·v4는 LoRA 설정이 동일(r32/α64)한데
    왜 이렇게 뛰었나 — 기준선을 **같은 하니스로 재평가**(81.5%)하고 클래스별로 보니 답이 나온다:
    `inclusion`만 **7%**(3/45)로 붕괴, 나머지 5개 클래스는 ~100%였다. 교정 32건은 정확히 이 약한
    클래스에 집중(inclusion 22 · rolled-in_scale 2)됐고, v4에서 **inclusion 7→76%,
    rolled-in_scale 82→100%**로 오르면서 나머지 100%를 유지해 +14.4%p가 전부 설명된다.

    | 클래스 | best_exp(v1) | v4 |
    |--------|:---:|:---:|
    | inclusion | 7% (3/45) | **76%** (34/45) |
    | rolled-in_scale | 82% | **100%** |
    | crazing·patches·pitted·scratches | 100% | 100% (유지) |

    즉 점프는 레시피 마법이 아니라 **모델이 못 맞히던 클래스를 사람이 교정→재학습으로 메우고, 나머지는
    퇴보시키지 않은** active learning의 정상 결과다.
- 통과할 때까지 하이퍼파라미터를 바꿔 후보를 찍어내면 **고정 평가셋에 과적합(test 누수)** 되어 게이트가
  무의미해진다 — v2·v3·v4는 그렇게 만든 게 아니라 "레시피를 기준선급으로 고친다"는 원칙적 수정의
  결과이고, 체크포인트는 항상 val로 골랐다. 게이트의 무결성이 자동화의 목적보다 우선한다.
- **v5 = 게이트가 정확도를 이기는 장면.** v4의 남은 약점은 `inclusion` 하나(75.6%, 오류 11건 중 10건이
  →`scratches`)였다. 이를 메우려 `inclusion`을 ×3 오버샘플(복제본마다 다른 증강 뷰)해 재학습하니
  **inclusion 75.6→100%, 전체 정확도 95.9→99.3%** 로 역대 최고가 나왔다. 그런데 **승격은 거부됐다** —
  `rolled-in_scale`에서 2건이 `crazing`으로 빠지며 **심각도 과소평가(miss)** 가 새로 생겨, miss에 10배
  비용을 매기는 위험점수가 0.0041→0.0074로 **현행보다 악화**했기 때문이다. 정확도가 +3.4%p 더 높아도
  안전지표가 퇴보하면 올리지 않는다 — D4 게이트는 정확도가 아니라 **위험**으로 판정한다. (어댑터·평가
  CSV는 보관 → 다음 루프에서 inclusion 이득은 살리고 rolled-in_scale 회귀만 없애는 후보로 재도전 가능.)

> 이 루프의 가치는 **"검증을 통과하지 못한 모델은 운영에 올라가지 않고, 통과한 모델만 올라간다"** 를
> 코드로 보장하는 데 있다. v2·v3 거부(레시피 약함)도, v4 승격(안전 개선)도, **v5 거부(정확도는 최고지만
> 안전 회귀)** 도 모두 같은 게이트의 정상 동작이다.

---

## 양산 현실성 — 처리량/지연 벤치마크

"정확도"만으로는 라인에 못 올린다. 운영 모델(v4)을 실제 추론 경로(`scripts/benchmark_latency.py`)로
측정한 결과(RTX 4080 SUPER, nf4 4bit, 단건·batch=1):

| 지표 | 값 |
|------|----|
| 단건 latency | 평균 **14.7s** (p50 11.3 / p90 24.2 / p99 41.1) |
| 처리량 | **0.068 img/s** (≈ 4.3 tok/s, 평균 63.5 tok/응답) |
| GPU peak mem | **6.0 GB** / 16 GB |
| 모델 로드 | 23s (1회) |

**이 수치는 느리다 — 그래서 원인을 진단했다:**
- 비전 토큰 폭증 아님 (200×200 이미지 → prompt_len 196), attention 이미 `sdpa`
- confidence 계산(`output_scores`) 오버헤드는 측정 노이즈 범위(±0) — 우리 코드 탓 아님
- 병목은 **bitsandbytes nf4 4bit의 autoregressive 디코딩**(batch=1, 메모리바운드 dequant).
  bnb 4bit은 *모델을 16GB에 욱여넣는* 데 최적이지 *생성 속도*엔 불리하다.

**최적화 경로(미적용, 추정):** LoRA 병합 후 fp16 서빙(7B fp16≈14GB로 16GB에 적재 가능) ·
AWQ/GPTQ + Marlin 커널 · vLLM/TensorRT-LLM — 통상 5–10× → 단건 ~1.5–3s 수준.

**아키텍처 판단:** 최적화해도 7B VLM은 웨이퍼 ms급 인라인 검사엔 부적합하다. 따라서 VLM은
**설명가능 트리아지·감사·샘플검사**에 두고, 100% 인라인 분류는 컴팩트 CNN으로 분리하는 것이 옳다
(→ 아래 *엣지 배포·경량화*가 이 대비를 **같은 NEU 결함 과제에서** 실측으로 보여준다).

---

## 엣지 배포·경량화 — VLM vs 인라인 CNN, 그리고 INT8 압축

위에서 7B VLM은 인라인엔 부적합하다고 진단했다. 그럼 라인에 올릴 모델은 무엇인가 — 말이 아니라
숫자로 답했다. **VLM과 똑같은 NEU 분할**(`data/processed`, test 270건, 누수 0)로 경량 CNN을 학습하고
(`scripts/train_edge_cnn.py`), ONNX 익스포트 + INT8 정적 양자화 후 정확도·지연·크기를 조합별로 실측했다
(`scripts/benchmark_edge.py`).

| 모델 | test 정확도 | 단건 latency | 처리량 | 크기 | 비고 |
|------|:---:|:---:|:---:|:---:|------|
| Qwen2.5-VL 7B (QLoRA) | 95.9% | 14,681 ms | 0.07 img/s | ~6 GB(GPU) | 설명가능·콜드스타트·OOD |
| ResNet18 (fp32, GPU) | **99.6%** | **2.7 ms** | 365 img/s | 44.8 MB | — |
| MobileNetV3-S (fp32, **CPU**) | **99.6%** | **1.8 ms** | **562 img/s** | 6.1 MB | GPU 없이도 인라인 |

**발견 1 — 폐쇄셋에선 1.5M 파라미터 CNN이 7B VLM을 이긴다.** 같은 6클래스·같은 test에서 MobileNetV3-Small
(1.52M, VLM의 약 1/4600 파라미터)이 99.6%로 VLM 95.9%를 앞서고, **GPU 없이 CPU 단일코어에서 1.8 ms**
(VLM보다 약 8,000× 빠름)에 돈다. 이건 VLM 폄하가 아니라 **역할 분담**이다 — VLM의 값어치는 폐쇄셋 정확도가
아니라 라벨 없는 콜드스타트·신규결함 탐지([OOD 트랙](#신규-결함ood-탐지--open-set-안전성-실측))·자연어
설명이고, 라벨이 쌓인 100% 인라인 분류는 경량 CNN이 정답이다.

**발견 2 — INT8 압축은 공짜가 아니다(아키텍처 의존).** 같은 PTQ(per-channel 정적 양자화, train 128장 캘리브)를
두 모델에 똑같이 적용했는데 결과가 갈렸다:

| 모델 | fp32 정확도 | INT8 정확도 | 크기 | 판정 |
|------|:---:|:---:|:---:|------|
| ResNet18 | 99.6% | **99.6%** | 44.7 → **11.3 MB** (4×↓) | 무손실 압축 — 엣지 적합 |
| MobileNetV3-S | 99.6% | **43.3%** | 6.1 → 1.9 MB | **붕괴** — 순진한 PTQ 실패 |

MobileNetV3의 depthwise conv·hard-swish·SE 블록은 활성값 분포가 채널마다 극단적이라 정적 PTQ가
정확도를 무너뜨리고(99.6→43.3%), x86에선 INT8 depthwise가 fp32보다 느리기까지 했다. **교훈: 압축은
반드시 검증 후 채택한다.** MobileNetV3는 PTQ가 아니라 QAT(양자화 인지 학습)가 필요하고, 애초에 fp32에서
이미 6 MB·562 img/s라 양자화 없이도 엣지에 충분하다. 반대로 ResNet18은 INT8이 무손실로 4× 작아져
저장·메모리 제약 엣지에 바로 쓸 수 있다.

**아키텍처 판단:** 운영 권장은 **MobileNetV3-S fp32(6 MB, CPU 562 img/s)를 인라인 분류기로,
ResNet18-INT8을 메모리 빠듯한 엣지의 대안**으로 둔다. 양산 GPU 타깃이라면 TensorRT INT8이 다음 단계지만,
Windows 재현성을 위해 여기선 ONNX Runtime로 측정했다. 산출물 `data/results/edge_deploy_*.json`.

**라이브 데모:** 이 경량 CNN(MobileNetV3-S fp32 ONNX, 6 MB)을 그대로 Gradio로 띄운 데모가 `space/`에
있다 — CPU만으로 도는 덕에 무료 Hugging Face Spaces에 배포 가능하다(7B VLM은 6 GB라 불가). 이미지를
올리면 결함유형·심각도·신뢰도와 추론시간(ms)을 즉시 보여주고, confidence < 0.80이면 운영 게이트가
사람 검토 큐로 보낸다. <!-- 배포 후: [▶ 라이브 데모](HF_SPACE_URL) -->`python space/app.py`로 로컬 실행.

---

## 반도체 도메인 전이 — 웨이퍼맵 결함분류 (ImageNet → fab)

위 벤치마크가 내린 결론("VLM은 인라인 부적합 → 컴팩트 CNN 분리")을 **실제 반도체 fab
데이터로 실증**한다. [WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
(811,457장 웨이퍼맵, 라벨된 25,519장·8개 결함패턴)에 ResNet18(64×64)을 학습해
두 가지를 측정했다: **(1) 사전학습 전이의 가치**, **(2) 인라인 추론 속도**.
(`scripts/{fetch,prep}_wm811k.py`, `scripts/train_wafer_cnn.py --sweep`)

### (1) 전이 입증 — 같은 모델, 사전학습 vs 무작위 초기화

라벨 양을 바꿔가며 ImageNet 사전학습 가중치로 시작한 모델과 스크래치 모델을 비교
(test 5,104장 고정, macro-F1는 희소 클래스까지 반영):

| 라벨 비율 | 학습수 | 사전학습 acc / macroF1 | 스크래치 acc / macroF1 | macroF1 Δ |
|----------|-------|----------------------|----------------------|-----------|
| **5%**   | 1,022 | 0.815 / **0.724** | 0.785 / 0.662 | **+6.1%p** |
| **10%**  | 2,041 | 0.862 / **0.797** | 0.813 / 0.702 | **+9.5%p** |
| **25%**  | 5,104 | 0.873 / **0.774** | 0.848 / 0.756 | +1.9%p |
| **100%** | 20,415| 0.917 / **0.875** | 0.903 / 0.858 | +1.7%p |

**핵심:** 전이 효과는 **라벨이 적을수록 크다**(10%에서 macroF1 +9.5%p). 데이터가 충분하면
격차는 좁혀진다(100% +1.7%p). 이는 fab의 현실 — **신규 공정·신제품·신규 결함은 라벨이
귀하다** — 과 정확히 맞물린다. 적은 교정 라벨로 빠르게 올리는 게 관건인 환경에서 사전학습
전이는 단순한 정확도 향상이 아니라 **라벨링 비용 절감**이다.

### (2) 인라인 적합성 — VLM 대비 속도

| | 단건 latency | 처리량 |
|--|------------|-------|
| 7B VLM (QLoRA, nf4) | 14.7s | 0.068 img/s |
| **웨이퍼 CNN (ResNet18)** | **3.3ms** | **301 img/s** |

같은 GPU에서 **약 4,400배** 빠르다. "VLM은 설명가능 트리아지·감사·샘플검사, 컴팩트 CNN은
100% 인라인 분류"라는 역할 분리가 정량적으로 옳음을 보여준다.

> **정직한 범위:** 이 트랙은 *ImageNet→웨이퍼* 전이를 입증한 것이다(NEU 금속결함→웨이퍼
> 같은 교차도메인 주장이 아니다 — 금속 표면사진과 이진 웨이퍼맵은 모달리티가 달라 과대주장하지
> 않는다). 두 트랙의 공통 메시지는 **"태스크에 맞는 모델 크기를 고르고, 그 선택을 데이터로
> 검증한다"** 이다.

---

## 신규 결함(OOD) 탐지 — open-set 안전성 실측

fab에는 **학습 때 본 적 없는 새로운 결함**이 등장한다. 모델이 이를 "모르겠다"고 걸러
사람검토로 보내는가, 아니면 아는 클래스로 자신 있게 오분류해 **조용히 흘려보내는가?**
이를 측정하기 위해 `inclusion` 유형을 학습에서 **완전히 제외**(train 1,260→1,050)하고
재학습한 뒤, 그 모델로 test 270건(제외된 inclusion 45 + 기존 5클래스 225)을 추론했다.
제외된 inclusion이 곧 '신규 결함'이다. (`retrain_lora.py --holdout-class`, `scripts/eval_ood.py`)

| 지표 | 값 | 읽는 법 |
|------|----|--------|
| 기존 5클래스 정확도 | **99.6%** | holdout이 본래 성능을 깨지 않음(sanity) |
| 평균 confidence (신규 vs 기존) | **0.816 vs 0.827** | 신규에서 거의 안 떨어짐 |
| AUROC (confidence가 신규 분리) | **0.68** | 우연(0.5)보다 약간 나을 뿐 |
| 임계값 0.80에서 신규 적발률 | **33%** (15/45) | 신규 결함의 2/3를 놓침 |
| 임계값 0.80에서 기존 오경보율 | 2.7% (6/225) | 기존은 거의 안 흘림 |

**정직한 결과 — 이게 핵심이다:** 모델은 처음 보는 inclusion 결함을 "불확실"로 표시하지
않고 **`scratches`(23건)·`rolled-in_scale`(16건)** 등 아는 클래스로 *기존만큼 자신 있게*
오분류했다. 즉 **생성 confidence는 신뢰할 만한 OOD 탐지기가 아니다**(신경망의 OOD 과신은
잘 알려진 실패 모드다). 운영 임계값에서 신규 결함의 1/3만 검토 큐에 걸린다.

**그래서 무엇을 하는가:**
- **사람검토(human-in-the-loop)를 제거할 수 없음**을 데이터로 확인 — confidence 게이트만
  믿고 무인 운영하면 신규 결함을 놓친다. 기존 audit/`needs_review` 레이어의 존재 이유가 실증됨.
- **전용 OOD 점수를 도입해 실제로 해결한다** — 아래 참조.

### 전용 OOD 점수 — confidence를 넘어서

같은 holdout 어댑터·같은 test 270건에서 OOD 점수 네 가지를 한 번에 뽑아 공정 비교했다
(`scripts/ood_scores.py`). generate 1회로 로짓 기반 점수(confidence·energy·entropy)를,
별도 forward 1회로 **마지막 레이어 hidden state**를 뽑아 — known 5클래스 train 특징으로
적합한 PCA(64)+수축공분산 공간에서의 최소 클래스 **Mahalanobis 거리**를 계산한다
(known-only 적합이라 test 누수 없음).

| OOD 점수 | AUROC | TPR@FPR5% | FPR95 | 읽는 법 |
|----------|------:|----------:|------:|--------|
| confidence (기준선) | 0.679 | 38% | 0.61 | 로짓 기반 — 앞 표의 0.68 재현(검증) |
| energy (−logsumexp) | 0.393 | 11% | 0.80 | **우연보다도 나쁨** — 로짓이 OOD를 못 가림 |
| entropy | 0.604 | 38% | 0.81 | 약함 |
| **Mahalanobis (특징공간)** | **0.968** | **84%** | **0.08** | **신뢰할 만한 OOD 탐지기** |
| ensemble (4점수 평균) | 0.690 | 49% | 0.79 | 약한 점수들이 희석 → Mahalanobis 단독만 못함 |

- **특징공간 거리가 답이다**: AUROC 0.68 → **0.97**, 기존 오경보 5%만 허용해도 신규 적발
  38% → **84%**. 모델의 *출력 확신*(로짓·confidence)은 OOD에 과신하지만, *내부 표현*은
  신규 결함이 known 클래스 어디서도 멀리 떨어져 있음을 안다.
- **로짓 기반은 신뢰 불가** — energy는 0.393으로 랜덤(0.5)보다도 못하고, 단순 ensemble은
  약한 로짓 점수가 Mahalanobis를 희석시켜 오히려 손해다. "점수를 많이 섞으면 낫다"는 착각의
  반례.
- **운영 함의**: confidence 게이트는 그대로 두되(오경보 통제), 그 뒤에 **Mahalanobis OOD
  게이트를 한 겹 더** 두면 신규 결함의 84%를 사람검토로 끌어올 수 있다 — 무인 흘림을 막는
  실효 레이어.

> 다른 두 트랙처럼, 여기서도 결론은 수치를 **꾸미지 않는** 것이다. confidence만 보면
> "OOD를 못 잡는다(0.68)"가 정직한 한계였고, 전용 특징공간 점수를 붙이니 "특징공간에선
> 잡힌다(0.97)"가 그 해법이다 — **한계를 드러내고, 그 한계를 실제로 메운다.**

---

## 기술 스택

`Qwen2.5-VL` · `QLoRA` · `PEFT` · `bitsandbytes` · `albumentations` · `PyTorch` · `FastAPI` · `Gradio` · `Docker`

---

## 관련 레포

- [autonomous-cv-pipeline](https://github.com/MJHolics/autonomous-cv-pipeline) — TensorRT FP16 + QLoRA 자율주행 파이프라인
- [multimodal-rag](https://github.com/MJHolics/multimodal-rag) — BGE-M3 + Qwen2.5-VL 기술문서 RAG
