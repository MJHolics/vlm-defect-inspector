# 자가개선 루프 실행 런북 (한 바퀴 돌리기)

설계는 [active_learning_design.md](active_learning_design.md), 각 결정 근거는
[decisions.md](decisions.md). 이 문서는 **실제로 루프를 한 번 도는 운영 절차**다.

CPU 단계(시딩·라벨추출·게이트)는 어디서나 돌고, **추론·재학습(3단계)만 GPU**가
필요하다. CPU 환경에서는 `--mock`으로 배선을 점검할 수 있다.

## 방법론 — test셋 순수성 (가장 중요)
- 교정 라벨은 **유입 풀(기본 `val.json`)** 에서만 모은다. val은 train에 포함되지
  않으므로 '새로 유입된 현장 이미지'를 대신한다.
- **고정 평가셋 `test.json`은 교정·재학습에 절대 쓰지 않는다.** 승격 게이트(D4)는
  이 셋으로만 평가한다. test를 학습에 섞으면 train-on-test 누수로 게이트가 무의미해진다.
- `seed_corrections.py`는 풀과 test의 교집합을 발견하면 즉시 중단한다.

## 단계별 절차

### 1. (GPU) 교정 라벨 시딩 — 사람 검토 입력 생성
현행 모델로 유입 풀을 추론하고, 오답을 정답으로 교정해 감사 DB에 쌓는다.
```bash
python scripts/seed_corrections.py --limit 80        # val에서 80장
# CPU 점검만:  python scripts/seed_corrections.py --mock --limit 80 --db /tmp/t.db
```
교정 라벨이 `RETRAIN_LABEL_THRESHOLD`(기본 20)건 이상 쌓이면 트리거가 켜진다.

### 2. (CPU) 트리거 확인 + 매니페스트 추출
```bash
python scripts/retrain_pipeline.py --check-only      # 트리거 판정만
python scripts/export_labels.py                       # 교정분 → retrain_manifest.jsonl
```

### 3. (GPU, 노트북) 재학습 + 고정 평가셋 추론
`notebooks/03_finetune.ipynb` (또는 `05_experiments.ipynb`)에서:
- `data/processed/retrain_manifest.jsonl`의 `(image_path, label)`를 **원본 train에 합쳐** LoRA 재학습 (D7)
- 어댑터 저장 (예: `models/checkpoints/cand_v2/`)
- `notebooks/04_evaluation.ipynb` 방식으로 **`test.json` 고정셋 추론** → 후보 CSV 저장
  (컬럼: `gt_type, gt_severity, pred_type, pred_severity`)

### 4·5·6. (CPU) 수용평가 → 안전 승격 게이트 → 기준시각 갱신
```bash
python scripts/retrain_pipeline.py \
    --candidate-version v2 \
    --adapter-path models/checkpoints/cand_v2 \
    --candidate-csv data/results/cand_v2_eval_results.csv
```
- 후보가 자체 수용기준(PASS)과 승격 게이트(현행 대비 위험 비악화 AND 정확도 비퇴보)를
  모두 통과하면 `models/registry.json`의 `current`가 교체된다.
- 악화되면 `rejected`로 기록만 되고 운영 모델은 유지된다(롤백 불필요).

## 롤백
```bash
curl -X POST localhost:8000/registry/rollback   # 또는 app.registry.rollback()
```

## 알려진 정합성 메모
- `models/registry.json`의 `v1-bootstrap.adapter_path`가 `models/checkpoints/best_exp/best`로
  적혀 있으나 실제 어댑터는 `models/checkpoints/best_exp/`에 있고, 서빙(`app/main.py`)은
  `models/checkpoints/best`를 로드한다. 다음 승격 시 후보 어댑터 경로를 정확히 기입해
  레지스트리-서빙 경로를 일치시킬 것.
