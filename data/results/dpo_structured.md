# DPO 구조화 JSON 정렬 — base vs DPO (실측)

- 모델: `Qwen/Qwen2.5-0.5B-Instruct` · 4-bit QLoRA + DPO(beta=0.1) · train 120쌍 · eval 24프롬프트 · epochs 3

### 행동 지표 (생성물 채점, greedy)

| 지표 | base | DPO | Δ |
|---|---|---|---|
| JSON 유효율 | 0.333 | 1.000 | +0.667 |
| 필드 충족률 | 0.000 | 1.000 | +1.000 |

### DPO-native 지표 (held-out 선호, 정책 vs 참조 implicit reward)

| 지표 | 학습 전(참조=정책) | DPO | 의미 |
|---|---|---|---|
| 선호정확도 | 0.500 | 1.000 | margin>0 비율(동률 0.5) |
| 평균 reward margin | 0.000 | +7.648 | r(chosen)−r(rejected) |

> 합성 선호쌍(chosen=유효 JSON / rejected=산문·깨진 JSON·필드누락) 기반의 정직한 데모. SFT(QLoRA)에 더해 RL 계열(DPO)로 출력 형식을 정렬할 수 있음을 작은 모델로 실측. 학습 전에는 정책=참조라 margin=0·선호정확도=0.5(구성상). held-out(seed 999, 미학습 프롬프트)에서 측정하므로 암기가 아닌 선호의 일반화를 본다.
