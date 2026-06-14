"""검증 가능한 AI 운영 설정.

제조/의료 같은 규제·고위험 환경에서 모델을 운영할 때 필요한 임계값과
수용기준(acceptance criteria)을 한곳에 모은다. 모두 환경변수로 덮어쓸 수 있다.
"""
import os

# ── Human-in-the-loop 반려 임계값 ──────────────────────
# confidence(생성 토큰 로그확률 기하평균)가 이 값 미만이면
# 자동 승인하지 않고 "사람 검토 필요(needs_review)"로 분류한다.
CONFIDENCE_THRESHOLD = float(os.getenv("VLM_CONFIDENCE_THRESHOLD", "0.80"))

# ── 수용기준 (acceptance criteria) ─────────────────────
# 정확도 한 숫자가 아니라 "오류의 비용"으로 모델을 평가한다.
# 제조 검사에서 불량을 정상이라 놓치는 것(miss)은, 정상을 불량이라
# 오검(false alarm)하는 것보다 훨씬 위험하므로 가중치를 크게 둔다.
COST_MISS = float(os.getenv("VLM_COST_MISS", "10.0"))          # 불량 미검출 (위험)
COST_FALSE_ALARM = float(os.getenv("VLM_COST_FALSE_ALARM", "1.0"))  # 불량 오검출 (재검토 비용)

# 배포 합격선: 비용가중 위험점수가 이 값 이하여야 출고 가능으로 본다.
# (위험점수 = 비용가중 오류합 / 전건이 최악(miss)일 때의 비용. 0=완벽, 1=전부 miss)
ACCEPTANCE_MAX_RISK = float(os.getenv("VLM_ACCEPTANCE_MAX_RISK", "0.15"))
# 불량 유형 분류 정확도 최소선 (별도 게이트)
ACCEPTANCE_TYPE_ACC_GATE = float(os.getenv("VLM_ACCEPTANCE_TYPE_ACC", "0.80"))

# ── 드리프트 모니터링 ──────────────────────────────────
# 최근 N건을 기준 분포와 비교한다.
DRIFT_WINDOW = int(os.getenv("VLM_DRIFT_WINDOW", "50"))
# PSI(Population Stability Index) 경보 기준: 0.1 미만 안정 / 0.1~0.25 주의 / 0.25+ 경보
DRIFT_PSI_WARN = float(os.getenv("VLM_DRIFT_PSI_WARN", "0.10"))
DRIFT_PSI_ALERT = float(os.getenv("VLM_DRIFT_PSI_ALERT", "0.25"))
# 최근 구간 평균 confidence가 기준 대비 이 값 이상 떨어지면 경보
DRIFT_CONF_DROP = float(os.getenv("VLM_DRIFT_CONF_DROP", "0.10"))

# ── 자가개선 루프 (Active Learning + 재학습) ───────────
# D1: 교정 라벨이 이만큼 쌓이면 재학습 트리거 (드리프트 alert와 OR). "작게" 설정.
RETRAIN_LABEL_THRESHOLD = int(os.getenv("VLM_RETRAIN_LABEL_THRESHOLD", "20"))
# D4: 후보 모델 승격 시 유형정확도 허용 퇴보폭 (이만큼 이내면 비퇴보로 간주).
PROMOTE_TYPE_ACC_EPS = float(os.getenv("VLM_PROMOTE_TYPE_ACC_EPS", "0.01"))

DEFECT_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]
