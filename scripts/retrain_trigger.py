"""재학습 트리거 판단 (D1).

다음 중 하나면 재학습을 권고한다 (먼저 오는 것):
  (a) 마지막 재학습 이후 교정 라벨이 임계값(RETRAIN_LABEL_THRESHOLD) 이상 쌓임
  (b) 데이터 드리프트가 'alert' 상태

CI/스케줄러가 주기적으로 호출하는 용도. exit code 0 = 재학습 필요, 1 = 불필요.

사용:
    python scripts/retrain_trigger.py
    python scripts/retrain_trigger.py --mark   # 재학습 완료 후 기준 시각 갱신
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import audit, config  # noqa: E402

LAST_RETRAIN_KEY = "last_retrain_at"


def check() -> dict:
    last = audit.get_meta(LAST_RETRAIN_KEY)
    new_labels = audit.count_corrected_since(last)
    drift = audit.drift_report()

    reason = []
    if new_labels >= config.RETRAIN_LABEL_THRESHOLD:
        reason.append(f"교정 라벨 {new_labels}건 ≥ 임계값 {config.RETRAIN_LABEL_THRESHOLD}")
    if drift.get("status") == "alert":
        reason.append(f"드리프트 alert (PSI={drift.get('class_distribution_psi')}, "
                      f"conf_drop={drift.get('confidence_drop')})")

    return {
        "should_retrain": bool(reason),
        "reasons": reason,
        "new_labels_since_last": new_labels,
        "threshold": config.RETRAIN_LABEL_THRESHOLD,
        "drift_status": drift.get("status"),
        "last_retrain_at": last,
    }


def main():
    ap = argparse.ArgumentParser(description="재학습 필요 여부 판단")
    ap.add_argument("--mark", action="store_true", help="재학습 완료로 표시(기준 시각 갱신)")
    args = ap.parse_args()

    audit.init_db()

    if args.mark:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        audit.set_meta(LAST_RETRAIN_KEY, now)
        print(f"재학습 기준 시각 갱신: {now}")
        return

    r = check()
    print("재학습 필요" if r["should_retrain"] else "재학습 불필요")
    print(f"  - 새 교정 라벨: {r['new_labels_since_last']} (임계값 {r['threshold']})")
    print(f"  - 드리프트: {r['drift_status']}")
    for reason in r["reasons"]:
        print(f"  → {reason}")
    sys.exit(0 if r["should_retrain"] else 1)


if __name__ == "__main__":
    main()
