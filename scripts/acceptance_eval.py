"""수용기준(Acceptance Criteria) 기반 평가.

정확도 한 숫자로는 "이 모델을 현장에 내보내도 되는가"를 답할 수 없다.
제조 검사에서 중요한 건 *어떤 종류의 오류*인가다.

- 위험한 과소평가(miss)  : 실제 high 심각도를 낮게 판정 → 불량이 출고될 위험
- 보수적 과대평가(false alarm) : 실제 low를 높게 판정 → 불필요한 재검토 비용

미검출에 큰 비용(COST_MISS)을, 오검출에 작은 비용(COST_FALSE_ALARM)을 매겨
비용가중 위험점수를 구하고, 사전 정의한 합격선과 비교해 PASS/FAIL을 판정한다.

사용:
    python scripts/acceptance_eval.py
    python scripts/acceptance_eval.py --csv data/results/exp_best_eval_results.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
DEFAULT_CSV = ROOT / "data" / "results" / "exp_best_eval_results.csv"


def _rank(sev: str) -> int | None:
    return SEVERITY_RANK.get((sev or "").strip().lower())


def evaluate(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    n = len(rows)
    if n == 0:
        raise SystemExit(f"빈 결과 파일: {csv_path}")

    type_correct = 0
    sev_correct = 0
    misses = 0          # 과소평가 (위험)
    false_alarms = 0    # 과대평가 (비용)
    sev_scored = 0      # 심각도 비교 가능 건수
    per_class_miss: dict[str, int] = {}

    for r in rows:
        if r.get("pred_type", "").strip().lower() == r.get("gt_type", "").strip().lower():
            type_correct += 1

        g, p = _rank(r.get("gt_severity")), _rank(r.get("pred_severity"))
        if g is None or p is None:
            continue
        sev_scored += 1
        if p == g:
            sev_correct += 1
        elif p < g:  # 실제보다 낮게 → miss
            misses += 1
            cls = r.get("gt_type", "?")
            per_class_miss[cls] = per_class_miss.get(cls, 0) + 1
        else:        # 실제보다 높게 → false alarm
            false_alarms += 1

    type_acc = type_correct / n
    sev_acc = sev_correct / sev_scored if sev_scored else 0.0

    # 비용가중 위험점수 (0=완벽, 1=전건 miss)
    weighted_cost = config.COST_MISS * misses + config.COST_FALSE_ALARM * false_alarms
    max_cost = config.COST_MISS * sev_scored
    risk_score = weighted_cost / max_cost if max_cost else 0.0

    passed_risk = risk_score <= config.ACCEPTANCE_MAX_RISK
    passed_type = type_acc >= config.ACCEPTANCE_TYPE_ACC_GATE
    verdict = "PASS" if (passed_risk and passed_type) else "FAIL"

    return {
        "csv": str(csv_path.relative_to(ROOT)) if csv_path.is_relative_to(ROOT) else str(csv_path),
        "n_samples": n,
        "type_accuracy": round(type_acc, 4),
        "severity_accuracy": round(sev_acc, 4),
        "severity_scored": sev_scored,
        "misses_underestimate": misses,
        "false_alarms_overestimate": false_alarms,
        "cost_weights": {"miss": config.COST_MISS, "false_alarm": config.COST_FALSE_ALARM},
        "risk_score": round(risk_score, 4),
        "gates": {
            "max_risk": config.ACCEPTANCE_MAX_RISK,
            "min_type_accuracy": config.ACCEPTANCE_TYPE_ACC_GATE,
            "passed_risk": passed_risk,
            "passed_type_accuracy": passed_type,
        },
        "per_class_dangerous_miss": dict(sorted(per_class_miss.items(), key=lambda x: -x[1])),
        "verdict": verdict,
    }


def _print_report(rep: dict) -> None:
    print("=" * 56)
    print(" 수용기준 평가 (Acceptance Criteria Report)")
    print("=" * 56)
    print(f" 데이터        : {rep['csv']}  ({rep['n_samples']}건)")
    print(f" 유형 정확도   : {rep['type_accuracy']:.1%}  (게이트 {rep['gates']['min_type_accuracy']:.0%})")
    print(f" 심각도 정확도 : {rep['severity_accuracy']:.1%}")
    print("-" * 56)
    print(f" 위험한 과소평가(miss)      : {rep['misses_underestimate']}건  ×{rep['cost_weights']['miss']}")
    print(f" 보수적 과대평가(false alarm): {rep['false_alarms_overestimate']}건  ×{rep['cost_weights']['false_alarm']}")
    print(f" 비용가중 위험점수          : {rep['risk_score']:.4f}  (합격선 ≤ {rep['gates']['max_risk']})")
    if rep["per_class_dangerous_miss"]:
        print(" 위험 과소평가 클래스별:")
        for cls, c in rep["per_class_dangerous_miss"].items():
            print(f"   - {cls}: {c}건")
    print("-" * 56)
    print(f" 최종 판정     : {rep['verdict']}")
    print("=" * 56)


def main():
    ap = argparse.ArgumentParser(description="수용기준 기반 모델 출고 합격 판정")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="평가 결과 CSV 경로")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "results" / "acceptance_report.json")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"결과 파일을 찾을 수 없습니다: {args.csv}")

    rep = evaluate(args.csv)
    _print_report(rep)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n리포트 저장: {args.out.relative_to(ROOT)}")
    sys.exit(0 if rep["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
