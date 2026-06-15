"""자가개선 루프의 수용평가·승격게이트 스모크 테스트 (GPU·외부의존 없음).

retrain_pipeline 오케스트레이터가 의존하는 두 순수 로직을 합성 데이터로 검증한다:
  1) acceptance_eval.evaluate — 비용가중 위험점수·PASS/FAIL 판정 (수용기준)
  2) registry.should_promote   — D4 안전 승격 게이트 (현행 대비 비퇴보)

레지스트리 파일이나 GPU를 건드리지 않으므로 CI/로컬에서 즉시 돌릴 수 있다.
실행: python tests/test_promotion_gate.py   (pytest 불필요)
"""
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, registry          # noqa: E402
from scripts import acceptance_eval        # noqa: E402

TYPES = ["scratches", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "crazing"]
SEV = ["low", "medium", "high"]


def _make_csv(dirpath: Path, name: str, n: int, misses: int) -> Path:
    """n행 합성 평가 CSV. 모두 유형 정답이고, 앞쪽 `misses`건만 심각도를 high→low로 과소평가."""
    rows = [[TYPES[i % len(TYPES)], SEV[i % 3]] for i in range(n)]
    forced = 0
    for r in rows:
        r += [r[0], r[1]]              # 기본은 정답
        if forced < misses and r[1] != "low":
            r[3] = "low"               # 위험 과소평가 1건
            forced += 1
    p = dirpath / name
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_type", "gt_severity", "pred_type", "pred_severity"])
        w.writerows(rows)
    return p


def test_acceptance_metrics():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        perfect = acceptance_eval.evaluate(_make_csv(d, "perfect.csv", 20, misses=0))
        assert perfect["risk_score"] == 0.0, perfect
        assert perfect["type_accuracy"] == 1.0, perfect
        assert perfect["verdict"] == "PASS", perfect

        # 위험 과소평가 1건: risk = COST_MISS*1 / (COST_MISS*sev_scored)
        one_miss = acceptance_eval.evaluate(_make_csv(d, "one.csv", 20, misses=1))
        assert one_miss["misses_underestimate"] == 1, one_miss
        assert one_miss["verdict"] == "PASS", one_miss          # 합격선 0.15 이내
        assert one_miss["risk_score"] > 0.0, one_miss

        # 과소평가 다수 → 합격선 초과 → FAIL
        many = acceptance_eval.evaluate(_make_csv(d, "many.csv", 20, misses=4))
        assert many["risk_score"] > config.ACCEPTANCE_MAX_RISK, many
        assert many["verdict"] == "FAIL", many
    print("  ✓ test_acceptance_metrics")


def test_promotion_gate():
    cur = {"type_accuracy": 0.8259, "risk_score": 0.0196}

    # 첫 모델(현행 없음) → 무조건 등록
    ok, _ = registry.should_promote(None, {"type_accuracy": 0.5, "risk_score": 0.9})
    assert ok is True

    # 위험 개선 + 정확도 유지 → 승격
    ok, reasons = registry.should_promote(cur, {"type_accuracy": 0.83, "risk_score": 0.01})
    assert ok is True, reasons

    # 위험 악화 → 거부 (수용기준은 통과해도 현행보다 나쁘면 막아야 한다)
    ok, reasons = registry.should_promote(cur, {"type_accuracy": 0.99, "risk_score": 0.05})
    assert ok is False, reasons

    # 정확도 EPS 초과 퇴보 → 거부
    worse = cur["type_accuracy"] - config.PROMOTE_TYPE_ACC_EPS - 0.01
    ok, reasons = registry.should_promote(cur, {"type_accuracy": worse, "risk_score": 0.0})
    assert ok is False, reasons

    # 정확도 EPS 이내 퇴보 + 위험 유지 → 통과 (잡음 허용)
    edge = cur["type_accuracy"] - config.PROMOTE_TYPE_ACC_EPS + 0.001
    ok, reasons = registry.should_promote(cur, {"type_accuracy": edge, "risk_score": 0.0196})
    assert ok is True, reasons
    print("  ✓ test_promotion_gate")


if __name__ == "__main__":
    print("자가개선 루프 게이트 스모크 테스트")
    test_acceptance_metrics()
    test_promotion_gate()
    print("전체 통과 ✅")
