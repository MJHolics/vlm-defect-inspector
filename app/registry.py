"""모델 레지스트리 + 승격 게이트 (D4, D5, D8).

어떤 모델이 어떤 데이터로 나와서 어떤 성능이었고, 지금 무엇이 운영 중인지를
추적한다(ISO식 추적성). 후보 모델은 안전 게이트를 통과해야만 승격된다.

승격 기준(D4): 고정 평가셋에서
    후보 위험점수 ≤ 현행 위험점수  AND  후보 유형정확도 ≥ 현행 − PROMOTE_TYPE_ACC_EPS
첫 모델은 비교 대상이 없으므로 무조건 등록·승격.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config

REGISTRY_PATH = Path(__file__).parent.parent / "models" / "registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"current": None, "models": []}


def save(reg: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def current(reg: dict | None = None) -> dict | None:
    reg = reg or load()
    cur_id = reg.get("current")
    return next((m for m in reg["models"] if m["version"] == cur_id), None)


def should_promote(current_metrics: dict | None, candidate_metrics: dict) -> tuple[bool, list[str]]:
    """D4 안전 게이트. (승격여부, 사유) 반환."""
    if current_metrics is None:
        return True, ["현행 모델 없음 → 첫 모델로 등록"]

    reasons = []
    ok = True
    cand_risk = candidate_metrics["risk_score"]
    cur_risk = current_metrics["risk_score"]
    if cand_risk <= cur_risk:
        reasons.append(f"[통과] 위험점수 개선/유지 {cand_risk} <= {cur_risk}")
    else:
        ok = False
        reasons.append(f"[불충족] 위험점수 악화 {cand_risk} > {cur_risk}")

    cand_acc = candidate_metrics["type_accuracy"]
    cur_acc = current_metrics["type_accuracy"]
    if cand_acc >= cur_acc - config.PROMOTE_TYPE_ACC_EPS:
        reasons.append(f"[통과] 유형정확도 비퇴보 {cand_acc} >= {cur_acc}-{config.PROMOTE_TYPE_ACC_EPS}")
    else:
        ok = False
        reasons.append(f"[불충족] 유형정확도 퇴보 {cand_acc} < {cur_acc}-{config.PROMOTE_TYPE_ACC_EPS}")

    return ok, reasons


def register_and_maybe_promote(
    version: str, metrics: dict, data_ref: str, adapter_path: str
) -> dict:
    """후보 모델을 등록하고, 게이트를 통과하면 승격한다."""
    reg = load()
    cur = current(reg)
    promote, reasons = should_promote(cur["metrics"] if cur else None, metrics)

    entry = {
        "version": version,
        "created_at": _now(),
        "metrics": metrics,           # {type_accuracy, risk_score, ...}
        "data_ref": data_ref,         # 학습 데이터 스냅샷 식별자 (D8)
        "adapter_path": adapter_path,
        "status": "active" if promote else "rejected",
        "gate_reasons": reasons,
    }
    reg["models"].append(entry)

    if promote:
        if cur:
            cur["status"] = "archived"   # 직전 모델 보관 → 롤백 가능 (D5)
        reg["current"] = version

    save(reg)
    return {"promoted": promote, "reasons": reasons, "entry": entry,
            "previous": cur["version"] if cur else None}


def rollback() -> dict:
    """직전 archived 모델로 되돌린다 (D5)."""
    reg = load()
    cur = current(reg)
    archived = [m for m in reg["models"] if m["status"] == "archived"]
    if not archived:
        return {"ok": False, "detail": "되돌릴 보관 모델이 없습니다"}
    prev = archived[-1]
    if cur:
        cur["status"] = "rejected"
    prev["status"] = "active"
    reg["current"] = prev["version"]
    save(reg)
    return {"ok": True, "current": prev["version"], "rolled_back_from": cur["version"] if cur else None}
