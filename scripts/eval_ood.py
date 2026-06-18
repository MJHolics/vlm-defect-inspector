"""신규 결함(OOD) 탐지 평가 — open-set 안전성 트랙 (GPU 필요).

특정 결함유형을 학습에서 완전히 제외(holdout)한 어댑터로 test.json 전체를
추론한다. 제외된 클래스는 모델이 한 번도 본 적 없는 '신규 결함'이다.
핵심 질문: 모델이 신규 결함을 아는 클래스로 자신 있게 오분류하는가, 아니면
confidence가 떨어져 '검토 필요'로 걸러지는가?

confidence(생성 토큰 로그확률 기하평균)를 in-distribution 점수로 보고:
  - AUROC: confidence가 신규 vs 기존을 분리하는 능력
  - 운영 임계값(config.CONFIDENCE_THRESHOLD)에서:
      신규결함 적발률(=낮은 confidence로 검토 큐에 걸리는 비율)
      기존클래스 오경보율(=멀쩡한 기존 결함이 검토로 새는 비율)
  - 기존 클래스 정확도(holdout 학습이 본래 성능을 깨지 않았는지 sanity)

이는 app/audit.decide_review 의 needs_review 게이트가 '본 적 없는 결함'에도
안전하게 작동하는지를 실측으로 보여준다.

사용 (GPU):
    python scripts/eval_ood.py --adapter models/checkpoints/ood_no_inclusion \
        --holdout-class inclusion
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data" / "processed"


def _gt_type(ex: dict) -> str:
    for t in ex.get("conversations", []):
        if t.get("role") == "assistant":
            try:
                return (json.loads(t["content"]).get("type") or "").lower()
            except Exception:
                pass
    return ""


def main():
    ap = argparse.ArgumentParser(description="신규 결함(OOD) 탐지 평가")
    ap.add_argument("--adapter", required=True, help="holdout 학습된 LoRA 어댑터")
    ap.add_argument("--holdout-class", required=True, help="학습에서 제외된 신규 결함 유형")
    ap.add_argument("--testset", type=Path, default=DATA / "test.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "results" / "ood_eval.csv")
    ap.add_argument("--json-out", type=Path, default=ROOT / "data" / "results" / "ood_eval.json")
    args = ap.parse_args()

    from app import config
    from app import main as appmain
    appmain.LORA_PATH = Path(args.adapter)
    appmain.USE_LORA = appmain.LORA_PATH.exists()
    if not appmain.USE_LORA:
        raise SystemExit(f"어댑터 경로가 없습니다: {args.adapter}")
    print(f"어댑터 로드: {args.adapter}  (신규 결함 = '{args.holdout_class}')")
    appmain.load_model()

    novel = args.holdout_class.lower()
    data = json.loads(args.testset.read_text(encoding="utf-8"))

    rows = []
    for i, ex in enumerate(data):
        p = Path(ex["image"])
        ap_ = (ROOT / p) if not p.is_absolute() else p
        img = Image.open(ap_).convert("RGB")
        parsed, _, _, conf = appmain.run_inference(img)
        pt = appmain._normalize_type(parsed) or ""
        gt = _gt_type(ex)
        rows.append({
            "id": ex["id"], "gt_type": gt, "pred_type": pt,
            "confidence": round(float(conf), 4), "is_novel": int(gt == novel),
        })
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(data)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "gt_type", "pred_type", "confidence", "is_novel"])
        w.writeheader()
        w.writerows(rows)

    # ── 지표 계산 ──
    from sklearn.metrics import roc_auc_score

    thr = config.CONFIDENCE_THRESHOLD
    novel_rows = [r for r in rows if r["is_novel"]]
    known_rows = [r for r in rows if not r["is_novel"]]
    n_novel, n_known = len(novel_rows), len(known_rows)
    if n_novel == 0:
        raise SystemExit(f"test셋에 신규 클래스 '{novel}' 샘플이 없음 — holdout-class 확인")

    y_true = [r["is_novel"] for r in rows]
    ood_score = [1.0 - r["confidence"] for r in rows]  # 높을수록 OOD
    auroc = roc_auc_score(y_true, ood_score)

    flagged_novel = sum(1 for r in novel_rows if r["confidence"] < thr)
    flagged_known = sum(1 for r in known_rows if r["confidence"] < thr)
    # holdout으로 신규가 된 클래스를 제외한 기존 클래스 정확도(sanity)
    known_acc = (sum(1 for r in known_rows if r["pred_type"] == r["gt_type"])
                 / n_known) if n_known else 0.0
    novel_conf = sum(r["confidence"] for r in novel_rows) / n_novel
    known_conf = sum(r["confidence"] for r in known_rows) / n_known if n_known else 0.0

    report = {
        "adapter": args.adapter,
        "novel_class": novel,
        "n_test": len(rows),
        "n_novel": n_novel,
        "n_known": n_known,
        "confidence_threshold": thr,
        "auroc_confidence_separates_novel": round(float(auroc), 4),
        "mean_confidence_novel": round(novel_conf, 4),
        "mean_confidence_known": round(known_conf, 4),
        "novel_detection_rate_at_thr": round(flagged_novel / n_novel, 4),
        "known_false_flag_rate_at_thr": round(flagged_known / n_known, 4) if n_known else None,
        "known_class_accuracy": round(known_acc, 4),
    }
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== OOD(신규 결함) 탐지 결과 ===")
    print(f"신규 클래스       : {novel} ({n_novel}건) / 기존 {n_known}건")
    print(f"AUROC(confidence) : {report['auroc_confidence_separates_novel']}")
    print(f"평균 confidence   : 신규 {novel_conf:.3f}  vs  기존 {known_conf:.3f}")
    print(f"임계값 {thr}에서   : 신규 적발률 {report['novel_detection_rate_at_thr']:.1%}, "
          f"기존 오경보율 {report['known_false_flag_rate_at_thr']:.1%}")
    print(f"기존클래스 정확도 : {known_acc:.1%} (holdout이 본래 성능 깼는지 sanity)")
    print(f"\n저장: {args.out} / {args.json_out}")


if __name__ == "__main__":
    main()
