"""고정 평가셋(test.json) 추론 → 후보 CSV 생성 (자가개선 루프 3단계 산출물, GPU 필요).

주어진 LoRA 어댑터로 test.json 전체를 추론해 acceptance_eval/retrain_pipeline이
먹는 CSV(gt_type, gt_severity, pred_type, pred_severity)를 만든다.
notebooks/04_evaluation.ipynb의 추론을 스크립트화한 것으로, app.main의 추론 경로를
그대로 재사용한다.

사용 (GPU):
    python scripts/eval_on_test.py --adapter models/checkpoints/cand_v2 \
        --out data/results/cand_v2_eval_results.csv
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


def _gt(ex: dict) -> tuple[str, str]:
    for t in ex.get("conversations", []):
        if t.get("role") == "assistant":
            try:
                j = json.loads(t["content"])
                return (j.get("type") or "").lower(), (j.get("severity") or "").lower()
            except Exception:
                pass
    return "", ""


def main():
    ap = argparse.ArgumentParser(description="test.json 추론 → 후보 CSV")
    ap.add_argument("--adapter", required=True, help="평가할 LoRA 어댑터 경로")
    ap.add_argument("--testset", type=Path, default=DATA / "test.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "results" / "candidate_eval_results.csv")
    ap.add_argument("--limit", type=int, default=None, help="표본 수 제한(디버그)")
    args = ap.parse_args()

    from app import main as appmain
    appmain.LORA_PATH = Path(args.adapter)
    appmain.USE_LORA = appmain.LORA_PATH.exists()
    if not appmain.USE_LORA:
        raise SystemExit(f"어댑터 경로가 없습니다: {args.adapter}")
    print(f"어댑터 로드: {args.adapter}")
    appmain.load_model()

    data = json.loads(args.testset.read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_type_ok = 0
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "image", "gt_type", "gt_severity", "pred_type", "pred_severity"])
        for i, ex in enumerate(data):
            p = Path(ex["image"])
            ap_ = (ROOT / p) if not p.is_absolute() else p
            img = Image.open(ap_).convert("RGB")
            parsed, _, _, _ = appmain.run_inference(img)
            pt = appmain._normalize_type(parsed)
            ps = (parsed.get("severity") if parsed else None) or ""
            gt_t, gt_s = _gt(ex)
            if pt == gt_t:
                n_type_ok += 1
            w.writerow([ex["id"], ex["image"], gt_t, gt_s, pt or "", ps.lower()])
            if (i + 1) % 30 == 0:
                print(f"  {i+1}/{len(data)}  (유형정확도 누적 {n_type_ok/(i+1):.1%})")

    print(f"\n완료: {len(data)}건, 유형정확도 {n_type_ok/len(data):.1%}")
    print(f"CSV 저장: {args.out}")


if __name__ == "__main__":
    main()
