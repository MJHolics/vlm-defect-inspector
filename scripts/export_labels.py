"""교정 라벨 → 재학습 데이터 매니페스트 추출 (D2).

사람이 검토 큐에서 확정한 정답(corrected)을, 보관해 둔 이미지와 묶어
재학습용 매니페스트(jsonl)로 내보낸다. 실제 재학습 단계(03_finetune)는
이 매니페스트의 (image_path, true_type)를 원본 train에 합쳐 사용한다.

사용:
    python scripts/export_labels.py
    python scripts/export_labels.py --out data/processed/retrain_manifest.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import audit  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "processed" / "retrain_manifest.jsonl"


def main():
    ap = argparse.ArgumentParser(description="교정 라벨을 재학습 매니페스트로 추출")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    audit.init_db()
    items = audit.corrected_for_training()
    usable = [x for x in items if x["image_exists"]]
    missing = [x for x in items if not x["image_exists"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for x in usable:
            f.write(json.dumps(
                {"image_path": x["image_path"], "label": x["true_type"]},
                ensure_ascii=False,
            ) + "\n")

    print(f"교정 라벨 총 {len(items)}건 중 이미지 보유 {len(usable)}건 추출")
    if missing:
        print(f"⚠ 이미지 없음 {len(missing)}건 (보관 전 추론분이거나 정리됨) — 재학습 제외")
    # 클래스별 분포
    dist: dict[str, int] = {}
    for x in usable:
        dist[x["label"]] = dist.get(x["label"], 0) + 1
    if dist:
        print("클래스별:", dict(sorted(dist.items(), key=lambda x: -x[1])))
    print(f"매니페스트: {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
