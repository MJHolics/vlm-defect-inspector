"""교정 라벨 시딩 — 자가개선 루프의 '사람 검토' 단계를 실데이터로 채운다 (D2 전단계).

자가개선 루프가 한 바퀴 돌려면 *모델이 틀린 건을 사람이 정답으로 교정한* 데이터가
감사 DB에 쌓여 있어야 한다. 이 스크립트는 그 상태를 실제로 만든다:

  유입 이미지 풀 → 현행 모델 추론 → 감사 DB 기록(+이미지 보관)
                → 오답인 건만 정답(true_type)으로 교정 입력

방법론 (중요):
  - 풀(--pool)은 기본적으로 **검증셋(val.json)** 을 쓴다. 이는 train에 포함되지
    않은 데이터로, '새로 유입된 현장 이미지'를 대신한다.
  - **고정 평가셋(test.json)은 절대 풀에 들어가지 않는다.** 교집합이 있으면 즉시
    중단한다. test셋을 교정→재학습에 쓰면 train-on-test 누수가 되어 승격 게이트가
    무의미해지기 때문이다(검증 가능한 AI의 핵심 전제).

GPU 필요: 추론은 4-bit Qwen2.5-VL을 로드하므로 CUDA 환경(노트북/서버)에서 실행한다.
  CPU 환경에서 배선만 점검하려면 --mock 으로 모델 없이 가짜 추론을 돌릴 수 있다.

사용:
    # GPU 환경에서 실제 시딩 (val에서 80장 추론 → 오답 교정)
    python scripts/seed_corrections.py --limit 80

    # CPU에서 DB 배선만 점검 (임시 DB, 실데이터 미오염)
    python scripts/seed_corrections.py --mock --limit 40 --db /tmp/seed_test.db

이후:
    python scripts/export_labels.py                  # 교정분 → 매니페스트
    python scripts/retrain_pipeline.py --check-only   # 트리거 점검
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audit, config  # noqa: E402


def _load_pool(pool_json: Path, holdout_json: Path) -> list[dict]:
    """풀 샘플 목록 [{id, image, gt_type}]. holdout(test)과의 교집합은 금지."""
    def _imgset(p: Path) -> set[str]:
        return {Path(ex["image"]).as_posix() for ex in json.loads(p.read_text(encoding="utf-8"))}

    holdout = _imgset(holdout_json)
    pool = json.loads(pool_json.read_text(encoding="utf-8"))

    items = []
    leaked = []
    for ex in pool:
        img = Path(ex["image"]).as_posix()
        if img in holdout:
            leaked.append(img)
            continue
        gt = _gt_type(ex)
        if gt:
            items.append({"id": ex["id"], "image": img, "gt_type": gt})
    if leaked:
        raise SystemExit(
            f"❌ 누수 차단: 풀({pool_json.name})이 고정 평가셋({holdout_json.name})과 "
            f"{len(leaked)}건 겹칩니다. test셋은 교정 풀에 쓸 수 없습니다."
        )
    return items


def _gt_type(ex: dict) -> str | None:
    """val/test 항목의 assistant 정답 JSON에서 type을 뽑는다. 폴더명으로 보강."""
    for turn in ex.get("conversations", []):
        if turn.get("role") == "assistant":
            try:
                t = (json.loads(turn["content"]).get("type") or "").strip().lower()
                if t in config.DEFECT_CLASSES:
                    return t
            except Exception:
                pass
    folder = Path(ex["image"]).parent.name.lower()
    return folder if folder in config.DEFECT_CLASSES else None


def _mock_predict(gt_type: str, idx: int) -> tuple[str, str, float]:
    """모델 없이 가짜 추론(배선 점검용). 매 4번째 건을 일부러 오답으로 만든다."""
    wrong = idx % 4 == 0
    others = [c for c in config.DEFECT_CLASSES if c != gt_type]
    pred = others[idx % len(others)] if wrong else gt_type
    return pred, "medium", 0.55 if wrong else 0.93


def run(args) -> int:
    # --db 로 임시 DB 지정 시 모듈 전역을 갈아끼워 실데이터를 보호한다.
    if args.db:
        audit.DB_PATH = Path(args.db)
        audit.REVIEW_IMG_DIR = audit.DB_PATH.parent / "images"
    audit.init_db()

    items = _load_pool(args.pool, args.holdout)
    random.Random(args.seed).shuffle(items)
    items = items[: args.limit]
    print(f"풀 {args.pool.name} → 누수검사 통과, {len(items)}건 추론 예정 "
          f"(DB: {audit.DB_PATH})")

    predict = None
    if not args.mock:
        from app import main  # 무거운 import는 실제 실행 때만
        if args.adapter:
            main.LORA_PATH = Path(args.adapter)
            main.USE_LORA = main.LORA_PATH.exists()
        print(f"모델 로드 중… (LoRA={main.USE_LORA}, {main.LORA_PATH})")
        main.load_model()
        from PIL import Image

        def predict(path: str, gt: str, i: int):
            raw = Path(path).read_bytes()
            img = Image.open(Path(path)).convert("RGB")
            parsed, _, _, conf = main.run_inference(img)
            ptype = main._normalize_type(parsed)
            sev = parsed.get("severity") if parsed else None
            return raw, img, ptype, sev, round(conf, 4)

    processed = corrected = auto_ok = 0
    per_class_corr: dict[str, int] = {}
    for i, it in enumerate(items):
        if args.mock:
            raw = Path(it["image"]).read_bytes()
            from PIL import Image
            img = Image.open(Path(it["image"])).convert("RGB")
            ptype, sev, conf = _mock_predict(it["gt_type"], i)
        else:
            raw, img, ptype, sev, conf = predict(it["image"], it["gt_type"], i)

        sha = audit.image_hash(raw)
        review = audit.decide_review(conf, ptype)
        # 교정에 쓸 수 있도록 이미지를 항상 보관한다(앱은 needs_review만 저장).
        audit.save_review_image(sha, img)
        rid = audit.log_inference(
            image_sha256=sha, predicted_type=ptype, severity=sev,
            confidence=conf, latency_ms=0.0,
            model_version="seed:" + ("mock" if args.mock else "current"),
            review_status=review, raw_output="",
        )
        processed += 1
        if ptype != it["gt_type"]:
            audit.submit_correction(rid, it["gt_type"], reviewer=args.reviewer)
            corrected += 1
            per_class_corr[it["gt_type"]] = per_class_corr.get(it["gt_type"], 0) + 1
        else:
            auto_ok += 1

    print(f"\n처리 {processed}건  | 정답(교정불요) {auto_ok}건  | 오답→교정 {corrected}건")
    if per_class_corr:
        print("교정 클래스별:", dict(sorted(per_class_corr.items(), key=lambda x: -x[1])))
    print(f"누적 교정 라벨(corrected_for_training): {len(audit.corrected_for_training())}건")
    print(f"재학습 트리거 임계값: {config.RETRAIN_LABEL_THRESHOLD}건")
    return 0


def main():
    ap = argparse.ArgumentParser(description="교정 라벨 시딩 (자가개선 루프 입력 생성)")
    ap.add_argument("--pool", type=Path, default=ROOT / "data" / "processed" / "val.json",
                    help="유입 이미지 풀 (기본=val.json, 현장 유입 대용)")
    ap.add_argument("--holdout", type=Path, default=ROOT / "data" / "processed" / "test.json",
                    help="절대 교정에 쓰면 안 되는 고정 평가셋 (누수 차단용)")
    ap.add_argument("--limit", type=int, default=80, help="추론할 이미지 수")
    ap.add_argument("--seed", type=int, default=42, help="샘플링 시드")
    ap.add_argument("--reviewer", default="seed-bot", help="교정 입력자 식별자")
    ap.add_argument("--adapter", help="현행 모델 LoRA 어댑터 경로 (기본=app.main 설정)")
    ap.add_argument("--mock", action="store_true", help="모델 없이 가짜 추론(배선 점검)")
    ap.add_argument("--db", help="감사 DB 경로 오버라이드(테스트용 임시 DB)")
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
