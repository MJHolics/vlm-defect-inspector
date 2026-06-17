"""자가개선 루프 오케스트레이터 (end-to-end).

사람 검토 → 라벨 축적 → 재학습 → 수용기준 평가 → 안전 승격 → 레지스트리 기록을
한 번의 실행으로 묶는다. 각 단계의 결정 근거는 docs/decisions.md (D1, D2, D4, D8).

흐름:
  1. 트리거 판정 (D1)        — 교정 라벨 ≥ 임계값 OR 드리프트 alert  [scripts/retrain_trigger]
  2. 라벨 추출 (D2)          — 교정분 → 재학습 매니페스트            [scripts/export_labels]
  3. ▶ 재학습 + 평가 (GPU)   — base에서 LoRA 재학습(D7) 후 고정 평가셋 추론 → 후보 CSV
  4. 수용기준 평가           — 비용가중 위험점수로 후보 메트릭 산출   [scripts/acceptance_eval]
  5. 안전 승격 게이트 (D4)   — 위험점수 ≤ 현행 AND 유형정확도 비퇴보면 교체 [app/registry]
  6. 기준 시각 갱신 (D1)     — 다음 트리거를 위해 last_retrain_at 갱신

3단계(학습+추론)는 GPU가 필요하므로 노트북(03_finetune / 05_experiments)이 담당한다.
이 스크립트는 노트북이 만들어 둔 산출물(어댑터 + 평가 CSV)을 받아 나머지 전 단계를
실제로 실행한다. --candidate-csv 를 주면 3단계를 건너뛰고 이어서 진행(resume)한다.

사용:
    # 트리거만 확인 (GPU 없이)
    python scripts/retrain_pipeline.py --check-only

    # 노트북에서 학습/추론을 끝낸 뒤, 후보 산출물로 루프 이어서 실행
    python scripts/retrain_pipeline.py \
        --candidate-version v2 \
        --adapter-path models/checkpoints/cand_v4 \
        --candidate-csv data/results/exp_best_eval_results.csv

    # 트리거 미충족이어도 강제로 진행
    python scripts/retrain_pipeline.py --force --candidate-version v2 \
        --adapter-path <p> --candidate-csv <c>
"""
import argparse
import sys
from pathlib import Path

# 윈도우 콘솔(cp949)은 em-dash(—) 등 일부 유니코드를 인코딩하지 못해 print에서
# UnicodeEncodeError로 죽는다. 출력 스트림을 UTF-8로 고정해 어디서든 안전하게 찍는다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audit, registry  # noqa: E402
from scripts import acceptance_eval, export_labels, retrain_trigger  # noqa: E402


def _hr(title: str) -> None:
    print(f"\n{'─' * 60}\n▶ {title}\n{'─' * 60}")


def run(args) -> int:
    audit.init_db()

    # ── 1. 트리거 판정 (D1) ────────────────────────────────
    _hr("1. 재학습 트리거 판정 (D1)")
    trig = retrain_trigger.check()
    print(f"   새 교정 라벨: {trig['new_labels_since_last']} (임계값 {trig['threshold']})")
    print(f"   드리프트: {trig['drift_status']}")
    for r in trig["reasons"]:
        print(f"   → {r}")
    if not trig["should_retrain"] and not args.force:
        print("\n재학습 불필요. (--force 로 강제 진행 가능)")
        return 0
    if not trig["should_retrain"]:
        print("\n⚠ 트리거 미충족이지만 --force 로 진행합니다.")
    if args.check_only:
        print("\n--check-only: 트리거가 켜졌습니다. 여기서 종료.")
        return 0

    # ── 2. 라벨 추출 (D2) ──────────────────────────────────
    _hr("2. 교정 라벨 추출 → 재학습 매니페스트 (D2)")
    manifest = args.manifest or (ROOT / "data" / "processed" / "retrain_manifest.jsonl")
    sys.argv = ["export_labels", "--out", str(manifest)]
    try:
        export_labels.main()
    except SystemExit:
        pass
    data_ref = args.data_ref or manifest.name
    print(f"   data_ref: {data_ref}")

    # ── 3. 재학습 + 고정 평가셋 추론 (GPU) ─────────────────
    _hr("3. LoRA 재학습 + 평가 (GPU — 노트북 담당)")
    if not args.candidate_csv:
        print(
            "   후보 산출물(--candidate-csv)이 없습니다. 이 단계는 GPU가 필요합니다.\n"
            "   다음을 실행해 어댑터와 고정 평가셋 추론 CSV를 만든 뒤 이 명령을 재실행하세요:\n"
            f"     1) notebooks/03_finetune.ipynb (또는 05_experiments.ipynb) 실행\n"
            f"        → 매니페스트({manifest.relative_to(ROOT)})를 원본 train에 합쳐 재학습\n"
            f"        → 어댑터 저장 (예: models/checkpoints/cand_v4)\n"
            f"     2) 04_evaluation 방식으로 고정 test 셋 추론 → CSV 저장\n"
            f"        (컬럼: gt_type, gt_severity, pred_type, pred_severity)\n"
            f"     3) python scripts/retrain_pipeline.py --candidate-version <v> \\\n"
            f"          --adapter-path <어댑터경로> --candidate-csv <CSV경로>"
        )
        return 2
    candidate_csv = Path(args.candidate_csv)
    if not candidate_csv.exists():
        print(f"   후보 CSV를 찾을 수 없습니다: {candidate_csv}")
        return 2
    print(f"   후보 산출물 사용 (학습 단계 건너뜀): {candidate_csv}")
    if not args.adapter_path:
        print("   ⚠ --adapter-path 미지정 — 레지스트리에 경로가 비어 기록됩니다.")

    # ── 4. 수용기준 평가 ───────────────────────────────────
    _hr("4. 후보 모델 수용기준 평가")
    cand_metrics = acceptance_eval.evaluate(candidate_csv)
    print(f"   유형정확도: {cand_metrics['type_accuracy']:.1%}  "
          f"위험점수: {cand_metrics['risk_score']}  판정: {cand_metrics['verdict']}")
    if cand_metrics["verdict"] != "PASS":
        print("   후보가 자체 수용기준(PASS)을 통과하지 못했습니다 — 승격 후보로 부적격.")
        if not args.force:
            print("   중단. (--force 로 게이트 평가까지 강행 가능)")
            return 1

    # ── 5. 안전 승격 게이트 (D4) ───────────────────────────
    _hr("5. 안전 승격 게이트 (D4)")
    result = registry.register_and_maybe_promote(
        version=args.candidate_version,
        metrics={
            "type_accuracy": cand_metrics["type_accuracy"],
            "risk_score": cand_metrics["risk_score"],
            "severity_accuracy": cand_metrics["severity_accuracy"],
            "n_samples": cand_metrics["n_samples"],
            "verdict": cand_metrics["verdict"],
        },
        data_ref=data_ref,
        adapter_path=args.adapter_path or "",
    )
    for r in result["reasons"]:
        print(f"   {r}")
    print(f"\n   결과: {'✅ 승격(promoted)' if result['promoted'] else '⛔ 거부(rejected)'}"
          f"  (이전 운영: {result['previous']})")
    print(f"   레지스트리: {registry.REGISTRY_PATH.relative_to(ROOT)}")

    # ── 6. 기준 시각 갱신 (D1) ─────────────────────────────
    if args.mark:
        _hr("6. 재학습 기준 시각 갱신 (D1)")
        sys.argv = ["retrain_trigger", "--mark"]
        retrain_trigger.main()

    print("\n루프 1회 완료.")
    return 0 if result["promoted"] else 1


def main():
    ap = argparse.ArgumentParser(description="자가개선 루프 오케스트레이터")
    ap.add_argument("--candidate-version", default="candidate",
                    help="후보 모델 버전 식별자 (예: v2, 2026-06-15-r1)")
    ap.add_argument("--adapter-path", help="후보 LoRA 어댑터 경로 (레지스트리 기록용)")
    ap.add_argument("--candidate-csv", help="후보 모델의 고정 평가셋 추론 CSV (3단계 산출물)")
    ap.add_argument("--manifest", type=Path, help="재학습 매니페스트 출력 경로")
    ap.add_argument("--data-ref", help="학습 데이터 스냅샷 식별자 (D8, 기본=매니페스트 파일명)")
    ap.add_argument("--force", action="store_true", help="트리거/수용 미충족이어도 진행")
    ap.add_argument("--check-only", action="store_true", help="트리거 판정까지만 하고 종료")
    ap.add_argument("--no-mark", dest="mark", action="store_false",
                    help="완료 후 재학습 기준 시각을 갱신하지 않음")
    ap.set_defaults(mark=True)
    args = ap.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
