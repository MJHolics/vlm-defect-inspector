"""KD 라벨효율 스윕 다중 seed 집계 → 평균±표준편차 (단일 seed 노이즈 제거).

train_edge_kd.py 를 seed 별로 돌린 결과 JSON 들을 모아 (N, mode)별 test 정확도의
평균·표준편차와 KD−hard 델타를 구한다. 단일 seed 곡선은 early-stop 변동으로
노이즈가 커서(예: N=210 hard 한 seed가 ep4에 멈춰 97.8%로 추락) 신호/노이즈를
가르려면 여러 seed 평균이 필요하다.

사용:
    python scripts/kd_aggregate.py data/results/kd_label_efficiency.json \
        data/results/kd_label_efficiency_seed1.json \
        data/results/kd_label_efficiency_seed2.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "results" / "kd_label_efficiency_agg.json"


def main():
    files = [Path(a) for a in sys.argv[1:]]
    if not files:
        files = sorted((ROOT / "data" / "results").glob("kd_label_efficiency*.json"))
        files = [f for f in files if "agg" not in f.name]
    if not files:
        raise SystemExit("집계할 seed JSON이 없습니다")

    acc = defaultdict(list)   # (n, mode) -> [test_acc per seed]
    seeds, meta = [], None
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        seeds.append(d.get("seed"))
        meta = meta or d
        for r in d["runs"]:
            acc[(r["n_per_class"], r["mode"])].append(r["test_accuracy"])

    ns = sorted({n for (n, _) in acc})
    rows = []
    print(f"seeds {seeds} | alpha {meta['alpha']} T {meta['temperature']} | "
          f"teacher {meta['teacher_argmax_accuracy_train']} | arch {meta['arch']}")
    print(f"{'N/cls':>6} | {'hard mean±sd':>16} | {'kd mean±sd':>16} | {'Δ(kd-hard)':>12}")
    for n in ns:
        h, k = np.array(acc[(n, "hard")]), np.array(acc[(n, "kd")])
        delta = k.mean() - h.mean()
        rows.append({
            "n_per_class": n, "n_seeds": len(h),
            "hard_mean": round(float(h.mean()), 4), "hard_std": round(float(h.std()), 4),
            "kd_mean": round(float(k.mean()), 4), "kd_std": round(float(k.std()), 4),
            "delta_mean": round(float(delta), 4),
            "hard_runs": [round(x, 4) for x in h.tolist()],
            "kd_runs": [round(x, 4) for x in k.tolist()],
        })
        print(f"{n:>6} | {h.mean():.4f}±{h.std():.4f} | {k.mean():.4f}±{k.std():.4f} | "
              f"{delta:+.4f}")

    OUT.write_text(json.dumps({
        "arch": meta["arch"],
        "teacher": meta["teacher"],
        "teacher_argmax_accuracy_train": meta["teacher_argmax_accuracy_train"],
        "alpha": meta["alpha"], "temperature": meta["temperature"],
        "seeds": seeds, "n_seeds": len(seeds),
        "vlm_v4_test_accuracy": meta["vlm_v4_test_accuracy"],
        "cnn_fulldata_test_accuracy": meta["cnn_fulldata_test_accuracy"],
        "by_n": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n집계 저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
