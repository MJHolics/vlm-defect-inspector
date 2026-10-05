"""불량 한 장을 알려 주면 문턱이 얼마나 나아지는가 — fewshot_teach_rule.py가 남긴 점수로 다시 계산.

정상 5장만으로 잡은 문턱은 물건마다 크게 빗나간다(test 오경보 0.25·놓침 0.23). 체험에서는 사용자가
"이건 불량"이라고 한 장을 알려 줄 수 있다. 그 한 장의 점수 d와 가르친 사진의 LOO 통계량 n 사이에
문턱을 둔다: th = n^(1-a) · d^a. 알려 준 한 장은 평가에서 뺀다. 불량 한 장은 run마다 20번 무작위로 뽑는다.
통계량과 a는 dev 5종에서 고르고 test 10종에서 보고한다.

산출: reports/fewshot_teach_feedback.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATS = {"min": np.min, "median": np.median, "max": np.max}
ALPHAS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
DRAWS = 20


def errors(runs: list[dict], stat: str, a: float) -> tuple[float, float]:
    fa, miss = [], []
    for r in runs:
        good, defect = np.array(r["good"]), np.array(r["defect"])
        rng = np.random.default_rng(r["seed"])
        n = STATS[stat](r["loo"])
        for i in rng.choice(len(defect), size=min(DRAWS, len(defect)), replace=False):
            d = max(defect[i], n)                       # 알려 준 불량이 정상보다 낮게 나오면 정상 쪽 값을 쓴다
            th = n ** (1 - a) * d ** a
            rest = np.delete(defect, i)
            fa.append(np.mean(good > th))
            miss.append(np.mean(rest <= th))
    return float(np.mean(fa)), float(np.mean(miss))


def main() -> None:
    data = json.loads((ROOT / "reports" / "fewshot_teach_rule.json").read_text(encoding="utf-8"))
    out = {}
    for b in sorted({r["backbone"] for r in data["runs"]}):
        dev = [r for r in data["runs"] if r["backbone"] == b and r["split"] == "dev"]
        test = [r for r in data["runs"] if r["backbone"] == b and r["split"] == "test"]
        grid = {(s, a): errors(dev, s, a) for s in STATS for a in ALPHAS}
        (stat, a), (dfa, dmiss) = min(grid.items(), key=lambda kv: sum(kv[1]))
        tfa, tmiss = errors(test, stat, a)
        base = data["summary"][b]["test"]
        per = {}
        for c in sorted({r["category"] for r in test}):
            fa, miss = errors([r for r in test if r["category"] == c], stat, a)
            per[c] = {"false_alarm": round(fa, 3), "miss": round(miss, 3)}
        out[b] = {"rule": {"stat": stat, "alpha": a}, "dev": {"false_alarm": dfa, "miss": dmiss},
                  "test": {"false_alarm": tfa, "miss": tmiss},
                  "test_normals_only": {"false_alarm": base["false_alarm"], "miss": base["miss"]},
                  "per_category_test": per}
        print(f"{b}: LOO {stat}, a={a} | dev 오경보 {dfa:.3f} 놓침 {dmiss:.3f} | test 오경보 {tfa:.3f} 놓침 {tmiss:.3f} "
              f"(정상만: {base['false_alarm']:.3f} · {base['miss']:.3f})")
        for c, v in per.items():
            print(f"   {c:12s} 오경보 {v['false_alarm']:.3f} 놓침 {v['miss']:.3f}")
    (ROOT / "reports" / "fewshot_teach_feedback.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8")
    main()
