"""브라우저판 판정 규칙 정하기 — 가르친 사진만으로 "이상" 문턱을 어떻게 잡는가.

체험에서는 결함 사진이 없으니 문턱을 정상 몇 장에서 뽑아야 한다. 가르친 사진을 한 장씩 빼고
나머지로 만든 뱅크에 대 본 점수(leave-one-out) K개에서 통계량 하나를 뽑고 여유 배수를 곱한다.
점수는 브라우저와 같은 정의: 32×32 거리 지도(제곱거리)를 3×3 평균한 뒤의 최댓값.

- MVTec AD 15종. **규칙은 dev 5종(이름순 앞 5개)에서 고르고 나머지 10종에서 보고한다.** 분할은 실행 전에 고정.
- k=5, 가르친 사진은 장당 4배(원본 + 회전 ±15°·이동 ±5% 3장), 시드 5개
- 후보: 통계량 {min, median, mean, max} × 배수 {0.6 … 2.0}. 고르는 기준 = dev 평균 (오경보 + 놓침) / 2

    python scripts/fewshot_teach_rule.py --root data/mvtec/_full
산출: reports/fewshot_teach_rule.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fewshot_teach_bench import ROOT, SIZE, Backbone, features, load, load_mask  # noqa: E402
from fewshot_teach_jitter import jitter  # noqa: E402

K, AUG, SEEDS = 5, 3, [0, 1, 2, 3, 4]
STATS = {"min": np.min, "median": np.median, "mean": np.mean, "max": np.max}
MARGINS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
N_DEV = 5


@torch.no_grad()
def web_maps(bank: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    n, h, w, d = feats.shape
    out = []
    for i in range(n):
        dist = torch.cdist(feats[i].reshape(-1, d), bank).pow(2).min(1).values.reshape(1, 1, h, w)
        out.append(F.avg_pool2d(dist, 3, 1, 1, count_include_pad=False)[0, 0])
    return torch.stack(out)


def find_categories(root: Path) -> list[Path]:
    return sorted(p.parent.parent for p in root.rglob("train/good") if p.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "mvtec" / "_full"))
    ap.add_argument("--backbones", nargs="+", default=["mobilenet_v3_small", "resnet18"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cats = find_categories(Path(args.root))
    dev = [c.name for c in cats[:N_DEV]]
    print("dev:", dev, "/ test:", [c.name for c in cats[N_DEV:]])

    runs = []          # 카테고리·백본·시드별: 시험 점수와 LOO 점수 원본을 남긴다
    nets = {b: Backbone(b).to(device) for b in args.backbones}
    for d in cats:
        train_paths = sorted((d / "train" / "good").glob("*.png"))
        test, labels, small = [], [], []
        for sub in sorted((d / "test").iterdir()):
            for p in sorted(sub.glob("*.png")):
                test.append(load(p))
                labels.append(0 if sub.name == "good" else 1)
                m = (np.zeros((SIZE, SIZE), np.uint8) if sub.name == "good"
                     else load_mask(d / "ground_truth" / sub.name / f"{p.stem}_mask.png"))
                small.append(F.max_pool2d(torch.from_numpy(m)[None, None].float(), 8)[0, 0].numpy())
        labels = np.array(labels)
        test = torch.stack(test)
        for bname, net in nets.items():
            tf = features(net, test, device)
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                shots = [load(train_paths[i]) for i in rng.choice(len(train_paths), K, replace=False)]
                groups = [[s] + [jitter(s, None, 15, 0.05, rng)[0] for _ in range(AUG)] for s in shots]
                gf = [features(net, torch.stack(g), device) for g in groups]
                dim = gf[0].shape[-1]
                loo = [float(web_maps(torch.cat([g.reshape(-1, dim) for j, g in enumerate(gf) if j != i]),
                                      gf[i][:1]).max()) for i in range(K)]
                maps = web_maps(torch.cat([g.reshape(-1, dim) for g in gf]), tf).cpu().numpy()
                scores = maps.reshape(len(maps), -1).max(1)
                hits = [int(sm[np.unravel_index(int(m.argmax()), m.shape)] > 0)
                        for m, lab, sm in zip(maps, labels, small) if lab == 1 and sm.any()]
                runs.append({"category": d.name, "split": "dev" if d.name in dev else "test", "backbone": bname,
                             "seed": seed, "loo": loo, "good": scores[labels == 0].tolist(),
                             "defect": scores[labels == 1].tolist(),
                             "image_auroc": float(roc_auc_score(labels, scores)), "pointing": float(np.mean(hits))})
        print(f"[{d.name}] " + " · ".join(
            f"{b} AUROC {np.mean([r['image_auroc'] for r in runs if r['category'] == d.name and r['backbone'] == b]):.3f} "
            f"적중 {np.mean([r['pointing'] for r in runs if r['category'] == d.name and r['backbone'] == b]):.3f}"
            for b in nets))

    def errors(sel: list[dict], stat: str, mg: float) -> tuple[float, float]:
        fa, miss = [], []
        for r in sel:
            th = STATS[stat](r["loo"]) * mg
            fa.append(np.mean(np.array(r["good"]) > th))
            miss.append(np.mean(np.array(r["defect"]) <= th))
        return float(np.mean(fa)), float(np.mean(miss))

    summary = {}
    for b in nets:
        dev_runs = [r for r in runs if r["backbone"] == b and r["split"] == "dev"]
        test_runs = [r for r in runs if r["backbone"] == b and r["split"] == "test"]
        grid = {(s, m): errors(dev_runs, s, m) for s in STATS for m in MARGINS}
        (stat, mg), (dfa, dmiss) = min(grid.items(), key=lambda kv: sum(kv[1]))
        tfa, tmiss = errors(test_runs, stat, mg)
        per_cat = {}
        for c in sorted({r["category"] for r in test_runs}):
            sel = [r for r in test_runs if r["category"] == c]
            fa, miss = errors(sel, stat, mg)
            per_cat[c] = {"false_alarm": round(fa, 3), "miss": round(miss, 3),
                          "image_auroc": round(float(np.mean([r["image_auroc"] for r in sel])), 3),
                          "pointing": round(float(np.mean([r["pointing"] for r in sel])), 3)}
        summary[b] = {"rule": {"stat": stat, "margin": mg}, "dev": {"false_alarm": dfa, "miss": dmiss},
                      "test": {"false_alarm": tfa, "miss": tmiss,
                               "image_auroc": float(np.mean([r["image_auroc"] for r in test_runs])),
                               "pointing": float(np.mean([r["pointing"] for r in test_runs]))},
                      "test_max_loo_x1.0": dict(zip(["false_alarm", "miss"], errors(test_runs, "max", 1.0))),
                      "per_category_test": per_cat}
        print(f"\n{b}: 고른 규칙 = LOO {stat} × {mg} | dev 오경보 {dfa:.3f} 놓침 {dmiss:.3f} | "
              f"test 오경보 {tfa:.3f} 놓침 {tmiss:.3f} (AUROC {summary[b]['test']['image_auroc']:.3f}, "
              f"적중 {summary[b]['test']['pointing']:.3f})")
        for c, v in per_cat.items():
            print(f"   {c:12s} 오경보 {v['false_alarm']:.3f} 놓침 {v['miss']:.3f} AUROC {v['image_auroc']:.3f} 적중 {v['pointing']:.3f}")

    out = ROOT / "reports" / "fewshot_teach_rule.json"
    out.write_text(json.dumps({"k": K, "aug": AUG, "seeds": SEEDS, "dev": dev, "summary": summary, "runs": runs},
                              ensure_ascii=False), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8")
    main()
