"""손으로 들고 찍으면 생기는 어긋남 — 시험 사진을 돌리고 밀었을 때 몇 장 가르치기가 얼마나 무너지는가.

`fewshot_teach_bench.py`는 MVTec의 정렬된 사진 그대로 쟀다. 휴대폰 체험에서는 물건이 매번 조금씩
돌아가고 밀린다. 시험 사진(과 정답 마스크)에 회전·이동을 주고, 가르치는 사진을 그대로 쓴 경우와
가르치는 사진도 같은 범위로 흔들어 늘린 경우(장당 8배)를 비교한다. metal_nut, k=5, 시드 3개.

산출: reports/fewshot_teach_jitter.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fewshot_teach_bench import (ROOT, SIZE, Backbone, anomaly_maps, category_dir, evaluate,  # noqa: E402
                                 features, load, load_mask, upsample)

LEVELS = [(0, 0.0), (5, 0.02), (15, 0.05), (45, 0.10), (180, 0.10)]   # (회전 ±도, 이동 ±비율)
K, SEEDS, AUG = 5, [0, 1, 2], 8


def jitter(img: torch.Tensor, mask, deg: float, shift: float, rng) -> tuple:
    a = float(rng.uniform(-deg, deg))
    t = [int(rng.uniform(-shift, shift) * SIZE), int(rng.uniform(-shift, shift) * SIZE)]
    # 빈 자리는 가장자리 색으로 채운다(검은 모서리가 이상으로 잡히지 않게)
    fill = [float(v) for v in img[:, 0, 0]]
    out = TF.affine(img, a, t, 1.0, 0.0, interpolation=TF.InterpolationMode.BILINEAR, fill=fill)
    if mask is None:
        return out, None
    m = TF.affine(torch.from_numpy(mask)[None], a, t, 1.0, 0.0)[0].numpy()
    return out, m


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = category_dir("metal_nut")
    train_paths = sorted((d / "train" / "good").glob("*.png"))
    test, labels, masks = [], [], []
    for sub in sorted((d / "test").iterdir()):
        for p in sorted(sub.glob("*.png")):
            test.append(load(p))
            labels.append(0 if sub.name == "good" else 1)
            masks.append(np.zeros((SIZE, SIZE), np.uint8) if sub.name == "good"
                         else load_mask(d / "ground_truth" / sub.name / f"{p.stem}_mask.png"))
    labels = np.array(labels)

    rows = []
    for bname in ["mobilenet_v3_small", "resnet18"]:
        net = Backbone(bname).to(device)
        for deg, shift in LEVELS:
            rng_t = np.random.default_rng(100)
            jt = [jitter(im, mk, deg, shift, rng_t) for im, mk in zip(test, masks)]
            timgs, tmasks = torch.stack([j[0] for j in jt]), np.stack([j[1] for j in jt])
            keep = np.array([lab == 0 or mk.any() for lab, mk in zip(labels, tmasks)])   # 결함이 화면 밖으로 나간 장 제외
            tf = features(net, timgs[keep], device)
            for mode in ["그대로", "흔들어 늘림"]:
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    shots = [load(train_paths[i]) for i in rng.choice(len(train_paths), K, replace=False)]
                    if mode == "흔들어 늘림" and deg > 0:
                        shots = [jitter(s, None, deg, shift, rng)[0] for s in shots for _ in range(AUG)]
                    bank = features(net, torch.stack(shots), device)
                    bank = bank.reshape(-1, bank.shape[-1])
                    r = evaluate(upsample(anomaly_maps(bank, tf)), labels[keep], tmasks[keep])
                    r.update(backbone=bname, rot_deg=deg, shift=shift, teach=mode, seed=seed,
                             bank_mb=round(bank.numel() * 4 / 1e6, 2))
                    rows.append(r)
            for mode in ["그대로", "흔들어 늘림"]:
                g = [r for r in rows if (r["backbone"], r["rot_deg"], r["teach"]) == (bname, deg, mode)]
                print(f"{bname:20s} 회전±{deg:3d}° 이동±{shift:.0%} {mode:7s} "
                      f"image {np.mean([r['image_auroc'] for r in g]):.3f} "
                      f"적중 {np.mean([r['pointing'] for r in g]):.3f} "
                      f"(무작위 {g[0]['pointing_random']:.3f}, 결함 {g[0]['n_defect']}장) 뱅크 {g[0]['bank_mb']}MB")
    out = ROOT / "reports" / "fewshot_teach_jitter.json"
    out.write_text(json.dumps({"k": K, "aug": AUG, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out)


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):
        _s.reconfigure(encoding="utf-8")
    main()
