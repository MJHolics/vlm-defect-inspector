"""정상 사진 몇 장으로 가르치는 이상 표시 — 브라우저에 넣을 만한 크기의 백본으로 되는가.

「손 안 대고 쓰는 검사기」의 체험은 "휴대폰으로 정상 물건을 몇 장 찍어 가르치고, 다른 물건의 이상 부위를
표시한다"이다. 기존 무지도 트랙(`anomaly_detect.py`)은 정상 320장·wide_resnet50_2 기준이라 이 조건의 답이 아니다.
여기서는 조건을 체험 쪽으로 옮겨서 잰다.

- 가르치는 장수 k = 1, 3, 5, 10, 20 (학습 정상 이미지에서 무작위, 시드 3개)
- 백본 = mobilenet_v3_small(가중치 약 10MB) · resnet18(약 45MB). 입력 256, ImageNet 가중치 그대로
- 방법 = 패치 특징 메모리 뱅크 + 최근접 거리(PatchCore 방식, coreset 없음 — k가 작아 뱅크가 작다)
- 지표 = image AUROC · pixel AUROC · 표시 지점 적중(이상 지도의 최댓값이 정답 마스크 안에 드는 비율)
- 비용 = 뱅크 크기(MB, float32) · 이미지 한 장 CPU 시간(백본 + 거리 계산, 1스레드)

사용:
    python scripts/fewshot_teach_bench.py --categories metal_nut screw
산출: reports/fewshot_teach.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
SIZE = 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def category_dir(cat: str) -> Path:
    d = ROOT / "data" / "mvtec" / cat
    return d if (d / "train").exists() else d / cat


def load(path: Path) -> torch.Tensor:
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float().div(255)


def load_mask(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return (np.asarray(im) > 0).astype(np.uint8)


class Backbone(torch.nn.Module):
    """stride 8·16 두 층의 특징을 stride 8 격자로 맞춰 이어 붙인다."""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name == "mobilenet_v3_small":
            m = models.mobilenet_v3_small(weights="IMAGENET1K_V1").features
            self.a, self.b = m[:4], m[4:9]
        elif name == "resnet18":
            m = models.resnet18(weights="IMAGENET1K_V1")
            self.a = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
            self.b = m.layer3
        else:
            raise ValueError(name)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        fa = self.a(x)
        fb = self.b(fa)
        fa = F.avg_pool2d(fa, 3, 1, 1)
        fb = F.avg_pool2d(fb, 3, 1, 1)
        fb = F.interpolate(fb, size=fa.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([fa, fb], 1)                    # (B, D, H, W)
        return f.permute(0, 2, 3, 1)                  # (B, H, W, D)


def n_params_mb(net: Backbone) -> float:
    return sum(p.numel() for p in net.parameters()) * 4 / 1e6


@torch.no_grad()
def features(net: Backbone, imgs: torch.Tensor, device: str, bs: int = 32) -> torch.Tensor:
    out = [net(imgs[i:i + bs].to(device)) for i in range(0, len(imgs), bs)]
    return torch.cat(out)


@torch.no_grad()
def anomaly_maps(bank: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    """feats (N,H,W,D) → 패치별 최근접 거리 (N,H,W)."""
    n, h, w, d = feats.shape
    out = []
    for i in range(n):
        dist = torch.cdist(feats[i].reshape(-1, d), bank)      # (HW, M)
        out.append(dist.min(1).values.reshape(h, w))
    return torch.stack(out)


def upsample(maps: torch.Tensor) -> np.ndarray:
    m = F.interpolate(maps[:, None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    m = F.avg_pool2d(m, 9, 1, 4)                                 # 가벼운 평활
    return m[:, 0].cpu().numpy()


def evaluate(maps: np.ndarray, labels: np.ndarray, masks: np.ndarray) -> dict:
    scores = maps.reshape(len(maps), -1).max(1)
    out = {"image_auroc": float(roc_auc_score(labels, scores)),
           "pixel_auroc": float(roc_auc_score(masks.reshape(-1), maps.reshape(-1)))}
    hits = []
    for m, lab, gt in zip(maps, labels, masks):
        if lab == 1 and gt.any():
            y, x = np.unravel_index(int(m.argmax()), m.shape)
            hits.append(int(gt[y, x]))
    out["pointing"] = float(np.mean(hits))
    out["n_defect"] = len(hits)
    # 기준선: 무작위로 한 점을 찍을 때 기대 적중 = 정답 마스크 면적 비율의 평균
    out["pointing_random"] = float(np.mean([gt.mean() for lab, gt in zip(labels, masks) if lab == 1 and gt.any()]))
    return out


def cpu_latency_ms(net: Backbone, bank: torch.Tensor, img: torch.Tensor, reps: int = 15) -> dict:
    """한 장 처리 시간(CPU 1스레드). 백본과 거리 계산을 나눠 잰다."""
    torch.set_num_threads(1)
    net_c, bank_c, x = net.cpu(), bank.cpu(), img[None]
    with torch.no_grad():
        for _ in range(3):
            f = net_c(x)
        tb, td = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            f = net_c(x)
            t1 = time.perf_counter()
            torch.cdist(f.reshape(-1, f.shape[-1]), bank_c).min(1)
            t2 = time.perf_counter()
            tb.append((t1 - t0) * 1e3)
            td.append((t2 - t1) * 1e3)
    return {"backbone_ms": float(np.median(tb)), "knn_ms": float(np.median(td))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["metal_nut", "screw"])
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--backbones", nargs="+", default=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--out", default=str(ROOT / "reports" / "fewshot_teach.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows, meta = [], {"size": SIZE, "device": device, "backbone_weights_mb": {}}
    for cat in args.categories:
        d = category_dir(cat)
        train_paths = sorted((d / "train" / "good").glob("*.png"))
        test, labels, masks = [], [], []
        for sub in sorted((d / "test").iterdir()):
            for p in sorted(sub.glob("*.png")):
                test.append(load(p))
                if sub.name == "good":
                    labels.append(0)
                    masks.append(np.zeros((SIZE, SIZE), np.uint8))
                else:
                    labels.append(1)
                    masks.append(load_mask(d / "ground_truth" / sub.name / f"{p.stem}_mask.png"))
        test, labels, masks = torch.stack(test), np.array(labels), np.stack(masks)
        print(f"[{cat}] 학습 정상 {len(train_paths)} · 시험 {len(test)}(결함 {int(labels.sum())})")

        for bname in args.backbones:
            net = Backbone(bname).to(device)
            meta["backbone_weights_mb"][bname] = round(n_params_mb(net), 1)
            test_feats = features(net, test, device)
            for k in args.shots:
                for seed in args.seeds:
                    rng = np.random.default_rng(seed)
                    pick = rng.choice(len(train_paths), size=k, replace=False)
                    shots = torch.stack([load(train_paths[i]) for i in pick])
                    bank = features(net, shots, device)
                    bank = bank.reshape(-1, bank.shape[-1])
                    maps = upsample(anomaly_maps(bank, test_feats))
                    r = evaluate(maps, labels, masks)
                    r.update(category=cat, backbone=bname, shots=k, seed=seed,
                             bank_patches=int(bank.shape[0]), bank_dim=int(bank.shape[1]),
                             bank_mb=round(bank.numel() * 4 / 1e6, 2))
                    if seed == args.seeds[0]:
                        r.update(cpu_latency_ms(net, bank, test[0]))
                        net.to(device)
                    rows.append(r)
                    print(f"  {bname:20s} k={k:2d} seed={seed} image {r['image_auroc']:.3f} "
                          f"pixel {r['pixel_auroc']:.3f} 적중 {r['pointing']:.3f} 뱅크 {r['bank_mb']}MB")

    Path(args.out).write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", args.out)

    print("\n요약(시드 평균)")
    keys = sorted({(r["category"], r["backbone"], r["shots"]) for r in rows})
    for cat, b, k in keys:
        g = [r for r in rows if (r["category"], r["backbone"], r["shots"]) == (cat, b, k)]
        lat = next((r for r in g if "backbone_ms" in r), {})
        print(f"{cat:10s} {b:20s} k={k:2d} image {np.mean([r['image_auroc'] for r in g]):.3f} "
              f"pixel {np.mean([r['pixel_auroc'] for r in g]):.3f} "
              f"적중 {np.mean([r['pointing'] for r in g]):.3f}"
              f"(범위 {min(r['pointing'] for r in g):.3f}~{max(r['pointing'] for r in g):.3f}, 무작위 {g[0]['pointing_random']:.3f}) "
              f"뱅크 {g[0]['bank_mb']}MB CPU {lat.get('backbone_ms', 0):.0f}+{lat.get('knn_ms', 0):.0f}ms")


if __name__ == "__main__":
    main()
