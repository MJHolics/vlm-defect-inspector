"""컴팩트 CNN 웨이퍼맵 결함분류 + 전이 입증 — 반도체 도메인 전이 트랙.

같은 ResNet18을 (a) ImageNet 사전학습 가중치 vs (b) 무작위 초기화로 학습해
**사전학습 전이의 가치**를 라벨 양(5/10/25/100%)별 데이터효율 곡선으로 입증한다.
또한 단건 추론 latency를 측정해 VLM(14.7s) 대비 컴팩트 CNN의 인라인 적합성을
실측으로 보여준다(README 아키텍처 판단의 정량 근거).

웨이퍼맵 픽셀 {0,1,2} → [0,1] 정규화 후 3채널 복제 + ImageNet 정규화.

사용 (GPU):
    python scripts/train_wafer_cnn.py --sweep        # 전체 매트릭스 + 결과 JSON
    python scripts/train_wafer_cnn.py --frac 1.0 --pretrained --epochs 20  # 단일 실행
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WM = ROOT / "data" / "wm811k"
OUT = ROOT / "data" / "results" / "wafer_transfer.json"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(X):
    """(N,S,S) uint8{0,1,2} → (N,3,S,S) float32 ImageNet 정규화."""
    import torch
    x = X.astype(np.float32) / 2.0                      # [0,1]
    x = np.repeat(x[:, None, :, :], 3, axis=1)          # 3채널
    x = (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    return torch.from_numpy(x)


def _subsample(y, frac, seed=42):
    """클래스 층화 부분추출 인덱스."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        n = max(1, int(round(len(ci) * frac)))
        idx.append(rng.choice(ci, size=n, replace=False))
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


def run_one(data, frac, pretrained, epochs, bs, device, lr=1e-3):
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import ResNet18_Weights, resnet18

    Xtr, ytr, Xte, yte, classes = data
    torch.manual_seed(42)
    np.random.seed(42)

    sel = _subsample(ytr, frac)
    xtr_t = _to_tensor(Xtr[sel])
    ytr_t = torch.from_numpy(ytr[sel])
    xte_t = _to_tensor(Xte)
    yte_t = torch.from_numpy(yte)

    tl = DataLoader(TensorDataset(xtr_t, ytr_t), batch_size=bs, shuffle=True, drop_last=False)

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    # 평가
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(xte_t), bs):
            xb = xte_t[i:i + bs].to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    yp = np.concatenate(preds)
    acc = accuracy_score(yte, yp)
    mf1 = f1_score(yte, yp, average="macro")

    tag = f"frac={frac:<4} init={'pretrained' if pretrained else 'scratch':<10}"
    print(f"  [{tag}] n_train={len(sel):>6,}  acc={acc:.4f}  macroF1={mf1:.4f}")
    return {
        "label_frac": frac,
        "init": "pretrained" if pretrained else "scratch",
        "n_train": int(len(sel)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(mf1), 4),
    }, model, xte_t


def measure_latency(model, xte_t, device, n=200, warmup=10):
    import torch
    model.eval()
    n = min(n, len(xte_t))
    lat = []
    with torch.no_grad():
        for i in range(warmup):
            _ = model(xte_t[i:i + 1].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        for i in range(n):
            x = xte_t[i:i + 1].to(device)
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat.append(time.perf_counter() - t0)
    lat = sorted(lat)
    mean = sum(lat) / len(lat)
    return {
        "batch1_latency_ms": {
            "mean": round(mean * 1000, 3),
            "p50": round(lat[len(lat) // 2] * 1000, 3),
            "p99": round(lat[min(len(lat) - 1, int(0.99 * len(lat)))] * 1000, 3),
        },
        "throughput_img_per_sec": round(1.0 / mean, 1),
        "n_samples": n,
    }


def main():
    ap = argparse.ArgumentParser(description="웨이퍼맵 CNN 전이 실험")
    ap.add_argument("--npz", type=Path, default=WM / "wafer_prepped.npz")
    ap.add_argument("--sweep", action="store_true",
                    help="frac×init 전체 매트릭스 실행 + 결과 JSON 저장")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.npz.exists():
        raise SystemExit(f"전처리 npz 없음: {args.npz} — 먼저 scripts/prep_wm811k.py 실행")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    npz = np.load(args.npz, allow_pickle=True)
    classes = [str(c) for c in npz["classes"]]
    data = (npz["X_train"], npz["y_train"], npz["X_test"], npz["y_test"], classes)
    print(f"클래스({len(classes)}): {classes}")
    print(f"train {len(npz['X_train']):,} / test {len(npz['X_test']):,}")

    if not args.sweep:
        res, _, _ = run_one(data, args.frac, args.pretrained, args.epochs,
                            args.bs, device)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    fracs = [0.05, 0.10, 0.25, 1.0]
    runs = []
    full_model = full_xte = None
    for frac in fracs:
        for pre in (True, False):
            res, model, xte_t = run_one(data, frac, pre, args.epochs, args.bs, device)
            runs.append(res)
            if pre and frac == 1.0:
                full_model, full_xte = model, xte_t

    latency = measure_latency(full_model, full_xte, device)
    print(f"\n단건 latency(pretrained,100%): "
          f"평균 {latency['batch1_latency_ms']['mean']}ms, "
          f"{latency['throughput_img_per_sec']} img/s")

    report = {
        "dataset": "WM-811K (MIR-WM811K)",
        "task": f"{len(classes)}-class 웨이퍼맵 결함패턴 분류",
        "classes": classes,
        "backbone": "ResNet18",
        "image_size": int(npz["X_train"].shape[1]),
        "epochs": args.epochs,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "data_efficiency": runs,
        "inference": latency,
        "vlm_latency_sec_ref": 14.681,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
