"""엣지용 경량 CNN 결함분류기 학습 — 엣지 배포·경량화 트랙.

플래그십 Qwen2.5-VL(7B, QLoRA)은 유형정확도 98.9%(v7)지만 1장에 14.7초·6GB라
인라인 검사에는 못 올린다(scripts/benchmark_latency.py 실측). 이 스크립트는
**같은 NEU 데이터·같은 분할**(data/processed)로 컴팩트 CNN을 학습해, 라벨이
쌓인 뒤 라인에 올릴 '엣지 모델'을 만든다. VLM은 신규결함·콜드스타트(OOD)용
유연한 교사, CNN은 인라인 처리량용 — 두 모델의 역할 분담을 실측으로 보인다.

test는 VLM과 동일한 270건(누수 0). 학습은 val 기준 early-stopping.

사용 (GPU):
    python scripts/train_edge_cnn.py --arch resnet18 --epochs 30
    python scripts/train_edge_cnn.py --arch mobilenet_v3_small --epochs 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config  # noqa: E402

SPLIT_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "models" / "checkpoints" / "edge_cnn"
CLASSES = config.DEFECT_CLASSES
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


def _load_split(name):
    """data/processed/{name}.json → [(abs_image_path, label_idx), ...]."""
    recs = json.loads((SPLIT_DIR / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for r in recs:
        label = json.loads(r["conversations"][1]["content"])["type"]
        out.append((str(ROOT / r["image"]), CLS2IDX[label]))
    return out


def _build_model(arch, num_classes, pretrained=True):
    import torch.nn as nn
    import torchvision.models as M

    if arch == "resnet18":
        w = M.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = M.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        w = M.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = M.mobilenet_v3_small(weights=w)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"미지원 arch: {arch}")
    return m


class _DS:
    """경로 리스트를 PIL→Tensor로 로딩. train이면 가벼운 증강."""

    def __init__(self, items, train, img_size=224):
        from torchvision import transforms as T

        self.items = items
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            self.tf = T.Compose([
                T.Grayscale(num_output_channels=3),
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(), norm,
            ])
        else:
            self.tf = T.Compose([
                T.Grayscale(num_output_channels=3),
                T.Resize((img_size, img_size)),
                T.ToTensor(), norm,
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), y


def _eval(model, loader, device):
    import torch

    model.eval()
    correct = 0
    n = 0
    per_class_tot = np.zeros(len(CLASSES), dtype=int)
    per_class_ok = np.zeros(len(CLASSES), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            y = y.numpy()
            correct += int((pred == y).sum())
            n += len(y)
            for gt, pr in zip(y, pred):
                per_class_tot[gt] += 1
                if gt == pr:
                    per_class_ok[gt] += 1
    acc = correct / n if n else 0.0
    per_class = {CLASSES[i]: round(float(per_class_ok[i]) / int(per_class_tot[i]), 4)
                 for i in range(len(CLASSES)) if per_class_tot[i]}
    return acc, per_class


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="엣지용 경량 CNN 결함분류기 학습")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6, help="val 미개선 허용 epoch")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr = _load_split("train")
    va = _load_split("val")
    te = _load_split("test")
    print(f"[data] train {len(tr)} | val {len(va)} | test {len(te)} | classes {len(CLASSES)}")

    nw = 4 if device == "cuda" else 0
    tr_loader = DataLoader(_DS(tr, True, args.img_size), batch_size=args.batch_size,
                           shuffle=True, num_workers=nw, pin_memory=(device == "cuda"))
    va_loader = DataLoader(_DS(va, False, args.img_size), batch_size=64,
                           shuffle=False, num_workers=nw)
    te_loader = DataLoader(_DS(te, False, args.img_size), batch_size=64,
                           shuffle=False, num_workers=nw)

    model = _build_model(args.arch, len(CLASSES), not args.no_pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CKPT_DIR / f"{args.arch}.pt"
    best_acc = 0.0
    best_epoch = -1
    no_improve = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(y)
        sched.step()
        va_acc, _ = _eval(model, va_loader, device)
        print(f"  epoch {ep:2d} | train_loss {tot_loss/len(tr):.4f} | val_acc {va_acc:.4f}"
              + ("  *best*" if va_acc > best_acc else ""))
        if va_acc > best_acc:
            best_acc = va_acc
            best_epoch = ep
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early-stop (val {args.patience}회 미개선)")
                break

    # best 체크포인트로 test 1회 평가
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, per_class = _eval(model, te_loader, device)
    size_mb = best_path.stat().st_size / 1e6
    train_sec = round(time.time() - t0, 1)

    print("\n" + "=" * 56)
    print(f" {args.arch}  best epoch {best_epoch} | val {best_acc:.4f}")
    print(f" test 유형정확도 : {test_acc:.4f}  (VLM v4 0.9593)")
    print(f" 파라미터        : {n_params/1e6:.2f}M | fp32 크기 {size_mb:.1f}MB")
    print(" 클래스별:", per_class)
    print("=" * 56)

    out = ROOT / "data" / "results" / f"edge_cnn_{args.arch}.json"
    out.write_text(json.dumps({
        "arch": args.arch,
        "pretrained": not args.no_pretrained,
        "img_size": args.img_size,
        "n_params_M": round(n_params / 1e6, 3),
        "fp32_size_mb": round(size_mb, 2),
        "best_epoch": best_epoch,
        "val_accuracy": round(best_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_per_class": per_class,
        "vlm_v4_test_accuracy": 0.9593,
        "train_seconds": train_sec,
        "device": device,
        "checkpoint": str(best_path.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"결과 저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
