"""VLM teacher → 엣지 CNN 지식증류, 라벨효율 스윕 (GPU 필요).

teacher(Qwen2.5-VL v4)는 폐쇄셋·풀데이터서 student CNN보다 약하다(95.9% vs 99.6%).
따라서 "증류로 정확도 향상"은 정직하지 않다 — student가 이미 이긴다. 이 스크립트는
증류의 값어치를 **라벨이 적을 때**(라벨예산 N/클래스)에서 검증한다:

  같은 서브셋·seed·val·test 에서 `hard-only` vs `hard+KD` 를 N∈{5,10,25,210} 으로
  비교 → 저N서 KD 우위 곡선이 나오는지 실측. (WM-811K 전이의 "저데이터일수록
  전이 큼"과 같은 축. VLM=콜드스타트 교사 / CNN=인라인 처리량 역할분담의 증명.)

KD 손실 = (1-α)·CE(hard, ls=0.1) + α·T²·KL( student/T || teacher_soft/T )
teacher_soft 는 kd_teacher_softlabels.py 가 저장한 클래스별 평균 logprob 을
온도 T 로 softmax 해서 매 배치 구성한다.

사용 (GPU):
    python scripts/kd_teacher_softlabels.py          # 먼저 soft label 캐시
    python scripts/train_edge_kd.py --arch mobilenet_v3_small
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train_edge_cnn as base  # noqa: E402  (모델/평가/분할 로더 재사용)

CLASSES = base.CLASSES
CLS2IDX = base.CLS2IDX
SOFT_DEFAULT = ROOT / "data" / "results" / "kd_teacher_softlabels.json"


def _load_softlabels(path):
    """image(상대경로) → 클래스별 평균 logprob 벡터(np.float32[6])."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    assert d["classes"] == CLASSES, "soft label 클래스 순서 불일치"
    out = {}
    for it in d["items"]:
        out[it["image"]] = np.array([it["mean_logp"][c] for c in CLASSES],
                                     dtype=np.float32)
    return out, d.get("teacher_argmax_accuracy")


def _load_train_with_soft(soft_map):
    """train.json → [(abs_path, y, soft_logp[6]), ...] (상대경로로 soft 매칭)."""
    recs = json.loads((base.SPLIT_DIR / "train.json").read_text(encoding="utf-8"))
    out = []
    for r in recs:
        y = CLS2IDX[json.loads(r["conversations"][1]["content"])["type"]]
        soft = soft_map.get(r["image"])
        if soft is None:
            raise SystemExit(f"soft label 없음: {r['image']} (soft label 먼저 생성)")
        out.append((str(ROOT / r["image"]), y, soft))
    return out


def _subset_per_class(items, n_per_class, seed):
    """클래스별 n_per_class 장 결정적 선택(seed 고정). hard/kd 가 같은 서브셋 사용."""
    rng = np.random.default_rng(seed)
    by_cls = {i: [] for i in range(len(CLASSES))}
    for it in items:
        by_cls[it[1]].append(it)
    chosen = []
    for c, lst in by_cls.items():
        idx = rng.permutation(len(lst))[:n_per_class]
        chosen.extend(lst[j] for j in idx)
    return chosen


class _KDDataset:
    def __init__(self, items, train, img_size=224):
        from torchvision import transforms as T
        self.items = items
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            self.tf = T.Compose([
                T.Grayscale(3), T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomRotation(15), T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(), norm])
        else:
            self.tf = T.Compose([
                T.Grayscale(3), T.Resize((img_size, img_size)), T.ToTensor(), norm])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        import torch
        path, y, soft = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), y, torch.from_numpy(soft)


def _train_one(arch, train_items, va_loader, te_loader, *, mode, alpha, temp,
               epochs, patience, lr, batch_size, img_size, device, seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)
    nw = 4 if device == "cuda" else 0
    tr_loader = DataLoader(_KDDataset(train_items, True, img_size),
                           batch_size=min(batch_size, len(train_items)),
                           shuffle=True, num_workers=nw, pin_memory=(device == "cuda"),
                           drop_last=False)

    model = base._build_model(arch, len(CLASSES), pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc, best_epoch, no_improve = 0.0, -1, 0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y, soft in tr_loader:
            x, y, soft = x.to(device), y.to(device), soft.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            if mode == "kd" and alpha > 0:
                t_soft = F.softmax(soft / temp, dim=1)
                s_logsoft = F.log_softmax(logits / temp, dim=1)
                kd = F.kl_div(s_logsoft, t_soft, reduction="batchmean") * (temp * temp)
                loss = (1 - alpha) * loss + alpha * kd
            loss.backward()
            opt.step()
        sched.step()
        va_acc, _ = base._eval(model, va_loader, device)
        if va_acc > best_acc:
            best_acc, best_epoch, no_improve = va_acc, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    test_acc, per_class = base._eval(model, te_loader, device)
    return {"val_accuracy": round(best_acc, 4), "test_accuracy": round(test_acc, 4),
            "best_epoch": best_epoch, "test_per_class": per_class}


def main():
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser(description="VLM→CNN 증류 라벨효율 스윕")
    ap.add_argument("--arch", default="mobilenet_v3_small",
                    choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--ns", default="5,10,25,210", help="클래스당 라벨 수(콤마)")
    ap.add_argument("--alpha", type=float, default=0.5, help="KD 손실 가중")
    ap.add_argument("--temp", type=float, default=4.0, help="증류 온도 T")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--softlabels", type=Path, default=SOFT_DEFAULT)
    ap.add_argument("--out", type=lambda s: Path(s).resolve(),
                    default=ROOT / "data" / "results" / "kd_label_efficiency.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ns = [int(x) for x in args.ns.split(",")]

    soft_map, teacher_acc = _load_softlabels(args.softlabels)
    train_items = _load_train_with_soft(soft_map)
    va = base._load_split("val")
    te = base._load_split("test")
    # val/test 는 hard 라벨만 필요 → base._DS(2-튜플)로 base._eval 과 맞춘다.
    nw = 4 if device == "cuda" else 0
    va_loader = DataLoader(base._DS(va, False, args.img_size),
                           batch_size=64, shuffle=False, num_workers=nw)
    te_loader = DataLoader(base._DS(te, False, args.img_size),
                           batch_size=64, shuffle=False, num_workers=nw)

    print(f"[setup] arch {args.arch} | teacher argmax acc {teacher_acc} | "
          f"alpha {args.alpha} T {args.temp} | device {device}")
    print(f"[setup] N/class {ns}  (train 풀 {len(train_items)}, val {len(va)}, test {len(te)})")

    runs = []
    t0 = time.time()
    for n in ns:
        subset = _subset_per_class(train_items, n, args.seed)
        for mode in ("hard", "kd"):
            r = _train_one(args.arch, subset, va_loader, te_loader, mode=mode,
                           alpha=args.alpha, temp=args.temp, epochs=args.epochs,
                           patience=args.patience, lr=args.lr,
                           batch_size=args.batch_size, img_size=args.img_size,
                           device=device, seed=args.seed)
            r.update({"n_per_class": n, "n_train": len(subset), "mode": mode})
            runs.append(r)
            print(f"  N={n:3d}/cls  {mode:4s}  val {r['val_accuracy']:.4f}  "
                  f"test {r['test_accuracy']:.4f}  (best ep {r['best_epoch']})")
        # 같은 N 의 hard vs kd 델타
        h = next(x for x in runs if x["n_per_class"] == n and x["mode"] == "hard")
        k = next(x for x in runs if x["n_per_class"] == n and x["mode"] == "kd")
        print(f"           Δtest(kd-hard) = {k['test_accuracy']-h['test_accuracy']:+.4f}")

    out = {
        "arch": args.arch,
        "teacher": "Qwen2.5-VL v4 (cand_v4)",
        "teacher_argmax_accuracy_train": teacher_acc,
        "alpha": args.alpha, "temperature": args.temp,
        "seed": args.seed, "device": device,
        "vlm_v4_test_accuracy": 0.9593,
        "cnn_fulldata_test_accuracy": 0.9963,
        "runs": runs,
        "total_seconds": round(time.time() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {args.out.relative_to(ROOT)}  ({out['total_seconds']}s)")


if __name__ == "__main__":
    main()
