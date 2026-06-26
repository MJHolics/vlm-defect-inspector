"""지도 vs 무지도, **같은 데이터·같은 테스트셋** 정면 비교 — 라벨 희소성 곡선.

트랙의 핵심 논지("어떤 상황에 어떤 모델을 왜")의 가장 약한 고리를 메운다.
기존 레포는 지도(NEU)·무지도(MVTec screw)가 *다른 도메인*이라 같은 이미지로
비교할 수 없었다(정직히 그렇게 적어둠). 이 스크립트는 **한 데이터셋(MVTec screw)**
에서 두 패러다임을 같은 테스트셋·같은 지표(image AUROC)로 붙인다.

질문: **"결함 라벨을 N개 줄 때, 지도학습이 무지도를 언제 이기기 시작하나?"**

설계(변수를 '감독 방식' 하나로 고립):
- 두 방법 모두 같은 ImageNet 사전학습 백본(resnet)을 쓴다 → 차이는 *오직 감독*.
- 두 방법 모두 같은 정상(양품) 이미지를 본다(정상은 싸다). 차이는 지도가
  **추가로 N개의 라벨된 결함**을 쓴다는 것뿐 — 무지도는 결함을 0개 본다.
- **무지도**(PatchCore): 정상-only memory bank, kNN 거리. 결함 라벨 미사용 → 수평선.
- **지도**(사전학습 CNN 미세조정): 정상 전부 + 결함 N개로 이진분류. N을 스윕 → 곡선.
- **고정 테스트셋**(seed별 층화분할, 균형). 정직한 분산 추정을 위해 **여러 seed**
  반복(KD 트랙 교훈: 단일-seed 가짜양성 회피), mean±std로 보고.

산출물: 라벨 희소성 곡선 그림 + CSV + **교차점**(지도 평균이 무지도를 넘는 최소 N).

사용:
    python scripts/head_to_head.py --smoke                 # 순수 로직 자기점검(데이터 불요)
    python scripts/head_to_head.py --seeds 5 --epochs 15   # 전체 실측(GPU)
    python scripts/head_to_head.py --seeds 3 --sweep 2,5,10,20  # 빠른 버전
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from scripts import anomaly_detect as ad  # noqa: E402

DATA = ROOT / "data" / "mvtec"
RESULTS = ROOT / "data" / "results"


# ───────────────────────── 순수 로직 (데이터·백본 불요, 테스트 대상) ─────────────────────────

def stratified_split(labels: np.ndarray, n_test_per_class: int, seed: int):
    """라벨 배열을 클래스별 동수로 층화분할 → (test_idx, pool_idx).

    각 클래스(0=정상,1=결함)에서 정확히 n_test_per_class개를 테스트로 뽑아
    **균형 테스트셋**을 만든다(image AUROC·정확도가 깨끗하게 정의됨). 나머지는 pool.

    labels: (N,) 0/1. 반환: (test_idx, pool_idx) 둘 다 정렬된 int64 배열.
    """
    rng = np.random.default_rng(seed)
    test, pool = [], []
    for c in (0, 1):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        if len(idx) < n_test_per_class:
            raise ValueError(f"클래스 {c} 표본 {len(idx)} < 테스트 요청 {n_test_per_class}")
        test.extend(idx[:n_test_per_class].tolist())
        pool.extend(idx[n_test_per_class:].tolist())
    return np.sort(np.array(test, dtype=np.int64)), np.sort(np.array(pool, dtype=np.int64))


def sample_defects(pool_defect_idx: np.ndarray, n: int, seed: int) -> np.ndarray:
    """pool의 결함 인덱스에서 n개를 무작위 추출(라벨 예산). n>=len면 전체."""
    rng = np.random.default_rng(seed + 1000)
    if n >= len(pool_defect_idx):
        return np.array(pool_defect_idx, dtype=np.int64)
    return np.sort(rng.choice(pool_defect_idx, size=n, replace=False))


def find_crossover(n_labels, sup_means, unsup_mean):
    """지도 평균이 무지도 평균을 처음으로 넘는(>=) 최소 라벨 수.

    n_labels: 오름차순 라벨 수 리스트. sup_means: 대응 지도 평균 점수.
    unsup_mean: 무지도 평균(상수). 끝까지 못 넘으면 None.
    """
    for n, s in zip(n_labels, sup_means):
        if s >= unsup_mean:
            return int(n)
    return None


def accuracy_at_threshold(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    """점수>=thr를 결함(1)으로 판정한 정확도."""
    pred = (scores >= thr).astype(int)
    return float((pred == labels).mean())


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC에서 Youden's J(=tpr-fpr) 최대화 임계값. 점수 분리 평가용."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


# ───────────────────────── 지도학습 이진분류기 (사전학습 CNN 미세조정) ─────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _make_transforms(train: bool, size: int = 224):
    import torch
    from torchvision.transforms import v2 as T

    base = [T.Resize((size, size))]
    if train:
        base += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15),
                 T.ColorJitter(brightness=0.2, contrast=0.2)]
    base += [T.ToImage(), T.ToDtype(torch.float32, scale=True),
             T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(base)


class _ImgDS:
    """(path, label) 리스트 → 텐서. train이면 가벼운 증강."""

    def __init__(self, items, train, size=224):
        self.items = items
        self.tf = _make_transforms(train, size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


def _build_classifier(arch):
    import torch.nn as nn
    import torchvision.models as M

    if arch == "resnet18":
        model = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif arch == "resnet50":
        model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError(f"미지원 arch: {arch}")
    return model


def train_supervised(good_paths, defect_paths, arch, epochs, device, seed, size=224):
    """정상 전부 + 결함 N개로 이진분류기 미세조정. 반환: 학습된 model.

    심한 불균형(정상≫결함 N개)을 **클래스 균형 샘플러**(소수 결함을 증강과 함께
    오버샘플)로 다뤄 분류기가 한 클래스로 무너지지 않게 한다. 가중 손실도 함께 건다.
    이건 약화시킨 허수아비가 아니라 실무가가 라벨이 적을 때 실제로 쓰는 강한 셋업 —
    공정한 비교의 전제다.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler

    torch.manual_seed(seed)
    items = [(p, 0) for p in good_paths] + [(p, 1) for p in defect_paths]
    targets = np.array([y for _p, y in items])

    model = _build_classifier(arch).to(device)
    n_good, n_def = len(good_paths), max(len(defect_paths), 1)
    w_good = (n_good + n_def) / (2.0 * n_good)
    w_def = (n_good + n_def) / (2.0 * n_def)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([w_good, w_def], device=device),
                               label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    # 클래스 균형 샘플러: 각 표본 가중 = 1/클래스빈도 → 배치당 정상·결함 ~50:50.
    cls_w = np.array([w_good, w_def])
    sample_w = cls_w[targets]
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=len(items), replacement=True,
                                    generator=g)
    bs = min(32, len(items))
    nw = 4 if device == "cuda" else 0
    loader = DataLoader(_ImgDS(items, True, size), batch_size=bs, sampler=sampler,
                        num_workers=nw, pin_memory=(device == "cuda"))

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
    return model


def supervised_scores(model, paths, device, size=224, batch=64):
    """테스트 경로별 결함확률 P(y=1). 반환: (N,) float."""
    import torch
    import torch.nn.functional as F

    tf = _make_transforms(False, size)
    from PIL import Image

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            imgs = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
            x = torch.stack(imgs).to(device)
            prob = F.softmax(model(x), dim=1)[:, 1]
            out.append(prob.cpu().numpy())
    return np.concatenate(out)


# ───────────────────────── 데이터 로딩 (MVTec screw, good vs defect) ─────────────────────────

def load_pool(category: str):
    """반환: (paths, labels). 정상=train/good + test/good, 결함=test/<defect>."""
    base = DATA / category
    cat = None
    for cand in [base / category, base]:
        if (cand / "train" / "good").is_dir():
            cat = cand
            break
    if cat is None:
        raise SystemExit(f"{base} 에 train/good 없음 — scripts/fetch_mvtec.py 먼저 실행")

    good = sorted((cat / "train" / "good").glob("*.png"))
    good += sorted((cat / "test" / "good").glob("*.png"))
    defect = []
    for d in sorted((cat / "test").iterdir()):
        if d.is_dir() and d.name != "good":
            defect += sorted(d.glob("*.png"))
    paths = [str(p) for p in good] + [str(p) for p in defect]
    labels = np.array([0] * len(good) + [1] * len(defect), dtype=np.int64)
    return paths, labels


# ───────────────────────── 무지도 (PatchCore, 정상-only) ─────────────────────────

def unsupervised_auroc(good_train_paths, test_paths, test_labels, backbone, coreset_ratio):
    """PatchCore를 정상-only로 적합하고 테스트 image AUROC와 점수 반환."""
    det = ad.PatchCore(backbone=backbone, coreset_ratio=coreset_ratio)
    det.fit(good_train_paths)
    scores, _maps, _lat = det.score(test_paths)
    auroc = ad.image_auroc(scores, test_labels)
    return auroc, scores


# ───────────────────────── 실험 루프 ─────────────────────────

def run(seeds, sweep, category, arch, epochs, backbone, coreset_ratio, n_test_per_class):
    import torch
    from sklearn.metrics import roc_auc_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths, labels = load_pool(category)
    n_good = int((labels == 0).sum())
    n_def = int((labels == 1).sum())
    print(f"MVTec '{category}' 풀: 정상 {n_good} · 결함 {n_def} (5종) | device={device}")
    print(f"테스트셋: 클래스별 {n_test_per_class}장(균형 {2*n_test_per_class}) | "
          f"seeds={seeds} | sweep={sweep} | 지도 arch={arch}/{epochs}ep | "
          f"무지도 PatchCore({backbone}, coreset={coreset_ratio})\n")

    rows = []  # 한 행 = (seed, paradigm, n_labels, image_auroc, accuracy)
    for seed in range(seeds):
        t_seed = time.perf_counter()
        test_idx, pool_idx = stratified_split(labels, n_test_per_class, seed)
        test_paths = [paths[i] for i in test_idx]
        test_labels = labels[test_idx]
        pool_labels = labels[pool_idx]
        pool_good = pool_idx[pool_labels == 0]
        pool_def = pool_idx[pool_labels == 1]
        good_train_paths = [paths[i] for i in pool_good]

        # 무지도 — 결함 라벨 0개, 정상 전부.
        u_auroc, u_scores = unsupervised_auroc(
            good_train_paths, test_paths, test_labels, backbone, coreset_ratio)
        u_thr = youden_threshold(u_scores, test_labels)
        u_acc = accuracy_at_threshold(u_scores, test_labels, u_thr)
        rows.append({"seed": seed, "paradigm": "unsupervised", "n_labels": 0,
                     "image_auroc": round(u_auroc, 4), "accuracy": round(u_acc, 4)})
        print(f"[seed {seed}] 무지도 PatchCore: AUROC {u_auroc:.4f} · acc {u_acc:.4f}")

        # 지도 — 정상 전부 + 결함 N개 스윕.
        for n in sweep:
            d_idx = sample_defects(pool_def, n, seed)
            n_eff = int(len(d_idx))
            defect_paths = [paths[i] for i in d_idx]
            model = train_supervised(good_train_paths, defect_paths, arch, epochs,
                                     device, seed)
            s_scores = supervised_scores(model, test_paths, device)
            s_auroc = float(roc_auc_score(test_labels, s_scores))
            s_acc = accuracy_at_threshold(s_scores, test_labels, 0.5)
            rows.append({"seed": seed, "paradigm": "supervised", "n_labels": n_eff,
                         "image_auroc": round(s_auroc, 4), "accuracy": round(s_acc, 4)})
            print(f"           지도 N={n_eff:>2}: AUROC {s_auroc:.4f} · acc {s_acc:.4f}")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        print(f"  (seed {seed} {time.perf_counter()-t_seed:.0f}s)\n")
    return rows


def aggregate(rows, sweep):
    """seed 평균/표준편차 집계 → (unsup_stats, sup_stats_by_n, crossover)."""
    def stats(vals):
        a = np.array(vals, dtype=float)
        return round(float(a.mean()), 4), round(float(a.std()), 4)

    u_auroc = [r["image_auroc"] for r in rows if r["paradigm"] == "unsupervised"]
    u_acc = [r["accuracy"] for r in rows if r["paradigm"] == "unsupervised"]
    unsup = {"image_auroc": stats(u_auroc), "accuracy": stats(u_acc)}

    sup = {}
    for n in sorted({r["n_labels"] for r in rows if r["paradigm"] == "supervised"}):
        a = [r["image_auroc"] for r in rows
             if r["paradigm"] == "supervised" and r["n_labels"] == n]
        ac = [r["accuracy"] for r in rows
              if r["paradigm"] == "supervised" and r["n_labels"] == n]
        sup[n] = {"image_auroc": stats(a), "accuracy": stats(ac)}

    ns = sorted(sup)
    cross = find_crossover(ns, [sup[n]["image_auroc"][0] for n in ns],
                           unsup["image_auroc"][0])
    return unsup, sup, cross


def save_outputs(rows, unsup, sup, cross, category):
    RESULTS.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f"head_to_head_{category}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "paradigm", "n_labels",
                                          "image_auroc", "accuracy"])
        w.writeheader()
        w.writerows(rows)

    # 집계 요약 CSV(그림과 동일 수치).
    sum_path = RESULTS / f"head_to_head_{category}_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paradigm", "n_labels", "auroc_mean", "auroc_std",
                    "acc_mean", "acc_std"])
        w.writerow(["unsupervised", 0, *unsup["image_auroc"], *unsup["accuracy"]])
        for n in sorted(sup):
            w.writerow(["supervised", n, *sup[n]["image_auroc"], *sup[n]["accuracy"]])

    fig_path = _plot(unsup, sup, cross, category)
    return csv_path, sum_path, fig_path


def _plot(unsup, sup, cross, category):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    ns = sorted(sup)
    means = [sup[n]["image_auroc"][0] for n in ns]
    stds = [sup[n]["image_auroc"][1] for n in ns]
    u_m, u_s = unsup["image_auroc"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # 무지도 = 수평 밴드(결함 라벨 0개).
    ax.axhline(u_m, color="#d1495b", lw=2, label=f"Unsupervised PatchCore (0 labels): {u_m:.3f}")
    ax.fill_between([min(ns) * 0.8, max(ns) * 1.2], u_m - u_s, u_m + u_s,
                    color="#d1495b", alpha=0.12)
    # 지도 = 라벨 수에 따른 곡선 + 오차밴드.
    ax.errorbar(ns, means, yerr=stds, marker="o", color="#2e4057", lw=2, capsize=3,
                label="Supervised CNN (N labeled defects)")
    if cross is not None:
        ax.axvline(cross, color="#3a7d44", ls="--", lw=1.5,
                   label=f"Crossover ≈ {cross} labels")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("# labeled defect images given to the supervised model (log)")
    ax.set_ylabel("Image AUROC  (same fixed test set)")
    ax.set_title(f"Supervised vs Unsupervised on the SAME data — MVTec {category}\n"
                 "below the crossover, labels are too few: unsupervised wins")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = RESULTS / f"head_to_head_{category}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ───────────────────────── smoke (순수 로직) ─────────────────────────

def _smoke() -> int:
    labels = np.array([0] * 20 + [1] * 10)
    test_idx, pool_idx = stratified_split(labels, n_test_per_class=5, seed=0)
    assert len(test_idx) == 10 and len(pool_idx) == 20, "분할 크기 오류"
    assert (labels[test_idx] == 0).sum() == 5 and (labels[test_idx] == 1).sum() == 5, \
        "테스트셋 균형 위배"
    assert set(test_idx).isdisjoint(set(pool_idx)), "test/pool 겹침"

    pool_def = pool_idx[labels[pool_idx] == 1]
    s = sample_defects(pool_def, 3, seed=0)
    assert len(s) == 3 and set(s).issubset(set(pool_def)), "결함 표본 오류"
    assert len(sample_defects(pool_def, 999, seed=0)) == len(pool_def), "n>=전체 처리 오류"

    # 교차점: 무지도 0.80, 지도 [0.6,0.75,0.85,0.9] → 처음 넘는 N=10.
    assert find_crossover([2, 5, 10, 20], [0.6, 0.75, 0.85, 0.9], 0.80) == 10
    assert find_crossover([2, 5], [0.6, 0.7], 0.80) is None, "못 넘으면 None"

    sc = np.array([0.1, 0.2, 0.8, 0.9])
    yl = np.array([0, 0, 1, 1])
    thr = youden_threshold(sc, yl)
    assert accuracy_at_threshold(sc, yl, thr) == 1.0, "완전분리 정확도 1.0 기대"
    print("smoke OK — split·sample·crossover·threshold 순수 로직 정상")
    return 0


def main():
    ap = argparse.ArgumentParser(description="지도 vs 무지도 head-to-head (MVTec)")
    ap.add_argument("--smoke", action="store_true", help="순수 로직 자기점검(데이터 불요)")
    ap.add_argument("--category", default="screw")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweep", default="2,5,10,20,40,80",
                    help="쉼표구분 결함 라벨 수(전체보다 크면 전체로 절단)")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--backbone", default="resnet18",
                    help="무지도 PatchCore 백본 — 기본은 지도 arch와 동일(resnet18)로 두어 "
                         "'감독 방식 하나만 다르다'를 통제. 강한 무지도를 보려면 wide_resnet50_2.")
    ap.add_argument("--coreset-ratio", type=float, default=0.10)
    ap.add_argument("--n-test-per-class", type=int, default=40)
    args = ap.parse_args()

    if args.smoke:
        raise SystemExit(_smoke())

    sweep = [int(x) for x in args.sweep.split(",") if x.strip()]
    rows = run(args.seeds, sweep, args.category, args.arch, args.epochs,
               args.backbone, args.coreset_ratio, args.n_test_per_class)
    unsup, sup, cross = aggregate(rows, sweep)
    csv_path, sum_path, fig_path = save_outputs(rows, unsup, sup, cross, args.category)

    print("=" * 64)
    print(f"무지도 PatchCore(0 labels): AUROC {unsup['image_auroc'][0]:.4f} "
          f"± {unsup['image_auroc'][1]:.4f} · acc {unsup['accuracy'][0]:.4f}")
    for n in sorted(sup):
        print(f"지도 N={n:>3}: AUROC {sup[n]['image_auroc'][0]:.4f} "
              f"± {sup[n]['image_auroc'][1]:.4f} · acc {sup[n]['accuracy'][0]:.4f}")
    if cross is not None:
        print(f"\n▶ 교차점: 결함 라벨 약 {cross}개부터 지도가 무지도를 넘어선다.")
        print(f"  → 라벨 {cross}개 미만이면 무지도(PatchCore)가 정답.")
    else:
        print("\n▶ 스윕 범위 내에선 지도가 무지도를 못 넘었다(라벨이 더 필요).")
    print("=" * 64)
    print(f"저장: {csv_path.relative_to(ROOT)} · {sum_path.relative_to(ROOT)} · "
          f"{fig_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
