"""능동 학습·자동 라벨링·라벨 오류 탐지 — 웨이퍼맵(WM-811K)에서 라벨 예산을 어디에 쓰나.

반도체 전이 트랙(train_wafer_cnn.py)은 "라벨이 적을수록 사전학습이 이득"을 보였지만, 라벨을
**어떤 샘플에** 붙일지는 층화 무작위로 고정했었다. 현장에서는 층화가 불가능하다(라벨을 붙이기
전엔 클래스를 모른다). 그래서 세 질문을 같은 데이터·같은 모델로 잰다.

1. 능동 학습(--part al): 무작위 100장에서 시작해 100장씩 1,500장까지 늘릴 때,
   random / entropy / margin / coreset(k-center) 중 무엇이 같은 라벨 수로 test macro-F1을
   더 올리나. 희소 클래스(Near-full 119장, Donut 444장)를 언제 찾아내나.
2. 자동 라벨링(--part auto): 라벨 1,000장 모델의 확신 높은 예측을 나머지 풀에 붙일 때,
   임계 τ는 라벨 있는 val에서만 고르고(val 정밀도 ≥ 0.98) 풀에서 실제 정밀도·커버리지를 잰다.
   붙인 라벨을 더해 재학습하면 test가 나아지나.
3. 라벨 오류 탐지(--part noise): 1,000장 중 10%의 라벨을 다른 클래스로 뒤집고, 5-fold
   out-of-fold 확률로 "주어진 라벨의 확신이 낮은" 100장을 사람 검토 대상으로 뽑는다.
   뽑은 것 중 실제 오류 비율, 고친 뒤 재학습 효과를 잰다.

모델은 기존 트랙과 같은 ResNet18(ImageNet 사전학습, 64x64)이고 매 라운드 처음부터 다시 학습한다.
test 5,104장은 보고에만 쓴다(전략·임계 선택에 쓰지 않음). 결과는 실행 단위마다 즉시 저장한다.

사용 (GPU):
    python scripts/active_learning_wafer.py --part al --seeds 0 1 2
    python scripts/active_learning_wafer.py --part auto --seeds 0 1 2
    python scripts/active_learning_wafer.py --part noise --seeds 0 1 2
    python scripts/active_learning_wafer.py --part al --smoke   # 2라운드·1에폭 배선 확인
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WM = ROOT / "data" / "wm811k" / "wafer_prepped.npz"
OUT = ROOT / "data" / "results" / "active_learning_wafer.json"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

N_VAL = 1000          # 라벨 있는 검증셋(임계·전략 판단용). 풀에서 미리 떼어 둔다.
INIT = 100
STEP = 100
BUDGET = 1500
STRATEGIES = ["random", "entropy", "margin", "coreset"]


# ---------------------------------------------------------------- 데이터·모델
def load():
    d = np.load(WM, allow_pickle=True)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"], [str(c) for c in d["classes"]]


def to_tensor(X):
    import torch
    x = X.astype(np.float32) / 2.0
    x = np.repeat(x[:, None, :, :], 3, axis=1)
    x = (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    return torch.from_numpy(x)


TRAIN_CFG = {"schedule": "cosine", "deterministic": True}   # noise_floor 결과로 정함(아래)


def train(X, y, n_cls, epochs, device, seed, bs=64, lr=1e-3, schedule=None, deterministic=None):
    """schedule: "const"(처음 실행) | "cosine"(마지막 에폭까지 lr을 0으로 감쇠).
    deterministic: cuDNN 결정적 모드. 같은 데이터·시드면 같은 모델이 나오게 한다."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.models import ResNet18_Weights, resnet18

    schedule = schedule or TRAIN_CFG["schedule"]
    deterministic = TRAIN_CFG["deterministic"] if deterministic is None else deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    m = m.to(device)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(to_tensor(X), torch.from_numpy(y)), batch_size=bs,
                    shuffle=True, generator=g)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    total = epochs * len(dl)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
             if schedule == "cosine" else None)
    crit = nn.CrossEntropyLoss()
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(m(xb), yb).backward()
            opt.step()
            if sched:
                sched.step()
    m.eval()
    return m


def infer(m, X, device, bs=1024, feats=False):
    """softmax 확률(그리고 원하면 avgpool 512차원 특징)."""
    import torch
    probs, fs = [], []
    body = torch.nn.Sequential(*list(m.children())[:-1]) if feats else None
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = to_tensor(X[i:i + bs]).to(device)
            if feats:
                f = body(xb).flatten(1)
                fs.append(f.cpu().numpy())
                logits = m.fc(f)
            else:
                logits = m(xb)
            probs.append(torch.softmax(logits, 1).cpu().numpy())
    P = np.concatenate(probs)
    return (P, np.concatenate(fs)) if feats else P


def evaluate(m, Xte, yte, device, n_cls):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    yp = infer(m, Xte, device).argmax(1)
    rec = recall_score(yte, yp, average=None, labels=list(range(n_cls)), zero_division=0)
    return {"acc": round(float(accuracy_score(yte, yp)), 4),
            "macro_f1": round(float(f1_score(yte, yp, average="macro")), 4),
            "recall": [round(float(r), 4) for r in rec]}


# ---------------------------------------------------------------- 질의 전략
def select(strategy, m, Xpool, pool_idx, lab_idx, k, device, rng):
    if strategy == "random":
        return rng.choice(pool_idx, size=k, replace=False)
    if strategy in ("entropy", "margin"):
        P = infer(m, Xpool[pool_idx], device)
        if strategy == "entropy":
            score = -(P * np.log(P + 1e-12)).sum(1)            # 클수록 불확실
        else:
            s = np.sort(P, 1)
            score = -(s[:, -1] - s[:, -2])                      # top1-top2 차이가 작을수록 불확실
        return pool_idx[np.argsort(-score)[:k]]
    if strategy == "coreset":
        import torch
        _, Fp = infer(m, Xpool[pool_idx], device, feats=True)
        _, Fl = infer(m, Xpool[lab_idx], device, feats=True)
        fp = torch.from_numpy(Fp).to(device)
        fl = torch.from_numpy(Fl).to(device)
        mind = torch.cdist(fp, fl).min(1).values               # 풀 각 점 → 가장 가까운 라벨 점
        picks = []
        for _ in range(k):                                     # k-center greedy
            j = int(torch.argmax(mind))
            picks.append(j)
            mind = torch.minimum(mind, torch.cdist(fp, fp[j:j + 1]).squeeze(1))
        return pool_idx[np.array(picks)]
    raise ValueError(strategy)


# ---------------------------------------------------------------- 저장
def save(key, value):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    data[key] = value
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def split_val(n, seed):
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(n)
    return perm[:N_VAL], perm[N_VAL:]


# ---------------------------------------------------------------- 1. 능동 학습
def part_al(args, device):
    Xtr, ytr, Xte, yte, classes = load()
    n_cls = len(classes)
    budget = 300 if args.smoke else BUDGET
    epochs = 1 if args.smoke else args.epochs
    for seed in args.seeds:
        val_idx, pool_all = split_val(len(ytr), seed)
        rng0 = np.random.default_rng(seed)
        init = rng0.choice(pool_all, size=INIT, replace=False)   # 전략 모두 같은 시작점
        for strat in (args.strategies or STRATEGIES):
            key = f"al/{strat}/seed{seed}" + ("/smoke" if args.smoke else "")
            rng = np.random.default_rng(seed * 100 + 7)
            lab = init.copy()
            curve = []
            t0 = time.time()
            while True:
                m = train(Xtr[lab], ytr[lab], n_cls, epochs, device, seed)
                ev = evaluate(m, Xte, yte, device, n_cls)
                cnt = np.bincount(ytr[lab], minlength=n_cls).tolist()
                curve.append({"n_labeled": int(len(lab)), **ev, "labeled_per_class": cnt})
                print(f"  {key} n={len(lab):>5} macroF1={ev['macro_f1']:.4f} acc={ev['acc']:.4f} "
                      f"Near-full={cnt[classes.index('Near-full')]} Donut={cnt[classes.index('Donut')]}",
                      flush=True)
                if len(lab) >= budget:
                    break
                pool = np.setdiff1d(pool_all, lab, assume_unique=False)
                new = select(strat, m, Xtr, pool, lab, STEP, device, rng)
                lab = np.concatenate([lab, new])
            save(key, {"curve": curve, "seconds": round(time.time() - t0, 1),
                       "epochs": epochs, "classes": classes})


# ---------------------------------------------------------------- 2. 자동 라벨링
def part_auto(args, device):
    from sklearn.metrics import f1_score
    Xtr, ytr, Xte, yte, classes = load()
    n_cls = len(classes)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for seed in args.seeds:
        val_idx, pool_all = split_val(len(ytr), seed)
        # 라벨 1,000장: 능동 학습 결과를 재현하지 않고, 같은 조건의 무작위 1,000장과
        # margin으로 고른 1,000장 두 가지를 출발점으로 쓴다(margin 집합은 al 결과에서 재구성 불가 →
        # 여기서 다시 돌린다).
        for start in ("random", "margin"):
            key = f"auto/{start}/seed{seed}"
            rng = np.random.default_rng(seed * 100 + 7)
            lab = np.random.default_rng(seed).choice(pool_all, size=INIT, replace=False)
            while len(lab) < 1000:
                m = train(Xtr[lab], ytr[lab], n_cls, args.epochs, device, seed)
                pool = np.setdiff1d(pool_all, lab)
                lab = np.concatenate([lab, select(start, m, Xtr, pool, lab, STEP, device, rng)])
            m = train(Xtr[lab], ytr[lab], n_cls, args.epochs, device, seed)
            base = evaluate(m, Xte, yte, device, n_cls)

            # τ는 val에서만 고른다. val을 확신 내림차순으로 줄 세워 상위 k개의 정밀도가 목표 이상인
            # 가장 큰 k를 찾고, 그 k번째 확신을 τ로 쓴다(k ≥ 20). 목표별로 따로 잰다.
            Pv = infer(m, Xtr[val_idx], device)
            conf_v, pred_v = Pv.max(1), Pv.argmax(1)
            order = np.argsort(-conf_v)
            correct = (pred_v[order] == ytr[val_idx][order]).astype(float)
            cum_prec = np.cumsum(correct) / np.arange(1, len(correct) + 1)
            pool = np.setdiff1d(pool_all, lab)
            Pp = infer(m, Xtr[pool], device)
            conf_p, pred_p = Pp.max(1), Pp.argmax(1)
            true_p = ytr[pool]
            res = {"n_labeled": int(len(lab)), "pool_size": int(len(pool)),
                   "val_overall_acc": round(float(correct.mean()), 4),
                   "test_labeled_only": base, "targets": {}}
            print(f"  {key} labeled-only macroF1={base['macro_f1']:.4f} val acc={correct.mean():.4f}", flush=True)
            for target in (0.99, 0.98, 0.95, 0.90):
                ok = np.where((cum_prec >= target) & (np.arange(1, len(correct) + 1) >= 20))[0]
                if len(ok) == 0:
                    res["targets"][str(target)] = {"reachable": False}
                    print(f"  {key} target={target} tau=unreachable on val", flush=True)
                    continue
                k = int(ok.max()) + 1
                tau = float(conf_v[order][k - 1])
                take = conf_p >= tau
                prec = float((pred_p[take] == true_p[take]).mean()) if take.any() else None
                per_cls = {}
                for c in range(n_cls):
                    tc = take & (pred_p == c)
                    per_cls[classes[c]] = {"n_auto": int(tc.sum()),
                                           "precision": (round(float((true_p[tc] == c).mean()), 4)
                                                         if tc.sum() else None),
                                           "pool_true": int((true_p == c).sum()),
                                           "covered_true": int((take & (true_p == c)).sum())}
                Xa = np.concatenate([Xtr[lab], Xtr[pool][take]])
                ya = np.concatenate([ytr[lab], pred_p[take]])
                m2 = train(Xa, ya, n_cls, args.epochs, device, seed)
                aug = evaluate(m2, Xte, yte, device, n_cls)
                res["targets"][str(target)] = {
                    "reachable": True, "tau": round(tau, 6), "val_coverage": round(k / len(correct), 4),
                    "n_auto": int(take.sum()), "coverage": round(float(take.mean()), 4),
                    "auto_precision": round(prec, 4) if prec is not None else None,
                    "per_class": per_cls, "test_with_auto": aug}
                print(f"  {key} target={target} tau={tau:.5f} val_cov={k/len(correct):.3f} auto={take.sum()} "
                      f"cov={take.mean():.3f} prec={prec:.4f} macroF1 {base['macro_f1']:.4f} -> {aug['macro_f1']:.4f}",
                      flush=True)
            save(key, res)


# ---------------------------------------------------------------- 3. 라벨 오류 탐지
def part_noise(args, device, rate=0.10, n_flag=100, folds=5):
    from sklearn.model_selection import StratifiedKFold
    Xtr, ytr, Xte, yte, classes = load()
    n_cls = len(classes)
    for seed in args.seeds:
        key = f"noise/seed{seed}"
        _, pool_all = split_val(len(ytr), seed)
        rng = np.random.default_rng(500 + seed)
        lab = rng.choice(pool_all, size=1000, replace=False)
        y_true = ytr[lab].copy()
        y_noisy = y_true.copy()
        flip = rng.choice(len(lab), size=int(rate * len(lab)), replace=False)
        for i in flip:
            y_noisy[i] = rng.choice([c for c in range(n_cls) if c != y_true[i]])
        is_err = y_noisy != y_true

        # out-of-fold: 각 샘플은 자기를 보지 않은 모델의 확률로 평가
        oof = np.zeros((len(lab), n_cls), dtype=np.float32)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        # 희소 클래스가 fold 수보다 적을 수 있어 층화 실패 시 일반 KFold로 내려간다
        try:
            splits = list(skf.split(np.zeros(len(lab)), y_noisy))
        except ValueError:
            from sklearn.model_selection import KFold
            splits = list(KFold(folds, shuffle=True, random_state=seed).split(lab))
        for tr, va in splits:
            mf = train(Xtr[lab][tr], y_noisy[tr], n_cls, args.epochs, device, seed)
            oof[va] = infer(mf, Xtr[lab][va], device)
        self_conf = oof[np.arange(len(lab)), y_noisy]
        flagged = np.argsort(self_conf)[:n_flag]                 # 주어진 라벨 확신이 낮은 순
        hit = int(is_err[flagged].sum())

        y_fix = y_noisy.copy()
        y_fix[flagged] = y_true[flagged]                         # 사람이 100장만 검토해 고침
        keep = np.setdiff1d(np.arange(len(lab)), flagged)        # 또는 그냥 뺌
        # 대조군: 무작위 100장을 검토했을 때
        rand_flag = rng.choice(len(lab), size=n_flag, replace=False)
        y_rand = y_noisy.copy()
        y_rand[rand_flag] = y_true[rand_flag]

        out = {"n": int(len(lab)), "n_errors": int(is_err.sum()), "n_flagged": n_flag,
               "flag_precision": round(hit / n_flag, 4), "flag_recall": round(hit / max(1, is_err.sum()), 4),
               "random_review_hits": int(is_err[rand_flag].sum())}
        for name, (X_, y_) in {"clean": (Xtr[lab], y_true), "noisy": (Xtr[lab], y_noisy),
                               "noisy_fixed_flagged": (Xtr[lab], y_fix),
                               "noisy_drop_flagged": (Xtr[lab][keep], y_noisy[keep]),
                               "noisy_fixed_random": (Xtr[lab], y_rand)}.items():
            m = train(X_, y_, n_cls, args.epochs, device, seed)
            out[name] = evaluate(m, Xte, yte, device, n_cls)
            print(f"  {key} {name:<22} macroF1={out[name]['macro_f1']:.4f}", flush=True)
        print(f"  {key} flagged {n_flag}: errors found {hit}/{is_err.sum()} "
              f"(random review finds {out['random_review_hits']})", flush=True)
        save(key, out)


# ---------------------------------------------------------------- 0. 학습 잡음 바닥
def part_floor(args, device, repeats=5):
    """같은 라벨 집합·같은 시드로 여러 번 학습해 macro-F1이 얼마나 흔들리는지 잰다.
    첫 능동 학습 실행에서 전략 넷이 같은 100장·같은 시드로 시작했는데 첫 라운드가
    0.454~0.572로 갈렸다 — 전략 차이를 말하기 전에 이 잡음부터 재야 한다."""
    Xtr, ytr, Xte, yte, classes = load()
    n_cls = len(classes)
    _, pool_all = split_val(len(ytr), 0)
    configs = {"const": ("const", False), "cosine": ("cosine", False),
               "cosine_det": ("cosine", True)}
    for n in (100, 500, 1500):
        lab = np.random.default_rng(0).choice(pool_all, size=n, replace=False)
        for name, (sch, det) in configs.items():
            f1s = []
            for r in range(repeats):
                m = train(Xtr[lab], ytr[lab], n_cls, args.epochs, device, seed=0,
                          schedule=sch, deterministic=det)
                f1s.append(evaluate(m, Xte, yte, device, n_cls)["macro_f1"])
            print(f"  floor n={n} {name:<10} macroF1 {np.mean(f1s):.4f} ± {np.std(f1s):.4f} "
                  f"(min {min(f1s):.4f} max {max(f1s):.4f})", flush=True)
            save(f"floor/n{n}/{name}", {"macro_f1": f1s, "mean": round(float(np.mean(f1s)), 4),
                                         "std": round(float(np.std(f1s)), 4)})


# ---------------------------------------------------------------- 집계
def part_report(args, device=None):
    """al/* 결과를 전략별로 묶는다: 라운드별 평균 macro-F1, 곡선 아래 면적(평균),
    random의 1,500장 성능에 처음 도달한 라벨 수, 희소 클래스 수집량, random 대비 짝지음 비교."""
    from scipy.stats import wilcoxon
    data = json.loads(OUT.read_text(encoding="utf-8"))
    runs = {}
    for k, v in data.items():
        parts = k.split("/")
        if parts[0] != "al" or len(parts) != 3:
            continue
        runs.setdefault(parts[1], {})[parts[2]] = v["curve"]
    if not runs:
        print("al 결과 없음")
        return
    classes = next(v["classes"] for k, v in data.items() if k.startswith("al/"))
    seeds = sorted(set.intersection(*[set(r) for r in runs.values()]))
    ns = [c["n_labeled"] for c in runs["random"][seeds[0]]]
    f1 = {s: np.array([[c["macro_f1"] for c in runs[s][sd]] for sd in seeds]) for s in runs}
    target = f1["random"][:, -1].mean()
    summary = {"seeds": seeds, "n_labeled": ns, "random_final_mean": round(float(target), 4)}
    print(f"seeds={seeds}  random@{ns[-1]} 평균 macro-F1 = {target:.4f}")
    nf = classes.index("Near-full") if "Near-full" in classes else 5
    dn = classes.index("Donut") if "Donut" in classes else 1
    for s in runs:
        mean_curve = f1[s].mean(0)
        reach = next((n for n, v in zip(ns, mean_curve) if v >= target), None)
        rare_nf = np.mean([runs[s][sd][-1]["labeled_per_class"][nf] for sd in seeds])
        rare_dn = np.mean([runs[s][sd][-1]["labeled_per_class"][dn] for sd in seeds])
        rec_nf = np.mean([runs[s][sd][-1]["recall"][nf] for sd in seeds])
        row = {"curve_mean": [round(float(x), 4) for x in mean_curve],
               "aulc": round(float(mean_curve.mean()), 4),
               "final_mean": round(float(mean_curve[-1]), 4),
               "final_std": round(float(f1[s][:, -1].std()), 4),
               "labels_to_reach_random_final": reach,
               "near_full_labeled_at_end": round(float(rare_nf), 1),
               "donut_labeled_at_end": round(float(rare_dn), 1),
               "near_full_recall_at_end": round(float(rec_nf), 4)}
        if s != "random":
            diff = (f1[s] - f1["random"]).ravel()           # 라운드×시드 짝
            row["wins_vs_random"] = int((diff > 0).sum())
            row["losses_vs_random"] = int((diff < 0).sum())
            row["wilcoxon_p_vs_random"] = float(wilcoxon(diff).pvalue) if np.any(diff) else 1.0
        summary[s] = row
        print(f"  {s:<8} AULC {row['aulc']:.4f}  final {row['final_mean']:.4f}±{row['final_std']:.4f}  "
              f"reach@{reach}  Near-full {rare_nf:.0f}장(recall {rec_nf:.2f})  Donut {rare_dn:.0f}장"
              + (f"  vs random {row['wins_vs_random']}:{row['losses_vs_random']} p={row['wilcoxon_p_vs_random']:.3g}"
                 if s != "random" else ""))
    save("al_summary", summary)


def main():
    ap = argparse.ArgumentParser(description="웨이퍼맵 능동 학습·자동 라벨링·라벨 오류 탐지")
    ap.add_argument("--part", choices=["floor", "al", "auto", "noise", "report"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--strategies", nargs="*")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    {"floor": part_floor, "al": part_al, "auto": part_auto, "noise": part_noise,
     "report": part_report}[args.part](args, device)


if __name__ == "__main__":
    main()
