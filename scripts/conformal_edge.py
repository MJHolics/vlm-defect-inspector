"""Conformal Prediction — 엣지 CNN에 '통계적으로 보장되는' 불확실성.

confidence 임계값(임의 휴리스틱)이나 OOD 점수를 넘어, **분포가정 없이 유한표본에서
커버리지(true label ∈ 예측집합)를 1-α 이상으로 수학적으로 보장**하는 예측 집합을 만든다.
검사·의료 도메인에서 "이 판정을 믿어도 되는가"를 휴리스틱이 아니라 보장으로 답한다.

두 가지 split-conformal 방법을 구현·비교한다(모두 순수 numpy 함수 → 검증 용이):
  - LAC   : 비순응점수 s = 1 - softmax(true). 집합 = {y : softmax(y) ≥ 1-q̂}. 집합이 작다(효율).
  - APS   : 정렬 누적합 점수. 적응적 집합(쉬운 입력은 1개, 애매하면 커진다) → 조건부 커버리지 우수.

교환성(exchangeability): test(270)는 학습·모델선택에 전혀 쓰지 않았으므로, test를 stratified로
calib/eval 반분해 calib로 보정하고 eval로 커버리지를 측정한다. 단일 split의 운(noise)을 없애려
N회 무작위 반복해 평균±표준편차를 보고한다.

운영 연결: 예측집합 크기 > 1 = '모델이 후보를 좁히지 못함 → 사람검토'를, 임의 임계값이 아니라
**1-α 보장 하에** 라우팅한다. 기존 confidence·OOD 게이트의 통계적 상위호환.

사용 (GPU 권장, CPU도 가능):
    python scripts/conformal_edge.py --arch resnet18 --alpha 0.1 --repeats 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_edge_cnn import CLASSES, _build_model, _load_split  # noqa: E402

CKPT_DIR = ROOT / "models" / "checkpoints" / "edge_cnn"
OUT_DIR = ROOT / "data" / "results" / "conformal"
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Conformal 핵심 (순수 numpy — 네트워크·모델 무관, 단위 검증 가능)
# ---------------------------------------------------------------------------
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """유한표본 보정 분위수: k=⌈(n+1)(1-α)⌉ 번째 순서통계량. k>n이면 +∞(전체 포함)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def lac_calibrate(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    scores = 1.0 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    return conformal_quantile(scores, alpha)


def lac_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """집합 마스크 (m, C): softmax(y) ≥ 1-q̂ 인 y 포함."""
    return probs >= (1.0 - qhat)


def aps_scores_true(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """각 표본의 APS 비순응점수 = 내림차순으로 true 클래스까지의 확률 누적합."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    # true 라벨이 정렬상 몇 번째인지
    rank = (order == labels[:, None]).argmax(axis=1)
    return csum[np.arange(len(labels)), rank]


def aps_calibrate(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float) -> float:
    return conformal_quantile(aps_scores_true(cal_probs, cal_labels), alpha)


def aps_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """내림차순 누적합이 q̂를 넘는 시점까지(그 클래스 포함) 집합에 넣는다."""
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    csum = np.cumsum(sorted_p, axis=1)
    take = csum - sorted_p < qhat  # 이 클래스 직전 누적이 q̂ 미만이면 포함
    mask = np.zeros_like(probs, dtype=bool)
    np.put_along_axis(mask, order, take, axis=1)
    return mask


def evaluate_sets(sets: np.ndarray, labels: np.ndarray) -> dict:
    incl = sets[np.arange(len(labels)), labels]
    sizes = sets.sum(axis=1)
    per_class = {}
    for c in range(len(CLASSES)):
        m = labels == c
        if m.any():
            per_class[CLASSES[c]] = round(float(incl[m].mean()), 4)
    return {
        "coverage": float(incl.mean()),
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "ambiguous_rate": float((sizes > 1).mean()),
        "empty_rate": float((sizes == 0).mean()),
        "per_class_coverage": per_class,
    }


# ---------------------------------------------------------------------------
# 모델 추론 → softmax
# ---------------------------------------------------------------------------
def _softmax_on_split(model, split_name: str, img_size: int, device: str):
    import torch
    from PIL import Image
    from torchvision import transforms as T

    tf = T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])
    items = _load_split(split_name)
    probs, labels = [], []
    model.eval()
    with torch.no_grad():
        for path, y in items:
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            p = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            probs.append(p)
            labels.append(y)
    return np.array(probs), np.array(labels)


def _stratified_halves(labels: np.ndarray, rng: np.random.Generator):
    """클래스별로 절반씩 calib/eval 인덱스로 나눈다(stratified)."""
    cal, ev = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        cal.extend(idx[:h])
        ev.extend(idx[h:])
    return np.array(cal), np.array(ev)


def _repeated(probs, labels, alpha, repeats, seed):
    """N회 stratified 반분 반복 → LAC/APS 커버리지·집합크기 평균±표준편차."""
    rng = np.random.default_rng(seed)
    acc = {"lac": [], "aps": []}
    for _ in range(repeats):
        cal, ev = _stratified_halves(labels, rng)
        cp, cl = probs[cal], labels[cal]
        ep, el = probs[ev], labels[ev]
        acc["lac"].append(evaluate_sets(lac_sets(ep, lac_calibrate(cp, cl, alpha)), el))
        acc["aps"].append(evaluate_sets(aps_sets(ep, aps_calibrate(cp, cl, alpha)), el))

    def agg(key):
        out = {}
        for metric in ("coverage", "avg_set_size", "singleton_rate", "ambiguous_rate", "empty_rate"):
            vals = [r[metric] for r in acc[key]]
            out[metric] = round(float(np.mean(vals)), 4)
            out[metric + "_std"] = round(float(np.std(vals)), 4)
        # 클래스별 커버리지 평균
        pc = {}
        for c in CLASSES:
            vals = [r["per_class_coverage"].get(c) for r in acc[key] if c in r["per_class_coverage"]]
            if vals:
                pc[c] = round(float(np.mean(vals)), 4)
        out["per_class_coverage"] = pc
        return out

    return {"lac": agg("lac"), "aps": agg("aps")}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="엣지 CNN Conformal Prediction")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--alpha", type=float, default=0.1, help="목표 오류율(커버리지=1-α)")
    ap.add_argument("--repeats", type=int, default=100, help="stratified 반분 반복 횟수")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = CKPT_DIR / f"{args.arch}.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt} (먼저 train_edge_cnn.py 실행)")

    model = _build_model(args.arch, len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    probs, labels = _softmax_on_split(model, "test", args.img_size, device)
    top1 = float((probs.argmax(1) == labels).mean())
    print(f"[data] test {len(labels)}장 · top-1 정확도 {top1:.4f}")

    # 주 결과(α 지정) — N회 반복
    main_res = _repeated(probs, labels, args.alpha, args.repeats, args.seed)
    print(f"\n=== α={args.alpha} (목표 커버리지 {1-args.alpha:.0%}) · {args.repeats}회 반복 ===")
    for name in ("lac", "aps"):
        r = main_res[name]
        print(f" {name.upper():4s} 커버리지 {r['coverage']:.3f}±{r['coverage_std']:.3f}"
              f" · 평균집합크기 {r['avg_set_size']:.2f}"
              f" · 단일{r['singleton_rate']:.0%}/애매{r['ambiguous_rate']:.0%}")

    # α 스윕 — 보장(커버리지가 1-α를 추종하는지) 검증
    sweep = {}
    for a in (0.01, 0.05, 0.10, 0.20):
        sweep[a] = _repeated(probs, labels, a, args.repeats, args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 플롯: (1) 경험적 커버리지 vs 목표(대각선), (2) 평균 집합크기 vs α
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 라벨 렌더(Windows)
    plt.rcParams["axes.unicode_minus"] = False

    alphas = sorted(sweep.keys())
    targets = [1 - a for a in alphas]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot([0.78, 1.0], [0.78, 1.0], "k--", lw=1, label="목표(=1-α)")
    for name, mk in (("lac", "o-"), ("aps", "s-")):
        cov = [sweep[a][name]["coverage"] for a in alphas]
        ax1.plot(targets, cov, mk, label=name.upper())
    ax1.set_xlabel("목표 커버리지 1-α"); ax1.set_ylabel("경험적 커버리지(test)")
    ax1.set_title("커버리지 보장 검증"); ax1.legend(); ax1.grid(alpha=0.3)
    for name, mk in (("lac", "o-"), ("aps", "s-")):
        sz = [sweep[a][name]["avg_set_size"] for a in alphas]
        ax2.plot(targets, sz, mk, label=name.upper())
    ax2.set_xlabel("목표 커버리지 1-α"); ax2.set_ylabel("평균 예측집합 크기")
    ax2.set_title("효율(집합이 작을수록 유용)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f"Conformal Prediction · {args.arch} (NEU test, {args.repeats}회 반복)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path = OUT_DIR / f"conformal_{args.arch}.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    out = OUT_DIR / f"conformal_{args.arch}.json"
    out.write_text(json.dumps({
        "arch": args.arch,
        "test_n": int(len(labels)),
        "top1_accuracy": round(top1, 4),
        "repeats": args.repeats,
        "alpha": args.alpha,
        "main": main_res,
        "sweep": {str(a): sweep[a] for a in alphas},
        "plot": str(plot_path.relative_to(ROOT)),
        "note": "test를 stratified 반분(calib/eval)해 보정·평가, N회 반복 평균. "
                "test는 학습·모델선택에 미사용(교환성 충족).",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과: {out.relative_to(ROOT)} · 플롯: {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
