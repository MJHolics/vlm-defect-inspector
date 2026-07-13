"""Calibration — 엣지 CNN 확률의 '신뢰도'를 보정한다 (ECE + Temperature Scaling).

Conformal(conformal_edge.py)은 **커버리지**(정답이 예측집합에 있음)를 보장하지만, softmax
*확률값 자체*가 믿을 만한지는 말하지 않는다. 현대 딥넷은 **과신(overconfident)** 경향이라
"confidence 0.99"가 실제론 90%만 맞을 수 있다 — 임계 라우팅·사람검토 판단이 이 확률에 기대므로
검사·의료 도메인에선 확률의 *정직성*이 곧 신뢰다.

측정: **ECE(Expected Calibration Error)** = confidence 구간별 |정확도 − 평균confidence|의 가중합.
교정: **Temperature Scaling(Guo+2017)** — 로짓을 스칼라 T로 나눠 NLL을 최소화하는 후처리.
  - T>1이면 확률을 부드럽게(과신 완화), T<1이면 날카롭게. **argmax 불변 → 정확도 그대로**.
  - 단일 파라미터 → 과적합 위험 최소·재현 쉬움. (Platt/isotonic은 파라미터 多·단조성 붕괴 위험.)

교환성: test(NEU)를 stratified 반분해 calib로 T를 적합, eval로 ECE를 측정한다. 단일 split의
운을 없애려 N회 반복 평균±표준편차. test는 학습·모델선택에 미사용.

핵심(ECE·NLL·temperature 적합)은 **순수 numpy 함수** → 네트워크 없이 단위 검증 가능.

사용:
    python scripts/calibrate_edge.py --arch resnet18 --repeats 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.conformal_edge import _stratified_halves  # noqa: E402  (stratified 반분 재사용)
from scripts.train_edge_cnn import CLASSES, _build_model, _load_split  # noqa: E402

CKPT_DIR = ROOT / "models" / "checkpoints" / "edge_cnn"
OUT_DIR = ROOT / "data" / "results" / "calibration"
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# 보정 핵심 (순수 numpy — 로짓만 받으면 됨, 단위 검증 가능)
# ---------------------------------------------------------------------------
def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """온도 T를 적용한 softmax (T>1 = 완만/과신완화, T<1 = 날카롭게)."""
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max(axis=1, keepdims=True)          # 수치안정
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    """온도 T 하 true 클래스 음의 로그우도(작을수록 보정 잘 됨)."""
    p = softmax(logits, T)
    true_p = p[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true_p, 1e-12, 1.0)).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (confidence 기준, 등간격 bin).

    각 bin에서 |평균정확도 − 평균confidence| 를 표본비중으로 가중합. 0에 가까울수록 정직.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += abs(correct[m].mean() - conf[m].mean()) * (m.sum() / n)
    return float(e)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """다중분류 Brier score(제곱오차) — 보정+정확도를 함께 보는 proper score."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lo: float = 0.05, hi: float = 10.0, iters: int = 60) -> float:
    """NLL(T)를 최소화하는 T를 golden-section 탐색으로 적합(결정적, scipy 불요).

    NLL(T)는 T>0에서 볼록에 가깝다 → 구간 탐색으로 전역최소에 수렴. seed·데이터 같으면 같은 T.
    """
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = nll(logits, labels, c), nll(logits, labels, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(logits, labels, d)
    return float((a + b) / 2)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


# ---------------------------------------------------------------------------
# 모델 추론 → 로짓
# ---------------------------------------------------------------------------
def _logits_on_split(model, split_name: str, img_size: int, device: str):
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
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for path, y in items:
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            logits.append(model(x)[0].cpu().numpy())
            labels.append(y)
    return np.array(logits), np.array(labels)


def _repeated(logits, labels, repeats, seed, n_bins):
    """N회 stratified 반분: calib로 T 적합, eval로 보정 전/후 ECE·NLL·Brier·정확도."""
    rng = np.random.default_rng(seed)
    rows = {"ece_before": [], "ece_after": [], "nll_before": [], "nll_after": [],
            "brier_before": [], "brier_after": [], "acc_before": [], "acc_after": [], "T": []}
    for _ in range(repeats):
        cal, ev = _stratified_halves(labels, rng)
        cl, cy = logits[cal], labels[cal]
        el, ey = logits[ev], labels[ev]
        T = fit_temperature(cl, cy)
        p0, p1 = softmax(el, 1.0), softmax(el, T)
        rows["T"].append(T)
        rows["ece_before"].append(ece(p0, ey, n_bins))
        rows["ece_after"].append(ece(p1, ey, n_bins))
        rows["nll_before"].append(nll(el, ey, 1.0))
        rows["nll_after"].append(nll(el, ey, T))
        rows["brier_before"].append(brier(p0, ey))
        rows["brier_after"].append(brier(p1, ey))
        rows["acc_before"].append(accuracy(el, ey))          # argmax 불변 → after와 동일해야
        rows["acc_after"].append(float((p1.argmax(1) == ey).mean()))
    return {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in rows.items()}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="엣지 CNN Calibration (ECE + Temperature Scaling)")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = CKPT_DIR / f"{args.arch}.pt"
    if not ckpt.exists():
        raise SystemExit(f"체크포인트 없음: {ckpt} (먼저 train_edge_cnn.py 실행)")

    model = _build_model(args.arch, len(CLASSES), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    logits, labels = _logits_on_split(model, "test", args.img_size, device)
    print(f"[data] test {len(labels)}장 · top-1 정확도 {accuracy(logits, labels):.4f} "
          f"· 전체 ECE(보정 전) {ece(softmax(logits), labels, args.n_bins):.4f}")

    res = _repeated(logits, labels, args.repeats, args.seed, args.n_bins)
    T_mean, T_std = res["T"]
    print(f"\n=== Temperature Scaling · {args.repeats}회 stratified 반분 (calib 적합 → eval 측정) ===")
    print(f"  적합 온도 T = {T_mean:.3f} ± {T_std:.3f}  ({'과신 완화(T>1)' if T_mean > 1 else '과소→날카롭게(T<1)'})")
    for metric, label in (("ece", "ECE"), ("nll", "NLL"), ("brier", "Brier")):
        b, bs = res[f"{metric}_before"]
        a, as_ = res[f"{metric}_after"]
        red = (1 - a / b) * 100 if b else 0.0
        print(f"  {label:6s} 보정 전 {b:.4f}±{bs:.4f} → 후 {a:.4f}±{as_:.4f}  ({red:+.1f}%)")
    ab, _ = res["acc_before"]
    aa, _ = res["acc_after"]
    print(f"  정확도  보정 전 {ab:.4f} → 후 {aa:.4f}  (argmax 불변 = 동일해야 정상)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 신뢰성 다이어그램(전체 test, 보정 전/후) — confidence bin별 정확도 vs 이상선.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
        T = fit_temperature(logits, labels)   # 전체로 시각화용 T
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for ax, (p, ttl) in zip(axes, ((softmax(logits, 1.0), "보정 전 (T=1)"),
                                        (softmax(logits, T), f"보정 후 (T={T:.2f})"))):
            conf, pred = p.max(1), p.argmax(1)
            correct = (pred == labels).astype(float)
            edges = np.linspace(0, 1, args.n_bins + 1)
            xs, ys = [], []
            for i in range(args.n_bins):
                m = (conf > edges[i]) & (conf <= edges[i + 1])
                if m.any():
                    xs.append(conf[m].mean()); ys.append(correct[m].mean())
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="완벽 보정")
            ax.plot(xs, ys, "o-", label="경험적")
            ax.set_title(f"{ttl} · ECE {ece(p, labels, args.n_bins):.3f}")
            ax.set_xlabel("평균 confidence"); ax.set_ylabel("정확도"); ax.legend(); ax.grid(alpha=0.3)
        fig.suptitle(f"신뢰성 다이어그램 · {args.arch} (NEU test)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plot_path = OUT_DIR / f"calibration_{args.arch}.png"
        fig.savefig(plot_path, dpi=120); plt.close(fig)
        plot_rel = str(plot_path.relative_to(ROOT))
    except Exception as e:  # matplotlib 없거나 폰트 문제 — 수치는 유지.
        print(f"  (플롯 생략: {e})")
        plot_rel = None

    out = OUT_DIR / f"calibration_{args.arch}.json"
    out.write_text(json.dumps({
        "arch": args.arch, "test_n": int(len(labels)), "repeats": args.repeats,
        "n_bins": args.n_bins, "temperature": {"mean": T_mean, "std": T_std},
        "metrics": {k: {"before": res[f"{k}_before"], "after": res[f"{k}_after"]}
                    for k in ("ece", "nll", "brier", "acc")},
        "plot": plot_rel,
        "note": "test stratified 반분(calib 적합/eval 측정), N회 반복 평균±표준편차. "
                "Temperature Scaling은 argmax 불변 → 정확도 그대로, 확률의 정직성(ECE)만 개선.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과: {out.relative_to(ROOT)}" + (f" · 플롯: {plot_rel}" if plot_rel else ""))


if __name__ == "__main__":
    main()
