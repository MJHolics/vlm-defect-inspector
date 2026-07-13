"""Conformal Prediction 핵심 로직의 결정적 단위 테스트 (GPU·모델·네트워크 없음).

`scripts/conformal_edge.py`의 순수 numpy 함수만 검증한다. 이 트랙의 핵심 주장은
"분포 가정 없이 유한 표본에서 커버리지(정답 ∈ 예측집합)를 1-α 이상으로 보장한다"이다.
README의 LAC 커버리지 표(0.902 등)는 실제 모델 softmax로 측정한 값이지만, 여기서는
**합성 데이터로 그 보장 성질 자체가 코드에서 성립하는지**를 재현 가능하게 못박는다.

실행: python tests/test_conformal.py   (pytest 불필요, GPU 불필요)
"""
import sys
from pathlib import Path

# Windows 기본 콘솔(cp949)에서도 유니코드 출력이 깨지지 않게 한다.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import conformal_edge as cp  # noqa: E402

C = 6  # NEU 6클래스


def _synthetic(n: int, seed: int):
    """교환 가능한 (probs, labels): softmax 확률을 만들고 그 분포에서 라벨을 샘플.

    라벨이 probs의 *참* 조건부 분포에서 나오므로 conformal 교환성 전제가 성립하고,
    이론상 LAC 주변 커버리지는 1-α 이상이어야 한다.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=1.5, size=(n, C))
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    labels = np.array([rng.choice(C, p=probs[i]) for i in range(n)])
    return probs, labels


def test_conformal_quantile_finite_sample():
    """k = ⌈(n+1)(1-α)⌉ 번째 순서통계량. k>n이면 +∞."""
    s = np.linspace(0.0, 1.0, 100)
    # n=100, α=0.1 → k=⌈101·0.9⌉=⌈90.9⌉=91 → sorted[90]
    assert abs(cp.conformal_quantile(s, 0.1) - np.sort(s)[90]) < 1e-9
    # n=100, α=0.5 → k=⌈101·0.5⌉=⌈50.5⌉=51 → sorted[50]
    assert abs(cp.conformal_quantile(s, 0.5) - np.sort(s)[50]) < 1e-9
    # α가 너무 작아 k>n → +∞ (전체 포함)
    assert cp.conformal_quantile(s, 0.001) == np.inf
    print("  ✓ test_conformal_quantile_finite_sample")


def test_lac_marginal_coverage_guarantee():
    """LAC 경험적 주변 커버리지가 1-α 보장을 따르는지 (다중 split 평균)."""
    probs, labels = _synthetic(n=2000, seed=0)
    rng = np.random.default_rng(7)
    for alpha in (0.1, 0.2):
        covs = []
        for _ in range(200):
            idx = rng.permutation(len(labels))
            cal, ev = idx[: len(idx) // 2], idx[len(idx) // 2:]
            qhat = cp.lac_calibrate(probs[cal], labels[cal], alpha)
            sets = cp.lac_sets(probs[ev], qhat)
            covs.append(cp.evaluate_sets(sets, labels[ev])["coverage"])
        mean_cov = float(np.mean(covs))
        # 보장: 평균 커버리지 ≥ 1-α (유한표본 상한 1-α+1/(n+1) 근처에서 약간 위)
        assert mean_cov >= (1 - alpha) - 0.01, (alpha, mean_cov)
        assert mean_cov <= (1 - alpha) + 0.05, (alpha, mean_cov)
    print("  ✓ test_lac_marginal_coverage_guarantee")


def test_aps_is_conservative():
    """APS는 LAC보다 보수적: 같은 α에서 커버리지가 LAC 이상, 집합도 더 크다."""
    probs, labels = _synthetic(n=2000, seed=1)
    idx = np.arange(len(labels))
    cal, ev = idx[::2], idx[1::2]
    alpha = 0.1
    lac_q = cp.lac_calibrate(probs[cal], labels[cal], alpha)
    aps_q = cp.aps_calibrate(probs[cal], labels[cal], alpha)
    lac = cp.evaluate_sets(cp.lac_sets(probs[ev], lac_q), labels[ev])
    aps = cp.evaluate_sets(cp.aps_sets(probs[ev], aps_q), labels[ev])
    assert aps["coverage"] >= lac["coverage"] - 1e-9, (lac["coverage"], aps["coverage"])
    assert aps["avg_set_size"] >= lac["avg_set_size"] - 1e-9
    assert aps["coverage"] >= (1 - alpha) - 0.02
    print("  ✓ test_aps_is_conservative")


def _synthetic_uneven_difficulty(n_per: int, seed: int):
    """클래스 0만 '어려운'(모델이 자신 없는) 데이터. 나머지 5클래스는 쉽다.

    주변(marginal) LAC는 쉬운 다수 클래스에 가려 어려운 클래스 0을 과소 커버한다.
    클래스 내부에서는 (prob, label=c) 쌍이 교환 가능하므로 Mondrian은 클래스별 보장을 회복한다.
    """
    rng = np.random.default_rng(seed)
    probs_list, labels_list = [], []
    for c in range(C):
        for _ in range(n_per):
            ptrue = rng.uniform(0.20, 0.55) if c == 0 else rng.uniform(0.85, 0.98)
            others = [j for j in range(C) if j != c]
            w = rng.dirichlet(np.ones(len(others))) * (1.0 - ptrue)
            p = np.empty(C)
            p[c] = ptrue
            for j, wj in zip(others, w):
                p[j] = wj
            probs_list.append(p)
            labels_list.append(c)
    return np.array(probs_list), np.array(labels_list)


def test_lac_mondrian_fixes_per_class_undercoverage():
    """Mondrian(클래스조건부) LAC가 주변 LAC의 클래스별 미달을 회복하는지.

    주변 LAC: 어려운 클래스 0이 1-α 미달. Mondrian: 클래스별 보정으로 0도 1-α 이상.
    """
    probs, labels = _synthetic_uneven_difficulty(n_per=400, seed=3)
    alpha = 0.1
    rng = np.random.default_rng(11)
    # 클래스별 커버리지를 split에 걸쳐 누적 → 각 클래스의 평균 커버리지로 집계.
    # (Mondrian 보장은 클래스별 E[coverage]≥1-α이지 split별 최소의 평균이 아니다.)
    marg_pc = {c: [] for c in cp.CLASSES}
    mond_pc = {c: [] for c in cp.CLASSES}
    for _ in range(80):
        cal, ev = cp._stratified_halves(labels, rng)
        cpb, clb = probs[cal], labels[cal]
        epb, elb = probs[ev], labels[ev]
        marg = cp.evaluate_sets(cp.lac_sets(epb, cp.lac_calibrate(cpb, clb, alpha)), elb)
        mond = cp.evaluate_sets(
            cp.lac_sets_mondrian(epb, cp.lac_calibrate_mondrian(cpb, clb, alpha)), elb
        )
        for c in cp.CLASSES:
            marg_pc[c].append(marg["per_class_coverage"][c])
            mond_pc[c].append(mond["per_class_coverage"][c])
    marg_mean = {c: float(np.mean(v)) for c, v in marg_pc.items()}
    mond_mean = {c: float(np.mean(v)) for c, v in mond_pc.items()}
    hard = cp.CLASSES[0]
    # 주변 LAC는 어려운 클래스 0을 과소 커버한다(< 1-α).
    assert marg_mean[hard] < (1 - alpha), f"주변 LAC가 클래스0을 미달시키지 않음: {marg_mean[hard]:.3f}"
    # Mondrian은 클래스 0을 보장 수준으로 끌어올린다.
    assert mond_mean[hard] >= marg_mean[hard], (marg_mean[hard], mond_mean[hard])
    assert mond_mean[hard] >= (1 - alpha) - 0.03, f"Mondrian 클래스0 커버리지 부족: {mond_mean[hard]:.3f}"
    # Mondrian은 모든 클래스의 평균 커버리지를 보장 근처로 끌어올린다.
    assert min(mond_mean.values()) >= (1 - alpha) - 0.03, (
        f"Mondrian 최소 클래스(평균) 커버리지 부족: {min(mond_mean.values()):.3f}"
    )
    print(f"  ✓ test_lac_mondrian_fixes_per_class_undercoverage "
          f"(클래스0: 주변 {marg_mean[hard]:.3f} → Mondrian {mond_mean[hard]:.3f})")


def test_evaluate_sets_arithmetic():
    """집합 마스크 → 커버리지/집합크기/싱글톤/공집합 회계가 정확한지 (손계산 대조)."""
    # 3표본 × 6클래스. 정답 = [0, 1, 2]
    sets = np.zeros((3, C), dtype=bool)
    sets[0, 0] = True                 # 정답 포함, 싱글톤
    sets[1, [0, 1, 2]] = True         # 정답 포함, 크기 3 (애매)
    sets[2, :] = False                # 공집합 → 미포함
    labels = np.array([0, 1, 2])
    r = cp.evaluate_sets(sets, labels)
    assert abs(r["coverage"] - 2 / 3) < 1e-9
    assert abs(r["avg_set_size"] - (1 + 3 + 0) / 3) < 1e-9
    assert abs(r["singleton_rate"] - 1 / 3) < 1e-9
    assert abs(r["ambiguous_rate"] - 1 / 3) < 1e-9
    assert abs(r["empty_rate"] - 1 / 3) < 1e-9
    print("  ✓ test_evaluate_sets_arithmetic")


if __name__ == "__main__":
    print("Conformal Prediction 핵심 로직 단위 테스트")
    test_conformal_quantile_finite_sample()
    test_lac_marginal_coverage_guarantee()
    test_aps_is_conservative()
    test_lac_mondrian_fixes_per_class_undercoverage()
    test_evaluate_sets_arithmetic()
    print("전체 통과 ✅")
