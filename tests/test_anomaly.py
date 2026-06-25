"""무지도 이상탐지 핵심 로직의 결정적 단위 테스트 (GPU·백본·네트워크 없음).

`scripts/anomaly_detect.py`의 순수 numpy 함수만 검증한다. 이 트랙의 핵심 주장은
"정상 feature 분포로부터의 이탈을 점수화하면 이상이 정상보다 높은 점수를 받는다"와
"coreset로 memory bank를 압축해도(무거움→가벼움) 그 순서가 유지된다"이다.
README의 AUROC 표는 실제 백본 feature로 측정한 값이지만, 여기서는 **합성 데이터로
그 성질이 코드에서 성립하는지**를 재현 가능하게 못박는다.

실행: python tests/test_anomaly.py   (pytest 불필요, GPU 불필요)
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import anomaly_detect as ad  # noqa: E402


def _clusters(seed: int, d: int = 16):
    """정상(원점 클러스터)·이상(멀리 떨어진) feature 생성."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(0, 1, size=(300, d)).astype(np.float32)
    return rng, normal


def test_coreset_size_and_coverage():
    """coreset 인덱스 개수가 ratio*N이고, 중복 없이 고른 점들을 뽑는다."""
    _, normal = _clusters(0)
    for ratio in (0.1, 0.25, 0.5):
        idx = ad.greedy_coreset_indices(normal, ratio=ratio, seed=1)
        assert len(idx) == int(np.ceil(len(normal) * ratio))
        assert len(set(idx.tolist())) == len(idx), "coreset에 중복 선택"
    # ratio>=1이면 전체 반환.
    assert len(ad.greedy_coreset_indices(normal, ratio=1.0)) == len(normal)
    print("OK coreset 크기·중복없음")


def test_knn_anomaly_separates():
    """이상 패치 점수가 정상보다 높다 — 압축률을 낮춰도 순서 유지."""
    rng, normal = _clusters(1)
    normal_test = rng.normal(0, 1, size=(20, 16)).astype(np.float32)
    anom_test = rng.normal(7, 1, size=(20, 16)).astype(np.float32)
    for ratio in (1.0, 0.25, 0.05):
        bank = normal[ad.greedy_coreset_indices(normal, ratio=ratio, seed=2)]
        _, s_norm = ad.knn_anomaly_scores(normal_test, bank)
        _, s_anom = ad.knn_anomaly_scores(anom_test, bank)
        assert s_anom > s_norm, f"ratio={ratio}: 이상({s_anom}) <= 정상({s_norm})"
    print("OK kNN 이상>정상 (압축률 1.0/0.25/0.05 모두)")


def test_padim_mahalanobis():
    """PaDiM 가우시안 적합 + Mahalanobis가 이상을 분리한다."""
    rng, _ = _clusters(2)
    feats = rng.normal(0, 1, size=(150, 3, 8)).astype(np.float32)  # (N, L=3, D=8)
    mean, inv_cov = ad.fit_padim_gaussian(feats)
    assert mean.shape == (3, 8) and inv_cov.shape == (3, 8, 8)
    _, m_norm = ad.mahalanobis_scores(rng.normal(0, 1, (3, 8)), mean, inv_cov)
    _, m_anom = ad.mahalanobis_scores(np.full((3, 8), 7.0), mean, inv_cov)
    assert m_anom > m_norm, "Mahalanobis 이상<=정상"
    print("OK PaDiM Mahalanobis 이상>정상")


def test_image_auroc():
    """완전 분리면 AUROC=1.0, 무작위면 라벨 한쪽뿐일 때 nan."""
    scores = np.array([0.1, 0.2, 0.15, 0.9, 1.0, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert abs(ad.image_auroc(scores, labels) - 1.0) < 1e-9
    assert np.isnan(ad.image_auroc(scores, np.zeros_like(labels)))
    print("OK image AUROC")


def _run_all():
    test_coreset_size_and_coverage()
    test_knn_anomaly_separates()
    test_padim_mahalanobis()
    test_image_auroc()
    print("\n전체 통과 — 무지도 AD 순수 로직 4개")


if __name__ == "__main__":
    _run_all()
