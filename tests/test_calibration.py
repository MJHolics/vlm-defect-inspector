"""Calibration 핵심(ECE·NLL·temperature) 단위테스트 — 순수 numpy, 네트워크·데이터 불요.

불변식:
  1) 과신(overconfident) 로짓은 ECE가 높고, temperature scaling이 이를 낮춘다.
  2) 과신 모델의 적합 온도 T>1 (확률을 부드럽게).
  3) Temperature scaling은 argmax 불변 → 정확도 그대로.
  4) 완벽 보정 확률의 ECE는 0에 가깝다.
  5) fit_temperature는 결정적(같은 입력 → 같은 T).
"""
from __future__ import annotations

import numpy as np

from scripts.calibrate_edge import (
    accuracy, brier, ece, fit_temperature, nll, softmax,
)


def _overconfident_logits(n=600, C=6, seed=0):
    """정답률은 ~75%인데 로짓 스케일이 커서 확률이 과신되는 합성 데이터."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, C, size=n)
    logits = rng.normal(0, 1, size=(n, C))
    # 75% 표본만 true 클래스에 큰 마진 → 나머지는 틀리지만 여전히 큰 마진(과신).
    for i in range(n):
        target = labels[i] if rng.random() < 0.75 else (labels[i] + 1) % C
        logits[i, target] += 8.0        # 큰 마진 = 과신 유도
    return logits, labels


def test_overconfident_has_high_ece_and_temperature_reduces_it():
    logits, labels = _overconfident_logits()
    p0 = softmax(logits, 1.0)
    e0 = ece(p0, labels)
    T = fit_temperature(logits, labels)
    e1 = ece(softmax(logits, T), labels)
    assert e0 > 0.10, f"과신 데이터 ECE가 낮음: {e0}"
    assert e1 < e0, f"temperature가 ECE를 못 낮춤: {e0} → {e1}"
    assert T > 1.0, f"과신인데 T<=1: {T}"


def test_temperature_preserves_accuracy():
    logits, labels = _overconfident_logits(seed=3)
    T = fit_temperature(logits, labels)
    assert accuracy(logits, labels) == float((softmax(logits, T).argmax(1) == labels).mean())


def test_nll_minimized_near_fitted_temperature():
    logits, labels = _overconfident_logits(seed=7)
    T = fit_temperature(logits, labels)
    base = nll(logits, labels, T)
    assert base <= nll(logits, labels, 1.0)          # 보정이 NLL 개선
    assert base <= nll(logits, labels, T * 1.5) + 1e-9
    assert base <= nll(logits, labels, T * 0.6) + 1e-9


def test_perfect_calibration_low_ece():
    # 확률=정확도가 일치하도록 구성: confidence p인 표본이 정확히 p 비율로 맞음.
    rng = np.random.default_rng(1)
    C = 2
    probs = []
    labels = []
    for conf in np.linspace(0.55, 0.99, 12):
        for _ in range(200):
            correct = rng.random() < conf
            row = np.array([1 - conf, conf]) if True else None
            probs.append(row)
            labels.append(1 if correct else 0)   # 클래스1을 '맞음'으로 매핑
    probs = np.array(probs); labels = np.array(labels)
    assert ece(probs, labels, n_bins=12) < 0.05


def test_fit_temperature_deterministic():
    logits, labels = _overconfident_logits(seed=11)
    assert fit_temperature(logits, labels) == fit_temperature(logits, labels)


def test_brier_between_zero_and_two():
    logits, labels = _overconfident_logits(seed=5)
    b = brier(softmax(logits, 1.0), labels)
    assert 0.0 <= b <= 2.0
