"""head_to_head 순수 로직 단위테스트 — 백본·데이터·torch 불요.

층화분할의 균형·비겹침, 결함 표본 예산, 교차점 검출, 임계 정확도를 검증한다.
실험 자체(GPU 학습)는 테스트하지 않고, 결론을 좌우하는 *수치 로직*만 고정한다.
"""
import numpy as np

from scripts.head_to_head import (
    accuracy_at_threshold,
    find_crossover,
    sample_defects,
    stratified_split,
    youden_threshold,
)


def test_split_is_balanced_and_disjoint():
    labels = np.array([0] * 30 + [1] * 12)
    test_idx, pool_idx = stratified_split(labels, n_test_per_class=5, seed=1)
    assert len(test_idx) == 10
    assert (labels[test_idx] == 0).sum() == 5
    assert (labels[test_idx] == 1).sum() == 5
    assert set(test_idx).isdisjoint(set(pool_idx))
    # 모든 인덱스가 test 또는 pool에 정확히 한 번.
    assert sorted(test_idx.tolist() + pool_idx.tolist()) == list(range(len(labels)))


def test_split_raises_when_class_too_small():
    labels = np.array([0] * 10 + [1] * 3)
    try:
        stratified_split(labels, n_test_per_class=5, seed=0)
    except ValueError:
        return
    raise AssertionError("결함이 테스트 요청보다 적으면 ValueError 나야 함")


def test_split_is_deterministic_per_seed():
    labels = np.array([0] * 20 + [1] * 20)
    a = stratified_split(labels, 5, seed=7)
    b = stratified_split(labels, 5, seed=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_sample_defects_budget():
    pool_def = np.arange(10, 25)  # 15개 결함 인덱스
    s = sample_defects(pool_def, 4, seed=2)
    assert len(s) == 4
    assert set(s).issubset(set(pool_def.tolist()))
    # n >= 전체면 전체 반환.
    assert len(sample_defects(pool_def, 100, seed=2)) == 15


def test_find_crossover():
    # 무지도 0.80, 지도가 N=10에서 처음 0.85로 넘어섬.
    assert find_crossover([2, 5, 10, 20], [0.60, 0.75, 0.85, 0.92], 0.80) == 10
    # 첫 점이 이미 넘으면 그 N.
    assert find_crossover([2, 5], [0.81, 0.9], 0.80) == 2
    # 끝까지 못 넘으면 None.
    assert find_crossover([2, 5, 10], [0.5, 0.6, 0.7], 0.80) is None


def test_accuracy_and_youden():
    scores = np.array([0.05, 0.2, 0.6, 0.95])
    labels = np.array([0, 0, 1, 1])
    thr = youden_threshold(scores, labels)
    assert accuracy_at_threshold(scores, labels, thr) == 1.0
    # 임계가 너무 높으면 결함을 다 놓쳐 정확도 0.5(정상만 맞음).
    assert accuracy_at_threshold(scores, labels, 1.5) == 0.5
