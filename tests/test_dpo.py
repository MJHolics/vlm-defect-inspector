"""DPO 구조화 정렬의 순수 로직 단위테스트 — 모델·GPU 불요.

불변식:
  1) 선호쌍은 chosen=완전한 JSON, rejected=무효/필드누락(학습 신호 성립).
  2) parse_score는 산문·깨진 JSON을 무효로, 부분 JSON은 필드 비율로 채점.
  3) preference_accuracy는 margin 부호로 승패, 동률(==0)은 0.5로 계수.
  4) mean_margin은 평균, 학습 전(참조=정책, 전부 0)은 0.
  5) build_pairs는 결정적(같은 seed → 같은 쌍).
"""
from __future__ import annotations

from scripts.dpo_structured import (
    build_pairs, mean_margin, parse_score, preference_accuracy,
)


def test_pairs_have_valid_chosen_and_invalid_rejected():
    for p in build_pairs(18, seed=0):
        assert set(p) == {"prompt", "chosen", "rejected"}
        ok_c, cov_c = parse_score(p["chosen"][0]["content"])
        ok_r, cov_r = parse_score(p["rejected"][0]["content"])
        assert ok_c and cov_c == 1.0            # chosen = 필수필드 완비 JSON
        assert (not ok_r) or cov_r < 1.0        # rejected = 무효이거나 필드 누락


def test_parse_score_grades_partial_json():
    assert parse_score("그냥 산문 설명입니다")[0] is False
    assert parse_score('{"defect_type": "x", "severity": medium')[0] is False  # 깨진 JSON
    assert parse_score('{"defect_type":"x"}') == (True, 0.25)
    assert parse_score(
        '{"defect_type":"scratch","severity":"low","confidence":0.8,"evidence":"e"}'
    ) == (True, 1.0)


def test_preference_accuracy_counts_ties_as_half():
    assert preference_accuracy([0.5, -0.2, 0.0, 1.0]) == 0.625
    assert preference_accuracy([0.0, 0.0]) == 0.5   # 학습 전 = 참조와 동일
    assert preference_accuracy([1.0, 2.0, 3.0]) == 1.0
    assert preference_accuracy([-1.0]) == 0.0
    assert preference_accuracy([]) == 0.0


def test_mean_margin():
    assert abs(mean_margin([0.5, -0.2, 0.0, 1.0]) - 0.325) < 1e-9
    assert mean_margin([]) == 0.0
    assert mean_margin([2.0, -2.0]) == 0.0          # 상쇄


def test_build_pairs_deterministic():
    assert build_pairs(10, seed=0) == build_pairs(10, seed=0)
    assert build_pairs(10, seed=0) != build_pairs(10, seed=1)
