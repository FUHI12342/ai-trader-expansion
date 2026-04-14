"""DriftDetector と ABTestManager のテスト。"""
from __future__ import annotations

import dataclasses

import pytest

from src.learning.drift_detector import (
    DriftDetector, DriftResult, DriftSeverity, _FallbackDetector,
)
from src.learning.ab_test import ABTestManager, ABTestResult, _safe_mean, _welch_t_test


# ---------------------------------------------------------------------------
# DriftResult テスト
# ---------------------------------------------------------------------------

class TestDriftResult:

    def test_is_frozen(self) -> None:
        r = DriftResult(detected=False, method="test", severity=DriftSeverity.NONE, details={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.detected = True  # type: ignore[misc]

    def test_is_warning(self) -> None:
        r = DriftResult(detected=True, method="test", severity=DriftSeverity.WARNING, details={})
        assert r.is_warning
        assert not r.is_drift

    def test_is_drift(self) -> None:
        r = DriftResult(detected=True, method="test", severity=DriftSeverity.DRIFT, details={})
        assert r.is_drift
        assert not r.is_warning


# ---------------------------------------------------------------------------
# _FallbackDetector テスト
# ---------------------------------------------------------------------------

class TestFallbackDetector:

    def test_no_drift_during_warmup(self) -> None:
        det = _FallbackDetector(short_window=5, long_window=20)
        for i in range(15):
            result = det.update(1.0)
        assert not result.detected

    def test_no_drift_stable_data(self) -> None:
        det = _FallbackDetector(short_window=5, long_window=20)
        for i in range(100):
            result = det.update(1.0)
        assert not result.detected

    def test_drift_on_regime_change(self) -> None:
        det = _FallbackDetector(
            short_window=5, long_window=50,
            warning_threshold=1.5, drift_threshold=2.5,
        )
        # 安定期間
        for _ in range(60):
            det.update(0.0)

        # 急激な変化
        detected = False
        for _ in range(20):
            result = det.update(10.0)
            if result.detected:
                detected = True
                break
        assert detected

    def test_reset(self) -> None:
        det = _FallbackDetector(short_window=5, long_window=10)
        for i in range(20):
            det.update(float(i))
        det.reset()
        assert len(det._short) == 0
        assert len(det._long) == 0


# ---------------------------------------------------------------------------
# DriftDetector テスト
# ---------------------------------------------------------------------------

class TestDriftDetector:

    def test_init_default(self) -> None:
        det = DriftDetector()
        assert len(det.methods) >= 1
        assert det.update_count == 0

    def test_init_fallback(self) -> None:
        det = DriftDetector(methods=["fallback"])
        assert "fallback" in det.methods

    def test_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="未対応のメソッド"):
            DriftDetector(methods=["nonexistent"])

    def test_update_increments_count(self) -> None:
        det = DriftDetector(methods=["fallback"])
        det.update(1.0)
        det.update(2.0)
        assert det.update_count == 2

    def test_no_drift_stable(self) -> None:
        det = DriftDetector(methods=["fallback"])
        for _ in range(200):
            result = det.update(1.0)
        assert not result.detected

    def test_reset(self) -> None:
        det = DriftDetector(methods=["fallback"])
        for _ in range(50):
            det.update(1.0)
        det.reset()
        assert det.update_count == 0


# ---------------------------------------------------------------------------
# ABTestResult テスト
# ---------------------------------------------------------------------------

class TestABTestResult:

    def test_is_frozen(self) -> None:
        r = ABTestResult(
            test_id="t1", champion_id="a", challenger_id="b",
            champion_mean=1.0, challenger_mean=2.0,
            p_value=0.01, n_samples=50, winner="challenger",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.winner = "champion"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ABTestManager テスト
# ---------------------------------------------------------------------------

class TestABTestManager:

    def test_start_test(self) -> None:
        mgr = ABTestManager()
        tid = mgr.start_test("model_v1", "model_v2")
        assert tid.startswith("ab_")
        assert mgr.test_count == 1

    def test_start_test_custom_id(self) -> None:
        mgr = ABTestManager()
        tid = mgr.start_test("a", "b", test_id="custom_test")
        assert tid == "custom_test"

    def test_update(self) -> None:
        mgr = ABTestManager()
        tid = mgr.start_test("a", "b")
        mgr.update(tid, champion_pnl=100.0, challenger_pnl=110.0)
        # エラーなし

    def test_update_nonexistent_raises(self) -> None:
        mgr = ABTestManager()
        with pytest.raises(KeyError):
            mgr.update("nonexistent", 1.0, 2.0)

    def test_evaluate_insufficient_samples(self) -> None:
        mgr = ABTestManager(min_samples=30)
        tid = mgr.start_test("a", "b")
        for _ in range(10):
            mgr.update(tid, 1.0, 2.0)

        result = mgr.evaluate(tid)
        assert result.winner == "inconclusive"
        assert result.n_samples == 10

    def test_evaluate_challenger_wins(self) -> None:
        mgr = ABTestManager(min_samples=30, significance_level=0.05)
        tid = mgr.start_test("a", "b")

        import random
        rng = random.Random(42)
        for _ in range(100):
            mgr.update(tid, rng.gauss(0, 1), rng.gauss(2, 1))

        result = mgr.evaluate(tid)
        assert result.challenger_mean > result.champion_mean
        # 明確な差があるのでchallengerが勝つはず
        assert result.winner == "challenger"

    def test_evaluate_champion_wins(self) -> None:
        mgr = ABTestManager(min_samples=30, significance_level=0.05)
        tid = mgr.start_test("a", "b")

        import random
        rng = random.Random(42)
        for _ in range(100):
            mgr.update(tid, rng.gauss(3, 1), rng.gauss(0, 1))

        result = mgr.evaluate(tid)
        assert result.champion_mean > result.challenger_mean
        assert result.winner == "champion"

    def test_conclude_test(self) -> None:
        mgr = ABTestManager(min_samples=5)
        tid = mgr.start_test("a", "b")
        for _ in range(10):
            mgr.update(tid, 1.0, 1.0)

        result = mgr.conclude(tid)
        assert isinstance(result, ABTestResult)

        # 終了後のupdateはエラー
        with pytest.raises(ValueError, match="既に終了"):
            mgr.update(tid, 1.0, 1.0)

    def test_get_active_tests(self) -> None:
        mgr = ABTestManager(min_samples=5)
        tid1 = mgr.start_test("a", "b")
        tid2 = mgr.start_test("c", "d")

        assert len(mgr.get_active_tests()) == 2

        for _ in range(10):
            mgr.update(tid1, 1.0, 1.0)
        mgr.conclude(tid1)

        assert len(mgr.get_active_tests()) == 1
        assert tid2 in mgr.get_active_tests()


# ---------------------------------------------------------------------------
# ヘルパー関数テスト
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_safe_mean_empty(self) -> None:
        assert _safe_mean([]) == 0.0

    def test_safe_mean_values(self) -> None:
        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0

    def test_welch_t_test_same_distribution(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        p = _welch_t_test(a, b)
        assert p > 0.5  # 同一分布なのでp値は大きい

    def test_welch_t_test_different_distribution(self) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(0, 1) for _ in range(50)]
        b = [rng.gauss(5, 1) for _ in range(50)]
        p = _welch_t_test(a, b)
        assert p < 0.05  # 明確に異なるのでp値は小さい

    def test_welch_t_test_insufficient_samples(self) -> None:
        assert _welch_t_test([1.0], [2.0]) == 1.0
