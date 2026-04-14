"""PositionSizer (Kelly Criterion + VIX + 日次損失制限) のテスト。"""
from __future__ import annotations

import dataclasses
import pytest

from src.risk.position_sizer import PositionSizer, SizingResult


class TestSizingResult:
    def test_is_frozen(self) -> None:
        r = SizingResult(position_pct=0.1, position_amount=10000,
                         kelly_raw=0.2, kelly_fraction=0.5, reason="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.position_pct = 0.5  # type: ignore[misc]


class TestKellyCalculation:
    def test_basic_kelly(self) -> None:
        ps = PositionSizer(kelly_fraction=1.0, max_position_pct=1.0)
        # W=0.6, avg_win=2000, avg_loss=1000 → R=2.0
        # Kelly = 0.6 - 0.4/2.0 = 0.4
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000)
        assert abs(r.kelly_raw - 0.4) < 0.001
        assert not r.blocked

    def test_half_kelly(self) -> None:
        ps = PositionSizer(kelly_fraction=0.5, max_position_pct=1.0)
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000)
        assert abs(r.position_pct - 0.2) < 0.001  # 0.4 × 0.5

    def test_quarter_kelly(self) -> None:
        ps = PositionSizer(kelly_fraction=0.25, max_position_pct=1.0)
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000)
        assert abs(r.position_pct - 0.1) < 0.001  # 0.4 × 0.25

    def test_negative_kelly_blocked(self) -> None:
        ps = PositionSizer()
        # 勝率30%, R=1.0 → Kelly = 0.3 - 0.7/1.0 = -0.4
        r = ps.calculate(capital=100_000, win_rate=0.3, avg_win=1000, avg_loss=1000)
        assert r.blocked
        assert r.position_pct == 0.0
        assert r.kelly_raw < 0

    def test_position_amount(self) -> None:
        ps = PositionSizer(kelly_fraction=0.5, max_position_pct=1.0)
        r = ps.calculate(capital=200_000, win_rate=0.6, avg_win=2000, avg_loss=1000)
        # position_pct = 0.2 → amount = 200,000 × 0.2 = 40,000
        assert r.position_amount == 40_000.0

    def test_zero_avg_loss(self) -> None:
        ps = PositionSizer()
        r = ps.calculate(capital=100_000, win_rate=0.8, avg_win=1000, avg_loss=0)
        assert r.blocked


class TestMaxPositionLimit:
    def test_capped_at_max(self) -> None:
        ps = PositionSizer(kelly_fraction=1.0, max_position_pct=0.3)
        # Kelly = 0.6 - 0.4/2.0 = 0.4 → capped to 0.3
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000)
        assert r.position_pct == 0.3

    def test_below_minimum_blocked(self) -> None:
        ps = PositionSizer(kelly_fraction=0.01, min_position_pct=0.05)
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000)
        assert r.blocked
        assert r.position_pct == 0.0


class TestDailyLossLimit:
    def test_blocked_after_daily_loss(self) -> None:
        ps = PositionSizer(daily_loss_limit_pct=0.02)
        ps.record_trade_pnl(-0.025)  # -2.5% → 制限超過
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000)
        assert r.blocked
        assert r.position_pct == 0.0

    def test_not_blocked_within_limit(self) -> None:
        ps = PositionSizer(daily_loss_limit_pct=0.02)
        ps.record_trade_pnl(-0.01)  # -1% → まだ余裕
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000)
        assert not r.blocked

    def test_reset_daily(self) -> None:
        ps = PositionSizer(daily_loss_limit_pct=0.02)
        ps.record_trade_pnl(-0.03)
        assert ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000).blocked

        ps.reset_daily("2026-04-15")
        assert not ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000).blocked

    def test_daily_pnl_tracking(self) -> None:
        ps = PositionSizer()
        ps.record_trade_pnl(0.01)
        ps.record_trade_pnl(-0.005)
        assert abs(ps.daily_pnl - 0.005) < 0.0001


class TestVIXAdjustment:
    def test_low_vix_no_adjustment(self) -> None:
        ps = PositionSizer(kelly_fraction=0.5, max_position_pct=1.0)
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000, vix=12.0)
        # VIX < 15 → multiplier 1.0
        assert abs(r.position_pct - 0.2) < 0.001

    def test_medium_vix_reduction(self) -> None:
        ps = PositionSizer(kelly_fraction=0.5, max_position_pct=1.0)
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000, vix=20.0)
        # VIX 15-25 → multiplier 0.7 → 0.2 × 0.7 = 0.14
        assert abs(r.position_pct - 0.14) < 0.001

    def test_high_vix_large_reduction(self) -> None:
        ps = PositionSizer(kelly_fraction=0.5, max_position_pct=1.0)
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000, vix=30.0)
        # VIX 25-35 → multiplier 0.4 → 0.2 × 0.4 = 0.08
        assert abs(r.position_pct - 0.08) < 0.001

    def test_extreme_vix_minimal_position(self) -> None:
        ps = PositionSizer(kelly_fraction=0.5, max_position_pct=1.0)
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000, vix=40.0)
        # VIX > 35 → multiplier 0.1 → 0.2 × 0.1 = 0.02
        assert abs(r.position_pct - 0.02) < 0.001

    def test_no_vix_provided(self) -> None:
        ps = PositionSizer(kelly_fraction=0.5, max_position_pct=1.0)
        r = ps.calculate(capital=100_000, win_rate=0.6, avg_win=2000, avg_loss=1000, vix=None)
        # VIX省略 → 調整なし
        assert abs(r.position_pct - 0.2) < 0.001


class TestBollingerSPYScenario:
    """実データに基づくシナリオテスト (Bollinger SPY 5年)。"""

    def test_bollinger_spy_sizing(self) -> None:
        ps = PositionSizer(kelly_fraction=0.5, max_position_pct=0.3)
        # Bollinger SPY: 勝率80%, 平均勝ち8441円, 平均負け9186円
        r = ps.calculate(
            capital=200_000, win_rate=0.8,
            avg_win=8441, avg_loss=9186,
        )
        # Kelly = 0.8 - 0.2/0.919 = 0.582 → Half Kelly = 0.291
        assert not r.blocked
        assert 0.25 < r.position_pct < 0.30  # max_position_pct=0.3でキャップ
        assert r.position_amount > 0
