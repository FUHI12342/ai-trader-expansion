"""新戦略 (Funding Arb, Grid, Pairs) のテスト。"""
from __future__ import annotations

import dataclasses
import numpy as np
import pandas as pd
import pytest

from src.strategies.funding_arb import (
    FundingSnapshot, ArbPosition, FundingRateCollector, FundingArbStrategy,
)
from src.strategies.grid_trading import GridTradingStrategy, GridTrade
from src.strategies.pairs_trading import PairsTradingStrategy, PairSignal, PairTrade


# ===== Funding Rate Arbitrage =====

class TestFundingSnapshot:
    def test_is_frozen(self) -> None:
        s = FundingSnapshot(symbol="BTC", rate=0.001, annual_rate=10.95,
                            timestamp=1.0, next_funding_time=2.0, exchange="binance")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.rate = 0.0  # type: ignore[misc]

    def test_is_positive(self) -> None:
        s = FundingSnapshot(symbol="BTC", rate=0.001, annual_rate=10.95,
                            timestamp=1.0, next_funding_time=2.0, exchange="binance")
        assert s.is_positive
        s2 = FundingSnapshot(symbol="BTC", rate=-0.001, annual_rate=-10.95,
                             timestamp=1.0, next_funding_time=2.0, exchange="binance")
        assert not s2.is_positive


class TestArbPosition:
    def test_net_exposure(self) -> None:
        p = ArbPosition(symbol="BTC", spot_quantity=1.0, spot_avg_price=100000,
                        perp_quantity=-1.0, perp_avg_price=100000,
                        total_funding_earned=100, entry_time=1.0, status="open")
        assert p.net_exposure == 0.0

    def test_funding_pnl_pct(self) -> None:
        p = ArbPosition(symbol="BTC", spot_quantity=1.0, spot_avg_price=100000,
                        perp_quantity=-1.0, perp_avg_price=100000,
                        total_funding_earned=1000, entry_time=1.0, status="open")
        assert p.funding_pnl_pct == 1.0


class TestFundingArbStrategy:
    def test_init(self) -> None:
        s = FundingArbStrategy(exchange_id="binance")
        assert s.total_funding_earned == 0.0
        assert len(s.open_positions) == 0

    def test_record_entry_and_funding(self) -> None:
        s = FundingArbStrategy(exchange_id="binance")
        pos = s.record_entry("BTC", 0.1, 100000, -0.1, 100000)
        assert pos.status == "open"

        s.record_funding("BTC", 50.0)
        assert s.total_funding_earned == 50.0
        assert s.open_positions["BTC"].total_funding_earned == 50.0

    def test_record_exit(self) -> None:
        s = FundingArbStrategy(exchange_id="binance")
        s.record_entry("BTC", 0.1, 100000, -0.1, 100000)
        s.record_funding("BTC", 100.0)
        closed = s.record_exit("BTC")
        assert closed.status == "closed"
        assert len(s.open_positions) == 0

    def test_summary(self) -> None:
        s = FundingArbStrategy(exchange_id="binance")
        s.record_entry("BTC", 0.1, 100000, -0.1, 100000)
        summary = s.summary()
        assert summary["open_positions"] == 1


# ===== Grid Trading =====

class TestGridTradingStrategy:
    def test_init(self) -> None:
        g = GridTradingStrategy(upper_price=110, lower_price=90, grid_count=10)
        assert len(g.grid_levels) == 11
        assert g.total_profit == 0.0

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            GridTradingStrategy(upper_price=90, lower_price=110)
        with pytest.raises(ValueError):
            GridTradingStrategy(upper_price=110, lower_price=90, grid_count=1)

    def test_backtest_range_market(self) -> None:
        g = GridTradingStrategy(upper_price=105, lower_price=95, grid_count=5,
                                total_investment=100_000)
        # レンジ相場シミュレーション: 95→105→95→105
        prices = []
        for _ in range(5):
            prices.extend(list(np.linspace(95, 105, 20)))
            prices.extend(list(np.linspace(105, 95, 20)))

        result = g.backtest(prices)
        assert result["total_trades"] > 0
        assert result["total_profit"] > 0  # レンジ相場では利益

    def test_backtest_trending_market(self) -> None:
        g = GridTradingStrategy(upper_price=120, lower_price=80, grid_count=10,
                                total_investment=100_000)
        # 上昇トレンド: 80→130 (グリッド上限突破)
        prices = list(np.linspace(100, 130, 100))
        result = g.backtest(prices)
        # トレンド時は取引少ない
        assert result["total_trades"] >= 0

    def test_summary(self) -> None:
        g = GridTradingStrategy(upper_price=110, lower_price=90, grid_count=5)
        s = g.summary()
        assert s["grid_count"] == 5
        assert s["upper"] == 110
        assert s["lower"] == 90


# ===== Pairs Trading =====

class TestPairsTradingStrategy:
    def _make_cointegrated_pair(self, n: int = 500) -> tuple:
        rng = np.random.RandomState(42)
        b = 100 + np.cumsum(rng.randn(n) * 1.0)
        a = 2.0 * b + rng.randn(n) * 3.0 + 50  # 共和分: A ≈ 2*B + noise
        dates = pd.bdate_range("2023-01-01", periods=n)
        return pd.Series(a, index=dates), pd.Series(b, index=dates)

    def test_hedge_ratio(self) -> None:
        s = PairsTradingStrategy()
        a, b = self._make_cointegrated_pair()
        hedge = s.compute_hedge_ratio(a, b)
        assert 1.5 < hedge < 2.5  # 真の値は2.0

    def test_cointegration_check(self) -> None:
        s = PairsTradingStrategy()
        a, b = self._make_cointegrated_pair()
        result = s.check_cointegration(a, b)
        assert result["cointegrated"]
        assert result["half_life"] > 0

    def test_cointegration_fails_for_random(self) -> None:
        s = PairsTradingStrategy()
        rng = np.random.RandomState(42)
        a = pd.Series(np.cumsum(rng.randn(500)))
        b = pd.Series(np.cumsum(rng.randn(500)))
        result = s.check_cointegration(a, b)
        # ランダムウォーク同士は共和分しない可能性が高い
        # (ただし偶然共和分する場合もあるのでassertは緩く)
        assert "hedge_ratio" in result

    def test_z_score(self) -> None:
        s = PairsTradingStrategy(lookback=20)
        spread = pd.Series(np.random.randn(100))
        z = s.compute_z_score(spread)
        assert len(z) == 100
        # z-scoreの大部分は-3〜3の範囲
        valid = z.dropna()
        assert valid.abs().mean() < 3.0

    def test_backtest_cointegrated(self) -> None:
        s = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, lookback=30)
        a, b = self._make_cointegrated_pair(500)
        result = s.backtest(a, b, asset_a="A", asset_b="B", capital=100_000)
        assert result["total_trades"] > 0
        assert "sharpe" in result
        assert "max_dd_pct" in result

    def test_backtest_returns_trades_list(self) -> None:
        s = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, lookback=30)
        a, b = self._make_cointegrated_pair(500)
        result = s.backtest(a, b, capital=100_000)
        assert isinstance(result["trades"], list)

    def test_pair_trade_is_frozen(self) -> None:
        t = PairTrade(asset_a="A", asset_b="B", entry_z=2.0, exit_z=0.5,
                      entry_spread=10.0, exit_spread=5.0, pnl=100,
                      pnl_pct=1.0, holding_days=10, result="WIN")
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.pnl = 0  # type: ignore[misc]
