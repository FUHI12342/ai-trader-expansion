"""Phase 1 Part B — ブローカー拡張・評価アセットクラス対応のテスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.brokers.base import Order, OrderSide, OrderType, Position
from src.brokers.paper_broker import PaperBroker
from src.evaluation.backtester import run_backtest
from src.strategies.base import BaseStrategy, SignalType


# ============================================================
# ヘルパー
# ============================================================

def _make_ohlcv(n: int = 300, seed: int = 42, trend: float = 0.0003) -> pd.DataFrame:
    """テスト用OHLCVデータを生成する。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start="2022-01-04", periods=n, freq="B")
    returns = rng.normal(trend, 0.015, n)
    close = 1000.0 * np.cumprod(1 + returns)
    intraday = close * rng.uniform(0.005, 0.025, n)
    high = close + intraday * rng.uniform(0, 1, n)
    low = close - intraday * rng.uniform(0, 1, n)
    open_ = close - intraday * rng.uniform(-0.5, 0.5, n)
    volume = rng.integers(100_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class _BuyAndHoldStrategy(BaseStrategy):
    """常にBUYシグナルを返すテスト戦略。"""

    @property
    def name(self) -> str:
        return "BuyAndHold"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(SignalType.FLAT, index=data.index)
        if len(signals) > 0:
            signals.iloc[0] = SignalType.BUY
        return signals


# ============================================================
# Task 1: Order の新フィールド（後方互換）
# ============================================================

class TestOrderNewFields:
    """Order dataclass の新フィールドテスト。"""

    def test_order_defaults_backward_compat(self) -> None:
        """既存コードで Order を作成できること（新フィールドのデフォルトが機能する）。"""
        order = Order(
            order_id="test001",
            symbol="7203.T",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None,
            status="filled",
        )
        assert order.asset_class == ""
        assert order.exchange == ""
        assert order.leverage == 1.0

    def test_order_with_new_fields(self) -> None:
        """新フィールドを明示的に設定できること。"""
        order = Order(
            order_id="test002",
            symbol="BTC/JPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            price=5_000_000.0,
            status="open",
            asset_class="CRYPTO",
            exchange="bitFlyer",
            leverage=2.0,
        )
        assert order.asset_class == "CRYPTO"
        assert order.exchange == "bitFlyer"
        assert order.leverage == 2.0

    def test_order_to_dict_includes_new_fields(self) -> None:
        """to_dict() が新フィールドを含むこと。"""
        order = Order(
            order_id="test003",
            symbol="7203.T",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            quantity=50.0,
            price=900.0,
            status="open",
            asset_class="STOCK",
            exchange="TSE",
            leverage=1.0,
        )
        d = order.to_dict()
        assert d["asset_class"] == "STOCK"
        assert d["exchange"] == "TSE"
        assert d["leverage"] == 1.0

    def test_order_is_immutable_with_new_fields(self) -> None:
        """frozen=True が維持されること。"""
        order = Order(
            order_id="test004",
            symbol="7203.T",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None,
            status="filled",
            asset_class="STOCK",
        )
        with pytest.raises((AttributeError, TypeError)):
            order.asset_class = "CRYPTO"  # type: ignore[misc]


# ============================================================
# Task 1: Position の新フィールド（後方互換）
# ============================================================

class TestPositionNewFields:
    """Position dataclass の新フィールドテスト。"""

    def test_position_defaults_backward_compat(self) -> None:
        """既存コードで Position を作成できること。"""
        pos = Position(
            symbol="7203.T",
            quantity=100.0,
            avg_entry_price=1000.0,
            current_price=1100.0,
            unrealized_pnl=10000.0,
            unrealized_pnl_pct=10.0,
        )
        assert pos.asset_class == ""
        assert pos.exchange == ""
        assert pos.leverage == 1.0
        assert pos.liquidation_price == 0.0
        assert pos.margin_requirement == 0.0

    def test_position_with_new_fields(self) -> None:
        """新フィールドを明示的に設定できること。"""
        pos = Position(
            symbol="BTC/JPY",
            quantity=1.0,
            avg_entry_price=5_000_000.0,
            current_price=5_500_000.0,
            unrealized_pnl=500_000.0,
            unrealized_pnl_pct=10.0,
            asset_class="CRYPTO",
            exchange="GMO",
            leverage=5.0,
            liquidation_price=3_000_000.0,
            margin_requirement=1_000_000.0,
        )
        assert pos.asset_class == "CRYPTO"
        assert pos.exchange == "GMO"
        assert pos.leverage == 5.0
        assert pos.liquidation_price == 3_000_000.0
        assert pos.margin_requirement == 1_000_000.0

    def test_position_to_dict_includes_new_fields(self) -> None:
        """to_dict() が新フィールドを含むこと。"""
        pos = Position(
            symbol="NK225",
            quantity=1.0,
            avg_entry_price=38_000.0,
            current_price=39_000.0,
            unrealized_pnl=1_000.0,
            unrealized_pnl_pct=2.6,
            asset_class="FUTURES",
            exchange="OSE",
            leverage=10.0,
            liquidation_price=30_000.0,
            margin_requirement=380_000.0,
        )
        d = pos.to_dict()
        assert d["asset_class"] == "FUTURES"
        assert d["liquidation_price"] == 30_000.0
        assert d["margin_requirement"] == 380_000.0


# ============================================================
# Task 1: OrderType.STOP_LIMIT
# ============================================================

class TestOrderTypeStopLimit:
    """OrderType.STOP_LIMIT のテスト。"""

    def test_stop_limit_exists(self) -> None:
        """STOP_LIMIT バリアントが存在すること。"""
        assert hasattr(OrderType, "STOP_LIMIT")
        assert OrderType.STOP_LIMIT == "stop_limit"

    def test_all_order_types_present(self) -> None:
        """既存の全OrderTypeが維持されていること。"""
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
        assert OrderType.STOP == "stop"
        assert OrderType.STOP_LIMIT == "stop_limit"


# ============================================================
# Task 2: PaperBroker fee_schedule
# ============================================================

class TestPaperBrokerFeeSchedule:
    """PaperBroker の fee_schedule テスト。"""

    def test_fee_schedule_not_required_backward_compat(self) -> None:
        """fee_schedule なしで従来通り動作すること。"""
        broker = PaperBroker(initial_balance=1_000_000.0, fee_rate=0.001)
        broker.update_price("7203.T", 1000.0)
        order = broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        # 手数料は fee_rate=0.001 ベースで計算される
        expected_fee = 1000.0 * 100 * 0.001
        assert abs(order.fee - expected_fee) < 1.0  # スリッページによる微差を許容

    def test_fee_schedule_stock_rate(self) -> None:
        """STOCK アセットクラスの fee_schedule が適用されること。"""
        # ここでは _get_fee_rate をユニットテストとして直接検証する
        broker = PaperBroker(
            fee_rate=0.001,
            fee_schedule={"STOCK": 0.0008, "CRYPTO": 0.0005},
        )
        assert broker._get_fee_rate("STOCK") == 0.0008

    def test_fee_schedule_crypto_rate(self) -> None:
        """CRYPTO アセットクラスの fee_schedule が適用されること。"""
        broker = PaperBroker(
            fee_rate=0.001,
            fee_schedule={"STOCK": 0.0008, "CRYPTO": 0.0005},
        )
        assert broker._get_fee_rate("CRYPTO") == 0.0005

    def test_fee_schedule_fallback_to_default(self) -> None:
        """fee_schedule に存在しないアセットクラスはデフォルト rate を使うこと。"""
        broker = PaperBroker(
            fee_rate=0.001,
            fee_schedule={"STOCK": 0.0008},
        )
        assert broker._get_fee_rate("FUTURES") == 0.001

    def test_fee_schedule_empty_asset_class_fallback(self) -> None:
        """asset_class が空文字の場合はデフォルト rate を使うこと。"""
        broker = PaperBroker(
            fee_rate=0.002,
            fee_schedule={"STOCK": 0.0005},
        )
        assert broker._get_fee_rate("") == 0.002

    def test_no_fee_schedule_default_rate(self) -> None:
        """fee_schedule=None の場合はデフォルト rate を返すこと。"""
        broker = PaperBroker(fee_rate=0.0015)
        assert broker._get_fee_rate("CRYPTO") == 0.0015


# ============================================================
# Task 3: run_backtest periods_per_year
# ============================================================

class TestRunBacktestPeriodsPerYear:
    """run_backtest の periods_per_year テスト。"""

    def test_periods_per_year_none_defaults_to_252(self) -> None:
        """periods_per_year=None がデフォルト 252 として動作すること。"""
        data = _make_ohlcv(n=300)
        strategy = _BuyAndHoldStrategy()
        result_none = run_backtest(strategy, data, periods_per_year=None)
        result_252 = run_backtest(strategy, data, periods_per_year=252)
        assert result_none.metrics.annualized_return_pct == result_252.metrics.annualized_return_pct
        assert result_none.metrics.sharpe_ratio == result_252.metrics.sharpe_ratio

    def test_periods_per_year_365_differs_from_252(self) -> None:
        """periods_per_year=365（暗号資産）と 252 では年率指標が異なること。"""
        data = _make_ohlcv(n=300)
        strategy = _BuyAndHoldStrategy()
        result_stock = run_backtest(strategy, data, periods_per_year=252)
        result_crypto = run_backtest(strategy, data, periods_per_year=365)
        # 年率指標は異なるはず（365日基準のほうが annualized return が大きくなる傾向）
        assert result_stock.metrics.annualized_return_pct != result_crypto.metrics.annualized_return_pct

    def test_periods_per_year_365_higher_sharpe(self) -> None:
        """上昇トレンドデータで periods_per_year=365 の Sharpe が 252 より大きいこと。"""
        rng = np.random.default_rng(1)
        n = 300
        dates = pd.date_range("2022-01-04", periods=n, freq="B")
        # 強い上昇トレンドを作成
        returns = rng.normal(0.002, 0.005, n)
        close = 1000.0 * np.cumprod(1 + returns)
        data = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1_000_000.0},
            index=dates,
        )
        strategy = _BuyAndHoldStrategy()
        result_252 = run_backtest(strategy, data, periods_per_year=252)
        result_365 = run_backtest(strategy, data, periods_per_year=365)
        # Sharpe は sqrt(periods_per_year) に比例するため 365 > 252 のはず
        assert result_365.metrics.sharpe_ratio > result_252.metrics.sharpe_ratio
