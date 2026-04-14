"""Phase 6: ポートフォリオ最適化 + リスク管理のテスト。"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.risk.portfolio_optimizer import PortfolioOptimizer, PortfolioAllocation, RebalanceOrder
from src.risk.drawdown_controller import DrawdownController, DrawdownAction, DrawdownStatus
from src.risk.correlation_monitor import CorrelationMonitor, CorrelationAlert


# ---------------------------------------------------------------------------
# テストデータヘルパー
# ---------------------------------------------------------------------------

def _make_returns(n: int = 252, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    data = rng.randn(n, n_assets) * 0.01 + 0.0003
    cols = [f"asset_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_equity(n: int = 252, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    returns = rng.randn(n) * 0.01 + 0.0003
    equity = 1_000_000 * np.cumprod(1 + returns)
    return pd.Series(equity, index=pd.bdate_range("2023-01-01", periods=n))


# ---------------------------------------------------------------------------
# PortfolioAllocation テスト
# ---------------------------------------------------------------------------

class TestPortfolioAllocation:

    def test_is_frozen(self) -> None:
        pa = PortfolioAllocation(
            weights={"a": 0.5, "b": 0.5},
            expected_return=0.1, expected_risk=0.15,
            sharpe_ratio=0.67, method="equal_weight",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            pa.method = "hrp"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PortfolioOptimizer テスト
# ---------------------------------------------------------------------------

class TestPortfolioOptimizer:

    def test_init_invalid_method(self) -> None:
        with pytest.raises(ValueError, match="未対応の手法"):
            PortfolioOptimizer(method="nonexistent")

    def test_equal_weight(self) -> None:
        returns = _make_returns(252, 4)
        opt = PortfolioOptimizer(method="equal_weight")
        result = opt.optimize(returns)

        assert result.method == "equal_weight"
        assert len(result.weights) == 4
        for w in result.weights.values():
            assert abs(w - 0.25) < 1e-10
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10

    def test_risk_parity(self) -> None:
        returns = _make_returns(252, 3)
        opt = PortfolioOptimizer(method="risk_parity")
        result = opt.optimize(returns)

        assert result.method == "risk_parity"
        assert len(result.weights) == 3
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10
        # 全ウェイトが正
        for w in result.weights.values():
            assert w > 0

    def test_hrp(self) -> None:
        returns = _make_returns(252, 3)
        opt = PortfolioOptimizer(method="hrp")
        result = opt.optimize(returns)

        assert result.method == "hrp"
        assert abs(sum(result.weights.values()) - 1.0) < 1e-10

    def test_optimize_empty_raises(self) -> None:
        opt = PortfolioOptimizer()
        with pytest.raises(ValueError, match="リターンデータが空"):
            opt.optimize(pd.DataFrame())

    def test_sharpe_ratio_computed(self) -> None:
        returns = _make_returns(252, 3)
        opt = PortfolioOptimizer(risk_free_rate=0.0)
        result = opt.optimize(returns)
        # シャープレシオは数値として計算される
        assert isinstance(result.sharpe_ratio, float)

    def test_rebalance_generates_orders(self) -> None:
        opt = PortfolioOptimizer()
        target = PortfolioAllocation(
            weights={"A": 0.5, "B": 0.3, "C": 0.2},
            expected_return=0.1, expected_risk=0.15,
            sharpe_ratio=0.67, method="equal_weight",
        )
        current = {"A": 0.4, "B": 0.4, "C": 0.2}
        orders = opt.rebalance(current, target, threshold=0.05)

        assert len(orders) == 3
        for o in orders:
            assert isinstance(o, RebalanceOrder)

        # A: 0.4 → 0.5 = buy (delta=0.1 > threshold)
        a_order = [o for o in orders if o.asset == "A"][0]
        assert a_order.action == "buy"

        # B: 0.4 → 0.3 = sell
        b_order = [o for o in orders if o.asset == "B"][0]
        assert b_order.action == "sell"

        # C: 0.2 → 0.2 = hold
        c_order = [o for o in orders if o.asset == "C"][0]
        assert c_order.action == "hold"

    def test_rebalance_new_asset(self) -> None:
        opt = PortfolioOptimizer()
        target = PortfolioAllocation(
            weights={"A": 0.5, "B": 0.5},
            expected_return=0.1, expected_risk=0.15,
            sharpe_ratio=0.67, method="equal_weight",
        )
        # D は target に無い → sell
        current = {"A": 0.5, "D": 0.5}
        orders = opt.rebalance(current, target, threshold=0.02)

        d_order = [o for o in orders if o.asset == "D"][0]
        assert d_order.action == "sell"

        b_order = [o for o in orders if o.asset == "B"][0]
        assert b_order.action == "buy"


# ---------------------------------------------------------------------------
# DrawdownController テスト
# ---------------------------------------------------------------------------

class TestDrawdownController:

    def test_init_invalid_thresholds(self) -> None:
        with pytest.raises(ValueError, match="閾値は"):
            DrawdownController(reduce_threshold=0.1, halt_threshold=0.05)

    def test_normal_state(self) -> None:
        ctrl = DrawdownController()
        equity = pd.Series([100, 101, 102, 103, 104])
        status = ctrl.check(equity)

        assert status.action == DrawdownAction.NORMAL
        assert status.exposure_ratio == 1.0
        assert status.current_drawdown == 0.0

    def test_reduce_exposure(self) -> None:
        ctrl = DrawdownController(
            reduce_threshold=0.05, halt_threshold=0.10, emergency_threshold=0.15,
        )
        # 100 → 94 = 6% drawdown (> reduce, < halt)
        equity = pd.Series([100, 100, 94])
        status = ctrl.check(equity)

        assert status.action == DrawdownAction.REDUCE_EXPOSURE
        assert 0.0 < status.exposure_ratio < 1.0

    def test_halt_trading(self) -> None:
        ctrl = DrawdownController(
            reduce_threshold=0.05, halt_threshold=0.10, emergency_threshold=0.15,
        )
        equity = pd.Series([100, 100, 90])  # 10% dd
        status = ctrl.check(equity)

        assert status.action == DrawdownAction.HALT_TRADING
        assert status.exposure_ratio == 0.0

    def test_emergency_exit(self) -> None:
        ctrl = DrawdownController(
            reduce_threshold=0.05, halt_threshold=0.10, emergency_threshold=0.15,
        )
        equity = pd.Series([100, 100, 84])  # 16% dd
        status = ctrl.check(equity)

        assert status.action == DrawdownAction.EMERGENCY_EXIT
        assert status.exposure_ratio == 0.0

    def test_empty_equity(self) -> None:
        ctrl = DrawdownController()
        status = ctrl.check(pd.Series(dtype=float))
        assert status.action == DrawdownAction.NORMAL

    def test_max_drawdown_tracking(self) -> None:
        ctrl = DrawdownController()
        # 100 → 92 → 95: current_dd = 5%, max_dd = 8%
        equity = pd.Series([100, 92, 95])
        status = ctrl.check(equity)

        assert abs(status.max_drawdown - 0.08) < 0.001
        assert abs(status.current_drawdown - 0.05) < 0.001

    def test_drawdown_status_is_frozen(self) -> None:
        status = DrawdownStatus(
            current_drawdown=0.05, max_drawdown=0.08,
            peak_equity=100.0, current_equity=95.0,
            action=DrawdownAction.NORMAL, exposure_ratio=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            status.action = DrawdownAction.HALT_TRADING  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CorrelationMonitor テスト
# ---------------------------------------------------------------------------

class TestCorrelationMonitor:

    def test_compute_basic(self) -> None:
        returns = _make_returns(100, 3)
        monitor = CorrelationMonitor()
        snapshot = monitor.compute(returns)

        assert snapshot.matrix.shape == (3, 3)
        assert snapshot.avg_correlation >= 0
        assert monitor.snapshot_count == 1

    def test_no_alert_low_correlation(self) -> None:
        # 独立なリターンデータ
        rng = np.random.RandomState(42)
        returns = pd.DataFrame({
            "a": rng.randn(200) * 0.01,
            "b": rng.randn(200) * 0.01,
        })
        monitor = CorrelationMonitor(alert_threshold=0.9)
        snapshot = monitor.compute(returns)

        assert len(snapshot.alerts) == 0

    def test_alert_high_correlation(self) -> None:
        # 高相関なデータ
        rng = np.random.RandomState(42)
        base = rng.randn(200) * 0.01
        returns = pd.DataFrame({
            "a": base,
            "b": base + rng.randn(200) * 0.001,  # ほぼ同じ
        })
        monitor = CorrelationMonitor(alert_threshold=0.8)
        snapshot = monitor.compute(returns)

        assert len(snapshot.alerts) >= 1
        assert snapshot.alerts[0].correlation > 0.8

    def test_empty_returns(self) -> None:
        monitor = CorrelationMonitor()
        snapshot = monitor.compute(pd.DataFrame())
        assert len(snapshot.alerts) == 0
        assert snapshot.avg_correlation == 0.0

    def test_single_asset(self) -> None:
        returns = _make_returns(100, 1)
        monitor = CorrelationMonitor()
        snapshot = monitor.compute(returns)
        assert len(snapshot.alerts) == 0

    def test_get_latest(self) -> None:
        monitor = CorrelationMonitor()
        assert monitor.get_latest() is None

        returns = _make_returns(100, 2)
        monitor.compute(returns)
        assert monitor.get_latest() is not None

    def test_alert_is_frozen(self) -> None:
        alert = CorrelationAlert(
            pair=("a", "b"), correlation=0.9,
            threshold=0.8, message="test",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            alert.correlation = 0.5  # type: ignore[misc]
