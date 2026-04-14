"""AutoTrader 統合テスト。"""
from __future__ import annotations

import pytest

from scripts.auto_trader import (
    DEFAULT_CONFIG, _load_strategies, _setup_notifications,
    run_paper_validation, AutoTrader,
)


class TestAutoTraderConfig:

    def test_default_config_keys(self) -> None:
        required = ["mode", "symbols", "strategies", "initial_capital",
                     "capital_pct", "interval_seconds"]
        for key in required:
            assert key in DEFAULT_CONFIG

    def test_load_strategies(self) -> None:
        strategies = _load_strategies()
        assert len(strategies) >= 3
        assert "ma_crossover" in strategies

    def test_setup_notifications_log(self) -> None:
        router = _setup_notifications({"notification_channels": ["log"]})
        assert router.channel_count >= 1


class TestAutoTraderInit:

    def test_init_paper_mode(self) -> None:
        config = dict(DEFAULT_CONFIG)
        config["mode"] = "paper"
        trader = AutoTrader(config)
        assert trader._running is False
        assert trader._current_capital_pct == 5.0

    def test_init_components(self) -> None:
        config = dict(DEFAULT_CONFIG)
        trader = AutoTrader(config)
        assert trader._oms is not None
        assert trader._drift is not None
        assert trader._ab is not None
        assert trader._dd_ctrl is not None
        assert trader._corr_monitor is not None
        assert trader._portfolio is not None
        assert trader._notifier is not None

    def test_stop(self) -> None:
        config = dict(DEFAULT_CONFIG)
        trader = AutoTrader(config)
        trader._running = True
        trader.stop()
        assert trader._running is False


class TestScaleUp:

    def test_scale_up_insufficient_data(self) -> None:
        config = dict(DEFAULT_CONFIG)
        trader = AutoTrader(config)
        trader._daily_pnls = [100.0] * 10  # 30日未満
        initial_pct = trader._current_capital_pct

        from unittest.mock import MagicMock
        broker = MagicMock()
        trader._maybe_scale_up(broker)

        assert trader._current_capital_pct == initial_pct  # 変化なし

    def test_scale_up_triggered(self) -> None:
        config = dict(DEFAULT_CONFIG)
        config["scale_up_sharpe_threshold"] = 0.1
        trader = AutoTrader(config)
        # 一貫して正のPnL → 高Sharpe
        trader._daily_pnls = [1000.0 + i * 10 for i in range(30)]
        initial_pct = trader._current_capital_pct

        from unittest.mock import MagicMock
        broker = MagicMock()
        trader._maybe_scale_up(broker)

        assert trader._current_capital_pct > initial_pct
