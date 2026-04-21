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
        assert trader._current_capital_pct == config["capital_pct"]

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
        config["capital_pct"] = 5.0
        config["scale_up_max_capital_pct"] = 50.0
        trader = AutoTrader(config)
        # 一貫して正のPnL → 高Sharpe
        trader._daily_pnls = [1000.0 + i * 10 for i in range(30)]
        initial_pct = trader._current_capital_pct

        from unittest.mock import MagicMock
        broker = MagicMock()
        trader._maybe_scale_up(broker)

        assert trader._current_capital_pct > initial_pct


class TestVerificationStatus:

    def test_paper_status(self) -> None:
        s = AutoTrader._verification_status(5, 60.0, 0.5)
        assert s["level"] == "paper"
        assert not s["ready"]

    def test_minimum_status(self) -> None:
        s = AutoTrader._verification_status(30, 55.0, 0.3)
        assert s["level"] == "minimum"
        assert s["ready"]

    def test_recommended_status(self) -> None:
        s = AutoTrader._verification_status(50, 60.0, 0.5)
        assert s["level"] == "recommended"
        assert s["ready"]

    def test_ideal_status(self) -> None:
        s = AutoTrader._verification_status(100, 65.0, 0.8)
        assert s["level"] == "ideal"
        assert s["ready"]

    def test_high_trades_low_winrate_stays_paper(self) -> None:
        s = AutoTrader._verification_status(50, 40.0, 0.5)
        assert s["level"] == "paper"
        assert not s["ready"]

    def test_high_trades_negative_sharpe_stays_paper(self) -> None:
        s = AutoTrader._verification_status(30, 55.0, -0.1)
        assert s["level"] == "paper"
        assert not s["ready"]


class TestAutoTraderDeFi:
    """DeFi 待機資金統合のテスト。"""

    def test_defi_disabled_by_default(self) -> None:
        """デフォルトでは DeFi 無効。"""
        trader = AutoTrader(dict(DEFAULT_CONFIG))
        assert trader._aave is None
        assert trader._defi_mgr is None

    def test_defi_enabled_initializes_components(self) -> None:
        """defi_enabled=True で Aave + Manager 生成。"""
        config = dict(DEFAULT_CONFIG)
        config["defi_enabled"] = True
        config["defi_apy"] = 0.06
        trader = AutoTrader(config)
        assert trader._aave is not None
        assert trader._defi_mgr is not None
        assert trader._aave.apy == 0.06

    def test_rebalance_defi_noop_when_disabled(self) -> None:
        """DeFi 無効時は何も起きない。"""
        trader = AutoTrader(dict(DEFAULT_CONFIG))

        class StubBroker:
            def get_balance(self) -> float:
                return 200000.0
            def get_positions(self) -> dict:
                return {}

        # 例外なく完了
        trader._maybe_rebalance_defi(StubBroker())

    def test_milestone_no_promotion_initially(self) -> None:
        """初回呼び出しは last_level を記録するだけで通知なし。"""
        trader = AutoTrader(dict(DEFAULT_CONFIG))
        from unittest.mock import MagicMock
        trader._notifier = MagicMock()
        trader._notify_milestone_if_promoted({"level": "paper", "detail": "", "ready": False})
        trader._notifier.send.assert_not_called()
        assert trader._last_verification_level == "paper"

    def test_milestone_notifies_on_promotion(self) -> None:
        """paper→minimum への昇格でnotifyが発火する。"""
        trader = AutoTrader(dict(DEFAULT_CONFIG))
        from unittest.mock import MagicMock
        trader._notifier = MagicMock()
        # 初回
        trader._notify_milestone_if_promoted({"level": "paper", "detail": "", "ready": False})
        # 昇格
        trader._notify_milestone_if_promoted({"level": "minimum", "detail": "30取引達成", "ready": True})
        assert trader._notifier.send.call_count == 1
        assert trader._last_verification_level == "minimum"

    def test_milestone_no_notify_on_same_level(self) -> None:
        """同レベル継続なら通知しない。"""
        trader = AutoTrader(dict(DEFAULT_CONFIG))
        from unittest.mock import MagicMock
        trader._notifier = MagicMock()
        trader._notify_milestone_if_promoted({"level": "minimum", "detail": "", "ready": True})
        trader._notify_milestone_if_promoted({"level": "minimum", "detail": "", "ready": True})
        trader._notifier.send.assert_not_called()

    def test_milestone_no_notify_on_demotion(self) -> None:
        """降格 (recommended→paper) では通知しない。"""
        trader = AutoTrader(dict(DEFAULT_CONFIG))
        from unittest.mock import MagicMock
        trader._notifier = MagicMock()
        trader._notify_milestone_if_promoted({"level": "recommended", "detail": "", "ready": True})
        trader._notify_milestone_if_promoted({"level": "paper", "detail": "", "ready": False})
        trader._notifier.send.assert_not_called()

    def test_rebalance_defi_deposits_when_flat_high(self) -> None:
        """FLAT比率高で Aave に預入される。"""
        config = dict(DEFAULT_CONFIG)
        config["defi_enabled"] = True
        config["defi_min_rebalance"] = 10.0
        trader = AutoTrader(config)

        class StubBroker:
            _balance = 200000.0
            def get_balance(self) -> float:
                return self._balance
            def get_positions(self) -> dict:
                return {}

        broker = StubBroker()
        trader._maybe_rebalance_defi(broker)
        assert trader._aave is not None
        assert trader._aave.balance > 0.0
        assert broker._balance < 200000.0

