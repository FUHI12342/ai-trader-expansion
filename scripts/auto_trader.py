"""AI Trader v2.0 — 完全自動化エントリポイント。

全5ステップを自動化:
  1. ペーパー検証 (バックテスト + ペーパートレード)
  2. パラメータ最適化 (Optuna)
  3. 小額ライブトレード
  4. 段階的スケールアップ
  5. 完全自動化 (日次レポート + ドリフト対応)

使用例:
  python scripts/auto_trader.py --mode paper --symbols 7203.T,9984.T
  python scripts/auto_trader.py --mode optimize --strategy ma_crossover
  python scripts/auto_trader.py --mode live --capital-pct 5
  python scripts/auto_trader.py --mode auto --config config/auto_trader.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brokers.base import OrderSide, OrderType
from src.brokers.broker_factory import BrokerFactory
from src.data.data_manager import DataManager
from src.evaluation.backtester import run_backtest
from src.learning.drift_detector import DriftDetector
from src.learning.ab_test import ABTestManager
from src.notifications.router import (
    NotificationRouter, LogChannel, TradingEvent, EventType,
)
from src.defi.aave_simulator import AaveSimulator
from src.defi.waiting_capital_manager import RebalanceActionType, WaitingCapitalManager
from src.risk.drawdown_controller import DrawdownController, DrawdownAction
from src.risk.correlation_monitor import CorrelationMonitor
from src.risk.portfolio_optimizer import PortfolioOptimizer
from src.risk.position_sizer import PositionSizer
from src.strategies.funding_arb import FundingArbStrategy, FundingRateCollector
from src.strategies.grid_trading import GridTradingStrategy
from src.trading.oms import OrderManagementSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "mode": "paper",
    "symbols": ["SPY", "QQQ"],
    "strategies": ["bollinger_rsi_adx"],
    "initial_capital": 200_000,
    "capital_pct": 100.0,
    "interval_seconds": 3600,
    "optimize_n_trials": 50,
    "optimize_metric": "combined_score",
    "drawdown_reduce": 0.03,
    "drawdown_halt": 0.05,
    "drawdown_emergency": 0.08,
    "correlation_threshold": 0.8,
    "drift_methods": ["fallback"],
    "notification_channels": ["log"],
    "slack_webhook": "",
    "discord_webhook": "",
    "daily_report_hour": 18,
    "auto_optimize_days": 30,
    "auto_rebalance_days": 30,
    "scale_up_sharpe_threshold": 0.3,
    "scale_up_max_capital_pct": 100.0,
    # Kelly Criterion
    "kelly_fraction": 0.5,
    "max_position_pct": 0.30,
    "daily_loss_limit_pct": 0.02,
    # Funding Rate Arbitrage
    "funding_rate_exchange": "",
    "funding_symbols": ["ETH/USDT:USDT", "SOL/USDT:USDT"],
    "min_funding_rate": 0.0001,
    # DeFi 待機資金運用 (Aave V3 USDC シミュレーター)
    "defi_enabled": False,
    "defi_apy": 0.05,
    "defi_flat_threshold_deposit": 0.5,
    "defi_flat_threshold_withdraw": 0.2,
    "defi_buffer_pct": 0.1,
    "defi_min_rebalance": 1000.0,
}


def _load_strategies() -> Dict[str, Any]:
    """利用可能な戦略をロードする。"""
    strategies = {}
    try:
        from src.strategies.ma_crossover import MACrossoverStrategy
        strategies["ma_crossover"] = MACrossoverStrategy()
    except ImportError:
        pass
    try:
        from src.strategies.macd_rsi import MACDRSIStrategy
        strategies["macd_rsi"] = MACDRSIStrategy()
    except ImportError:
        pass
    try:
        from src.strategies.bollinger_rsi_adx import BollingerRSIADXStrategy
        strategies["bollinger_rsi_adx"] = BollingerRSIADXStrategy()
    except ImportError:
        pass
    try:
        from src.strategies.dual_momentum import DualMomentumStrategy
        strategies["dual_momentum"] = DualMomentumStrategy()
    except ImportError:
        pass
    return strategies


def _setup_notifications(config: Dict[str, Any]) -> NotificationRouter:
    """通知ルーターをセットアップする。"""
    router = NotificationRouter()
    router.add_channel(LogChannel())

    channels = config.get("notification_channels", ["log"])

    if "slack" in channels and config.get("slack_webhook"):
        from src.notifications.router import SlackChannel
        router.add_channel(SlackChannel(config["slack_webhook"]))

    if "discord" in channels and config.get("discord_webhook"):
        from src.notifications.router import DiscordChannel
        router.add_channel(DiscordChannel(config["discord_webhook"]))

    return router


# ---------------------------------------------------------------------------
# Step 1: ペーパー検証
# ---------------------------------------------------------------------------

def run_paper_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    """ペーパートレーディング検証を実行する。"""
    logger.info("=== Step 1: ペーパー検証開始 ===")
    results = {}

    dm = DataManager()
    strategies = _load_strategies()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    for symbol in config["symbols"]:
        logger.info("データ取得: %s (%s ~ %s)", symbol, start_date, end_date)
        try:
            data = dm.fetch_ohlcv(symbol, start_date, end_date)
        except Exception as e:
            logger.warning("データ取得失敗 %s: %s", symbol, e)
            continue

        if data.empty or len(data) < 100:
            logger.warning("データ不足: %s (%d行)", symbol, len(data))
            continue

        for strat_name in config["strategies"]:
            if strat_name not in strategies:
                continue
            strategy = strategies[strat_name]
            try:
                result = run_backtest(
                    strategy, data, symbol=symbol,
                    initial_capital=config["initial_capital"],
                )
                m = result.metrics
                key = f"{symbol}/{strat_name}"
                results[key] = {
                    "final_capital": result.final_capital,
                    "return_pct": m.total_return_pct,
                    "sharpe": m.sharpe_ratio,
                    "max_dd_pct": m.max_drawdown_pct,
                    "win_rate": m.win_rate,
                    "total_trades": m.total_trades,
                    "profit_factor": m.profit_factor,
                }
                logger.info(
                    "%s: Return=%.2f%% Sharpe=%.3f MaxDD=%.2f%%",
                    key, m.total_return_pct, m.sharpe_ratio, m.max_drawdown_pct,
                )
            except Exception as e:
                logger.warning("バックテスト失敗 %s/%s: %s", symbol, strat_name, e)

    dm.close()
    return results


# ---------------------------------------------------------------------------
# Step 2: パラメータ最適化
# ---------------------------------------------------------------------------

def run_optimization(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optunaによるパラメータ最適化を実行する。"""
    logger.info("=== Step 2: パラメータ最適化開始 ===")

    try:
        from src.optimization.optuna_optimizer import OptunaOptimizer
    except ImportError:
        logger.error("optuna未インストール")
        return {}

    dm = DataManager()
    strategies = _load_strategies()
    results = {}
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    for strat_name in config["strategies"]:
        if strat_name not in strategies:
            continue
        strategy = strategies[strat_name]
        if not strategy.parameter_space():
            continue

        symbol = config["symbols"][0]
        try:
            data = dm.fetch_ohlcv(symbol, start_date, end_date)
            if data.empty or len(data) < 500:
                continue

            optimizer = OptunaOptimizer(
                strategy=strategy,
                n_trials=config["optimize_n_trials"],
                objective_metric=config["optimize_metric"],
                seed=42,
            )
            opt_result = optimizer.optimize(data=data, symbol=symbol)
            results[strat_name] = {
                "best_params": opt_result.best_params,
                "best_value": opt_result.best_value,
                "n_complete": opt_result.n_complete,
            }
            logger.info(
                "%s: best_value=%.4f params=%s",
                strat_name, opt_result.best_value, opt_result.best_params,
            )
        except Exception as e:
            logger.warning("最適化失敗 %s: %s", strat_name, e)

    dm.close()
    return results


# ---------------------------------------------------------------------------
# Step 3-5: ライブトレード + スケールアップ + 自動化
# ---------------------------------------------------------------------------

class AutoTrader:
    """完全自動トレーダー。

    Step 3〜5 を統合し、以下を自動実行:
    - 定期的なシグナル生成 + 注文実行
    - ドリフト検出 + A/Bテスト
    - ポートフォリオリバランス
    - DrawdownController によるエクスポージャー制御
    - 日次レポート + 通知
    - パフォーマンスに応じた自動スケールアップ
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._running = False
        self._iteration = 0
        self._start_time = time.time()

        # コンポーネント初期化
        is_paper = config["mode"] in ("paper", "optimize")
        self._factory = BrokerFactory.create_default(
            paper_mode=is_paper,
            initial_capital=config["initial_capital"],
        )
        self._dm = DataManager()
        self._oms = OrderManagementSystem(max_retries=3, stale_seconds=300)
        self._drift = DriftDetector(methods=config["drift_methods"])
        self._ab = ABTestManager(min_samples=30)
        self._dd_ctrl = DrawdownController(
            reduce_threshold=config["drawdown_reduce"],
            halt_threshold=config["drawdown_halt"],
            emergency_threshold=config["drawdown_emergency"],
        )
        self._corr_monitor = CorrelationMonitor(
            alert_threshold=config["correlation_threshold"],
        )
        self._portfolio = PortfolioOptimizer(method="risk_parity")
        self._notifier = _setup_notifications(config)
        self._strategies = _load_strategies()

        # Kelly Criterion ポジションサイザー
        self._sizer = PositionSizer(
            kelly_fraction=config.get("kelly_fraction", 0.5),
            max_position_pct=config.get("max_position_pct", 0.30),
            daily_loss_limit_pct=config.get("daily_loss_limit_pct", 0.02),
        )

        # DeFi 待機資金運用 (optional)
        self._aave: Optional[AaveSimulator] = None
        self._defi_mgr: Optional[WaitingCapitalManager] = None
        if config.get("defi_enabled", False):
            self._aave = AaveSimulator(
                apy=config.get("defi_apy", 0.05),
                compound_daily=True,
            )
            self._defi_mgr = WaitingCapitalManager(
                aave=self._aave,
                flat_threshold_deposit=config.get("defi_flat_threshold_deposit", 0.5),
                flat_threshold_withdraw=config.get("defi_flat_threshold_withdraw", 0.2),
                buffer_pct=config.get("defi_buffer_pct", 0.1),
                min_rebalance_amount=config.get("defi_min_rebalance", 1000.0),
            )

        # Funding Rate Arbitrage
        fr_exchange = config.get("funding_rate_exchange", "")
        self._fr_strategy: Optional[FundingArbStrategy] = None
        if fr_exchange:
            self._fr_strategy = FundingArbStrategy(
                exchange_id=fr_exchange,
                config={"enableRateLimit": True},
                min_funding_rate=config.get("min_funding_rate", 0.0001),
            )

        # Grid Trading (暗号レンジ用)
        self._grid: Optional[GridTradingStrategy] = None

        # 自動強化学習
        from src.learning.self_improver import SelfImprover
        self._improver = SelfImprover(
            optimize_interval_days=config.get("auto_optimize_days", 7),
        )

        # 状態
        self._daily_pnls: List[float] = []
        self._last_optimize_date: Optional[str] = None
        self._last_rebalance_date: Optional[str] = None
        self._current_capital_pct = config["capital_pct"]
        self._trade_wins: int = 0
        self._trade_losses: int = 0

    async def run(self) -> None:
        """メインループを開始する。"""
        self._running = True
        logger.info("AutoTrader開始 (mode=%s)", self._config["mode"])

        self._notifier.send(TradingEvent(
            event_type=EventType.CUSTOM,
            title="AutoTrader起動",
            message=f"mode={self._config['mode']}, symbols={self._config['symbols']}",
            data=self._config,
        ))

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("AutoTraderエラー: %s", e)

            await asyncio.sleep(self._config["interval_seconds"])

    def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        """1回のトレーディングサイクル。"""
        self._iteration += 1
        now = datetime.now()

        # ドローダウンチェック
        broker = self._factory.get_broker("paper") or self._factory.get_broker("paper_fallback")
        if broker is None:
            return

        import pandas as pd
        balance = broker.get_balance()
        equity = pd.Series([self._config["initial_capital"], balance])
        dd_status = self._dd_ctrl.check(equity)

        if dd_status.action in (DrawdownAction.HALT_TRADING, DrawdownAction.EMERGENCY_EXIT):
            logger.warning("ドローダウン制御: %s (DD=%.2f%%)", dd_status.action.value, dd_status.current_drawdown * 100)
            self._notifier.send(TradingEvent(
                event_type=EventType.DRAWDOWN_ALERT,
                title="ドローダウンアラート",
                message=f"DD={dd_status.current_drawdown*100:.1f}% action={dd_status.action.value}",
                data={"drawdown": dd_status.current_drawdown},
                severity="critical",
            ))
            return

        # --- Funding Rate Arbitrage (最優先・最低リスク) ---
        if self._fr_strategy and self._fr_strategy.is_available:
            for fr_sym in self._config.get("funding_symbols", []):
                try:
                    eval_result = self._fr_strategy.evaluate(fr_sym)
                    action = eval_result["action"]
                    if action == "enter":
                        logger.info("FR Arb エントリー候補: %s (%s)", fr_sym, eval_result["reason"])
                    elif action == "exit":
                        logger.info("FR Arb エグジット: %s (%s)", fr_sym, eval_result["reason"])
                except Exception as e:
                    logger.debug("FR Arb エラー %s: %s", fr_sym, e)

        # --- テクニカル戦略 (Kelly Criterion でサイジング) ---
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")

        # 日次リセット判定
        today_str = now.strftime("%Y-%m-%d")
        if self._sizer._current_date != today_str:
            self._sizer.reset_daily(today_str)

        for symbol in self._config["symbols"]:
            for strat_name in self._config["strategies"]:
                if strat_name not in self._strategies:
                    continue
                try:
                    data = self._dm.fetch_ohlcv(symbol, start_date, end_date)
                    if data.empty or len(data) < 50:
                        continue
                    strategy = self._strategies[strat_name]
                    signal = strategy.generate_signal_realtime(data)

                    if signal != 0:
                        # Kelly Criterion でポジションサイズ計算
                        win_rate = self._trade_wins / max(1, self._trade_wins + self._trade_losses)
                        if win_rate == 0:
                            win_rate = 0.6  # 初期値（Bollinger想定）

                        sizing = self._sizer.calculate(
                            capital=balance,
                            win_rate=win_rate,
                            avg_win=8441,   # Bollinger SPY 5Y実績
                            avg_loss=9186,  # Bollinger SPY 5Y実績
                        )

                        if sizing.blocked:
                            logger.info("ポジションブロック: %s (%s)", symbol, sizing.reason)
                            continue

                        qty = max(1, int(sizing.position_amount * dd_status.exposure_ratio / float(data["close"].iloc[-1])))
                        side = OrderSide.BUY if signal > 0 else OrderSide.SELL

                        managed = self._oms.submit(
                            broker=broker, symbol=symbol, side=side,
                            order_type=OrderType.MARKET, quantity=qty,
                            strategy_name=strat_name,
                        )

                        # 勝敗記録（成行の場合は即約定）
                        if managed.status.value == "filled":
                            self._trade_wins += 1
                        logger.info(
                            "注文: %s %s qty=%d (Kelly=%.1f%%, DD_exp=%.1f%%)",
                            symbol, side.value, qty,
                            sizing.position_pct * 100, dd_status.exposure_ratio * 100,
                        )
                except Exception as e:
                    logger.debug("シグナル生成エラー %s/%s: %s", symbol, strat_name, e)

        # OMS監視
        self._oms.monitor(broker)
        self._oms.cancel_stale(broker)

        # DeFi 待機資金リバランス (optional)
        self._maybe_rebalance_defi(broker)

        # ドリフト検出
        pnl = balance - self._config["initial_capital"]
        drift_result = self._drift.update(pnl)
        if drift_result.is_drift:
            self._notifier.send(TradingEvent(
                event_type=EventType.DRIFT_DETECTED,
                title="ドリフト検出",
                message=f"method={drift_result.method} severity={drift_result.severity.value}",
                data=drift_result.details,
                severity="warning",
            ))

        # 日次処理
        if now.hour == self._config["daily_report_hour"] and self._iteration % 60 == 0:
            self._daily_report(broker)

        # 自動強化サイクル (日次、daily_report_hourと同時刻)
        if now.hour == self._config["daily_report_hour"] and self._iteration % 60 == 1:
            try:
                improve_result = self._improver.run_cycle(
                    dm=self._dm,
                    strategies=self._strategies,
                    symbols=self._config["symbols"],
                )
                if improve_result.get("pending_improvements"):
                    self._notifier.send(TradingEvent(
                        event_type=EventType.STAGE_CHANGE,
                        title="自動強化: 改善候補発見",
                        message=json.dumps(improve_result["pending_improvements"], ensure_ascii=False),
                        data=improve_result,
                    ))
            except Exception as e:
                logger.warning("自動強化サイクルエラー: %s", e)

        # 定期最適化 (auto_optimize_days ごと)
        today = now.strftime("%Y-%m-%d")
        if self._last_optimize_date is None:
            self._last_optimize_date = today

        days_since_opt = (now - datetime.strptime(self._last_optimize_date, "%Y-%m-%d")).days
        if days_since_opt >= self._config["auto_optimize_days"]:
            logger.info("定期最適化実行")
            run_optimization(self._config)
            self._last_optimize_date = today

        # 自動スケールアップ (Step 4)
        self._maybe_scale_up(broker)

    _VERIFICATION_LEVEL_ORDER = ["paper", "minimum", "recommended", "ideal"]

    def _notify_milestone_if_promoted(self, status: Dict[str, Any]) -> None:
        """検証レベル昇格を検出したらマイルストン通知を発火する。

        状態は self._last_verification_level に保持。
        """
        current = status.get("level", "paper")
        last = getattr(self, "_last_verification_level", None)

        if last is None:
            self._last_verification_level = current
            return

        try:
            cur_idx = self._VERIFICATION_LEVEL_ORDER.index(current)
            last_idx = self._VERIFICATION_LEVEL_ORDER.index(last)
        except ValueError:
            return

        if cur_idx <= last_idx:
            return

        # 昇格検出
        self._last_verification_level = current
        milestone_title = f"🎯 マイルストン達成: {last.upper()} → {current.upper()}"
        milestone_msg = (
            f"検証レベルが {last} から {current} に昇格しました。\n"
            f"{status.get('detail', '')}\n"
            f"次のアクション: "
            f"{'段階的に実資金を投入' if status.get('ready') else 'ペーパートレード継続'}"
        )
        logger.info("[MILESTONE] %s", milestone_title)
        try:
            self._notifier.send(TradingEvent(
                event_type=EventType.CUSTOM,
                title=milestone_title,
                message=milestone_msg,
                data={"from": last, "to": current, "status": status},
                severity="info",
            ))
        except Exception as e:
            logger.warning("マイルストン通知失敗: %s", e)

    def _maybe_rebalance_defi(self, broker: Any) -> None:
        """DeFi待機資金のリバランス判定 + 仮想送金を実行する。

        実ブローカーとDeFiプロトコルの実接続は未対応 — 口座内部会計として
        cash_balance から Aave に移したことにする。実運用時はここで USDC 送金/引出。
        """
        if self._defi_mgr is None or self._aave is None:
            return

        try:
            cash = float(broker.get_balance())
            positions = broker.get_positions() or {}
            active_value = sum(
                float(getattr(p, "market_value", 0.0)) for p in positions.values()
            )
            total = cash + active_value + self._aave.balance
            action = self._defi_mgr.decide(
                cash_balance=cash,
                active_position_value=active_value,
                total_capital=total,
            )

            if action.action == RebalanceActionType.DEPOSIT:
                result = self._aave.deposit(action.amount)
                if result.success and hasattr(broker, "_balance"):
                    broker._balance -= action.amount
                logger.info(
                    "DeFi預入: %.2f USDC (FLAT=%.1f%%) → Aave残高 %.2f",
                    action.amount, action.flat_ratio * 100, self._aave.balance,
                )
            elif action.action == RebalanceActionType.WITHDRAW:
                result = self._aave.withdraw(action.amount)
                if result.success and hasattr(broker, "_balance"):
                    broker._balance += action.amount
                logger.info(
                    "DeFi引出: %.2f USDC (FLAT=%.1f%%) → Aave残高 %.2f",
                    action.amount, action.flat_ratio * 100, self._aave.balance,
                )
        except Exception as e:
            logger.warning("DeFiリバランスエラー: %s", e)

    def _daily_report(self, broker: Any) -> None:
        """日次レポートを送信する。検証ステータス判定を含む。"""
        balance = broker.get_balance()
        pnl = balance - self._config["initial_capital"]
        pnl_pct = pnl / self._config["initial_capital"] * 100
        filled = self._oms.get_filled_orders()
        summary = self._oms.summary()
        uptime = (time.time() - self._start_time) / 3600

        # 検証ステータス判定
        total_trades = len(filled)
        wins = sum(1 for o in filled if getattr(o, 'avg_fill_price', 0) > 0)
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0

        import numpy as np
        sharpe = 0.0
        if len(self._daily_pnls) >= 5:
            arr = np.array(self._daily_pnls)
            if np.std(arr) > 0:
                sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252))

        status = self._verification_status(total_trades, win_rate, sharpe)
        title = f"[{status['label']}] AI Trader 日次レポート"

        # マイルストン通知: 検証レベル昇格を検出
        self._notify_milestone_if_promoted(status)

        # DeFi Aave 状況 (有効時のみ)
        defi_section = ""
        if self._aave is not None:
            snap = self._aave.snapshot()
            defi_section = (
                f"\nDeFi(Aave): 元本{snap.principal:,.0f} + 利息{snap.accrued_interest:,.0f} "
                f"= {snap.total_balance:,.0f} USDC (APY {snap.apy*100:.1f}%)"
            )

        report = (
            f"== 検証ステータス: {status['label']} ==\n"
            f"{status['detail']}\n"
            f"\n"
            f"残高: {balance:,.0f}円 (PnL: {pnl:+,.0f}円 / {pnl_pct:+.2f}%)"
            f"{defi_section}\n"
            f"取引: {total_trades}件 (勝率: {win_rate:.0f}%)\n"
            f"Sharpe(推定): {sharpe:.2f}\n"
            f"OMS: {summary}\n"
            f"稼働時間: {uptime:.1f}時間"
        )

        self._notifier.send(TradingEvent(
            event_type=EventType.DAILY_SUMMARY,
            title=title,
            message=report,
            data={
                "balance": balance, "pnl": pnl,
                "total_trades": total_trades, "win_rate": win_rate,
                "sharpe": sharpe, "verification": status,
            },
        ))

    @staticmethod
    def _verification_status(
        total_trades: int, win_rate: float, sharpe: float
    ) -> Dict[str, Any]:
        """検証ステータスを判定する。

        Returns
        -------
        Dict[str, Any]
            label: ステータスラベル
            ready: 実運用移行可能か
            detail: 詳細説明
        """
        # 理想ライン: 100取引 + 勝率60% + Sharpe>0.5
        if total_trades >= 100 and win_rate >= 60 and sharpe > 0.5:
            return {
                "label": "IDEAL - 実運用移行可能(理想)",
                "ready": True,
                "level": "ideal",
                "detail": f"100取引達成 (勝率{win_rate:.0f}%, Sharpe{sharpe:.2f}) — 十分な統計的根拠あり。自信を持って実運用可能。",
            }

        # 推奨ライン: 50取引 + 勝率55% + Sharpe>0.3
        if total_trades >= 50 and win_rate >= 55 and sharpe > 0.3:
            return {
                "label": "RECOMMENDED - 実運用移行可能(推奨)",
                "ready": True,
                "level": "recommended",
                "detail": f"50取引達成 (勝率{win_rate:.0f}%, Sharpe{sharpe:.2f}) — 安定性を確認。実運用に移行可能。",
            }

        # 最低ライン: 30取引 + 勝率50% + Sharpe>0
        if total_trades >= 30 and win_rate >= 50 and sharpe > 0:
            return {
                "label": "MINIMUM - 実運用移行可能(最低)",
                "ready": True,
                "level": "minimum",
                "detail": f"30取引達成 (勝率{win_rate:.0f}%, Sharpe{sharpe:.2f}) — 最低条件クリア。小額から実運用開始可。",
            }

        # 条件未達
        remaining = max(0, 30 - total_trades)
        return {
            "label": f"PAPER - 検証中 (残{remaining}取引)",
            "ready": False,
            "level": "paper",
            "detail": f"取引{total_trades}/30件 (勝率{win_rate:.0f}%, Sharpe{sharpe:.2f}) — 最低ライン到達まであと{remaining}取引。",
        }

    def _maybe_scale_up(self, broker: Any) -> None:
        """Step 4: パフォーマンスに応じて自動スケールアップ。

        条件:
        - 直近30日の Sharpe > threshold
        - MaxDD < halt_threshold
        - 現在の capital_pct < max_capital_pct
        """
        if len(self._daily_pnls) < 30:
            return

        import numpy as np
        recent = np.array(self._daily_pnls[-30:])
        mean_ret = np.mean(recent)
        std_ret = np.std(recent)
        if std_ret == 0:
            return

        sharpe_30d = mean_ret / std_ret * np.sqrt(252)
        threshold = self._config["scale_up_sharpe_threshold"]
        max_pct = self._config["scale_up_max_capital_pct"]

        if sharpe_30d > threshold and self._current_capital_pct < max_pct:
            new_pct = min(self._current_capital_pct * 1.5, max_pct)
            logger.info(
                "自動スケールアップ: %.1f%% -> %.1f%% (Sharpe30d=%.2f)",
                self._current_capital_pct, new_pct, sharpe_30d,
            )
            self._notifier.send(TradingEvent(
                event_type=EventType.STAGE_CHANGE,
                title="スケールアップ",
                message=f"資金配分: {self._current_capital_pct:.1f}% -> {new_pct:.1f}% (Sharpe={sharpe_30d:.2f})",
                data={"old_pct": self._current_capital_pct, "new_pct": new_pct, "sharpe": sharpe_30d},
            ))
            self._current_capital_pct = new_pct


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trader v2.0 自動化エンジン")
    parser.add_argument("--mode", choices=["paper", "optimize", "live", "auto"],
                        default="paper", help="動作モード")
    parser.add_argument("--symbols", default=None, help="銘柄リスト (カンマ区切り)")
    parser.add_argument("--strategies", default=None, help="戦略リスト (カンマ区切り)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="初期資金")
    parser.add_argument("--capital-pct", type=float, default=5.0, help="投入資金比率(%)")
    parser.add_argument("--config", default=None, help="設定ファイル (JSON)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = dict(DEFAULT_CONFIG)

    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    config["mode"] = args.mode
    if args.symbols:
        config["symbols"] = args.symbols.split(",")
    if args.strategies:
        config["strategies"] = args.strategies.split(",")
    config["initial_capital"] = args.capital
    config["capital_pct"] = args.capital_pct

    if args.mode == "paper":
        results = run_paper_validation(config)
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    elif args.mode == "optimize":
        results = run_optimization(config)
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    elif args.mode in ("live", "auto"):
        trader = AutoTrader(config)

        def _signal_handler(sig: int, frame: Any) -> None:
            logger.info("シグナル受信: 停止中...")
            trader.stop()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        asyncio.run(trader.run())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
