"""イベントドリブンバックテストエンジン。

手数料・スリッページを考慮したリアルな損益計算。
全てのDataFrameはimmutableパターンで処理する。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import EvaluationResult, calculate_metrics
from ..strategies.base import BaseStrategy, SignalType


@dataclass(frozen=True)
class Trade:
    """個別取引記録（immutable）。"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    side: int                # 1=ロング, -1=ショート
    pnl: float               # 絶対損益（円）
    pnl_pct: float           # 損益率（%）
    fee: float               # 手数料合計
    duration_days: int       # 保有期間（日数）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    """バックテスト結果（immutable）。"""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    metrics: EvaluationResult
    trades: List[Dict[str, Any]]
    equity_curve: List[float]      # 日次資産価値リスト
    equity_dates: List[str]        # 対応する日付リスト

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "metrics": self.metrics.to_dict(),
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "equity_dates": self.equity_dates,
        }


def run_backtest(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    symbol: str = "UNKNOWN",
    initial_capital: float = 1_000_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size_pct: float = 1.0,   # ポジションサイズ（資産の何%を使うか）
    allow_short: bool = False,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None,
) -> BacktestResult:
    """イベントドリブンバックテストを実行する。

    Parameters
    ----------
    strategy:
        バックテストする戦略インスタンス
    data:
        OHLCV DataFrame（DatetimeIndex）
    symbol:
        銘柄コード/名称
    initial_capital:
        初期資金
    fee_rate:
        片道手数料率（例: 0.001 = 0.1%）
    slippage_rate:
        スリッページ率（例: 0.0005 = 0.05%）
    position_size_pct:
        1取引で使用する資産比率（0〜1）
    allow_short:
        ショート取引を許可するか
    risk_free_rate:
        年率リスクフリーレート
    periods_per_year:
        年間取引日数

    Returns
    -------
    BacktestResult
        バックテスト結果
    """
    # 年間取引日数のデフォルト処理（株式: 252, 暗号資産呼び出し元: 365を渡す）
    if periods_per_year is None:
        periods_per_year = 252

    # シグナル生成（元データは変更しない）
    signals = strategy.generate_signals(data)

    df = data.copy()
    df["signal"] = signals

    capital = initial_capital
    cash = initial_capital
    position = 0.0        # 保有株数
    entry_price = 0.0
    entry_date = None
    entry_side = 0

    trades: List[Trade] = []
    equity_values: List[float] = []
    equity_dates: List[str] = []
    entry_idx: int = 0

    for i, (date, row) in enumerate(df.iterrows()):
        sig = int(row["signal"])
        close = float(row["close"])

        # スリッページを考慮した実行価格
        exec_price_buy = close * (1 + slippage_rate)
        exec_price_sell = close * (1 - slippage_rate)

        # ポジションクローズ条件
        should_close = (
            (position > 0 and sig == SignalType.SELL) or
            (position < 0 and sig == SignalType.BUY) or
            (position != 0 and sig == SignalType.SELL and position > 0) or
            (position != 0 and sig == SignalType.BUY and position < 0)
        )

        if should_close and position != 0:
            # クローズ処理
            exec_price = exec_price_sell if position > 0 else exec_price_buy
            close_fee = abs(position) * exec_price * fee_rate
            pnl = (exec_price - entry_price) * position - close_fee

            # エントリー手数料も含める
            entry_fee = abs(position) * entry_price * fee_rate
            total_fee = close_fee + entry_fee
            pnl_with_entry_fee = (exec_price - entry_price) * position - total_fee

            entry_date_str = str(entry_date) if entry_date else str(date)
            trade = Trade(
                entry_date=entry_date_str,
                exit_date=str(date),
                entry_price=entry_price,
                exit_price=exec_price,
                shares=abs(position),
                side=entry_side,
                pnl=pnl_with_entry_fee,
                pnl_pct=(pnl_with_entry_fee / (abs(position) * entry_price)) * 100,
                fee=total_fee,
                duration_days=max(1, i - entry_idx),
            )
            trades.append(trade)

            cash += abs(position) * exec_price - close_fee
            position = 0.0
            entry_price = 0.0
            entry_date = None
            entry_side = 0

        # ポジションオープン条件
        if position == 0 and sig in (SignalType.BUY,):
            # BUYエントリー
            invest = cash * position_size_pct
            exec_price = exec_price_buy
            fee = invest * fee_rate
            shares = (invest - fee) / exec_price
            if shares > 0:
                position = shares
                entry_price = exec_price
                entry_date = date
                entry_side = 1
                entry_idx = i
                cash -= invest

        elif position == 0 and allow_short and sig == SignalType.SELL:
            # SHORTエントリー
            invest = cash * position_size_pct
            exec_price = exec_price_sell
            fee = invest * fee_rate
            shares = -(invest - fee) / exec_price
            if abs(shares) > 0:
                position = shares
                entry_price = exec_price
                entry_date = date
                entry_side = -1
                entry_idx = i
                cash -= fee  # ショートは証拠金のみ

        # 現在の資産価値
        if position > 0:
            current_value = cash + position * close
        elif position < 0:
            current_value = cash + abs(position) * (entry_price - close)
        else:
            current_value = cash

        equity_values.append(current_value)
        equity_dates.append(str(date))

    # 残ポジションの強制クローズ（最終日）
    if position != 0 and len(df) > 0:
        last_row = df.iloc[-1]
        exec_price = float(last_row["close"]) * (1 - slippage_rate if position > 0 else 1 + slippage_rate)
        close_fee = abs(position) * exec_price * fee_rate
        entry_fee = abs(position) * entry_price * fee_rate
        total_fee = close_fee + entry_fee
        pnl_with_fee = (exec_price - entry_price) * position - total_fee

        trade = Trade(
            entry_date=str(entry_date),
            exit_date=str(df.index[-1]),
            entry_price=entry_price,
            exit_price=exec_price,
            shares=abs(position),
            side=entry_side,
            pnl=pnl_with_fee,
            pnl_pct=(pnl_with_fee / (abs(position) * entry_price)) * 100,
            fee=total_fee,
            duration_days=len(df) - entry_idx,
        )
        trades.append(trade)

        if equity_values:
            equity_values[-1] = cash + abs(position) * exec_price - close_fee

    final_capital = equity_values[-1] if equity_values else initial_capital

    # metrics計算用DataFrameを準備
    equity_series = pd.Series(
        equity_values,
        index=pd.Index(df.index[:len(equity_values)]),
    )

    trades_df = pd.DataFrame([t.to_dict() for t in trades]) if trades else pd.DataFrame()

    metrics = calculate_metrics(
        equity_curve=equity_series,
        trades=trades_df if not trades_df.empty else None,
        benchmark_prices=df["close"],
        initial_capital=initial_capital,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    return BacktestResult(
        strategy_name=strategy.name,
        symbol=symbol,
        start_date=str(df.index[0]) if len(df) > 0 else "",
        end_date=str(df.index[-1]) if len(df) > 0 else "",
        initial_capital=initial_capital,
        final_capital=final_capital,
        metrics=metrics,
        trades=[t.to_dict() for t in trades],
        equity_curve=equity_values,
        equity_dates=equity_dates,
    )
