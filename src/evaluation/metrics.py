"""パフォーマンスメトリクス計算モジュール。

Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Profit Factor等を計算する。
全ての計算結果はimmutableなdataclassとして返す。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationResult:
    """パフォーマンス評価結果（immutable）。"""
    total_return_pct: float          # 総リターン（%）
    annualized_return_pct: float     # 年率リターン（%）
    max_drawdown_pct: float          # 最大ドローダウン（%、負値）
    sharpe_ratio: float              # シャープレシオ
    sortino_ratio: float             # ソルティノレシオ
    calmar_ratio: float              # カルマーレシオ
    win_rate: float                  # 勝率（0〜1）
    profit_factor: float             # プロフィットファクター（総利益/総損失）
    avg_win_pct: float               # 平均利益（%）
    avg_loss_pct: float              # 平均損失（%）
    total_trades: int                # 総取引数
    max_consecutive_wins: int        # 最大連勝数
    max_consecutive_losses: int      # 最大連敗数
    exposure_pct: float              # 市場露出率（%）
    buy_hold_return_pct: float       # バイアンドホールドリターン（%）
    excess_return_pct: float         # 超過リターン（戦略 - B&H）
    volatility_pct: float            # 年率ボラティリティ（%）

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で返す。"""
        return asdict(self)


def _zero_result() -> EvaluationResult:
    """空/エラー時のゼロ結果。"""
    return EvaluationResult(
        total_return_pct=0.0,
        annualized_return_pct=0.0,
        max_drawdown_pct=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        total_trades=0,
        max_consecutive_wins=0,
        max_consecutive_losses=0,
        exposure_pct=0.0,
        buy_hold_return_pct=0.0,
        excess_return_pct=0.0,
        volatility_pct=0.0,
    )


def _max_consecutive(series: pd.Series, value: int) -> int:
    """指定した値の最大連続カウントを計算する。"""
    max_count = 0
    count = 0
    for v in series:
        if v == value:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count


def calculate_metrics(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    benchmark_prices: Optional[pd.Series] = None,
    initial_capital: float = 1_000_000.0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> EvaluationResult:
    """パフォーマンスメトリクスを包括的に計算する。

    Parameters
    ----------
    equity_curve:
        ポートフォリオ価値の時系列（DatetimeIndex推奨）
    trades:
        取引履歴DataFrame（pnl_pct カラム必須）
    benchmark_prices:
        バイアンドホールドのベンチマーク価格系列
    initial_capital:
        初期資金
    risk_free_rate:
        年率リスクフリーレート（例: 0.02 = 2%）
    periods_per_year:
        年間取引日数（株式: 252, 暗号資産: 365）

    Returns
    -------
    EvaluationResult
        全メトリクスを格納したimmutableオブジェクト
    """
    if equity_curve.empty or initial_capital <= 0:
        return _zero_result()

    # ------------------------------------------------------------------
    # 総リターン・年率リターン
    # ------------------------------------------------------------------
    total_return = (equity_curve.iloc[-1] / initial_capital - 1) * 100
    n_periods = len(equity_curve)
    ann_factor = periods_per_year / max(n_periods - 1, 1)
    ann_return = ((1 + total_return / 100) ** ann_factor - 1) * 100

    # ------------------------------------------------------------------
    # 最大ドローダウン
    # ------------------------------------------------------------------
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak * 100
    max_dd = float(drawdown.min())

    # ------------------------------------------------------------------
    # リターン系列
    # ------------------------------------------------------------------
    returns = equity_curve.pct_change().dropna()
    daily_rf = risk_free_rate / periods_per_year

    # ------------------------------------------------------------------
    # シャープレシオ
    # ------------------------------------------------------------------
    excess = returns - daily_rf
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(periods_per_year))
        if excess.std() > 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # ソルティノレシオ（下方偏差のみ）
    # ------------------------------------------------------------------
    downside_returns = returns[returns < daily_rf] - daily_rf
    # 下方偏差は全期間数ベース（ゼロを含む）で計算
    n_total = len(returns)
    downside_sq_sum = float((downside_returns ** 2).sum())
    downside_std = float(np.sqrt(downside_sq_sum / n_total)) if n_total > 0 else 0.0
    excess_mean = float(returns.mean()) - daily_rf
    sortino = (
        float(excess_mean / downside_std * np.sqrt(periods_per_year))
        if downside_std > 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # カルマーレシオ
    # ------------------------------------------------------------------
    calmar = (
        float(ann_return / abs(max_dd))
        if max_dd < 0
        else 0.0
    )

    # ------------------------------------------------------------------
    # ボラティリティ（年率）
    # ------------------------------------------------------------------
    volatility = float(returns.std() * np.sqrt(periods_per_year) * 100)

    # ------------------------------------------------------------------
    # 取引統計（tradesが提供された場合）
    # ------------------------------------------------------------------
    win_rate = 0.0
    profit_factor = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    total_trades = 0
    max_consec_wins = 0
    max_consec_losses = 0

    if trades is not None and not trades.empty and "pnl_pct" in trades.columns:
        pnl = trades["pnl_pct"].dropna()
        total_trades = len(pnl)

        if total_trades > 0:
            wins = pnl[pnl > 0]
            losses = pnl[pnl <= 0]

            win_rate = len(wins) / total_trades
            avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
            avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0

            total_profit = wins.sum()
            total_loss = abs(losses.sum())
            profit_factor = (
                float(total_profit / total_loss)
                if total_loss > 0
                else (float("inf") if total_profit > 0 else 0.0)
            )

            # 連勝・連敗
            win_series = (pnl > 0).astype(int)
            max_consec_wins = _max_consecutive(win_series, 1)
            max_consec_losses = _max_consecutive(win_series, 0)

    # ------------------------------------------------------------------
    # 市場露出率
    # ------------------------------------------------------------------
    exposure_pct = 0.0
    if trades is not None and not trades.empty and "position" in trades.columns:
        exposure_pct = float((trades["position"] != 0).mean() * 100)

    # ------------------------------------------------------------------
    # バイアンドホールド比較
    # ------------------------------------------------------------------
    buy_hold_return = 0.0
    if benchmark_prices is not None and len(benchmark_prices) >= 2:
        bh_start = float(benchmark_prices.iloc[0])
        bh_end = float(benchmark_prices.iloc[-1])
        if bh_start > 0:
            buy_hold_return = (bh_end / bh_start - 1) * 100

    excess_return = total_return - buy_hold_return

    return EvaluationResult(
        total_return_pct=round(total_return, 4),
        annualized_return_pct=round(ann_return, 4),
        max_drawdown_pct=round(max_dd, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        total_trades=total_trades,
        max_consecutive_wins=max_consec_wins,
        max_consecutive_losses=max_consec_losses,
        exposure_pct=round(exposure_pct, 2),
        buy_hold_return_pct=round(buy_hold_return, 4),
        excess_return_pct=round(excess_return, 4),
        volatility_pct=round(volatility, 4),
    )
