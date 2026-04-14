"""Pairs Trading (統計的裁定) 戦略。

共和分関係にある2銘柄のスプレッドが平均から乖離したら
ロング/ショートのペアポジションを構築し、平均回帰で利確する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PairSignal:
    """ペアトレーディングシグナル（immutable）。"""
    asset_a: str
    asset_b: str
    z_score: float
    spread: float
    hedge_ratio: float
    action: str  # "long_a_short_b", "short_a_long_b", "close", "hold"
    confidence: float


@dataclass(frozen=True)
class PairTrade:
    """ペアトレード取引記録（immutable）。"""
    asset_a: str
    asset_b: str
    entry_z: float
    exit_z: float
    entry_spread: float
    exit_spread: float
    pnl: float
    pnl_pct: float
    holding_days: int
    result: str  # "WIN", "LOSS"


class PairsTradingStrategy:
    """Pairs Trading 戦略。

    Parameters
    ----------
    entry_z:
        エントリーの Z-score 閾値 (デフォルト: 2.0)
    exit_z:
        エグジットの Z-score 閾値 (デフォルト: 0.5)
    stop_z:
        ストップロスの Z-score 閾値 (デフォルト: 3.5)
    lookback:
        スプレッド計算の lookback 期間 (デフォルト: 60)
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.0,
        lookback: int = 60,
        max_holding_days: int = 30,
        rolling_hedge: bool = True,
    ) -> None:
        self._entry_z = entry_z
        self._exit_z = exit_z
        self._stop_z = stop_z
        self._lookback = lookback
        self._max_holding_days = max_holding_days
        self._rolling_hedge = rolling_hedge

    def compute_hedge_ratio(self, prices_a: pd.Series, prices_b: pd.Series) -> float:
        """OLS回帰でヘッジ比率を計算する。"""
        if len(prices_a) < 10 or len(prices_b) < 10:
            return 1.0
        b = prices_b.values
        a = prices_a.values
        # β = Cov(A,B) / Var(B)
        cov = np.cov(a, b)
        if cov[1, 1] == 0:
            return 1.0
        return float(cov[0, 1] / cov[1, 1])

    def compute_spread(
        self, prices_a: pd.Series, prices_b: pd.Series, hedge_ratio: float
    ) -> pd.Series:
        """スプレッド = A - β * B"""
        return prices_a - hedge_ratio * prices_b

    def compute_z_score(self, spread: pd.Series) -> pd.Series:
        """ローリング Z-score を計算する。"""
        mean = spread.rolling(self._lookback, min_periods=20).mean()
        std = spread.rolling(self._lookback, min_periods=20).std()
        std = std.replace(0, np.nan)
        return (spread - mean) / std

    def check_cointegration(self, prices_a: pd.Series, prices_b: pd.Series) -> Dict[str, Any]:
        """簡易共和分検定（Engle-Granger 簡易版）。

        完全な ADF 検定には statsmodels が必要。
        ここではスプレッドの定常性をヒューリスティックに判定する。
        """
        hedge = self.compute_hedge_ratio(prices_a, prices_b)
        spread = self.compute_spread(prices_a, prices_b, hedge)

        # スプレッドの統計
        spread_clean = spread.dropna()
        if len(spread_clean) < 30:
            return {"cointegrated": False, "reason": "データ不足", "hedge_ratio": hedge}

        mean = spread_clean.mean()
        std = spread_clean.std()
        if std == 0:
            return {"cointegrated": False, "reason": "分散ゼロ", "hedge_ratio": hedge}

        # 平均回帰の指標: 半減期
        spread_lag = spread_clean.shift(1).dropna()
        spread_diff = spread_clean.iloc[1:].values - spread_lag.values
        spread_lag_vals = spread_lag.values - mean

        if len(spread_lag_vals) < 10:
            return {"cointegrated": False, "reason": "データ不足", "hedge_ratio": hedge}

        # AR(1) 係数
        beta_ar = np.sum(spread_lag_vals * spread_diff) / np.sum(spread_lag_vals ** 2)
        if beta_ar >= 0:
            return {"cointegrated": False, "reason": "平均回帰なし", "hedge_ratio": hedge}

        half_life = -np.log(2) / beta_ar

        return {
            "cointegrated": half_life < self._lookback,
            "hedge_ratio": round(hedge, 4),
            "half_life": round(half_life, 1),
            "spread_mean": round(mean, 4),
            "spread_std": round(std, 4),
        }

    def backtest(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        asset_a: str = "A",
        asset_b: str = "B",
        capital: float = 100_000,
    ) -> Dict[str, Any]:
        """ペアトレーディングのバックテスト。

        Parameters
        ----------
        prices_a:
            銘柄Aの価格系列
        prices_b:
            銘柄Bの価格系列
        asset_a:
            銘柄A名
        asset_b:
            銘柄B名
        capital:
            初期資金

        Returns
        -------
        Dict[str, Any]
            バックテスト結果
        """
        hedge = self.compute_hedge_ratio(prices_a, prices_b)
        spread = self.compute_spread(prices_a, prices_b, hedge)

        z_scores = self.compute_z_score(spread)

        trades: List[PairTrade] = []
        position = 0  # 0: なし, 1: long_a_short_b, -1: short_a_long_b
        entry_idx = 0
        entry_z = 0.0
        entry_spread = 0.0
        equity = capital
        equity_curve = [capital]

        for i in range(1, len(z_scores)):
            z = z_scores.iloc[i]
            if np.isnan(z):
                equity_curve.append(equity)
                continue

            current_spread = spread.iloc[i]

            # エントリー
            if position == 0:
                if z > self._entry_z:
                    position = -1  # short_a_long_b (スプレッド縮小を期待)
                    entry_idx = i
                    entry_z = z
                    entry_spread = current_spread
                elif z < -self._entry_z:
                    position = 1  # long_a_short_b (スプレッド拡大を期待)
                    entry_idx = i
                    entry_z = z
                    entry_spread = current_spread

            # エグジット
            elif position != 0:
                should_exit = False
                reason = ""
                holding_days = i - entry_idx

                if abs(z) < self._exit_z:
                    should_exit = True
                    reason = "平均回帰"
                elif abs(z) > self._stop_z:
                    should_exit = True
                    reason = "ストップロス"
                elif holding_days >= self._max_holding_days:
                    should_exit = True
                    reason = "最大保有期間超過"

                if should_exit:
                    spread_change = current_spread - entry_spread
                    pnl = -position * spread_change * (capital * 0.5 / abs(entry_spread)) if entry_spread != 0 else 0
                    pnl_pct = pnl / capital * 100

                    trades.append(PairTrade(
                        asset_a=asset_a, asset_b=asset_b,
                        entry_z=round(entry_z, 3),
                        exit_z=round(z, 3),
                        entry_spread=round(entry_spread, 4),
                        exit_spread=round(current_spread, 4),
                        pnl=round(pnl, 2),
                        pnl_pct=round(pnl_pct, 2),
                        holding_days=i - entry_idx,
                        result="WIN" if pnl > 0 else "LOSS",
                    ))
                    equity += pnl
                    position = 0

            equity_curve.append(equity)

        # 結果集計
        wins = sum(1 for t in trades if t.result == "WIN")
        losses = sum(1 for t in trades if t.result == "LOSS")
        total_pnl = sum(t.pnl for t in trades)

        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd = (peak - eq_arr) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(dd)) * 100

        returns = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0

        return {
            "total_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / (wins + losses) * 100 if (wins + losses) > 0 else 0, 1),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(total_pnl / capital * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_dd_pct": round(max_dd, 2),
            "hedge_ratio": round(hedge, 4),
            "final_equity": round(equity, 2),
            "trades": [
                {"entry_z": t.entry_z, "exit_z": t.exit_z, "pnl": t.pnl,
                 "pnl_pct": t.pnl_pct, "days": t.holding_days, "result": t.result}
                for t in trades
            ],
        }
