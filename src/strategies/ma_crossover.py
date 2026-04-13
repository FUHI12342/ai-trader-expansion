"""MA Crossover（移動平均クロス）戦略。

ゴールデンクロス → BUY、デッドクロス → SELL。
ロングオンリー。
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import BaseStrategy, SignalType


class MACrossoverStrategy(BaseStrategy):
    """移動平均クロス戦略。

    Parameters
    ----------
    short_window:
        短期移動平均の期間（デフォルト: 20）
    long_window:
        長期移動平均の期間（デフォルト: 100）
    price_col:
        使用する価格カラム名（デフォルト: "close"）
    """

    name = "ma_crossover"

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 100,
        price_col: str = "close",
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.price_col = price_col

    def parameter_space(self) -> Dict[str, tuple]:
        """最適化パラメータ空間を返す。"""
        return {
            "fast_period": (int, 5, 50),
            "slow_period": (int, 20, 200),
        }

    def _min_bars(self) -> int:
        return self.long_window + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """MAクロスシグナルを生成する。

        Parameters
        ----------
        data:
            OHLCV DataFrame

        Returns
        -------
        pd.Series
            シグナル系列（1=BUY, -1=SELL, 0=FLAT）
        """
        self.validate_data(data)
        df = data.copy()

        if self.price_col not in df.columns:
            raise ValueError(f"価格カラム '{self.price_col}' が見つかりません。")

        px = df[self.price_col].astype(float)
        ma_short = px.rolling(window=self.short_window, min_periods=self.short_window).mean()
        ma_long = px.rolling(window=self.long_window, min_periods=self.long_window).mean()

        # クロス判定（両MAが揃った時点から）
        valid = ma_short.notna() & ma_long.notna()
        cross_up = valid & (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        cross_dn = valid & (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

        signals = pd.Series(SignalType.FLAT, index=df.index, dtype=int)
        signals[cross_up] = SignalType.BUY
        signals[cross_dn] = SignalType.SELL

        return signals
