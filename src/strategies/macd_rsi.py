"""MACD + RSI 複合戦略。

- MACD (12, 26, 9) のゴールデンクロス/デッドクロスでエントリー候補を抽出
- RSI (14) で過売/過買フィルタ（買い: RSI < 70, 売り: RSI > 30）
- 両方のシグナルが一致した場合のみエントリー
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy, SignalType


def _ema(series: pd.Series, span: int) -> pd.Series:
    """指数移動平均（EMA）を計算する。"""
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI（相対力指数）を計算する。"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # データ不足時は中立値50


class MACDRSIStrategy(BaseStrategy):
    """MACD + RSI 複合戦略。

    Parameters
    ----------
    macd_fast:
        MACD短期EMA期間（デフォルト: 12）
    macd_slow:
        MACD長期EMA期間（デフォルト: 26）
    macd_signal:
        MACDシグナル線のEMA期間（デフォルト: 9）
    rsi_period:
        RSI計算期間（デフォルト: 14）
    rsi_overbought:
        RSI過買水準（デフォルト: 70）
    rsi_oversold:
        RSI過売水準（デフォルト: 30）
    """

    name = "macd_rsi"

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> None:
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def _min_bars(self) -> int:
        return self.macd_slow + self.macd_signal + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """MACD+RSI複合シグナルを生成する。

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
        close = df["close"].astype(float)

        # MACD計算
        ema_fast = _ema(close, self.macd_fast)
        ema_slow = _ema(close, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, self.macd_signal)

        # MACDクロス判定
        macd_cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        macd_cross_dn = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        # RSI計算
        rsi = _rsi(close, self.rsi_period)

        # RSIフィルタ（買い: 過買でない, 売り: 過売でない）
        rsi_buy_filter = rsi < self.rsi_overbought
        rsi_sell_filter = rsi > self.rsi_oversold

        # 複合シグナル（MACDクロスかつRSIフィルタをパス）
        buy_signal = macd_cross_up & rsi_buy_filter
        sell_signal = macd_cross_dn & rsi_sell_filter

        signals = pd.Series(SignalType.FLAT, index=df.index, dtype=int)
        signals[buy_signal] = SignalType.BUY
        signals[sell_signal] = SignalType.SELL

        return signals

    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """デバッグ用: MACD/RSI指標をDataFrameとして返す。"""
        df = data.copy()
        close = df["close"].astype(float)
        ema_fast = _ema(close, self.macd_fast)
        ema_slow = _ema(close, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, self.macd_signal)
        df = df.copy()
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = macd_line - signal_line
        df["rsi"] = _rsi(close, self.rsi_period)
        return df
