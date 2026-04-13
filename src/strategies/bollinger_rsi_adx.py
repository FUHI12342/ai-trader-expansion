"""ボリンジャーバンド + RSI + ADX 複合戦略。

- BB(20, 2) タッチでエントリー候補
- RSI(14) で方向確認（下バンドタッチ: RSI < 50, 上バンドタッチ: RSI > 50）
- ADX(14) > 25 でトレンド強度フィルタ（トレンド相場のみエントリー）
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .base import BaseStrategy, SignalType
from .macd_rsi import _rsi


def _bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ボリンジャーバンドを計算する。

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (upper, middle, lower) バンド
    """
    mid = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ADX（平均方向性指数）を計算する。

    ADX > 25 でトレンド相場と判定。
    """
    # True Range計算
    hl = high - low
    hc = (high - close.shift(1)).abs()
    lc = (low - close.shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    # Directional Movement計算
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    # Smoothed values (Wilder's smoothing)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s.replace(0, np.nan)

    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx.fillna(0)


class BollingerRSIADXStrategy(BaseStrategy):
    """ボリンジャーバンド + RSI + ADX 複合戦略。

    Parameters
    ----------
    bb_period:
        ボリンジャーバンドの期間（デフォルト: 20）
    bb_std:
        ボリンジャーバンドの標準偏差倍数（デフォルト: 2.0）
    rsi_period:
        RSI期間（デフォルト: 14）
    adx_period:
        ADX期間（デフォルト: 14）
    adx_threshold:
        ADXトレンド強度の閾値（デフォルト: 25）
    """

    name = "bollinger_rsi_adx"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def parameter_space(self) -> Dict[str, tuple]:
        """最適化パラメータ空間を返す。"""
        return {
            "bb_period": (int, 10, 30),
            "rsi_period": (int, 7, 21),
            "adx_period": (int, 7, 21),
            "adx_threshold": (float, 20.0, 35.0),
        }

    def _min_bars(self) -> int:
        return max(self.bb_period, self.adx_period) * 2 + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """ボリンジャー + RSI + ADX複合シグナルを生成する。

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
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # ボリンジャーバンド計算
        bb_upper, bb_mid, bb_lower = _bollinger_bands(close, self.bb_period, self.bb_std)

        # RSI計算
        rsi = _rsi(close, self.rsi_period)

        # ADX計算
        adx = _adx(high, low, close, self.adx_period)

        # トレンド強度フィルタ
        trend_filter = adx > self.adx_threshold

        # エントリー条件
        # 下バンドタッチ（低RSI + トレンドあり） → BUY
        touch_lower = close <= bb_lower
        rsi_bullish = rsi < 50
        buy_signal = touch_lower & rsi_bullish & trend_filter

        # 上バンドタッチ（高RSI + トレンドあり） → SELL
        touch_upper = close >= bb_upper
        rsi_bearish = rsi > 50
        sell_signal = touch_upper & rsi_bearish & trend_filter

        signals = pd.Series(SignalType.FLAT, index=df.index, dtype=int)
        signals[buy_signal] = SignalType.BUY
        signals[sell_signal] = SignalType.SELL

        return signals

    def get_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """デバッグ用: 指標をDataFrameとして返す。"""
        df = data.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        bb_upper, bb_mid, bb_lower = _bollinger_bands(close, self.bb_period, self.bb_std)
        df["bb_upper"] = bb_upper
        df["bb_mid"] = bb_mid
        df["bb_lower"] = bb_lower
        df["rsi"] = _rsi(close, self.rsi_period)
        df["adx"] = _adx(high, low, close, self.adx_period)
        return df
