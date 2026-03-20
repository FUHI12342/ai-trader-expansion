"""pytest共有フィクスチャ。"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(
    n: int = 300,
    start: str = "2022-01-04",
    seed: int = 42,
    trend: float = 0.0003,
) -> pd.DataFrame:
    """テスト用OHLCVデータを生成する。

    Parameters
    ----------
    n:
        行数（日数）
    start:
        開始日（"YYYY-MM-DD"）
    seed:
        乱数シード
    trend:
        1日あたりのトレンド（0=フラット, 正=上昇）

    Returns
    -------
    pd.DataFrame
        OHLCVデータ（DatetimeIndex）
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n, freq="B")

    # ランダムウォーク + トレンドで価格生成
    returns = rng.normal(trend, 0.015, n)
    close = 1000.0 * np.cumprod(1 + returns)

    # 日中変動でOHLを生成
    intraday_range = close * rng.uniform(0.005, 0.025, n)
    high = close + intraday_range * rng.uniform(0, 1, n)
    low = close - intraday_range * rng.uniform(0, 1, n)
    open_ = close - intraday_range * rng.uniform(-0.5, 0.5, n)

    volume = rng.integers(100_000, 10_000_000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


@pytest.fixture
def ohlcv_data() -> pd.DataFrame:
    """標準テストデータ（300日、シード42）。"""
    return _make_ohlcv(n=300)


@pytest.fixture
def ohlcv_long() -> pd.DataFrame:
    """長期テストデータ（800日、シード42）。"""
    return _make_ohlcv(n=800, seed=42)


@pytest.fixture
def ohlcv_trending_up() -> pd.DataFrame:
    """上昇トレンドのテストデータ。"""
    return _make_ohlcv(n=400, trend=0.002, seed=123)


@pytest.fixture
def ohlcv_trending_down() -> pd.DataFrame:
    """下落トレンドのテストデータ。"""
    return _make_ohlcv(n=400, trend=-0.002, seed=456)


@pytest.fixture
def simple_trade_returns() -> list[float]:
    """簡単な取引リターン列。"""
    return [0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.01, 0.025, -0.015, 0.03]


@pytest.fixture
def equity_curve() -> pd.Series:
    """テスト用エクイティカーブ。"""
    rng = np.random.default_rng(999)
    returns = rng.normal(0.0003, 0.01, 252)
    capital = 1_000_000.0
    values = [capital]
    for r in returns:
        values.append(values[-1] * (1 + r))
    dates = pd.date_range("2023-01-04", periods=len(values), freq="B")
    return pd.Series(values, index=dates)
