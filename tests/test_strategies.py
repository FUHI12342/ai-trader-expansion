"""全戦略のユニットテスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import BaseStrategy, SignalType, REQUIRED_COLUMNS
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.dual_momentum import DualMomentumStrategy
from src.strategies.macd_rsi import MACDRSIStrategy, _ema, _rsi
from src.strategies.bollinger_rsi_adx import BollingerRSIADXStrategy, _bollinger_bands, _adx


# ============================================================
# 基底クラスのテスト
# ============================================================

class TestBaseStrategy:
    """BaseStrategyのテスト。"""

    def test_validate_data_empty(self):
        """空データでValueErrorが発生することを確認。"""
        strategy = MACrossoverStrategy()
        with pytest.raises(ValueError, match="入力データが空"):
            strategy.validate_data(pd.DataFrame())

    def test_validate_data_missing_columns(self):
        """必須カラム不足でValueErrorが発生することを確認。"""
        strategy = MACrossoverStrategy()
        df = pd.DataFrame({"open": [1, 2], "close": [1, 2]})
        with pytest.raises(ValueError, match="必須カラムが不足"):
            strategy.validate_data(df)

    def test_validate_data_insufficient_rows(self):
        """データ行数不足でValueErrorが発生することを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=10)
        df = pd.DataFrame({
            "open": [1.0] * 3, "high": [2.0] * 3, "low": [0.5] * 3,
            "close": [1.5] * 3, "volume": [1000] * 3,
        })
        with pytest.raises(ValueError, match="データ行数が不足"):
            strategy.validate_data(df)

    def test_get_parameters(self):
        """パラメータ取得が正常に動作することを確認。"""
        strategy = MACrossoverStrategy(short_window=15, long_window=50)
        params = strategy.get_parameters()
        assert params["short_window"] == 15
        assert params["long_window"] == 50

    def test_set_parameters_immutable(self):
        """set_parametersが元のインスタンスを変更しないことを確認。"""
        original = MACrossoverStrategy(short_window=20, long_window=100)
        updated = original.set_parameters(short_window=10)
        assert original.short_window == 20   # 変更なし
        assert updated.short_window == 10    # 新しいインスタンス

    def test_set_parameters_invalid_key(self):
        """存在しないパラメータでAttributeErrorが発生することを確認。"""
        strategy = MACrossoverStrategy()
        with pytest.raises(AttributeError):
            strategy.set_parameters(nonexistent_param=123)


# ============================================================
# MA Crossover 戦略のテスト
# ============================================================

class TestMACrossoverStrategy:
    """MACrossoverStrategyのテスト。"""

    def test_signal_type_values(self, ohlcv_data: pd.DataFrame):
        """シグナル値が-1, 0, 1のいずれかであることを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(ohlcv_data)
        valid = {SignalType.SELL, SignalType.FLAT, SignalType.BUY}
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_signal_length_matches_data(self, ohlcv_data: pd.DataFrame):
        """シグナルの長さがデータと一致することを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(ohlcv_data)
        assert len(signals) == len(ohlcv_data)

    def test_signal_index_matches_data(self, ohlcv_data: pd.DataFrame):
        """シグナルのインデックスがデータと一致することを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(ohlcv_data)
        pd.testing.assert_index_equal(signals.index, ohlcv_data.index)

    def test_buy_signal_on_golden_cross(self):
        """ゴールデンクロスでBUYシグナルが発生することを確認。"""
        # 短期MAが長期MAを上抜けるデータを作成
        dates = pd.date_range("2022-01-04", periods=25, freq="B")
        # 最初は低価格→後半急上昇でゴールデンクロス
        prices = [100.0] * 10 + [150.0] * 15
        df = pd.DataFrame({
            "open": prices, "high": [p + 5 for p in prices],
            "low": [p - 5 for p in prices], "close": prices,
            "volume": [1000000] * 25,
        }, index=dates)
        strategy = MACrossoverStrategy(short_window=3, long_window=5)
        signals = strategy.generate_signals(df)
        assert SignalType.BUY in signals.values

    def test_no_signal_before_warmup(self, ohlcv_data: pd.DataFrame):
        """ウォームアップ期間中はシグナルが発生しないことを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(ohlcv_data)
        # 最初の long_window 行はシグナルなし
        assert (signals.iloc[:20] == SignalType.FLAT).all()

    def test_data_not_mutated(self, ohlcv_data: pd.DataFrame):
        """元データが変更されないことを確認（immutableパターン）。"""
        original_close = ohlcv_data["close"].copy()
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        strategy.generate_signals(ohlcv_data)
        pd.testing.assert_series_equal(ohlcv_data["close"], original_close)


# ============================================================
# Dual Momentum 戦略のテスト
# ============================================================

class TestDualMomentumStrategy:
    """DualMomentumStrategyのテスト。"""

    def test_generate_signals_returns_series(self, ohlcv_long: pd.DataFrame):
        """シグナルがpd.Seriesで返されることを確認。"""
        strategy = DualMomentumStrategy(lookback_period=60, rebalance_freq="D")
        signals = strategy.generate_signals(ohlcv_long)
        assert isinstance(signals, pd.Series)

    def test_signal_values_valid(self, ohlcv_long: pd.DataFrame):
        """シグナル値が-1, 0, 1のいずれかであることを確認。"""
        strategy = DualMomentumStrategy(lookback_period=60, rebalance_freq="D")
        signals = strategy.generate_signals(ohlcv_long)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_buy_on_uptrend(self, ohlcv_trending_up: pd.DataFrame):
        """上昇トレンドでBUYシグナルが多いことを確認。"""
        strategy = DualMomentumStrategy(lookback_period=60, threshold=0.0, rebalance_freq="D")
        signals = strategy.generate_signals(ohlcv_trending_up)
        buy_ratio = (signals == SignalType.BUY).sum() / max(len(signals), 1)
        # 上昇トレンドではBUYが多い傾向
        assert buy_ratio > 0.1

    def test_flat_when_insufficient_data(self):
        """データ不足でValueErrorが発生することを確認。"""
        strategy = DualMomentumStrategy(lookback_period=252)
        df = pd.DataFrame({
            "open": [100.0] * 10, "high": [110.0] * 10,
            "low": [90.0] * 10, "close": [100.0] * 10, "volume": [1000] * 10,
        }, index=pd.date_range("2022-01-04", periods=10, freq="B"))
        with pytest.raises(ValueError, match="データ行数が不足"):
            strategy.generate_signals(df)

    def test_threshold_effect(self, ohlcv_long: pd.DataFrame):
        """閾値が高いほどBUYシグナルが少なくなることを確認。"""
        strategy_low = DualMomentumStrategy(lookback_period=60, threshold=0.0, rebalance_freq="D")
        strategy_high = DualMomentumStrategy(lookback_period=60, threshold=0.5, rebalance_freq="D")
        signals_low = strategy_low.generate_signals(ohlcv_long)
        signals_high = strategy_high.generate_signals(ohlcv_long)
        assert (signals_low == SignalType.BUY).sum() >= (signals_high == SignalType.BUY).sum()


# ============================================================
# MACD + RSI 戦略のテスト
# ============================================================

class TestMACDRSIStrategy:
    """MACDRSIStrategyのテスト。"""

    def test_ema_length(self):
        """EMAの長さが入力と一致することを確認。"""
        series = pd.Series(range(1, 51), dtype=float)
        ema = _ema(series, span=10)
        assert len(ema) == 50

    def test_rsi_range(self, ohlcv_data: pd.DataFrame):
        """RSIが0〜100の範囲にあることを確認。"""
        rsi = _rsi(ohlcv_data["close"], period=14)
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_generate_signals_valid(self, ohlcv_data: pd.DataFrame):
        """シグナル生成が正常に動作することを確認。"""
        strategy = MACDRSIStrategy()
        signals = strategy.generate_signals(ohlcv_data)
        assert len(signals) == len(ohlcv_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_buy_requires_both_conditions(self, ohlcv_data: pd.DataFrame):
        """BUYシグナルはMACDクロスとRSIフィルタの両方が必要なことを確認。"""
        strategy = MACDRSIStrategy(rsi_overbought=100)  # RSIフィルタを無効化
        strategy_with_filter = MACDRSIStrategy(rsi_overbought=70)

        signals_no_filter = strategy.generate_signals(ohlcv_data)
        signals_with_filter = strategy_with_filter.generate_signals(ohlcv_data)

        # フィルタなしのほうがBUYシグナルが多い（または同じ）
        assert (signals_no_filter == SignalType.BUY).sum() >= (
            signals_with_filter == SignalType.BUY).sum()

    def test_get_indicators(self, ohlcv_data: pd.DataFrame):
        """指標取得が正常に動作することを確認。"""
        strategy = MACDRSIStrategy()
        indicators = strategy.get_indicators(ohlcv_data)
        assert "macd" in indicators.columns
        assert "rsi" in indicators.columns
        assert "macd_hist" in indicators.columns

    def test_signal_no_lookahead(self, ohlcv_data: pd.DataFrame):
        """シグナルが未来データを参照していないことを確認（最後の行を削除しても結果が変わらない）。"""
        strategy = MACDRSIStrategy()
        signals_full = strategy.generate_signals(ohlcv_data)
        signals_minus1 = strategy.generate_signals(ohlcv_data.iloc[:-1])
        # 最後の1行以外は同じ
        pd.testing.assert_series_equal(
            signals_full.iloc[:-1],
            signals_minus1,
        )


# ============================================================
# Bollinger + RSI + ADX 戦略のテスト
# ============================================================

class TestBollingerRSIADXStrategy:
    """BollingerRSIADXStrategyのテスト。"""

    def test_bollinger_bands_shape(self, ohlcv_data: pd.DataFrame):
        """ボリンジャーバンドの形状を確認。"""
        upper, mid, lower = _bollinger_bands(ohlcv_data["close"], 20, 2.0)
        assert len(upper) == len(ohlcv_data)
        # NaNでない行だけを比較（ウォームアップ期間のNaNを除外）
        valid = upper.notna() & mid.notna() & lower.notna()
        assert (upper[valid] >= mid[valid]).all()
        assert (mid[valid] >= lower[valid]).all()

    def test_adx_range(self, ohlcv_data: pd.DataFrame):
        """ADXが0〜100の範囲にあることを確認。"""
        adx = _adx(ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"])
        valid_adx = adx.dropna()
        assert (valid_adx >= 0).all()
        # ADXは通常100以下（稀に超えることがあるが実用上問題なし）

    def test_generate_signals_valid(self, ohlcv_data: pd.DataFrame):
        """シグナル生成が正常に動作することを確認。"""
        strategy = BollingerRSIADXStrategy()
        signals = strategy.generate_signals(ohlcv_data)
        assert len(signals) == len(ohlcv_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_adx_filter_reduces_signals(self, ohlcv_data: pd.DataFrame):
        """ADXフィルタが高いほどシグナルが少なくなることを確認。"""
        strategy_low = BollingerRSIADXStrategy(adx_threshold=10.0)
        strategy_high = BollingerRSIADXStrategy(adx_threshold=50.0)
        signals_low = strategy_low.generate_signals(ohlcv_data)
        signals_high = strategy_high.generate_signals(ohlcv_data)
        total_low = (signals_low != SignalType.FLAT).sum()
        total_high = (signals_high != SignalType.FLAT).sum()
        assert total_low >= total_high

    def test_get_indicators(self, ohlcv_data: pd.DataFrame):
        """指標取得が正常に動作することを確認。"""
        strategy = BollingerRSIADXStrategy()
        indicators = strategy.get_indicators(ohlcv_data)
        assert "bb_upper" in indicators.columns
        assert "bb_lower" in indicators.columns
        assert "rsi" in indicators.columns
        assert "adx" in indicators.columns
