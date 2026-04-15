"""KronosForecaster と KronosStrategy のユニットテスト。

Kronos はオプション依存（transformers/torch）なのでモックを使用する。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ============================================================
# テスト用ヘルパー
# ============================================================

def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    """テスト用 OHLCV DataFrame を生成する。"""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "open": close - rng.uniform(0, 1, n),
        "high": close + rng.uniform(0, 2, n),
        "low": close - rng.uniform(0, 2, n),
        "close": close,
        "volume": rng.integers(1000, 10000, n).astype(float),
    })


# ============================================================
# _quantize_kline のテスト
# ============================================================

class TestQuantizeKline:
    """OHLCV → トークン量子化のテスト。"""

    def test_output_shape(self) -> None:
        """出力が (N, 5) 形状のリストであることを確認。"""
        from src.forecasters.kronos_forecaster import _quantize_kline
        data = _make_ohlcv(20)
        tokens = _quantize_kline(data)
        assert len(tokens) == 20
        assert all(len(row) == 5 for row in tokens)

    def test_values_in_range(self) -> None:
        """トークン値が [0, n_bins-1] 範囲内であることを確認。"""
        from src.forecasters.kronos_forecaster import _quantize_kline
        data = _make_ohlcv(30)
        n_bins = 128
        tokens = _quantize_kline(data, n_bins=n_bins)
        for row in tokens:
            for val in row:
                assert 0 <= val < n_bins

    def test_constant_series(self) -> None:
        """全値が同一の場合、中央ビンが返ることを確認。"""
        from src.forecasters.kronos_forecaster import _quantize_kline
        data = pd.DataFrame({
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "close": [100.0] * 10,
            "volume": [5000.0] * 10,
        })
        tokens = _quantize_kline(data, n_bins=256)
        for row in tokens:
            for val in row:
                assert val == 128  # n_bins // 2


# ============================================================
# KronosForecaster のテスト（モック使用）
# ============================================================

class TestKronosForecasterAvailability:
    """依存パッケージ未インストール時の振る舞い。"""

    def test_import_error_when_unavailable(self) -> None:
        """_KRONOS_AVAILABLE=False 時に ImportError が発生する。"""
        with patch("src.forecasters.kronos_forecaster._KRONOS_AVAILABLE", False):
            from src.forecasters.kronos_forecaster import KronosForecaster
            with pytest.raises(ImportError, match="transformers"):
                KronosForecaster()

    def test_predict_raises_when_unavailable(self) -> None:
        """predict() が _KRONOS_AVAILABLE=False 時に ImportError を返す。"""
        with patch("src.forecasters.kronos_forecaster._KRONOS_AVAILABLE", True):
            from src.forecasters.kronos_forecaster import KronosForecaster
            forecaster = KronosForecaster.__new__(KronosForecaster)
            forecaster.model_name = "test"
            forecaster.device = "cpu"
            forecaster.n_bins = 256
            forecaster.revision = None
            forecaster._model = None
            forecaster._tokenizer = None

        with patch("src.forecasters.kronos_forecaster._KRONOS_AVAILABLE", False):
            with pytest.raises(ImportError):
                forecaster.predict(_make_ohlcv(), horizon=5)


class TestKronosForecasterValidation:
    """入力バリデーションのテスト。"""

    @patch("src.forecasters.kronos_forecaster._KRONOS_AVAILABLE", True)
    def test_empty_data_raises(self) -> None:
        """空 DataFrame で ValueError。"""
        from src.forecasters.kronos_forecaster import KronosForecaster
        forecaster = KronosForecaster.__new__(KronosForecaster)
        forecaster._model = None
        forecaster._tokenizer = None
        forecaster.n_bins = 256
        forecaster.model_name = "test"
        forecaster.device = "cpu"
        forecaster.revision = None
        with pytest.raises(ValueError, match="空"):
            forecaster.predict(pd.DataFrame(), horizon=5)

    @patch("src.forecasters.kronos_forecaster._KRONOS_AVAILABLE", True)
    def test_invalid_horizon_raises(self) -> None:
        """horizon < 1 で ValueError。"""
        from src.forecasters.kronos_forecaster import KronosForecaster
        forecaster = KronosForecaster.__new__(KronosForecaster)
        forecaster._model = None
        forecaster._tokenizer = None
        forecaster.n_bins = 256
        forecaster.model_name = "test"
        forecaster.device = "cpu"
        forecaster.revision = None
        with pytest.raises(ValueError, match="horizon"):
            forecaster.predict(_make_ohlcv(), horizon=0)

    @patch("src.forecasters.kronos_forecaster._KRONOS_AVAILABLE", True)
    def test_missing_columns_raises(self) -> None:
        """必須カラム不足で ValueError。"""
        from src.forecasters.kronos_forecaster import KronosForecaster
        forecaster = KronosForecaster.__new__(KronosForecaster)
        forecaster._model = None
        forecaster._tokenizer = None
        forecaster.n_bins = 256
        forecaster.model_name = "test"
        forecaster.device = "cpu"
        forecaster.revision = None
        with pytest.raises(ValueError, match="必須カラム"):
            forecaster.predict(pd.DataFrame({"close": [1, 2, 3]}), horizon=5)


class TestKronosForecasterPredict:
    """predict() の出力形式テスト（モデルモック）。"""

    def test_predict_output_shape(self) -> None:
        """predict() が正しい形状の DataFrame を返す。"""
        import torch
        from src.forecasters.kronos_forecaster import KronosForecaster

        mock_model = MagicMock()
        # generate() の戻り値をシミュレート
        input_len = 50 * 5  # 50行 × 5チャネル
        gen_len = 5 * 5     # horizon=5 × 5チャネル
        fake_output = torch.randint(0, 256, (1, input_len + gen_len))
        mock_model.generate.return_value = fake_output
        mock_model.device = "cpu"

        mock_tokenizer = MagicMock()

        with patch("src.forecasters.kronos_forecaster._KRONOS_AVAILABLE", True):
            forecaster = KronosForecaster.__new__(KronosForecaster)
            forecaster.model_name = "test"
            forecaster.device = "cpu"
            forecaster.n_bins = 256
            forecaster.revision = None
            forecaster._model = mock_model
            forecaster._tokenizer = mock_tokenizer

            result = forecaster.predict(_make_ohlcv(50), horizon=5)

        assert isinstance(result, pd.DataFrame)
        assert "forecast_close" in result.columns
        assert "forecast_direction" in result.columns
        assert len(result) == 5

    def test_predict_direction_returns_int(self) -> None:
        """predict_direction() が int を返す。"""
        import torch
        from src.forecasters.kronos_forecaster import KronosForecaster

        mock_model = MagicMock()
        input_len = 50 * 5
        gen_len = 5 * 5
        fake_output = torch.randint(0, 256, (1, input_len + gen_len))
        mock_model.generate.return_value = fake_output
        mock_model.device = "cpu"

        with patch("src.forecasters.kronos_forecaster._KRONOS_AVAILABLE", True):
            forecaster = KronosForecaster.__new__(KronosForecaster)
            forecaster.model_name = "test"
            forecaster.device = "cpu"
            forecaster.n_bins = 256
            forecaster.revision = None
            forecaster._model = mock_model
            forecaster._tokenizer = MagicMock()

            direction = forecaster.predict_direction(_make_ohlcv(50), horizon=5)

        assert direction in (-1, 0, 1)


# ============================================================
# KronosStrategy のテスト
# ============================================================

class TestKronosStrategyFallback:
    """依存未インストール時の FLAT フォールバック。"""

    def test_all_flat_when_unavailable(self) -> None:
        """_KRONOS_AVAILABLE=False 時に全シグナル FLAT。"""
        with patch("src.strategies.kronos_strategy._KRONOS_AVAILABLE", False):
            from src.strategies.kronos_strategy import KronosStrategy
            strategy = KronosStrategy.__new__(KronosStrategy)
            strategy.model_name = "test"
            strategy.horizon = 5
            strategy.var_confidence = 0.95
            strategy.capital = 1_000_000.0
            strategy.var_limit = 0.02
            strategy.revision = None
            strategy._forecaster = None
            strategy._var_calc = MagicMock()

            data = _make_ohlcv(50)
            with pytest.warns(RuntimeWarning, match="FLAT"):
                signals = strategy.generate_signals(data)

            assert (signals == 0).all()

    def test_validate_data_called(self) -> None:
        """generate_signals が validate_data を呼ぶ。"""
        with patch("src.strategies.kronos_strategy._KRONOS_AVAILABLE", False):
            from src.strategies.kronos_strategy import KronosStrategy
            strategy = KronosStrategy.__new__(KronosStrategy)
            strategy.model_name = "test"
            strategy.horizon = 5
            strategy.var_confidence = 0.95
            strategy.capital = 1_000_000.0
            strategy.var_limit = 0.02
            strategy.revision = None
            strategy._forecaster = None
            strategy._var_calc = MagicMock()

            with pytest.raises(ValueError, match="空"):
                strategy.generate_signals(pd.DataFrame())


class TestKronosStrategyMeta:
    """KronosStrategy メタ情報のテスト。"""

    def test_name(self) -> None:
        """戦略名が 'kronos' であること。"""
        from src.strategies.kronos_strategy import KronosStrategy
        assert KronosStrategy.name == "kronos"

    def test_min_bars(self) -> None:
        """最低バー数がhorizon*3以上であること。"""
        with patch("src.strategies.kronos_strategy._KRONOS_AVAILABLE", False):
            from src.strategies.kronos_strategy import KronosStrategy
            strategy = KronosStrategy.__new__(KronosStrategy)
            strategy.horizon = 10
            assert strategy._min_bars() >= 30
