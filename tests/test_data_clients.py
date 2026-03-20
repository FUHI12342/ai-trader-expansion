"""データクライアントのモックテスト。"""
from __future__ import annotations

import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.yfinance_client import YFinanceClient, _normalize_columns
from src.data.data_manager import DataManager
from config.settings import Settings, DatabaseSettings


# ============================================================
# yfinance クライアントのテスト
# ============================================================

class TestNormalizeColumns:
    """_normalize_columnsのテスト。"""

    def test_lowercase_columns(self):
        """カラム名が小文字に変換されることを確認。"""
        df = pd.DataFrame({
            "Open": [1.0], "High": [2.0], "Low": [0.5],
            "Close": [1.5], "Volume": [1000],
        })
        result = _normalize_columns(df)
        assert all(c == c.lower() for c in result.columns)

    def test_missing_required_column(self):
        """必須カラム不足でValueErrorが発生することを確認。"""
        df = pd.DataFrame({"Open": [1.0], "Close": [1.5]})
        with pytest.raises(ValueError, match="必須カラムがありません"):
            _normalize_columns(df)

    def test_adj_close_rename(self):
        """'adj close' が 'adj_close' に変換されることを確認。"""
        df = pd.DataFrame({
            "Open": [1.0], "High": [2.0], "Low": [0.5],
            "Close": [1.5], "Volume": [1000], "Adj Close": [1.4],
        })
        result = _normalize_columns(df)
        assert "adj_close" in result.columns


class TestYFinanceClient:
    """YFinanceClientのモックテスト。"""

    def test_fetch_ohlcv_returns_dataframe(self):
        """fetch_ohlcvがDataFrameを返すことをモックで確認。"""
        mock_df = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [105.0, 106.0],
                "Low": [99.0, 100.0],
                "Close": [103.0, 104.0],
                "Volume": [1_000_000, 1_200_000],
            },
            index=pd.date_range("2023-01-04", periods=2, freq="B"),
        )

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_df
            client = YFinanceClient()
            result = client.fetch_ohlcv("7203.T", "2023-01-04", "2023-01-10")

        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        assert not result.empty

    def test_fetch_ohlcv_empty_data_raises(self):
        """データが空の場合ValueErrorが発生することを確認。"""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            client = YFinanceClient()
            with pytest.raises(ValueError, match="データが取得できませんでした"):
                client.fetch_ohlcv("INVALID_SYMBOL")

    def test_memory_cache(self):
        """メモリキャッシュが機能することを確認。"""
        mock_df = pd.DataFrame(
            {
                "Open": [100.0], "High": [105.0], "Low": [99.0],
                "Close": [103.0], "Volume": [1_000_000],
            },
            index=pd.date_range("2023-01-04", periods=1, freq="B"),
        )

        call_count = 0

        def mock_history(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_df

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.side_effect = mock_history
            client = YFinanceClient(cache_enabled=True)
            client.fetch_ohlcv("7203.T", "2023-01-04", "2023-01-05")
            client.fetch_ohlcv("7203.T", "2023-01-04", "2023-01-05")

        # キャッシュヒットで2回目はAPIコールなし
        assert call_count == 1

    def test_clear_cache(self):
        """キャッシュクリアが機能することを確認。"""
        client = YFinanceClient(cache_enabled=True)
        client._memory_cache["test_key"] = pd.DataFrame()
        client.clear_cache()
        assert len(client._memory_cache) == 0


# ============================================================
# DataManager のテスト
# ============================================================

class TestDataManager:
    """DataManagerのテスト。"""

    def _make_settings_with_tempdb(self) -> Settings:
        """一時DBパスを使った設定を作成する。"""
        import os
        tmp = tempfile.mktemp(suffix=".db")
        db_settings = DatabaseSettings(cache_db_path=tmp)
        return Settings(database=db_settings)

    def test_init_creates_db(self):
        """初期化時にSQLiteDBが作成されることを確認。"""
        import os
        settings = self._make_settings_with_tempdb()
        dm = DataManager(settings=settings)
        assert os.path.exists(settings.database.cache_db_path)
        dm.close()

    def test_save_and_load_cache(self, ohlcv_data: pd.DataFrame):
        """キャッシュの保存と読み込みが正常に動作することを確認。"""
        settings = self._make_settings_with_tempdb()
        dm = DataManager(settings=settings, cache_expiry_days=365)

        dm._save_to_cache("TEST.T", ohlcv_data, "yfinance")
        loaded = dm._load_from_cache(
            "TEST.T",
            str(ohlcv_data.index[0])[:10],
            str(ohlcv_data.index[-1])[:10],
            "yfinance",
        )

        assert loaded is not None
        assert len(loaded) > 0
        dm.close()

    def test_cache_miss_returns_none(self):
        """キャッシュミス時にNoneが返ることを確認。"""
        settings = self._make_settings_with_tempdb()
        dm = DataManager(settings=settings)
        result = dm._load_from_cache("NOTFOUND.T", "2023-01-01", "2023-12-31", "yfinance")
        assert result is None
        dm.close()

    def test_fetch_ohlcv_with_yfinance_mock(self):
        """fetch_ohlcvがyfinanceモックから正常に取得できることを確認。"""
        settings = self._make_settings_with_tempdb()
        mock_df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [103.0, 104.0],
                "volume": [1_000_000, 1_200_000],
            },
            index=pd.date_range("2023-01-04", periods=2, freq="B"),
        )

        dm = DataManager(settings=settings, source="yfinance")
        with patch.object(dm, "_fetch_yfinance", return_value=mock_df):
            result = dm.fetch_ohlcv("7203.T", "2023-01-04", "2023-01-10", force_refresh=True)

        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        dm.close()

    def test_context_manager(self):
        """with構文でリソースが解放されることを確認。"""
        settings = self._make_settings_with_tempdb()
        with DataManager(settings=settings) as dm:
            assert dm._conn is not None
        # withブロック後は接続がクローズされている
        assert dm._conn is None

    def test_clear_cache_all(self, ohlcv_data: pd.DataFrame):
        """全キャッシュクリアが正常に動作することを確認。"""
        settings = self._make_settings_with_tempdb()
        dm = DataManager(settings=settings)
        dm._save_to_cache("TEST.T", ohlcv_data, "yfinance")
        dm.clear_cache()

        # クリア後はキャッシュヒットしない
        loaded = dm._load_from_cache(
            "TEST.T",
            str(ohlcv_data.index[0])[:10],
            str(ohlcv_data.index[-1])[:10],
            "yfinance",
        )
        assert loaded is None
        dm.close()
