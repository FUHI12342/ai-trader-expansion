"""DataManager v2 のテスト（プラグインレジストリ・インターバルキャッシュ等）。"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from config.settings import DatabaseSettings, Settings
from src.data.source_base import DataSourceBase


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 30, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start, periods=n, freq="B")
    close = 1000.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    return pd.DataFrame(
        {"open": close * 0.99, "high": close * 1.01, "low": close * 0.98, "close": close, "volume": 1e6},
        index=dates,
    )


class DummySource(DataSourceBase):
    """テスト用のダミーデータソース。"""

    def __init__(self, name: str, supported: bool = True, data: Optional[pd.DataFrame] = None) -> None:
        self.name = name
        self._supported = supported
        self._data = data if data is not None else _make_ohlcv()

    def fetch_ohlcv(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        return self._data.copy()

    def supports_symbol(self, symbol: str) -> bool:
        return self._supported

    def supported_intervals(self) -> list[str]:
        return ["1d", "1h"]


@pytest.fixture
def tmp_settings(tmp_path: Path) -> Settings:
    """一時ディレクトリを使う設定を返す。"""
    db_path = str(tmp_path / "test_cache.db")
    db_settings = DatabaseSettings(cache_db_path=db_path)
    settings = Settings(database=db_settings)
    return settings


@pytest.fixture
def manager(tmp_settings: Settings):
    """DataManagerインスタンスを返す。"""
    from src.data.data_manager import DataManager
    dm = DataManager(settings=tmp_settings, source="auto")
    yield dm
    dm.close()


# ---------------------------------------------------------------------------
# プラグインレジストリテスト
# ---------------------------------------------------------------------------

def test_register_source(manager):
    """register_sourceでソースが登録できる。"""
    src = DummySource("custom_source")
    manager.register_source(src)
    assert "custom_source" in manager._sources


def test_unregister_source(manager):
    """unregister_sourceでソースが削除できる。"""
    src = DummySource("to_remove")
    manager.register_source(src)
    manager.unregister_source("to_remove")
    assert "to_remove" not in manager._sources


def test_unregister_nonexistent_no_error(manager):
    """未登録のソースを削除してもエラーにならない。"""
    manager.unregister_source("nonexistent")  # エラーなし


def test_register_source_overwrites_same_name(manager):
    """同名ソースを登録すると上書きされる。"""
    src1 = DummySource("duplicate")
    src2 = DummySource("duplicate")
    manager.register_source(src1)
    manager.register_source(src2)
    assert manager._sources["duplicate"] is src2


# ---------------------------------------------------------------------------
# 自動ソース選択テスト
# ---------------------------------------------------------------------------

def test_auto_select_source_for_japanese_stock(manager):
    """日本株(.T)のとき、supports_symbolでTrueを返すソースが選ばれる。"""
    # yfinanceはすべての銘柄をサポートするので登録しておく
    yf_src = DummySource("yfinance", supported=True)
    manager.register_source(yf_src)
    selected = manager._auto_select_source("7203.T")
    assert selected is yf_src


def test_auto_select_source_returns_first_match(manager):
    """複数ソースのうち最初にsupports_symbol=Trueを返すものが選ばれる。"""
    src_a = DummySource("source_a", supported=False)
    src_b = DummySource("source_b", supported=True)
    manager.register_source(src_a)
    manager.register_source(src_b)
    selected = manager._auto_select_source("BTC/USDT")
    assert selected is src_b


def test_auto_select_source_fallback_to_yfinance(manager):
    """マッチするソースがない場合はyfinanceにフォールバックする。"""
    # yfinanceクライアントのフォールバックをテスト（実際にyfinanceが起動するのでモック）
    with patch.object(manager, "_get_yfinance") as mock_yf:
        mock_source = DummySource("yfinance", supported=False)
        mock_yf.return_value = mock_source
        selected = manager._auto_select_source("UNKNOWN_SYMBOL_XYZ")
        # _auto_select_source はフォールバックで _get_yfinance を呼ぶ
        mock_yf.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_ohlcvとキャッシュテスト
# ---------------------------------------------------------------------------

def test_fetch_ohlcv_uses_registered_source(manager):
    """fetch_ohlcvが登録済みソースを使う。"""
    expected_df = _make_ohlcv(30)
    src = DummySource("yfinance", supported=True, data=expected_df)
    manager.register_source(src)

    df = manager.fetch_ohlcv("7203.T", "2024-01-01", "2024-01-31", source="yfinance")
    assert not df.empty
    assert "close" in df.columns


def test_fetch_ohlcv_with_interval_parameter(manager):
    """fetch_ohlcvでintervalパラメータが渡せる。"""
    src = DummySource("yfinance", supported=True)
    manager.register_source(src)

    df = manager.fetch_ohlcv("7203.T", "2024-01-01", "2024-01-31", source="yfinance", interval="1h")
    assert not df.empty


def test_fetch_ohlcv_caches_result(manager):
    """fetch_ohlcvの結果がキャッシュに保存される。"""
    src = DummySource("yfinance", supported=True)
    manager.register_source(src)

    # 1回目の取得（キャッシュなし）
    manager.fetch_ohlcv("7203.T", "2024-01-01", "2024-01-10", source="yfinance")

    # キャッシュに何かデータがあることを確認
    assert manager._conn is not None
    cursor = manager._conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ohlcv_cache WHERE symbol=?", ("7203.T",))
    count = cursor.fetchone()[0]
    assert count > 0


def test_fetch_ohlcv_force_refresh_bypasses_cache(manager):
    """force_refresh=Trueでキャッシュをバイパスする。"""
    src = DummySource("yfinance", supported=True)
    manager.register_source(src)

    # 1回目
    manager.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-10", source="yfinance")
    # force_refreshで再取得
    df = manager.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-10", source="yfinance", force_refresh=True)
    assert not df.empty


def test_clear_cache_symbol(manager):
    """clear_cache(symbol)で特定銘柄のキャッシュが削除される。"""
    src = DummySource("yfinance", supported=True)
    manager.register_source(src)
    manager.fetch_ohlcv("7203.T", "2024-01-01", "2024-01-10", source="yfinance")
    manager.clear_cache("7203.T")

    cursor = manager._conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ohlcv_cache WHERE symbol=?", ("7203.T",))
    assert cursor.fetchone()[0] == 0


def test_clear_cache_all(manager):
    """clear_cache()で全キャッシュが削除される。"""
    src = DummySource("yfinance", supported=True)
    manager.register_source(src)
    manager.fetch_ohlcv("7203.T", "2024-01-01", "2024-01-10", source="yfinance")
    manager.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-10", source="yfinance")
    manager.clear_cache()

    cursor = manager._conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ohlcv_cache")
    assert cursor.fetchone()[0] == 0


def test_migrate_cache_schema_adds_interval_column(tmp_settings: Settings, tmp_path: Path):
    """マイグレーションでintervalカラムが追加される。"""
    from src.data.data_manager import DataManager

    # intervalカラムなしでテーブルを作成
    db_path = tmp_settings.database.cache_db_path
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE ohlcv_cache (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL, source TEXT, cached_at TEXT,
            PRIMARY KEY (symbol, date, source)
        )
    """)
    conn.commit()
    conn.close()

    # DataManagerを作成するとマイグレーションが走る
    dm = DataManager(settings=tmp_settings)

    # intervalカラムが存在することを確認
    cursor = dm._conn.cursor()
    cursor.execute("PRAGMA table_info(ohlcv_cache)")
    columns = {row[1] for row in cursor.fetchall()}
    assert "interval" in columns
    dm.close()


# ---------------------------------------------------------------------------
# コンテキストマネージャテスト
# ---------------------------------------------------------------------------

def test_context_manager(tmp_settings: Settings):
    """DataManagerがコンテキストマネージャとして使える。"""
    from src.data.data_manager import DataManager
    with DataManager(settings=tmp_settings) as dm:
        assert dm._conn is not None
    # close後は接続がNone
    assert dm._conn is None


def test_fetch_ccxt_raises_when_not_registered(manager):
    """CCXTソースが未登録のときfetch_ohlcvはValueErrorを発生させる。"""
    with pytest.raises(ValueError, match="CCXTソースが登録されていません"):
        manager.fetch_ohlcv("BTC/USDT", "2024-01-01", "2024-01-31", source="ccxt")


# ---------------------------------------------------------------------------
# subscribe_realtime テスト
# ---------------------------------------------------------------------------

def test_subscribe_realtime_returns_feed(manager):
    """subscribe_realtimeがWebSocketFeedを返す。"""
    from src.data.ws_feed import WebSocketFeed
    received = []
    feed = manager.subscribe_realtime(
        symbols=["BTC/USDT"],
        callback=lambda t: received.append(t),
    )
    assert isinstance(feed, WebSocketFeed)
    assert "BTC/USDT" in feed._subscriptions
    assert len(feed._callbacks) == 1
