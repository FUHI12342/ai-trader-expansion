"""データ取得・キャッシュ統合マネージャー。

データソースの切替とSQLiteローカルキャッシュを管理する。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Literal, Optional

import pandas as pd

from config.settings import Settings, get_settings
from .yfinance_client import YFinanceClient
from .jquants_client import JQuantsClient

logger = logging.getLogger(__name__)

DataSource = Literal["yfinance", "jquants", "auto"]


class DataManager:
    """データ取得・キャッシュ統合マネージャー。

    ソースをyfinanceまたはJ-Quantsで切り替えて株価データを取得し、
    SQLiteにキャッシュする。キャッシュヒット時はAPIコールを省略する。

    Parameters
    ----------
    settings:
        設定オブジェクト（省略時は環境変数から読み込み）
    source:
        デフォルトデータソース（"yfinance", "jquants", "auto"）
    cache_expiry_days:
        キャッシュ有効期限（日数、デフォルト: 1日）
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        source: DataSource = "auto",
        cache_expiry_days: int = 1,
    ) -> None:
        self._settings = settings or get_settings()
        self._source = source
        self._cache_expiry_days = cache_expiry_days
        self._yfinance: Optional[YFinanceClient] = None
        self._jquants: Optional[JQuantsClient] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """SQLiteキャッシュDBを初期化する。"""
        db_path = self._settings.database.cache_db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._write_lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT NOT NULL,
                    cached_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, date, source)
                )
            """)
            self._conn.commit()
        logger.debug(f"SQLiteキャッシュDB初期化: {db_path}")

    def _get_yfinance(self) -> YFinanceClient:
        """YFinanceClientを遅延初期化する。"""
        if self._yfinance is None:
            self._yfinance = YFinanceClient(cache_enabled=False)
        return self._yfinance

    def _get_jquants(self) -> JQuantsClient:
        """JQuantsClientを遅延初期化する。"""
        if self._jquants is None:
            self._jquants = JQuantsClient(self._settings.jquants)
        return self._jquants

    def _load_from_cache(
        self,
        symbol: str,
        start: str,
        end: str,
        source: str,
    ) -> Optional[pd.DataFrame]:
        """SQLiteキャッシュからデータを読み込む。

        キャッシュが期限切れまたは不完全な場合はNoneを返す。
        """
        if self._conn is None:
            return None

        expiry = (datetime.now() - timedelta(days=self._cache_expiry_days)).strftime("%Y-%m-%d %H:%M:%S")

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT date, open, high, low, close, volume
            FROM ohlcv_cache
            WHERE symbol = ? AND source = ? AND date >= ? AND date <= ?
              AND cached_at > ?
            ORDER BY date ASC
        """, (symbol, source, start, end, expiry))

        rows = cursor.fetchall()
        if not rows:
            return None

        df = pd.DataFrame(
            rows, columns=["date", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        logger.debug(f"キャッシュヒット: {symbol} ({len(df)}行)")
        return df.copy()

    def _save_to_cache(
        self,
        symbol: str,
        df: pd.DataFrame,
        source: str,
    ) -> None:
        """DataFrameをSQLiteキャッシュに保存する。"""
        if self._conn is None:
            return

        cached_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records = []

        for date_idx, row in df.iterrows():
            date_str = str(date_idx)[:10]  # YYYY-MM-DD形式
            records.append((
                symbol, date_str,
                float(row.get("open", 0) or 0),
                float(row.get("high", 0) or 0),
                float(row.get("low", 0) or 0),
                float(row.get("close", 0) or 0),
                float(row.get("volume", 0) or 0),
                source, cached_at,
            ))

        with self._write_lock:
            cursor = self._conn.cursor()
            cursor.executemany("""
                INSERT OR REPLACE INTO ohlcv_cache
                (symbol, date, open, high, low, close, volume, source, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            self._conn.commit()
        logger.debug(f"キャッシュ保存: {symbol} ({len(records)}行)")

    def fetch_ohlcv(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        source: Optional[DataSource] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """OHLCV データを取得する（キャッシュ優先）。

        Parameters
        ----------
        symbol:
            銘柄コード（例: "7203.T", "AAPL"）
        start:
            開始日（"YYYY-MM-DD"形式）
        end:
            終了日（"YYYY-MM-DD"形式）
        source:
            データソース（省略時はデフォルトソース）
        force_refresh:
            キャッシュを無視して再取得するか

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame
        """
        effective_source = source or self._source
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # キャッシュ確認（force_refreshでない場合）
        if not force_refresh:
            source_key = effective_source if effective_source != "auto" else "yfinance"
            cached = self._load_from_cache(symbol, start, end, source_key)
            if cached is not None and not cached.empty:
                return cached

        # データソース選択とデータ取得
        df = self._fetch_from_source(symbol, start, end, effective_source)

        # キャッシュに保存
        source_key = effective_source if effective_source != "auto" else "yfinance"
        self._save_to_cache(symbol, df, source_key)

        return df.copy()

    def _fetch_from_source(
        self,
        symbol: str,
        start: str,
        end: str,
        source: DataSource,
    ) -> pd.DataFrame:
        """指定ソースからデータを取得する。"""
        if source == "jquants":
            return self._fetch_jquants(symbol, start, end)
        elif source == "yfinance":
            return self._fetch_yfinance(symbol, start, end)
        else:
            # auto: J-Quantsを試して失敗したらyfinance
            try:
                return self._fetch_jquants(symbol, start, end)
            except Exception as e:
                logger.warning(f"J-Quants取得失敗（{symbol}）: {e}. yfinanceにフォールバック")
                return self._fetch_yfinance(symbol, start, end)

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """yfinanceからデータを取得する。"""
        client = self._get_yfinance()
        return client.fetch_ohlcv(symbol, start, end)

    def _fetch_jquants(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """J-Quantsからデータを取得する。"""
        client = self._get_jquants()
        # J-Quantsの銘柄コードは数字4桁（末尾の.Tを除去）
        code = symbol.replace(".T", "").replace(".JP", "")
        df = client.fetch_stock_prices(code, start, end)

        # adj_close または close を使用
        if "adj_close" in df.columns and df["adj_close"].notna().any():
            df = df.copy()
            df["close"] = df["adj_close"]

        return df

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """キャッシュをクリアする。

        Parameters
        ----------
        symbol:
            特定銘柄のキャッシュをクリア（省略時は全件削除）
        """
        if self._conn is None:
            return

        with self._write_lock:
            cursor = self._conn.cursor()
            if symbol:
                cursor.execute("DELETE FROM ohlcv_cache WHERE symbol = ?", (symbol,))
                logger.info(f"キャッシュクリア: {symbol}")
            else:
                cursor.execute("DELETE FROM ohlcv_cache")
                logger.info("全キャッシュクリア")
            self._conn.commit()

    def close(self) -> None:
        """リソースをクローズする。"""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._jquants:
            self._jquants.close()

    def __enter__(self) -> "DataManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
