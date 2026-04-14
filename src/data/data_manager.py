"""データ取得・キャッシュ統合マネージャー。

データソースのプラグインレジストリとSQLiteローカルキャッシュを管理する。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, Literal, Optional

import pandas as pd

from config.settings import Settings, get_settings
from .source_base import DataSourceBase
from .yfinance_client import YFinanceClient
from .jquants_client import JQuantsClient
from .jquants_official_client import JQuantsOfficialClient

logger = logging.getLogger(__name__)

DataSource = Literal["yfinance", "jquants", "auto", "ccxt"]


class DataManager:
    """データ取得・キャッシュ統合マネージャー。

    プラグインレジストリパターンでデータソースを管理し、
    銘柄に応じて最適なソースを自動選択する。
    SQLiteにキャッシュしてAPIコールを削減する。

    Parameters
    ----------
    settings:
        設定オブジェクト（省略時は環境変数から読み込み）
    source:
        デフォルトデータソース（"yfinance", "jquants", "auto", "ccxt"）
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
        # プラグインレジストリ: name → DataSourceBase
        self._sources: Dict[str, DataSourceBase] = {}
        self._conn: Optional[sqlite3.Connection] = None
        self._write_lock = threading.Lock()
        self._init_db()
        # デフォルトソースを遅延登録（使用時に初期化）
        self._jquants_raw: Optional[JQuantsClient | JQuantsOfficialClient] = None

    # ------------------------------------------------------------------
    # プラグインレジストリ
    # ------------------------------------------------------------------

    def register_source(self, source: DataSourceBase) -> None:
        """データソースをレジストリに登録する。

        Parameters
        ----------
        source:
            登録する DataSourceBase 実装
        """
        self._sources[source.name] = source
        logger.debug(f"データソース登録: {source.name}")

    def unregister_source(self, name: str) -> None:
        """データソースをレジストリから削除する。

        Parameters
        ----------
        name:
            削除するソース名
        """
        self._sources.pop(name, None)
        logger.debug(f"データソース削除: {name}")

    def _get_yfinance(self) -> YFinanceClient:
        """YFinanceClient を遅延初期化してレジストリに登録する。"""
        if "yfinance" not in self._sources:
            client = YFinanceClient(cache_enabled=False)
            self.register_source(client)
        source = self._sources["yfinance"]
        assert isinstance(source, YFinanceClient)
        return source

    def _get_jquants(self) -> JQuantsClient | JQuantsOfficialClient:
        """JQuantsClient を遅延初期化してレジストリに登録する。

        use_official_client が True の場合は公式SDKアダプタを使用する。
        """
        if self._jquants_raw is None:
            if self._settings.jquants.use_official_client:
                self._jquants_raw = JQuantsOfficialClient(self._settings.jquants)
            else:
                self._jquants_raw = JQuantsClient(self._settings.jquants)
            # JQuantsClient は DataSourceBase を実装しているが
            # JQuantsOfficialClient はレガシーのため型チェック後に登録
            if isinstance(self._jquants_raw, DataSourceBase):
                self.register_source(self._jquants_raw)
        return self._jquants_raw

    def _auto_select_source(self, symbol: str) -> DataSourceBase:
        """銘柄に最適なデータソースを自動選択する。

        登録済みソースの supports_symbol() を順に確認し、
        最初にマッチしたソースを返す。マッチしない場合は yfinance を使用。

        Parameters
        ----------
        symbol:
            銘柄シンボル

        Returns
        -------
        DataSourceBase
            選択されたデータソース
        """
        # 登録済みソースから supports_symbol() でマッチするものを探す
        for src in self._sources.values():
            if src.supports_symbol(symbol):
                return src

        # フォールバック: yfinance を遅延初期化して返す
        return self._get_yfinance()

    # ------------------------------------------------------------------
    # DB初期化
    # ------------------------------------------------------------------

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
                    interval TEXT NOT NULL DEFAULT '1d',
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    source TEXT NOT NULL,
                    cached_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, date, interval, source)
                )
            """)
            self._conn.commit()
        self._migrate_cache_schema()
        logger.debug(f"SQLiteキャッシュDB初期化: {db_path}")

    def _migrate_cache_schema(self) -> None:
        """既存キャッシュDBにintervalカラムを追加するマイグレーション。

        新規DBには不要だが、Phase1以前のDBとの後方互換のために実行する。
        """
        if self._conn is None:
            return
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA table_info(ohlcv_cache)")
        columns = {row[1] for row in cursor.fetchall()}

        if "interval" not in columns:
            logger.info("キャッシュDBマイグレーション: intervalカラムを追加")
            with self._write_lock:
                cursor.execute(
                    "ALTER TABLE ohlcv_cache ADD COLUMN interval TEXT NOT NULL DEFAULT '1d'"
                )
                self._conn.commit()

    # ------------------------------------------------------------------
    # キャッシュ読み書き
    # ------------------------------------------------------------------

    def _load_from_cache(
        self,
        symbol: str,
        start: str,
        end: str,
        source: str,
        interval: str = "1d",
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
            WHERE symbol = ? AND source = ? AND interval = ?
              AND date >= ? AND date <= ?
              AND cached_at > ?
            ORDER BY date ASC
        """, (symbol, source, interval, start, end, expiry))

        rows = cursor.fetchall()
        if not rows:
            return None

        df = pd.DataFrame(
            rows, columns=["date", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        logger.debug(f"キャッシュヒット: {symbol} interval={interval} ({len(df)}行)")
        return df.copy()

    def _save_to_cache(
        self,
        symbol: str,
        df: pd.DataFrame,
        source: str,
        interval: str = "1d",
    ) -> None:
        """DataFrameをSQLiteキャッシュに保存する。"""
        if self._conn is None:
            return

        cached_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        records = []

        for date_idx, row in df.iterrows():
            date_str = str(date_idx)[:10]  # YYYY-MM-DD形式
            records.append((
                symbol, date_str, interval,
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
                (symbol, date, interval, open, high, low, close, volume, source, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            self._conn.commit()
        logger.debug(f"キャッシュ保存: {symbol} interval={interval} ({len(records)}行)")

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        source: Optional[DataSource] = None,
        force_refresh: bool = False,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """OHLCV データを取得する（キャッシュ優先）。

        Parameters
        ----------
        symbol:
            銘柄コード（例: "7203.T", "AAPL", "BTC/USDT"）
        start:
            開始日（"YYYY-MM-DD"形式）
        end:
            終了日（"YYYY-MM-DD"形式）
        source:
            データソース（省略時はデフォルトソース）
        force_refresh:
            キャッシュを無視して再取得するか
        interval:
            データ間隔（"1d", "1h"等）

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

        # キャッシュキー用のソース名を決定
        source_key = self._resolve_source_key(symbol, effective_source)

        # キャッシュ確認（force_refreshでない場合）
        if not force_refresh:
            cached = self._load_from_cache(symbol, start, end, source_key, interval)
            if cached is not None and not cached.empty:
                return cached

        # データ取得
        df = self._fetch_from_source(symbol, start, end, effective_source, interval)

        # キャッシュに保存
        self._save_to_cache(symbol, df, source_key, interval)

        return df.copy()

    def _resolve_source_key(self, symbol: str, source: DataSource) -> str:
        """キャッシュキー用のソース名を解決する。

        Parameters
        ----------
        symbol:
            銘柄シンボル
        source:
            指定されたソース名

        Returns
        -------
        str
            キャッシュキーに使用するソース名
        """
        if source == "auto":
            # 自動選択ソースの名前を返す
            selected = self._auto_select_source(symbol)
            return selected.name
        return source

    def _fetch_from_source(
        self,
        symbol: str,
        start: str,
        end: str,
        source: DataSource,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """指定ソースからデータを取得する。"""
        if source == "jquants":
            return self._fetch_jquants(symbol, start, end)
        elif source == "yfinance":
            return self._fetch_yfinance(symbol, start, end, interval)
        elif source == "ccxt":
            return self._fetch_ccxt(symbol, start, end, interval)
        else:
            # auto: 銘柄に最適なソースを自動選択
            return self._fetch_auto(symbol, start, end, interval)

    def _fetch_auto(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """銘柄に最適なソースを自動選択してデータを取得する。"""
        # デフォルトソースを遅延登録
        self._get_yfinance()
        try:
            self._get_jquants()
        except Exception:
            pass  # J-Quants設定がない場合はスキップ

        selected = self._auto_select_source(symbol)
        logger.info(f"自動選択ソース: {selected.name} ({symbol})")

        try:
            return selected.fetch_ohlcv(symbol, start, end, interval)
        except Exception as e:
            # フォールバック: yfinanceを試みる
            if selected.name != "yfinance":
                logger.warning(
                    f"{selected.name}取得失敗（{symbol}）: {e}. yfinanceにフォールバック"
                )
                return self._fetch_yfinance(symbol, start, end, interval)
            raise

    def _fetch_yfinance(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """yfinanceからデータを取得する。"""
        client = self._get_yfinance()
        return client.fetch_ohlcv(symbol, start, end, interval)

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

    def _fetch_ccxt(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """CCXTからデータを取得する。"""
        if "ccxt" not in self._sources:
            raise ValueError(
                "CCXTソースが登録されていません。"
                "register_source(CCXTClient(...)) を先に呼び出してください。"
            )
        return self._sources["ccxt"].fetch_ohlcv(symbol, start, end, interval)

    # ------------------------------------------------------------------
    # キャッシュ管理
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # リソース管理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """リソースをクローズする。"""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._jquants_raw:
            self._jquants_raw.close()

    # ------------------------------------------------------------------
    # リアルタイムデータ
    # ------------------------------------------------------------------

    def subscribe_realtime(
        self,
        symbols: list[str],
        callback: "Callable[[TickData], None]",
        exchange_id: str = "binance",
        config: Optional[dict] = None,
    ) -> "WebSocketFeed":
        """リアルタイム価格フィードを開始する。

        銘柄の asset_class に応じてソースを自動判別する:
        - 暗号資産 → ccxt.pro WebSocket
        - その他 → ポーリング

        Parameters
        ----------
        symbols:
            購読する銘柄のリスト
        callback:
            TickData 受信時のコールバック
        exchange_id:
            取引所ID（デフォルト: "binance"）
        config:
            取引所設定

        Returns
        -------
        WebSocketFeed
            作成されたフィードインスタンス（stop()/close() で停止）
        """
        from .ws_feed import WebSocketFeed, TickData  # noqa: F811

        feed = WebSocketFeed(exchange_id=exchange_id, config=config)
        feed.subscribe(symbols)
        feed.add_callback(callback)
        return feed

    def __enter__(self) -> "DataManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
