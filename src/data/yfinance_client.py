"""yfinance データ取得クライアント。

yfinanceラッパー（ローカルキャッシュ付き）。
DataFrameはimmutableパターンで返す。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .source_base import DataSourceBase

logger = logging.getLogger(__name__)

# yfinanceはオプション依存
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logger.warning("yfinanceが未インストールです: pip install yfinance")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """カラム名を小文字に統一してOHLCV形式に揃える。"""
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    # yfinanceのカラム名マッピング
    rename_map = {
        "adj close": "adj_close",
        "stock splits": "stock_splits",
    }
    out = out.rename(columns=rename_map)

    # 必須カラムが揃っているか確認
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(out.columns)):
        missing = required - set(out.columns)
        raise ValueError(f"yfinanceデータに必須カラムがありません: {missing}")

    return out


class YFinanceClient(DataSourceBase):
    """yfinanceラッパークライアント。

    Parameters
    ----------
    cache_enabled:
        キャッシュを有効にするか（DataManagerのSQLiteキャッシュを使う場合はFalse）
    """

    name = "yfinance"

    def __init__(self, cache_enabled: bool = False) -> None:
        if not HAS_YFINANCE:
            raise ImportError("yfinanceが未インストールです: pip install yfinance")
        self._cache_enabled = cache_enabled
        self._memory_cache: dict[str, pd.DataFrame] = {}

    def fetch_ohlcv(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """OHLCV データを取得する。

        Parameters
        ----------
        symbol:
            ティッカーシンボル（例: "7203.T", "AAPL"）
        start:
            開始日（"YYYY-MM-DD"形式、省略時は1年前）
        end:
            終了日（"YYYY-MM-DD"形式、省略時は今日）
        interval:
            データ間隔（"1d", "1wk", "1mo"等）
        auto_adjust:
            配当・分割調整を行うか

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame（DatetimeIndex, 小文字カラム名）

        Raises
        ------
        ValueError
            データが取得できない場合
        """
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # メモリキャッシュ確認
        cache_key = f"{symbol}_{start}_{end}_{interval}"
        if self._cache_enabled and cache_key in self._memory_cache:
            logger.debug(f"メモリキャッシュから取得: {symbol}")
            return self._memory_cache[cache_key].copy()

        logger.info(f"yfinance: {symbol} ({start} 〜 {end}, {interval}) を取得")
        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
        )

        if raw.empty:
            raise ValueError(f"データが取得できませんでした: {symbol} ({start}〜{end})")

        df = _normalize_columns(raw)

        # 日次の場合はタイムゾーン除去してdate形式に統一
        if interval == "1d" and hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        if self._cache_enabled:
            self._memory_cache[cache_key] = df.copy()

        return df.copy()

    def fetch_multiple(
        self,
        symbols: list[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """複数銘柄のOHLCVを取得する。

        Parameters
        ----------
        symbols:
            ティッカーシンボルのリスト
        start:
            開始日
        end:
            終了日
        interval:
            データ間隔

        Returns
        -------
        dict[str, pd.DataFrame]
            {symbol: DataFrame} の辞書
        """
        results: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                results[sym] = self.fetch_ohlcv(sym, start, end, interval)
            except Exception as e:
                logger.warning(f"{sym} の取得に失敗しました: {e}")
        return results

    def supports_symbol(self, symbol: str) -> bool:
        """銘柄をサポートするか判定する。

        暗号資産（"/" を含む）以外の銘柄をサポートする。
        日本株（.T）、米国株、ETF等を対象とする。

        Parameters
        ----------
        symbol:
            銘柄シンボル

        Returns
        -------
        bool
            サポートする場合は True
        """
        # 暗号資産の "/" 形式はCCXT担当
        return "/" not in symbol

    def supported_intervals(self) -> list[str]:
        """サポートする時間足一覧を返す。

        Returns
        -------
        list[str]
            サポートする時間足のリスト
        """
        return ["1d", "1wk", "1mo"]

    def clear_cache(self) -> None:
        """メモリキャッシュをクリアする。"""
        self._memory_cache.clear()
