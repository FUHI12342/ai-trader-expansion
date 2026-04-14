"""WebSocket リアルタイム価格フィードマネージャー。

再接続ロジック（exponential backoff）、バックプレッシャー制御、
TickData → 1分足 OHLCV 集約機能を備える。
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TickData:
    """ティックデータ（immutable）。"""

    symbol: str
    price: float
    volume: float
    timestamp: float
    exchange: str = ""
    bid: float = 0.0
    ask: float = 0.0


Callback = Callable[[TickData], None]


@dataclass
class _BarBuilder:
    """TickData を 1分足 OHLCV に集約するビルダー。

    各銘柄ごとに1つ保持し、分が変わったら確定バーを返す。
    """

    symbol: str
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    tick_count: int = 0
    current_minute: int = -1

    def update(self, tick: TickData) -> Optional[dict]:
        """TickData を取り込み、分が変わったら確定バーを返す。

        Returns
        -------
        Optional[dict]
            確定した 1分足バー。分が変わっていない場合は None
        """
        minute = int(tick.timestamp // 60)

        if self.current_minute == -1:
            self._reset(tick, minute)
            return None

        if minute != self.current_minute:
            completed = {
                "timestamp": datetime.fromtimestamp(
                    self.current_minute * 60, tz=timezone.utc
                ),
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "tick_count": self.tick_count,
            }
            self._reset(tick, minute)
            return completed

        self.high = max(self.high, tick.price)
        self.low = min(self.low, tick.price)
        self.close = tick.price
        self.volume += tick.volume
        self.tick_count += 1
        return None

    def _reset(self, tick: TickData, minute: int) -> None:
        self.current_minute = minute
        self.open = tick.price
        self.high = tick.price
        self.low = tick.price
        self.close = tick.price
        self.volume = tick.volume
        self.tick_count = 1


class WebSocketFeed:
    """WebSocket リアルタイム価格フィード。

    ccxt.proが利用可能な場合はWebSocket、
    そうでない場合はポーリングにフォールバック。

    機能:
    - 自動再接続（exponential backoff: 1s → 2s → 4s ... 最大60s）
    - バックプレッシャー制御（バッファ上限超過で古いティックを破棄）
    - TickData → 1分足 OHLCV バー集約

    Parameters
    ----------
    exchange_id:
        取引所ID（例: "binance"）
    config:
        取引所設定（APIキー等）
    max_buffer_size:
        コールバック処理待ちバッファの上限（デフォルト: 1000）
    max_reconnect_delay:
        再接続の最大待機秒数（デフォルト: 60.0）
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        config: Optional[dict] = None,
        max_buffer_size: int = 1000,
        max_reconnect_delay: float = 60.0,
    ) -> None:
        self._exchange_id = exchange_id
        self._config = config or {}
        self._subscriptions: Set[str] = set()
        self._callbacks: list[Callback] = []
        self._bar_callbacks: list[Callable[[str, dict], None]] = []
        self._latest_prices: Dict[str, TickData] = {}
        self._running = False
        self._poll_interval: float = 5.0
        self._max_buffer_size = max_buffer_size
        self._max_reconnect_delay = max_reconnect_delay
        self._reconnect_count: int = 0
        self._bar_builders: Dict[str, _BarBuilder] = {}
        self._completed_bars: Dict[str, List[dict]] = defaultdict(list)

        # ccxt.proの利用可能性を確認
        try:
            import ccxt.pro as ccxtpro  # type: ignore[import-not-found]
            exchange_class = getattr(ccxtpro, exchange_id, None)
            if exchange_class:
                self._exchange = exchange_class(self._config)
                self._use_ws = True
            else:
                self._use_ws = False
                self._exchange = None
        except ImportError:
            self._use_ws = False
            self._exchange = None
            logger.info("ccxt.pro未インストール: ポーリングモードで動作")

    @property
    def reconnect_count(self) -> int:
        """再接続回数を返す。"""
        return self._reconnect_count

    def subscribe(self, symbols: list[str]) -> None:
        """銘柄をサブスクリプションに追加する。"""
        for s in symbols:
            self._subscriptions.add(s)

    def unsubscribe(self, symbols: list[str]) -> None:
        """銘柄をサブスクリプションから削除する。"""
        for s in symbols:
            self._subscriptions.discard(s)
            self._bar_builders.pop(s, None)

    def add_callback(self, callback: Callback) -> None:
        """ティックデータ受信時のコールバックを登録する。"""
        self._callbacks.append(callback)

    def add_bar_callback(self, callback: Callable[[str, dict], None]) -> None:
        """1分足バー確定時のコールバックを登録する。

        Parameters
        ----------
        callback:
            (symbol, bar_dict) を受け取るコールバック関数
        """
        self._bar_callbacks.append(callback)

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """最新価格を取得する。"""
        tick = self._latest_prices.get(symbol)
        return tick.price if tick else None

    def get_latest_tick(self, symbol: str) -> Optional[TickData]:
        """最新ティックデータを取得する。"""
        return self._latest_prices.get(symbol)

    def get_completed_bars(self, symbol: str) -> List[dict]:
        """確定済み1分足バーのリストを返す（コピー）。"""
        return list(self._completed_bars.get(symbol, []))

    def get_bars_as_dataframe(self, symbol: str) -> pd.DataFrame:
        """確定済み1分足バーをDataFrameとして返す。

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame。バーがない場合は空DataFrame
        """
        bars = self._completed_bars.get(symbol, [])
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(bars)
        df = df.set_index("timestamp")
        return df[["open", "high", "low", "close", "volume"]].copy()

    async def start(self) -> None:
        """フィードを開始する。"""
        self._running = True
        self._reconnect_count = 0
        if self._use_ws and self._exchange:
            await self._ws_loop_with_reconnect()
        else:
            await self._poll_loop()

    def stop(self) -> None:
        """フィードを停止する。"""
        self._running = False

    def _process_tick(self, tick: TickData) -> None:
        """ティックデータを処理する（コールバック発火 + バー集約）。

        バックプレッシャー: _latest_prices のサイズが上限を超えたら
        最も古いエントリを削除する。
        """
        # バックプレッシャー制御
        if len(self._latest_prices) > self._max_buffer_size:
            oldest_key = next(iter(self._latest_prices))
            del self._latest_prices[oldest_key]
            logger.warning(
                "バッファ上限超過: 最も古いエントリ %s を削除", oldest_key
            )

        self._latest_prices[tick.symbol] = tick

        for cb in self._callbacks:
            try:
                cb(tick)
            except Exception as e:
                logger.error("コールバックエラー: %s", e)

        # バー集約
        if tick.symbol not in self._bar_builders:
            self._bar_builders[tick.symbol] = _BarBuilder(symbol=tick.symbol)
        builder = self._bar_builders[tick.symbol]
        completed_bar = builder.update(tick)
        if completed_bar is not None:
            self._completed_bars[tick.symbol].append(completed_bar)
            for bar_cb in self._bar_callbacks:
                try:
                    bar_cb(tick.symbol, completed_bar)
                except Exception as e:
                    logger.error("バーコールバックエラー: %s", e)

    def _calc_backoff(self) -> float:
        """Exponential backoff の待機秒数を計算する。"""
        delay = min(2 ** self._reconnect_count, self._max_reconnect_delay)
        return delay

    async def _ws_loop_with_reconnect(self) -> None:
        """再接続付きWebSocketループ。"""
        while self._running and self._subscriptions:
            try:
                await self._ws_loop()
            except Exception as e:
                if not self._running:
                    break
                self._reconnect_count += 1
                delay = self._calc_backoff()
                logger.warning(
                    "WebSocket切断 (retry=%d, delay=%.1fs): %s",
                    self._reconnect_count, delay, e,
                )
                await asyncio.sleep(delay)

                # exchange を再生成
                try:
                    if self._exchange and hasattr(self._exchange, "close"):
                        await self._exchange.close()
                    import ccxt.pro as ccxtpro  # type: ignore[import-not-found]
                    exchange_class = getattr(ccxtpro, self._exchange_id)
                    self._exchange = exchange_class(self._config)
                except Exception as re_err:
                    logger.error("exchange再生成失敗: %s", re_err)

    async def _ws_loop(self) -> None:
        """WebSocketループ。"""
        while self._running and self._subscriptions:
            for symbol in list(self._subscriptions):
                ticker = await self._exchange.watch_ticker(symbol)
                tick = TickData(
                    symbol=symbol,
                    price=float(ticker.get("last", 0)),
                    volume=float(ticker.get("baseVolume", 0)),
                    timestamp=time.time(),
                    exchange=self._exchange_id,
                    bid=float(ticker.get("bid", 0)),
                    ask=float(ticker.get("ask", 0)),
                )
                self._process_tick(tick)

    async def _poll_loop(self) -> None:
        """ポーリングフォールバックループ。"""
        try:
            import ccxt as _ccxt  # type: ignore[import-not-found]
            exchange_class = getattr(_ccxt, self._exchange_id, None)
            if not exchange_class:
                logger.error("ポーリング: 取引所 %s が見つかりません", self._exchange_id)
                return
            exchange = exchange_class(self._config)
        except ImportError:
            logger.error("ccxt未インストール: ポーリング不可")
            return

        while self._running and self._subscriptions:
            for symbol in list(self._subscriptions):
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    tick = TickData(
                        symbol=symbol,
                        price=float(ticker.get("last", 0)),
                        volume=float(ticker.get("baseVolume", 0)),
                        timestamp=time.time(),
                        exchange=self._exchange_id,
                        bid=float(ticker.get("bid", 0)),
                        ask=float(ticker.get("ask", 0)),
                    )
                    self._process_tick(tick)
                except Exception as e:
                    logger.warning("ポーリングエラー (%s): %s", symbol, e)
            await asyncio.sleep(self._poll_interval)

    async def close(self) -> None:
        """リソースをクリーンアップする。"""
        self.stop()
        if self._exchange and hasattr(self._exchange, "close"):
            await self._exchange.close()
