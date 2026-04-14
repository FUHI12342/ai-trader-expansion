"""WebSocketFeed, TickData, _BarBuilder, OrderBook のテスト。"""
from __future__ import annotations

import dataclasses
import time

import pytest

from src.data.ws_feed import TickData, WebSocketFeed, _BarBuilder
from src.models.orderbook import OrderBookLevel, OrderBookSnapshot


# ---------------------------------------------------------------------------
# TickData テスト
# ---------------------------------------------------------------------------

def test_tick_data_is_frozen():
    """TickDataはfrozen dataclassで変更不可である。"""
    tick = TickData(symbol="BTC/USDT", price=42000.0, volume=10.0, timestamp=time.time())
    with pytest.raises(dataclasses.FrozenInstanceError):
        tick.price = 99999.0  # type: ignore[misc]


def test_tick_data_default_fields():
    """TickDataのデフォルトフィールドが正しい。"""
    tick = TickData(symbol="BTC/USDT", price=42000.0, volume=10.0, timestamp=1234567890.0)
    assert tick.exchange == ""
    assert tick.bid == 0.0
    assert tick.ask == 0.0


def test_tick_data_all_fields():
    """TickDataが全フィールドで正常に生成できる。"""
    tick = TickData(
        symbol="ETH/USDT",
        price=2500.0,
        volume=50.0,
        timestamp=1234567890.0,
        exchange="binance",
        bid=2499.0,
        ask=2501.0,
    )
    assert tick.symbol == "ETH/USDT"
    assert tick.price == 2500.0
    assert tick.exchange == "binance"


# ---------------------------------------------------------------------------
# WebSocketFeed テスト
# ---------------------------------------------------------------------------

def test_ws_feed_init():
    """WebSocketFeedが初期化できる。"""
    feed = WebSocketFeed(exchange_id="binance")
    assert feed is not None
    assert feed._running is False


def test_ws_feed_init_with_buffer_config():
    """WebSocketFeedがバッファ設定で初期化できる。"""
    feed = WebSocketFeed(max_buffer_size=500, max_reconnect_delay=30.0)
    assert feed._max_buffer_size == 500
    assert feed._max_reconnect_delay == 30.0


def test_subscribe_adds_symbols():
    """subscribe()で銘柄が追加される。"""
    feed = WebSocketFeed()
    feed.subscribe(["BTC/USDT", "ETH/USDT"])
    assert "BTC/USDT" in feed._subscriptions
    assert "ETH/USDT" in feed._subscriptions


def test_unsubscribe_removes_symbols():
    """unsubscribe()で銘柄が削除される。"""
    feed = WebSocketFeed()
    feed.subscribe(["BTC/USDT", "ETH/USDT"])
    feed.unsubscribe(["BTC/USDT"])
    assert "BTC/USDT" not in feed._subscriptions
    assert "ETH/USDT" in feed._subscriptions


def test_unsubscribe_cleans_bar_builder():
    """unsubscribe()でバービルダーも削除される。"""
    feed = WebSocketFeed()
    feed.subscribe(["BTC/USDT"])
    feed._bar_builders["BTC/USDT"] = _BarBuilder(symbol="BTC/USDT")
    feed.unsubscribe(["BTC/USDT"])
    assert "BTC/USDT" not in feed._bar_builders


def test_unsubscribe_nonexistent_symbol_no_error():
    """未登録の銘柄のunsubscribeはエラーにならない。"""
    feed = WebSocketFeed()
    feed.unsubscribe(["NONEXISTENT"])  # エラーなし


def test_get_latest_price_returns_none_when_no_data():
    """データがない場合get_latest_priceはNoneを返す。"""
    feed = WebSocketFeed()
    result = feed.get_latest_price("BTC/USDT")
    assert result is None


def test_get_latest_tick_returns_none_when_no_data():
    """データがない場合get_latest_tickはNoneを返す。"""
    feed = WebSocketFeed()
    result = feed.get_latest_tick("BTC/USDT")
    assert result is None


def test_add_callback():
    """コールバックが登録できる。"""
    feed = WebSocketFeed()
    received = []

    def cb(tick: TickData) -> None:
        received.append(tick)

    feed.add_callback(cb)
    assert len(feed._callbacks) == 1


def test_add_bar_callback():
    """バーコールバックが登録できる。"""
    feed = WebSocketFeed()
    bars = []

    def bar_cb(symbol: str, bar: dict) -> None:
        bars.append((symbol, bar))

    feed.add_bar_callback(bar_cb)
    assert len(feed._bar_callbacks) == 1


def test_stop_sets_running_to_false():
    """stop()で_runningがFalseになる。"""
    feed = WebSocketFeed()
    feed._running = True
    feed.stop()
    assert feed._running is False


def test_latest_price_after_manual_insert():
    """_latest_pricesに手動でデータを入れたあとget_latest_priceが値を返す。"""
    feed = WebSocketFeed()
    tick = TickData(symbol="BTC/USDT", price=42000.0, volume=10.0, timestamp=time.time())
    feed._latest_prices["BTC/USDT"] = tick
    assert feed.get_latest_price("BTC/USDT") == 42000.0


def test_process_tick_fires_callback():
    """_process_tick()がコールバックを発火する。"""
    feed = WebSocketFeed()
    received = []
    feed.add_callback(lambda t: received.append(t))

    tick = TickData(symbol="BTC/USDT", price=42000.0, volume=1.0, timestamp=time.time())
    feed._process_tick(tick)

    assert len(received) == 1
    assert received[0].price == 42000.0
    assert feed.get_latest_price("BTC/USDT") == 42000.0


def test_process_tick_backpressure():
    """バッファ上限超過で古いエントリが削除される。"""
    feed = WebSocketFeed(max_buffer_size=3)
    for i in range(5):
        tick = TickData(
            symbol=f"SYM{i}", price=float(i), volume=1.0, timestamp=time.time()
        )
        feed._process_tick(tick)
    # 最大3 + 最新1 = 4個あるが、超過分を削除するので <= max_buffer_size + 1
    assert len(feed._latest_prices) <= 4


def test_reconnect_count_starts_at_zero():
    """reconnect_countの初期値は0。"""
    feed = WebSocketFeed()
    assert feed.reconnect_count == 0


def test_calc_backoff():
    """exponential backoffの計算が正しい。"""
    feed = WebSocketFeed(max_reconnect_delay=60.0)
    feed._reconnect_count = 0
    assert feed._calc_backoff() == 1.0
    feed._reconnect_count = 1
    assert feed._calc_backoff() == 2.0
    feed._reconnect_count = 3
    assert feed._calc_backoff() == 8.0
    feed._reconnect_count = 10
    assert feed._calc_backoff() == 60.0  # max_reconnect_delay で上限


def test_get_completed_bars_empty():
    """バーがない場合空リストを返す。"""
    feed = WebSocketFeed()
    assert feed.get_completed_bars("BTC/USDT") == []


def test_get_bars_as_dataframe_empty():
    """バーがない場合空DataFrameを返す。"""
    feed = WebSocketFeed()
    df = feed.get_bars_as_dataframe("BTC/USDT")
    assert len(df) == 0
    assert "close" in df.columns


# ---------------------------------------------------------------------------
# _BarBuilder テスト
# ---------------------------------------------------------------------------

def test_bar_builder_first_tick_returns_none():
    """最初のティックではバーは確定しない。"""
    builder = _BarBuilder(symbol="BTC/USDT")
    tick = TickData(symbol="BTC/USDT", price=42000.0, volume=1.0, timestamp=1000.0 * 60)
    result = builder.update(tick)
    assert result is None
    assert builder.open == 42000.0


def test_bar_builder_same_minute_accumulates():
    """同一分内のティックは集約される。"""
    builder = _BarBuilder(symbol="BTC/USDT")
    base_ts = 1000 * 60  # 分 1000
    builder.update(TickData(symbol="BTC/USDT", price=100.0, volume=1.0, timestamp=base_ts))
    builder.update(TickData(symbol="BTC/USDT", price=110.0, volume=2.0, timestamp=base_ts + 10))
    builder.update(TickData(symbol="BTC/USDT", price=95.0, volume=3.0, timestamp=base_ts + 20))

    assert builder.open == 100.0
    assert builder.high == 110.0
    assert builder.low == 95.0
    assert builder.close == 95.0
    assert builder.volume == 6.0
    assert builder.tick_count == 3


def test_bar_builder_minute_change_returns_bar():
    """分が変わったら確定バーを返す。"""
    builder = _BarBuilder(symbol="BTC/USDT")
    min_0 = 1000 * 60
    min_1 = 1001 * 60

    builder.update(TickData(symbol="BTC/USDT", price=100.0, volume=1.0, timestamp=min_0))
    builder.update(TickData(symbol="BTC/USDT", price=110.0, volume=2.0, timestamp=min_0 + 30))

    bar = builder.update(TickData(symbol="BTC/USDT", price=105.0, volume=5.0, timestamp=min_1))
    assert bar is not None
    assert bar["open"] == 100.0
    assert bar["high"] == 110.0
    assert bar["low"] == 100.0
    assert bar["close"] == 110.0
    assert bar["volume"] == 3.0
    assert bar["tick_count"] == 2

    # 新しい分のビルダーがリセットされている
    assert builder.open == 105.0
    assert builder.current_minute == 1001


def test_process_tick_bar_aggregation():
    """_process_tick経由でバー集約が動作する。"""
    feed = WebSocketFeed()
    bar_results = []
    feed.add_bar_callback(lambda sym, bar: bar_results.append((sym, bar)))

    min_0 = 2000 * 60
    min_1 = 2001 * 60

    feed._process_tick(TickData(symbol="X", price=10.0, volume=1.0, timestamp=min_0))
    feed._process_tick(TickData(symbol="X", price=12.0, volume=1.0, timestamp=min_0 + 30))
    assert len(bar_results) == 0  # まだ確定していない

    feed._process_tick(TickData(symbol="X", price=15.0, volume=1.0, timestamp=min_1))
    assert len(bar_results) == 1
    assert bar_results[0][0] == "X"
    assert bar_results[0][1]["open"] == 10.0
    assert bar_results[0][1]["high"] == 12.0

    # completed_bars にも記録される
    assert len(feed.get_completed_bars("X")) == 1


# ---------------------------------------------------------------------------
# OrderBookSnapshot テスト
# ---------------------------------------------------------------------------

def test_orderbook_level_is_frozen():
    """OrderBookLevelはfrozenで変更不可。"""
    level = OrderBookLevel(price=100.0, quantity=5.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        level.price = 200.0  # type: ignore[misc]


def test_orderbook_snapshot_is_frozen():
    """OrderBookSnapshotはfrozenで変更不可。"""
    ob = OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=time.time(),
        bids=(OrderBookLevel(100.0, 1.0),),
        asks=(OrderBookLevel(101.0, 2.0),),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ob.symbol = "ETH"  # type: ignore[misc]


def test_orderbook_best_bid_ask():
    """best_bid / best_ask が正しい。"""
    ob = OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=1.0,
        bids=(OrderBookLevel(100.0, 5.0), OrderBookLevel(99.0, 10.0)),
        asks=(OrderBookLevel(101.0, 3.0), OrderBookLevel(102.0, 7.0)),
    )
    assert ob.best_bid == 100.0
    assert ob.best_ask == 101.0


def test_orderbook_spread():
    """スプレッド計算が正しい。"""
    ob = OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=1.0,
        bids=(OrderBookLevel(100.0, 5.0),),
        asks=(OrderBookLevel(101.0, 3.0),),
    )
    assert ob.spread == 1.0
    assert abs(ob.spread_bps - 99.50) < 1.0  # 約99.5 bps


def test_orderbook_mid_price():
    """仲値計算が正しい。"""
    ob = OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=1.0,
        bids=(OrderBookLevel(100.0, 5.0),),
        asks=(OrderBookLevel(102.0, 3.0),),
    )
    assert ob.mid_price == 101.0


def test_orderbook_total_volumes():
    """合計数量が正しい。"""
    ob = OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=1.0,
        bids=(OrderBookLevel(100.0, 5.0), OrderBookLevel(99.0, 10.0)),
        asks=(OrderBookLevel(101.0, 3.0), OrderBookLevel(102.0, 7.0)),
    )
    assert ob.total_bid_volume == 15.0
    assert ob.total_ask_volume == 10.0


def test_orderbook_empty_bids_asks():
    """bids/asks が空の場合のプロパティ。"""
    ob = OrderBookSnapshot(
        symbol="BTC/USDT", timestamp=1.0,
        bids=(), asks=(),
    )
    assert ob.best_bid == 0.0
    assert ob.best_ask == 0.0
    assert ob.spread == 0.0
    assert ob.spread_bps == 0.0
    assert ob.mid_price == 0.0
