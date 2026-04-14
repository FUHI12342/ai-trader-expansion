"""板情報（Order Book）データモデル。

リアルタイム板情報のスナップショットを immutable に保持する。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderBookLevel:
    """板情報の1価格レベル（immutable）。

    Parameters
    ----------
    price:
        価格
    quantity:
        数量
    """

    price: float
    quantity: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    """板情報のスナップショット（immutable）。

    Parameters
    ----------
    symbol:
        銘柄シンボル
    timestamp:
        Unix タイムスタンプ
    bids:
        買い注文（価格降順）
    asks:
        売り注文（価格昇順）
    exchange:
        取引所名
    """

    symbol: str
    timestamp: float
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]
    exchange: str = ""

    @property
    def best_bid(self) -> float:
        """最良買い気配値。"""
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """最良売り気配値。"""
        return self.asks[0].price if self.asks else 0.0

    @property
    def spread(self) -> float:
        """スプレッド（best_ask - best_bid）。"""
        if not self.bids or not self.asks:
            return 0.0
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        """スプレッド（ベーシスポイント）。"""
        mid = self.mid_price
        if mid == 0.0:
            return 0.0
        return (self.spread / mid) * 10_000

    @property
    def mid_price(self) -> float:
        """仲値。"""
        if not self.bids or not self.asks:
            return 0.0
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def total_bid_volume(self) -> float:
        """買い注文の合計数量。"""
        return sum(level.quantity for level in self.bids)

    @property
    def total_ask_volume(self) -> float:
        """売り注文の合計数量。"""
        return sum(level.quantity for level in self.asks)
