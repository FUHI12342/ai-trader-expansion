"""ブローカー基底クラス（ABC）。

全ブローカーはこの抽象クラスを継承してimplementする。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OrderSide(str, Enum):
    """注文方向。"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """注文種別。"""
    MARKET = "market"          # 成行
    LIMIT = "limit"            # 指値
    STOP = "stop"              # 逆指値
    STOP_LIMIT = "stop_limit"  # 逆指値＋指値（新規）


@dataclass(frozen=True)
class Order:
    """注文データ（immutable）。"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]           # 指値の場合
    status: str                      # "open", "filled", "cancelled", "partial"
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fee: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    asset_class: str = ""        # "STOCK", "CRYPTO", "FUTURES", "BOND_ETF"
    exchange: str = ""           # "TSE", "GMO", "bitFlyer", "OSE"
    leverage: float = 1.0        # 1.0 はスポット、>1 はマージン/先物

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Position:
    """ポジション情報（immutable）。"""
    symbol: str
    quantity: float                  # 保有株数（マイナス=ショート）
    avg_entry_price: float           # 平均取得価格
    current_price: float             # 現在価格
    unrealized_pnl: float            # 含み損益
    unrealized_pnl_pct: float        # 含み損益率（%）
    asset_class: str = ""
    exchange: str = ""
    leverage: float = 1.0
    liquidation_price: float = 0.0    # 0 = 非適用（スポット）
    margin_requirement: float = 0.0   # 必要証拠金（通貨単位）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BrokerBase(ABC):
    """ブローカー基底クラス（ABC）。

    全ブローカー実装はこのクラスを継承すること。
    """

    @abstractmethod
    def get_balance(self) -> float:
        """利用可能な現金残高を返す。"""
        ...

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """全ポジションを{symbol: Position}形式で返す。"""
        ...

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        """注文を発注する。

        Parameters
        ----------
        symbol:
            銘柄コード
        side:
            注文方向（buy/sell）
        order_type:
            注文種別（market/limit）
        quantity:
            株数
        price:
            指値価格（指値注文の場合）

        Returns
        -------
        Order
            発注された注文情報
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """注文をキャンセルする。

        Returns
        -------
        bool
            キャンセル成功の場合True
        """
        ...

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """注文状況を取得する。"""
        ...

    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """オープン注文一覧を取得する。"""
        ...

    def get_position(self, symbol: str) -> Optional[Position]:
        """特定銘柄のポジションを取得する。"""
        return self.get_positions().get(symbol)

    def is_connected(self) -> bool:
        """接続状況を確認する（デフォルト: True）。"""
        return True
