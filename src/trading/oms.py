"""Order Management System (OMS)。

注文のライフサイクル管理: 発注 → 監視 → 約定確認 → リトライ/キャンセル。
部分約定、ステイル注文の自動キャンセル、Reconciliation を統合する。
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, List, Optional

from src.brokers.base import BrokerBase, Order, OrderSide, OrderType

logger = logging.getLogger(__name__)


class OMSStatus(str, Enum):
    """OMS管理注文のステータス。"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    ERROR = "error"
    STALE = "stale"


@dataclass(frozen=True)
class ManagedOrder:
    """OMS管理注文（immutable）。

    Parameters
    ----------
    order_id:
        ブローカーから付与された注文ID（PENDING時は空）
    symbol:
        銘柄シンボル
    side:
        注文方向
    order_type:
        注文種別
    strategy_name:
        発注元の戦略名
    target_quantity:
        目標数量
    filled_quantity:
        約定済み数量
    avg_fill_price:
        平均約定価格
    status:
        OMS管理ステータス
    retries:
        リトライ回数
    max_retries:
        最大リトライ回数
    price:
        指値価格（成行の場合はNone）
    created_at:
        作成時刻（Unix timestamp）
    last_updated:
        最終更新時刻（Unix timestamp）
    error_message:
        エラーメッセージ（正常時は空）
    """

    order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    strategy_name: str = ""
    target_quantity: float = 0.0
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: OMSStatus = OMSStatus.PENDING
    retries: int = 0
    max_retries: int = 3
    price: Optional[float] = None
    created_at: float = 0.0
    last_updated: float = 0.0
    error_message: str = ""

    @property
    def remaining_quantity(self) -> float:
        """未約定数量。"""
        return max(0.0, self.target_quantity - self.filled_quantity)

    @property
    def is_terminal(self) -> bool:
        """最終状態（FILLED/CANCELLED/ERROR）か。"""
        return self.status in (OMSStatus.FILLED, OMSStatus.CANCELLED, OMSStatus.ERROR)

    @property
    def fill_ratio(self) -> float:
        """約定率（0.0〜1.0）。"""
        if self.target_quantity == 0:
            return 0.0
        return self.filled_quantity / self.target_quantity


class OrderManagementSystem:
    """注文管理システム。

    Parameters
    ----------
    max_retries:
        発注失敗時の最大リトライ回数（デフォルト: 3）
    stale_seconds:
        注文がstaleと判定されるまでの秒数（デフォルト: 300）
    """

    def __init__(
        self,
        max_retries: int = 3,
        stale_seconds: float = 300.0,
    ) -> None:
        self._max_retries = max_retries
        self._stale_seconds = stale_seconds
        self._orders: Dict[str, ManagedOrder] = {}
        self._order_counter: int = 0

    @property
    def order_count(self) -> int:
        """管理中の注文数。"""
        return len(self._orders)

    def submit(
        self,
        broker: BrokerBase,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        strategy_name: str = "",
        price: Optional[float] = None,
    ) -> ManagedOrder:
        """注文を発注する。

        Parameters
        ----------
        broker:
            発注先ブローカー
        symbol:
            銘柄シンボル
        side:
            注文方向
        order_type:
            注文種別
        quantity:
            注文数量
        strategy_name:
            発注元の戦略名
        price:
            指値価格

        Returns
        -------
        ManagedOrder
            管理注文（発注成功時は SUBMITTED、失敗時は ERROR）
        """
        now = time.time()
        self._order_counter += 1
        internal_id = f"oms_{self._order_counter}"

        managed = ManagedOrder(
            symbol=symbol,
            side=side,
            order_type=order_type,
            strategy_name=strategy_name,
            target_quantity=quantity,
            price=price,
            max_retries=self._max_retries,
            created_at=now,
            last_updated=now,
            status=OMSStatus.PENDING,
        )

        try:
            broker_order = broker.place_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
            )
            managed = replace(
                managed,
                order_id=broker_order.order_id,
                status=OMSStatus.SUBMITTED,
                filled_quantity=broker_order.filled_quantity,
                avg_fill_price=broker_order.avg_fill_price,
                last_updated=time.time(),
            )

            if broker_order.status == "filled":
                managed = replace(
                    managed,
                    status=OMSStatus.FILLED,
                    filled_quantity=broker_order.filled_quantity,
                    avg_fill_price=broker_order.avg_fill_price,
                )
            elif broker_order.status == "partial":
                managed = replace(managed, status=OMSStatus.PARTIAL)

            logger.info(
                "注文発注成功: %s %s %s %.2f @ %s (id=%s)",
                symbol, side.value, order_type.value,
                quantity, price, broker_order.order_id,
            )

        except Exception as e:
            managed = replace(
                managed,
                status=OMSStatus.ERROR,
                error_message=str(e)[:200],
                last_updated=time.time(),
            )
            logger.error("注文発注失敗: %s %s: %s", symbol, side.value, e)

        self._orders[internal_id] = managed
        return managed

    def monitor(self, broker: BrokerBase) -> List[ManagedOrder]:
        """未約定注文の状態を更新する。

        Parameters
        ----------
        broker:
            ブローカーインスタンス

        Returns
        -------
        List[ManagedOrder]
            更新された注文のリスト
        """
        updated: List[ManagedOrder] = []

        for key, managed in list(self._orders.items()):
            if managed.is_terminal or not managed.order_id:
                continue

            try:
                broker_order = broker.get_order(managed.order_id)
                if broker_order is None:
                    continue

                new_status = managed.status
                if broker_order.status == "filled":
                    new_status = OMSStatus.FILLED
                elif broker_order.status == "partial":
                    new_status = OMSStatus.PARTIAL
                elif broker_order.status == "cancelled":
                    new_status = OMSStatus.CANCELLED

                if (new_status != managed.status
                        or broker_order.filled_quantity != managed.filled_quantity):
                    managed = replace(
                        managed,
                        status=new_status,
                        filled_quantity=broker_order.filled_quantity,
                        avg_fill_price=broker_order.avg_fill_price,
                        last_updated=time.time(),
                    )
                    self._orders[key] = managed
                    updated.append(managed)

            except Exception as e:
                logger.warning("注文監視エラー (%s): %s", managed.order_id, e)

        return updated

    def cancel_stale(self, broker: BrokerBase) -> List[ManagedOrder]:
        """ステイル注文をキャンセルする。

        stale_seconds を超えて未約定の注文を自動キャンセルする。

        Returns
        -------
        List[ManagedOrder]
            キャンセルされた注文のリスト
        """
        now = time.time()
        cancelled: List[ManagedOrder] = []

        for key, managed in list(self._orders.items()):
            if managed.is_terminal or not managed.order_id:
                continue

            age = now - managed.created_at
            if age < self._stale_seconds:
                continue

            try:
                success = broker.cancel_order(managed.order_id)
                if success:
                    managed = replace(
                        managed,
                        status=OMSStatus.STALE,
                        last_updated=time.time(),
                    )
                    self._orders[key] = managed
                    cancelled.append(managed)
                    logger.info(
                        "ステイル注文キャンセル: %s (age=%.0fs)",
                        managed.order_id, age,
                    )
            except Exception as e:
                logger.warning("キャンセル失敗 (%s): %s", managed.order_id, e)

        return cancelled

    def retry_failed(self, broker: BrokerBase) -> List[ManagedOrder]:
        """失敗した注文をリトライする。

        Returns
        -------
        List[ManagedOrder]
            リトライされた注文のリスト
        """
        retried: List[ManagedOrder] = []

        for key, managed in list(self._orders.items()):
            if managed.status != OMSStatus.ERROR:
                continue
            if managed.retries >= managed.max_retries:
                continue

            try:
                broker_order = broker.place_order(
                    symbol=managed.symbol,
                    side=managed.side,
                    order_type=managed.order_type,
                    quantity=managed.remaining_quantity,
                    price=managed.price,
                )

                new_status = OMSStatus.SUBMITTED
                if broker_order.status == "filled":
                    new_status = OMSStatus.FILLED

                managed = replace(
                    managed,
                    order_id=broker_order.order_id,
                    status=new_status,
                    retries=managed.retries + 1,
                    filled_quantity=managed.filled_quantity + broker_order.filled_quantity,
                    avg_fill_price=broker_order.avg_fill_price,
                    error_message="",
                    last_updated=time.time(),
                )
                self._orders[key] = managed
                retried.append(managed)
                logger.info("リトライ成功: %s (retry=%d)", managed.symbol, managed.retries)

            except Exception as e:
                managed = replace(
                    managed,
                    retries=managed.retries + 1,
                    error_message=str(e)[:200],
                    last_updated=time.time(),
                )
                self._orders[key] = managed
                logger.warning("リトライ失敗: %s (retry=%d): %s", managed.symbol, managed.retries, e)

        return retried

    def get_active_orders(self) -> List[ManagedOrder]:
        """未完了（非terminal）の注文一覧を返す。"""
        return [m for m in self._orders.values() if not m.is_terminal]

    def get_all_orders(self) -> List[ManagedOrder]:
        """全管理注文を返す。"""
        return list(self._orders.values())

    def get_filled_orders(self) -> List[ManagedOrder]:
        """約定済み注文のみ返す。"""
        return [m for m in self._orders.values() if m.status == OMSStatus.FILLED]

    def summary(self) -> Dict[str, int]:
        """ステータス別の注文数サマリーを返す。"""
        counts: Dict[str, int] = {}
        for m in self._orders.values():
            counts[m.status.value] = counts.get(m.status.value, 0) + 1
        return counts
