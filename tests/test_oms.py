"""OMS (Order Management System) のテスト。"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.brokers.base import BrokerBase, Order, OrderSide, OrderType, Position
from src.trading.oms import ManagedOrder, OMSStatus, OrderManagementSystem


# ---------------------------------------------------------------------------
# テスト用モックブローカー
# ---------------------------------------------------------------------------

class MockBroker(BrokerBase):
    """テスト用のモックブローカー。"""

    def __init__(self) -> None:
        self._balance = 1_000_000.0
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self.place_order_raise: Optional[Exception] = None

    def get_balance(self) -> float:
        return self._balance

    def get_positions(self) -> Dict[str, Position]:
        return {}

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        if self.place_order_raise:
            raise self.place_order_raise

        self._order_counter += 1
        oid = f"mock_{self._order_counter}"
        # 成行は即約定
        if order_type == OrderType.MARKET:
            fill_price = price or 1000.0
            order = Order(
                order_id=oid, symbol=symbol, side=side,
                order_type=order_type, quantity=quantity,
                price=price, status="filled",
                filled_quantity=quantity,
                avg_fill_price=fill_price,
            )
        else:
            order = Order(
                order_id=oid, symbol=symbol, side=side,
                order_type=order_type, quantity=quantity,
                price=price, status="open",
            )
        self._orders[oid] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            old = self._orders[order_id]
            self._orders[order_id] = replace(old, status="cancelled")
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        return [o for o in self._orders.values() if o.status == "open"]

    def fill_order(self, order_id: str, qty: Optional[float] = None) -> None:
        """テスト用: 注文を約定させる。"""
        old = self._orders[order_id]
        fill_qty = qty or old.quantity
        status = "filled" if fill_qty >= old.quantity else "partial"
        self._orders[order_id] = replace(
            old, status=status, filled_quantity=fill_qty,
            avg_fill_price=old.price or 1000.0,
        )


# ---------------------------------------------------------------------------
# ManagedOrder テスト
# ---------------------------------------------------------------------------

class TestManagedOrder:

    def test_is_frozen(self) -> None:
        mo = ManagedOrder()
        with pytest.raises(AttributeError):
            mo.status = OMSStatus.FILLED  # type: ignore[misc]

    def test_remaining_quantity(self) -> None:
        mo = ManagedOrder(target_quantity=100.0, filled_quantity=30.0)
        assert mo.remaining_quantity == 70.0

    def test_remaining_quantity_zero(self) -> None:
        mo = ManagedOrder(target_quantity=100.0, filled_quantity=100.0)
        assert mo.remaining_quantity == 0.0

    def test_is_terminal_filled(self) -> None:
        assert ManagedOrder(status=OMSStatus.FILLED).is_terminal

    def test_is_terminal_cancelled(self) -> None:
        assert ManagedOrder(status=OMSStatus.CANCELLED).is_terminal

    def test_is_terminal_error(self) -> None:
        assert ManagedOrder(status=OMSStatus.ERROR).is_terminal

    def test_is_not_terminal_submitted(self) -> None:
        assert not ManagedOrder(status=OMSStatus.SUBMITTED).is_terminal

    def test_is_not_terminal_pending(self) -> None:
        assert not ManagedOrder(status=OMSStatus.PENDING).is_terminal

    def test_fill_ratio(self) -> None:
        mo = ManagedOrder(target_quantity=100.0, filled_quantity=50.0)
        assert mo.fill_ratio == 0.5

    def test_fill_ratio_zero_target(self) -> None:
        mo = ManagedOrder(target_quantity=0.0, filled_quantity=0.0)
        assert mo.fill_ratio == 0.0


# ---------------------------------------------------------------------------
# OMS テスト
# ---------------------------------------------------------------------------

class TestOrderManagementSystem:

    def test_submit_market_order_filled(self) -> None:
        """成行注文は即約定してFILLEDになる。"""
        oms = OrderManagementSystem()
        broker = MockBroker()

        mo = oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
            strategy_name="test_strategy",
        )

        assert mo.status == OMSStatus.FILLED
        assert mo.filled_quantity == 100
        assert mo.order_id.startswith("mock_")
        assert mo.strategy_name == "test_strategy"
        assert oms.order_count == 1

    def test_submit_limit_order_submitted(self) -> None:
        """指値注文はSUBMITTEDになる。"""
        oms = OrderManagementSystem()
        broker = MockBroker()

        mo = oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=2500.0,
        )

        assert mo.status == OMSStatus.SUBMITTED
        assert mo.filled_quantity == 0

    def test_submit_failure_returns_error(self) -> None:
        """発注失敗時はERRORステータス。"""
        oms = OrderManagementSystem()
        broker = MockBroker()
        broker.place_order_raise = RuntimeError("API timeout")

        mo = oms.submit(
            broker=broker, symbol="BTC/USDT", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=1.0,
        )

        assert mo.status == OMSStatus.ERROR
        assert "API timeout" in mo.error_message

    def test_monitor_detects_fill(self) -> None:
        """monitor()が約定を検出する。"""
        oms = OrderManagementSystem()
        broker = MockBroker()

        mo = oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=2500.0,
        )
        assert mo.status == OMSStatus.SUBMITTED

        # ブローカー側で約定させる
        broker.fill_order(mo.order_id)

        updated = oms.monitor(broker)
        assert len(updated) == 1
        assert updated[0].status == OMSStatus.FILLED

    def test_monitor_detects_partial_fill(self) -> None:
        """monitor()が部分約定を検出する。"""
        oms = OrderManagementSystem()
        broker = MockBroker()

        mo = oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=2500.0,
        )

        broker.fill_order(mo.order_id, qty=30)

        updated = oms.monitor(broker)
        assert len(updated) == 1
        assert updated[0].status == OMSStatus.PARTIAL
        assert updated[0].filled_quantity == 30

    def test_monitor_skips_terminal(self) -> None:
        """terminal状態の注文はmonitor()でスキップする。"""
        oms = OrderManagementSystem()
        broker = MockBroker()

        oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
        )

        updated = oms.monitor(broker)
        assert len(updated) == 0  # 既にFILLED

    def test_cancel_stale(self) -> None:
        """ステイル注文がキャンセルされる。"""
        oms = OrderManagementSystem(stale_seconds=0)  # 即stale
        broker = MockBroker()

        oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=2500.0,
        )

        cancelled = oms.cancel_stale(broker)
        assert len(cancelled) == 1
        assert cancelled[0].status == OMSStatus.STALE

    def test_cancel_stale_ignores_fresh(self) -> None:
        """新しい注文はキャンセルされない。"""
        oms = OrderManagementSystem(stale_seconds=9999)
        broker = MockBroker()

        oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=100, price=2500.0,
        )

        cancelled = oms.cancel_stale(broker)
        assert len(cancelled) == 0

    def test_retry_failed(self) -> None:
        """失敗した注文がリトライされる。"""
        oms = OrderManagementSystem(max_retries=3)
        broker = MockBroker()

        # 最初は失敗
        broker.place_order_raise = RuntimeError("timeout")
        mo = oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
        )
        assert mo.status == OMSStatus.ERROR

        # リトライ時は成功
        broker.place_order_raise = None
        retried = oms.retry_failed(broker)
        assert len(retried) == 1
        assert retried[0].status == OMSStatus.FILLED
        assert retried[0].retries == 1

    def test_retry_max_exceeded(self) -> None:
        """最大リトライ回数を超えるとリトライしない。"""
        oms = OrderManagementSystem(max_retries=1)
        broker = MockBroker()

        # 最初の発注は失敗 (retries=0, status=ERROR)
        broker.place_order_raise = RuntimeError("error")
        oms.submit(
            broker=broker, symbol="7203.T", side=OrderSide.BUY,
            order_type=OrderType.MARKET, quantity=100,
        )

        # リトライ1回目: まだ失敗（retries 0→1）、retried リストには
        # 失敗パス(except)で入らないが、内部的にretries=1に更新される
        oms.retry_failed(broker)

        # retries(1) >= max_retries(1) なので2回目はスキップ
        retried2 = oms.retry_failed(broker)
        assert len(retried2) == 0

        # 内部の注文を確認: retries == 1
        all_orders = oms.get_all_orders()
        assert all_orders[0].retries == 1
        assert all_orders[0].status == OMSStatus.ERROR

    def test_get_active_orders(self) -> None:
        """未完了注文の一覧取得。"""
        oms = OrderManagementSystem()
        broker = MockBroker()

        oms.submit(broker=broker, symbol="A", side=OrderSide.BUY,
                   order_type=OrderType.MARKET, quantity=10)  # → FILLED
        oms.submit(broker=broker, symbol="B", side=OrderSide.BUY,
                   order_type=OrderType.LIMIT, quantity=10, price=100)  # → SUBMITTED

        active = oms.get_active_orders()
        assert len(active) == 1
        assert active[0].symbol == "B"

    def test_get_filled_orders(self) -> None:
        """約定済み注文の一覧取得。"""
        oms = OrderManagementSystem()
        broker = MockBroker()

        oms.submit(broker=broker, symbol="A", side=OrderSide.BUY,
                   order_type=OrderType.MARKET, quantity=10)
        oms.submit(broker=broker, symbol="B", side=OrderSide.BUY,
                   order_type=OrderType.LIMIT, quantity=10, price=100)

        filled = oms.get_filled_orders()
        assert len(filled) == 1
        assert filled[0].symbol == "A"

    def test_summary(self) -> None:
        """ステータスサマリーが正しい。"""
        oms = OrderManagementSystem()
        broker = MockBroker()

        oms.submit(broker=broker, symbol="A", side=OrderSide.BUY,
                   order_type=OrderType.MARKET, quantity=10)
        oms.submit(broker=broker, symbol="B", side=OrderSide.BUY,
                   order_type=OrderType.LIMIT, quantity=10, price=100)

        s = oms.summary()
        assert s["filled"] == 1
        assert s["submitted"] == 1
