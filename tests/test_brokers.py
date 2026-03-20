"""ブローカーのテスト。"""
from __future__ import annotations

import pytest

from src.brokers.base import Order, OrderSide, OrderType, Position
from src.brokers.paper_broker import PaperBroker


# ============================================================
# PaperBroker のテスト
# ============================================================

class TestPaperBroker:
    """PaperBrokerのテスト。"""

    @pytest.fixture
    def broker(self) -> PaperBroker:
        """テスト用ブローカーを作成する。"""
        return PaperBroker(initial_balance=1_000_000.0, fee_rate=0.001, slippage_rate=0.0)

    def test_initial_balance(self, broker: PaperBroker):
        """初期残高が正しく設定されることを確認。"""
        assert broker.get_balance() == 1_000_000.0

    def test_buy_order_reduces_balance(self, broker: PaperBroker):
        """買い注文で残高が減少することを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        # 手数料込みで減少するはず
        assert broker.get_balance() < 1_000_000.0

    def test_buy_creates_position(self, broker: PaperBroker):
        """買い注文でポジションが作成されることを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        positions = broker.get_positions()
        assert "7203.T" in positions
        assert positions["7203.T"].quantity == 100.0

    def test_sell_closes_position(self, broker: PaperBroker):
        """売り注文でポジションがクローズされることを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        broker.update_price("7203.T", 1100.0)
        broker.place_order("7203.T", OrderSide.SELL, OrderType.MARKET, 100)
        positions = broker.get_positions()
        assert "7203.T" not in positions or positions["7203.T"].quantity == 0

    def test_profit_on_price_increase(self, broker: PaperBroker):
        """価格上昇時に利益が発生することを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        balance_after_buy = broker.get_balance()

        broker.update_price("7203.T", 1200.0)
        broker.place_order("7203.T", OrderSide.SELL, OrderType.MARKET, 100)
        final_balance = broker.get_balance()

        # 20%上昇 → 手数料引いても利益あり
        assert final_balance > 1_000_000.0

    def test_insufficient_balance_raises(self, broker: PaperBroker):
        """残高不足で注文失敗することを確認。"""
        broker.update_price("7203.T", 100_000.0)  # 非常に高価な銘柄
        with pytest.raises(ValueError, match="残高不足"):
            broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 1000)

    def test_oversell_raises(self, broker: PaperBroker):
        """保有数量超の売り注文でエラーになることを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        with pytest.raises(ValueError, match="保有株数不足"):
            broker.place_order("7203.T", OrderSide.SELL, OrderType.MARKET, 200)

    def test_no_price_set_raises(self, broker: PaperBroker):
        """価格が設定されていない場合にエラーになることを確認。"""
        with pytest.raises(ValueError, match="現在価格が設定されていません"):
            broker.place_order("UNKNOWN.T", OrderSide.BUY, OrderType.MARKET, 100)

    def test_average_price_on_multiple_buys(self, broker: PaperBroker):
        """複数回買いの平均取得価格が正しく計算されることを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        broker.update_price("7203.T", 1200.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)

        positions = broker.get_positions()
        pos = positions["7203.T"]
        assert pos.quantity == 200.0
        # 平均取得価格は1000〜1200の間
        assert 1000.0 <= pos.avg_entry_price <= 1200.0

    def test_get_equity_includes_positions(self, broker: PaperBroker):
        """get_equity()がポジションの時価を含むことを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        broker.update_price("7203.T", 2000.0)

        equity = broker.get_equity()
        # 現金 + ポジション時価評価 > 初期残高
        assert equity > 1_000_000.0

    def test_reset_clears_state(self, broker: PaperBroker):
        """reset()が全状態をクリアすることを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        broker.reset()

        assert broker.get_balance() == 1_000_000.0
        assert len(broker.get_positions()) == 0
        assert len(broker.get_order_history()) == 0

    def test_order_is_returned(self, broker: PaperBroker):
        """発注した注文が返されることを確認。"""
        broker.update_price("7203.T", 1000.0)
        order = broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        assert isinstance(order, Order)
        assert order.status == "filled"
        assert order.filled_quantity == 100.0

    def test_get_open_orders_empty(self, broker: PaperBroker):
        """ペーパーブローカーはオープン注文が常に空であることを確認。"""
        assert broker.get_open_orders() == []

    def test_order_history(self, broker: PaperBroker):
        """注文履歴が正しく記録されることを確認。"""
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 50)
        broker.place_order("7203.T", OrderSide.SELL, OrderType.MARKET, 50)
        history = broker.get_order_history()
        assert len(history) == 2


# ============================================================
# Order dataclass のテスト
# ============================================================

class TestOrder:
    """Order dataclassのテスト。"""

    def test_order_immutable(self):
        """Orderがimmutableであることを確認。"""
        order = Order(
            order_id="test001",
            symbol="7203.T",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None,
            status="filled",
        )
        with pytest.raises((AttributeError, TypeError)):
            order.quantity = 200.0  # type: ignore

    def test_to_dict(self):
        """to_dict()が正しく動作することを確認。"""
        order = Order(
            order_id="test001",
            symbol="7203.T",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None,
            status="filled",
        )
        d = order.to_dict()
        assert d["order_id"] == "test001"
        assert d["symbol"] == "7203.T"
        assert d["status"] == "filled"
