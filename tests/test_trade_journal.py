"""トレードジャーナル（SQLite永続化）のテスト。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest

from src.brokers.base import Order, OrderSide, OrderType
from src.brokers.paper_broker import PaperBroker
from src.brokers.trade_journal import TradeJournal


# ============================================================
# フィクスチャ
# ============================================================

@pytest.fixture
def journal(tmp_path: Path) -> Generator[TradeJournal, None, None]:
    """テスト用の一時DBを使ったジャーナル。"""
    db_path = str(tmp_path / "test_journal.db")
    j = TradeJournal(db_path=db_path)
    yield j
    j.close()


@pytest.fixture
def sample_order() -> Order:
    """テスト用サンプル注文。"""
    return Order(
        order_id="order001",
        symbol="7203.T",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        price=None,
        status="filled",
        filled_quantity=100.0,
        avg_fill_price=1000.0,
        fee=100.0,
        created_at="2024-01-10 09:00:00",
        updated_at="2024-01-10 09:00:01",
    )


@pytest.fixture
def broker_with_journal(tmp_path: Path) -> Generator[tuple[PaperBroker, TradeJournal], None, None]:
    """ジャーナル付きPaperBroker。"""
    db_path = str(tmp_path / "broker_journal.db")
    j = TradeJournal(db_path=db_path)
    broker = PaperBroker(
        initial_balance=1_000_000.0,
        fee_rate=0.001,
        slippage_rate=0.0,
        journal=j,
    )
    yield broker, j
    j.close()


# ============================================================
# テーブル作成テスト
# ============================================================

class TestTableCreation:
    """テーブル初期化のテスト。"""

    def test_tables_created_on_init(self, journal: TradeJournal) -> None:
        """初期化時に全テーブルが作成されることを確認。"""
        conn = journal._conn
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "orders" in tables
        assert "snapshots" in tables
        assert "daily_pnl" in tables

    def test_second_init_same_db_no_error(self, tmp_path: Path) -> None:
        """同一DBパスで2回初期化してもエラーにならない（IF NOT EXISTS）。"""
        db_path = str(tmp_path / "double_init.db")
        j1 = TradeJournal(db_path=db_path)
        j1.close()
        j2 = TradeJournal(db_path=db_path)
        j2.close()


# ============================================================
# save_order / load_orders テスト
# ============================================================

class TestOrderPersistence:
    """注文の保存と読み込みのテスト。"""

    def test_save_and_load_order(self, journal: TradeJournal, sample_order: Order) -> None:
        """save_order → load_orders のラウンドトリップ。"""
        journal.save_order(sample_order)
        orders = journal.load_orders()
        assert len(orders) == 1
        loaded = orders[0]
        assert loaded.order_id == sample_order.order_id
        assert loaded.symbol == sample_order.symbol
        assert loaded.side == sample_order.side
        assert loaded.order_type == sample_order.order_type
        assert loaded.quantity == sample_order.quantity
        assert loaded.status == sample_order.status

    def test_load_orders_empty(self, journal: TradeJournal) -> None:
        """注文がない場合は空リストを返す。"""
        assert journal.load_orders() == []

    def test_multiple_orders_loaded_in_created_order(self, journal: TradeJournal) -> None:
        """複数注文が created_at 昇順で返ることを確認。"""
        orders = [
            Order(
                order_id=f"ord{i:03d}",
                symbol="7203.T",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=10.0,
                price=None,
                status="filled",
                created_at=f"2024-01-{10 + i:02d} 09:00:00",
                updated_at=f"2024-01-{10 + i:02d} 09:00:01",
            )
            for i in range(3)
        ]
        for o in orders:
            journal.save_order(o)

        loaded = journal.load_orders()
        assert [o.order_id for o in loaded] == ["ord000", "ord001", "ord002"]

    def test_save_order_replace_on_duplicate_id(self, journal: TradeJournal, sample_order: Order) -> None:
        """同一order_idを再保存すると上書きされる（INSERT OR REPLACE）。"""
        journal.save_order(sample_order)
        import dataclasses
        updated = dataclasses.replace(sample_order, status="cancelled")
        journal.save_order(updated)
        orders = journal.load_orders()
        assert len(orders) == 1
        assert orders[0].status == "cancelled"

    def test_order_with_none_price(self, journal: TradeJournal) -> None:
        """price=None の注文を保存・復元できる。"""
        order = Order(
            order_id="ord_none_price",
            symbol="9984.T",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50.0,
            price=None,
            status="filled",
        )
        journal.save_order(order)
        loaded = journal.load_orders()[0]
        assert loaded.price is None

    def test_sell_order_roundtrip(self, journal: TradeJournal) -> None:
        """SELL注文のラウンドトリップ。"""
        order = Order(
            order_id="sell001",
            symbol="6758.T",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=200.0,
            price=1500.0,
            status="filled",
            filled_quantity=200.0,
            avg_fill_price=1498.0,
            fee=299.6,
            created_at="2024-02-01 10:30:00",
            updated_at="2024-02-01 10:30:05",
        )
        journal.save_order(order)
        loaded = journal.load_orders()[0]
        assert loaded.side == OrderSide.SELL
        assert loaded.order_type == OrderType.LIMIT
        assert loaded.avg_fill_price == pytest.approx(1498.0)


# ============================================================
# save_snapshot / load_latest_snapshot テスト
# ============================================================

class TestSnapshotPersistence:
    """スナップショットの保存と読み込みのテスト。"""

    def test_save_and_load_snapshot(self, journal: TradeJournal) -> None:
        """スナップショットの保存と最新取得。"""
        positions = {"7203.T": {"quantity": 100.0, "avg_price": 1000.0}}
        journal.save_snapshot(balance=900_000.0, equity=1_000_000.0, positions=positions)
        snap = journal.load_latest_snapshot()
        assert snap is not None
        assert snap["balance"] == pytest.approx(900_000.0)
        assert snap["equity"] == pytest.approx(1_000_000.0)
        assert "7203.T" in snap["positions"]  # type: ignore[operator]

    def test_load_latest_snapshot_returns_none_when_empty(self, journal: TradeJournal) -> None:
        """スナップショットがない場合は None を返す。"""
        assert journal.load_latest_snapshot() is None

    def test_load_latest_snapshot_returns_most_recent(self, journal: TradeJournal) -> None:
        """複数スナップショットがある場合、最新のものを返す。"""
        journal.save_snapshot(balance=800_000.0, equity=800_000.0, positions={})
        journal.save_snapshot(balance=950_000.0, equity=1_050_000.0, positions={})
        snap = journal.load_latest_snapshot()
        assert snap is not None
        assert snap["balance"] == pytest.approx(950_000.0)


# ============================================================
# save_daily_pnl / get_daily_pnl テスト
# ============================================================

class TestDailyPnl:
    """日次P&Lの保存と取得のテスト。"""

    def test_save_and_retrieve_daily_pnl(self, journal: TradeJournal) -> None:
        """日次P&Lの保存と取得。"""
        journal.save_daily_pnl("2024-01-10", open_eq=1_000_000.0, close_eq=1_020_000.0)
        records = journal.get_daily_pnl(days=30)
        assert len(records) == 1
        assert records[0]["date"] == "2024-01-10"
        assert records[0]["pnl"] == pytest.approx(20_000.0)
        assert records[0]["pnl_pct"] == pytest.approx(2.0)

    def test_daily_pnl_upsert(self, journal: TradeJournal) -> None:
        """同一日付の再保存は上書きされる。"""
        journal.save_daily_pnl("2024-01-10", open_eq=1_000_000.0, close_eq=1_010_000.0)
        journal.save_daily_pnl("2024-01-10", open_eq=1_000_000.0, close_eq=1_025_000.0)
        records = journal.get_daily_pnl()
        assert len(records) == 1
        assert records[0]["pnl"] == pytest.approx(25_000.0)

    def test_get_daily_pnl_respects_days_limit(self, journal: TradeJournal) -> None:
        """days パラメータが正しく機能する。"""
        for i in range(10):
            journal.save_daily_pnl(
                f"2024-01-{i + 1:02d}",
                open_eq=1_000_000.0,
                close_eq=1_001_000.0 * (1 + i * 0.001),
            )
        records = journal.get_daily_pnl(days=5)
        assert len(records) == 5

    def test_get_daily_pnl_ascending_order(self, journal: TradeJournal) -> None:
        """取得結果が date 昇順になっている。"""
        journal.save_daily_pnl("2024-01-03", open_eq=1_000_000.0, close_eq=1_005_000.0)
        journal.save_daily_pnl("2024-01-01", open_eq=1_000_000.0, close_eq=1_002_000.0)
        journal.save_daily_pnl("2024-01-02", open_eq=1_000_000.0, close_eq=1_003_000.0)
        records = journal.get_daily_pnl()
        dates = [r["date"] for r in records]
        assert dates == sorted(dates)


# ============================================================
# get_trade_summary テスト
# ============================================================

class TestTradeSummary:
    """トレードサマリーのテスト。"""

    def test_summary_empty(self, journal: TradeJournal) -> None:
        """注文がない場合のサマリー。"""
        summary = journal.get_trade_summary()
        assert summary["total_orders"] == 0
        assert summary["filled_orders"] == 0
        assert summary["net_pnl"] == pytest.approx(0.0)

    def test_summary_with_orders(self, journal: TradeJournal) -> None:
        """買い・売り注文がある場合のサマリー計算。"""
        buy = Order(
            order_id="b001",
            symbol="7203.T",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None,
            status="filled",
            filled_quantity=100.0,
            avg_fill_price=1000.0,
            fee=100.0,
        )
        sell = Order(
            order_id="s001",
            symbol="7203.T",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None,
            status="filled",
            filled_quantity=100.0,
            avg_fill_price=1200.0,
            fee=120.0,
        )
        journal.save_order(buy)
        journal.save_order(sell)
        summary = journal.get_trade_summary()
        assert summary["total_orders"] == 2
        assert summary["filled_orders"] == 2
        # net_pnl = sell - buy - fees = 120000 - 100000 - 220 = 19780
        assert summary["net_pnl"] == pytest.approx(19_780.0)
        assert summary["total_fees"] == pytest.approx(220.0)


# ============================================================
# PaperBroker + TradeJournal 統合テスト
# ============================================================

class TestPaperBrokerWithJournal:
    """PaperBroker と TradeJournal の統合テスト。"""

    def test_place_order_saves_to_journal(
        self, broker_with_journal: tuple[PaperBroker, TradeJournal]
    ) -> None:
        """place_order() がジャーナルに注文を保存することを確認。"""
        broker, journal = broker_with_journal
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        orders = journal.load_orders()
        assert len(orders) == 1
        assert orders[0].symbol == "7203.T"
        assert orders[0].status == "filled"

    def test_multiple_orders_all_saved(
        self, broker_with_journal: tuple[PaperBroker, TradeJournal]
    ) -> None:
        """複数の注文がすべてジャーナルに保存される。"""
        broker, journal = broker_with_journal
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 50)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 30)
        broker.update_price("7203.T", 1100.0)
        broker.place_order("7203.T", OrderSide.SELL, OrderType.MARKET, 80)
        orders = journal.load_orders()
        assert len(orders) == 3

    def test_save_snapshot_delegates_to_journal(
        self, broker_with_journal: tuple[PaperBroker, TradeJournal]
    ) -> None:
        """save_snapshot() がジャーナルにスナップショットを保存する。"""
        broker, journal = broker_with_journal
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 100)
        broker.save_snapshot()
        snap = journal.load_latest_snapshot()
        assert snap is not None
        assert snap["balance"] == pytest.approx(broker.get_balance())

    def test_save_snapshot_without_journal_is_noop(self) -> None:
        """journal=None の場合、save_snapshot() は何もしない（エラーなし）。"""
        broker = PaperBroker(initial_balance=1_000_000.0)
        broker.update_price("7203.T", 1000.0)
        broker.place_order("7203.T", OrderSide.BUY, OrderType.MARKET, 10)
        broker.save_snapshot()  # エラーにならないことを確認


# ============================================================
# PaperBroker.load_state テスト
# ============================================================

class TestPaperBrokerLoadState:
    """PaperBroker.load_state() のテスト。"""

    def test_load_state_empty_journal_uses_fallback(self, tmp_path: Path) -> None:
        """ジャーナルが空の場合、fallback_balance で作成される。"""
        db_path = str(tmp_path / "empty.db")
        journal = TradeJournal(db_path=db_path)
        broker = PaperBroker.load_state(
            journal=journal, fallback_balance=500_000.0
        )
        assert broker.get_balance() == pytest.approx(500_000.0)
        assert len(broker.get_order_history()) == 0
        journal.close()

    def test_load_state_restores_balance_from_snapshot(self, tmp_path: Path) -> None:
        """スナップショットがある場合、残高が復元される。"""
        db_path = str(tmp_path / "with_snap.db")
        journal = TradeJournal(db_path=db_path)
        journal.save_snapshot(balance=750_000.0, equity=800_000.0, positions={})
        broker = PaperBroker.load_state(journal=journal, fallback_balance=1_000_000.0)
        assert broker.get_balance() == pytest.approx(750_000.0)
        journal.close()

    def test_load_state_restores_order_history(self, tmp_path: Path) -> None:
        """注文履歴がジャーナルから復元される。"""
        db_path = str(tmp_path / "with_orders.db")
        journal = TradeJournal(db_path=db_path)
        order = Order(
            order_id="hist001",
            symbol="9984.T",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            price=None,
            status="filled",
            filled_quantity=10.0,
            avg_fill_price=2000.0,
            fee=20.0,
        )
        journal.save_order(order)
        broker = PaperBroker.load_state(journal=journal, fallback_balance=1_000_000.0)
        history = broker.get_order_history()
        assert len(history) == 1
        assert history[0].order_id == "hist001"
        journal.close()

    def test_load_state_restores_positions(self, tmp_path: Path) -> None:
        """ポジションがスナップショットから復元される。"""
        db_path = str(tmp_path / "with_positions.db")
        journal = TradeJournal(db_path=db_path)
        positions = {"7203.T": {"quantity": 200.0, "avg_price": 950.0}}
        journal.save_snapshot(balance=810_000.0, equity=1_000_000.0, positions=positions)
        broker = PaperBroker.load_state(journal=journal, fallback_balance=1_000_000.0)
        # ポジションが復元されていることを確認（価格設定後に取得）
        broker.update_price("7203.T", 950.0)
        pos = broker.get_positions()
        assert "7203.T" in pos
        assert pos["7203.T"].quantity == pytest.approx(200.0)
        journal.close()

    def test_load_state_new_orders_persist(self, tmp_path: Path) -> None:
        """復元後の新規注文がジャーナルに保存される。"""
        db_path = str(tmp_path / "persist_new.db")
        journal = TradeJournal(db_path=db_path)
        broker = PaperBroker.load_state(
            journal=journal, fallback_balance=1_000_000.0
        )
        broker.update_price("6758.T", 5000.0)
        broker.place_order("6758.T", OrderSide.BUY, OrderType.MARKET, 10)
        orders = journal.load_orders()
        assert len(orders) == 1
        assert orders[0].symbol == "6758.T"
        journal.close()
