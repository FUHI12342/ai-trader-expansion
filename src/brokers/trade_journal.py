"""トレードジャーナル（SQLite永続化）。

全ペーパートレードの注文履歴・ポジション・残高スナップショットを
追記型（append-only）で永続化する。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Dict, List, Optional

from .base import Order, OrderSide, OrderType

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "./trade_journal.db"


class TradeJournal:
    """SQLite永続化トレードジャーナル。

    Parameters
    ----------
    db_path:
        SQLiteデータベースファイルのパス。
        省略時は環境変数 ``TRADER_JOURNAL_DB_PATH`` 、
        それも未設定の場合は ``./trade_journal.db`` を使用する。
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or os.environ.get(
            "TRADER_JOURNAL_DB_PATH", _DEFAULT_DB_PATH
        )
        # check_same_thread=False でスレッドセーフを確保
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info("TradeJournal 初期化完了: %s", self._db_path)

    # ------------------------------------------------------------------
    # テーブル初期化
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """テーブルが存在しない場合に作成する。"""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id        TEXT PRIMARY KEY,
                    symbol          TEXT    NOT NULL,
                    side            TEXT    NOT NULL,
                    order_type      TEXT    NOT NULL,
                    quantity        REAL    NOT NULL,
                    price           REAL,
                    status          TEXT    NOT NULL,
                    filled_quantity REAL    NOT NULL DEFAULT 0.0,
                    avg_fill_price  REAL    NOT NULL DEFAULT 0.0,
                    fee             REAL    NOT NULL DEFAULT 0.0,
                    created_at      TEXT    NOT NULL DEFAULT '',
                    updated_at      TEXT    NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    balance         REAL    NOT NULL,
                    equity          REAL    NOT NULL,
                    positions_json  TEXT    NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date            TEXT    PRIMARY KEY,
                    open_equity     REAL    NOT NULL,
                    close_equity    REAL    NOT NULL,
                    pnl             REAL    NOT NULL,
                    pnl_pct         REAL    NOT NULL,
                    cumulative_pnl_pct REAL NOT NULL DEFAULT 0.0
                );
                """
            )

    # ------------------------------------------------------------------
    # 書き込みメソッド
    # ------------------------------------------------------------------

    def save_order(self, order: Order) -> None:
        """注文レコードをDBに挿入（既存のorder_idは上書き）。

        Parameters
        ----------
        order:
            保存する注文データ。
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO orders
                    (order_id, symbol, side, order_type, quantity, price,
                     status, filled_quantity, avg_fill_price, fee,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.order_id,
                    order.symbol,
                    order.side.value if isinstance(order.side, OrderSide) else order.side,
                    order.order_type.value if isinstance(order.order_type, OrderType) else order.order_type,
                    order.quantity,
                    order.price,
                    order.status,
                    order.filled_quantity,
                    order.avg_fill_price,
                    order.fee,
                    order.created_at,
                    order.updated_at,
                ),
            )
        logger.debug("注文保存: %s %s %s", order.order_id, order.symbol, order.side)

    def save_snapshot(
        self,
        balance: float,
        equity: float,
        positions: Dict[str, object],
    ) -> None:
        """ポートフォリオスナップショットをDBに挿入する。

        Parameters
        ----------
        balance:
            現金残高。
        equity:
            総資産（現金 + 含み）。
        positions:
            ポジション辞書（JSON形式で保存）。
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        positions_json = json.dumps(positions, ensure_ascii=False, default=str)
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO snapshots (timestamp, balance, equity, positions_json)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, balance, equity, positions_json),
            )
        logger.debug("スナップショット保存: balance=%s equity=%s", balance, equity)

    def save_daily_pnl(
        self,
        date: str,
        open_eq: float,
        close_eq: float,
    ) -> None:
        """日次P&LをDBに挿入または更新する。

        累積P&L率は直近レコードを基準に自動計算する。

        Parameters
        ----------
        date:
            日付文字列（YYYY-MM-DD）。
        open_eq:
            始値エクイティ（円）。
        close_eq:
            終値エクイティ（円）。
        """
        pnl = close_eq - open_eq
        pnl_pct = (pnl / open_eq * 100.0) if open_eq > 0 else 0.0

        # 直近の累積P&L率を取得して加算
        row = self._conn.execute(
            "SELECT cumulative_pnl_pct FROM daily_pnl ORDER BY date DESC LIMIT 1"
        ).fetchone()
        prev_cumulative = row["cumulative_pnl_pct"] if row else 0.0
        cumulative_pnl_pct = prev_cumulative + pnl_pct

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO daily_pnl
                    (date, open_equity, close_equity, pnl, pnl_pct, cumulative_pnl_pct)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    open_equity        = excluded.open_equity,
                    close_equity       = excluded.close_equity,
                    pnl                = excluded.pnl,
                    pnl_pct            = excluded.pnl_pct,
                    cumulative_pnl_pct = excluded.cumulative_pnl_pct
                """,
                (date, open_eq, close_eq, pnl, pnl_pct, cumulative_pnl_pct),
            )
        logger.debug("日次P&L保存: date=%s pnl=%.2f", date, pnl)

    # ------------------------------------------------------------------
    # 読み込みメソッド
    # ------------------------------------------------------------------

    def load_orders(self) -> List[Order]:
        """全注文レコードをDBから読み込む。

        Returns
        -------
        List[Order]
            保存されている全注文（created_at昇順）。
        """
        rows = self._conn.execute(
            "SELECT * FROM orders ORDER BY created_at ASC"
        ).fetchall()

        orders: List[Order] = []
        for row in rows:
            orders.append(
                Order(
                    order_id=row["order_id"],
                    symbol=row["symbol"],
                    side=OrderSide(row["side"]),
                    order_type=OrderType(row["order_type"]),
                    quantity=row["quantity"],
                    price=row["price"],
                    status=row["status"],
                    filled_quantity=row["filled_quantity"],
                    avg_fill_price=row["avg_fill_price"],
                    fee=row["fee"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return orders

    def load_latest_snapshot(self) -> Optional[Dict[str, object]]:
        """最新スナップショットをDBから読み込む。

        Returns
        -------
        Optional[Dict[str, object]]
            最新スナップショット（balance, equity, positions_json, timestamp）。
            レコードが存在しない場合は None。
        """
        row = self._conn.execute(
            "SELECT * FROM snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "balance": row["balance"],
            "equity": row["equity"],
            "positions": json.loads(row["positions_json"]),
        }

    def get_daily_pnl(self, days: int = 30) -> List[Dict[str, object]]:
        """直近N日分の日次P&Lを取得する。

        Parameters
        ----------
        days:
            取得する日数（デフォルト: 30日）。

        Returns
        -------
        List[Dict[str, object]]
            日次P&Lリスト（date昇順）。
        """
        rows = self._conn.execute(
            """
            SELECT date, open_equity, close_equity, pnl, pnl_pct, cumulative_pnl_pct
            FROM daily_pnl
            ORDER BY date DESC
            LIMIT ?
            """,
            (days,),
        ).fetchall()

        # 昇順に変換して返す
        result: List[Dict[str, object]] = [
            {
                "date": row["date"],
                "open_equity": row["open_equity"],
                "close_equity": row["close_equity"],
                "pnl": row["pnl"],
                "pnl_pct": row["pnl_pct"],
                "cumulative_pnl_pct": row["cumulative_pnl_pct"],
            }
            for row in reversed(rows)
        ]
        return result

    def get_trade_summary(self) -> Dict[str, object]:
        """集計トレード統計を返す。

        Returns
        -------
        Dict[str, object]
            total_orders, filled_orders, total_fees,
            total_buy_amount, total_sell_amount, net_pnl
        """
        row = self._conn.execute(
            """
            SELECT
                COUNT(*)                                              AS total_orders,
                SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END)  AS filled_orders,
                SUM(fee)                                              AS total_fees,
                SUM(CASE WHEN side = 'buy'  THEN avg_fill_price * filled_quantity ELSE 0 END)
                                                                      AS total_buy_amount,
                SUM(CASE WHEN side = 'sell' THEN avg_fill_price * filled_quantity ELSE 0 END)
                                                                      AS total_sell_amount
            FROM orders
            """
        ).fetchone()

        total_buy: float = row["total_buy_amount"] or 0.0
        total_sell: float = row["total_sell_amount"] or 0.0
        total_fees: float = row["total_fees"] or 0.0
        net_pnl = total_sell - total_buy - total_fees

        return {
            "total_orders": row["total_orders"] or 0,
            "filled_orders": row["filled_orders"] or 0,
            "total_fees": round(total_fees, 2),
            "total_buy_amount": round(total_buy, 2),
            "total_sell_amount": round(total_sell, 2),
            "net_pnl": round(net_pnl, 2),
        }

    # ------------------------------------------------------------------
    # クリーンアップ
    # ------------------------------------------------------------------

    def close(self) -> None:
        """DB接続を閉じる。"""
        self._conn.close()
        logger.debug("TradeJournal 接続クローズ: %s", self._db_path)
