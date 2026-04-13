"""ペーパートレードブローカー。

仮想的な注文・ポジション・残高を管理する。
実際の取引は発生しない（テスト・デモ用）。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from .base import BrokerBase, Order, OrderSide, OrderType, Position
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class PaperBroker(BrokerBase):
    """ペーパートレードブローカー。

    Parameters
    ----------
    initial_balance:
        初期残高（円）
    fee_rate:
        片道手数料率（デフォルト: 0.001 = 0.1%）
    slippage_rate:
        スリッページ率（デフォルト: 0.0005 = 0.05%）
    journal:
        SQLite永続化ジャーナル（省略時はインメモリのみ）
    """

    def __init__(
        self,
        initial_balance: float = 1_000_000.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        journal: Optional[TradeJournal] = None,
    ) -> None:
        self._balance = initial_balance
        self._initial_balance = initial_balance
        self._fee_rate = fee_rate
        self._slippage_rate = slippage_rate
        self._journal = journal

        # 可変状態（ミュータブルだがブローカー内部状態として管理）
        self._positions: Dict[str, dict] = {}      # {symbol: {"quantity": float, "avg_price": float}}
        self._orders: Dict[str, Order] = {}        # {order_id: Order}
        self._current_prices: Dict[str, float] = {}
        self._order_history: List[Order] = []

    def update_price(self, symbol: str, price: float) -> None:
        """現在価格を更新する（バックテストやライブ接続から呼び出す）。"""
        self._current_prices[symbol] = price

    def get_balance(self) -> float:
        """利用可能な現金残高を返す。"""
        return self._balance

    def get_equity(self) -> float:
        """総資産（現金 + 時価評価額）を返す。"""
        pos_value = sum(
            info["quantity"] * self._current_prices.get(sym, info["avg_price"])
            for sym, info in self._positions.items()
            if info["quantity"] > 0
        )
        return self._balance + pos_value

    def get_positions(self) -> Dict[str, Position]:
        """全ポジションを返す。"""
        result: Dict[str, Position] = {}
        for symbol, info in self._positions.items():
            qty = info["quantity"]
            if qty == 0:
                continue

            avg_price = info["avg_price"]
            current = self._current_prices.get(symbol, avg_price)
            pnl = (current - avg_price) * qty
            pnl_pct = (pnl / (avg_price * abs(qty))) * 100 if avg_price > 0 else 0.0

            result[symbol] = Position(
                symbol=symbol,
                quantity=qty,
                avg_entry_price=avg_price,
                current_price=current,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
            )
        return result

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        """仮想注文を発注・即時約定する。

        成行注文は即時約定。指値は価格を指定として記録するが、
        現在価格で即時約定する（簡略実装）。
        """
        if quantity <= 0:
            raise ValueError(f"株数は正の値にしてください: {quantity}")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        order_id = str(uuid.uuid4())[:8]

        # 実行価格（スリッページ適用）
        current_price = price or self._current_prices.get(symbol, 0.0)
        if current_price <= 0:
            raise ValueError(f"{symbol} の現在価格が設定されていません。update_price()を呼び出してください。")

        if side == OrderSide.BUY:
            exec_price = current_price * (1 + self._slippage_rate)
        else:
            exec_price = current_price * (1 - self._slippage_rate)

        fee = exec_price * quantity * self._fee_rate
        total_cost = exec_price * quantity

        if side == OrderSide.BUY:
            required = total_cost + fee
            if required > self._balance:
                raise ValueError(
                    f"残高不足: 必要={required:.0f}円, 残高={self._balance:.0f}円"
                )

            # ポジション更新
            if symbol in self._positions and self._positions[symbol]["quantity"] > 0:
                existing = self._positions[symbol]
                new_qty = existing["quantity"] + quantity
                new_avg = (
                    existing["avg_price"] * existing["quantity"] + exec_price * quantity
                ) / new_qty
                self._positions[symbol] = {"quantity": new_qty, "avg_price": new_avg}
            else:
                self._positions[symbol] = {"quantity": quantity, "avg_price": exec_price}

            self._balance -= required
            logger.info(f"BUY約定: {symbol} {quantity}株 @{exec_price:.0f}円 (手数料:{fee:.0f}円)")

        else:  # SELL
            pos = self._positions.get(symbol, {"quantity": 0, "avg_price": 0})
            if pos["quantity"] < quantity:
                raise ValueError(
                    f"保有株数不足: 必要={quantity}, 保有={pos['quantity']}"
                )

            new_qty = pos["quantity"] - quantity
            self._positions[symbol] = {"quantity": new_qty, "avg_price": pos["avg_price"]}

            proceeds = total_cost - fee
            self._balance += proceeds
            logger.info(f"SELL約定: {symbol} {quantity}株 @{exec_price:.0f}円 (手数料:{fee:.0f}円)")

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="filled",
            filled_quantity=quantity,
            avg_fill_price=exec_price,
            fee=fee,
            created_at=now,
            updated_at=now,
        )

        self._orders[order_id] = order
        self._order_history.append(order)

        if self._journal is not None:
            self._journal.save_order(order)

        return order

    def cancel_order(self, order_id: str) -> bool:
        """注文をキャンセルする（ペーパーでは即時約定するため常にFalse）。"""
        order = self._orders.get(order_id)
        if order is None or order.status != "open":
            return False

        # キャンセル済み注文に更新（immutableなのでreplaceで新規作成）
        import dataclasses
        cancelled = dataclasses.replace(order, status="cancelled")
        self._orders[order_id] = cancelled
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """注文状況を取得する。"""
        return self._orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """オープン注文一覧を返す（ペーパーでは常に空）。"""
        return []

    def get_order_history(self) -> List[Order]:
        """注文履歴を返す（immutableコピー）。"""
        return list(self._order_history)

    def save_snapshot(self) -> None:
        """現在のポートフォリオスナップショットをジャーナルに保存する。

        journal が None の場合は何もしない。
        """
        if self._journal is None:
            return
        positions_serializable = {
            sym: {"quantity": info["quantity"], "avg_price": info["avg_price"]}
            for sym, info in self._positions.items()
        }
        self._journal.save_snapshot(
            balance=self._balance,
            equity=self.get_equity(),
            positions=positions_serializable,
        )

    @classmethod
    def load_state(
        cls,
        journal: TradeJournal,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        fallback_balance: float = 1_000_000.0,
    ) -> "PaperBroker":
        """ジャーナルから状態を復元した PaperBroker を返す。

        最新スナップショットがあればその残高・ポジションを復元する。
        スナップショットがない場合は fallback_balance で新規作成する。

        Parameters
        ----------
        journal:
            復元元のジャーナル。
        fee_rate:
            手数料率。
        slippage_rate:
            スリッページ率。
        fallback_balance:
            スナップショットが存在しない場合の初期残高。

        Returns
        -------
        PaperBroker
            復元済みのブローカーインスタンス。
        """
        snapshot = journal.load_latest_snapshot()

        if snapshot is not None:
            balance = float(snapshot["balance"])  # type: ignore[arg-type]
            raw_positions: Dict[str, dict] = snapshot.get("positions", {})  # type: ignore[assignment]
        else:
            balance = fallback_balance
            raw_positions = {}

        broker = cls(
            initial_balance=balance,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            journal=journal,
        )

        # ポジションを復元
        for sym, info in raw_positions.items():
            broker._positions[sym] = {
                "quantity": float(info.get("quantity", 0.0)),
                "avg_price": float(info.get("avg_price", 0.0)),
            }

        # 注文履歴を復元
        for order in journal.load_orders():
            broker._orders[order.order_id] = order
            broker._order_history.append(order)

        logger.info(
            "PaperBroker 状態復元: balance=%.0f, positions=%d, orders=%d",
            balance,
            len(broker._positions),
            len(broker._order_history),
        )
        return broker

    def reset(self) -> None:
        """ブローカー状態をリセットする（テスト用）。"""
        self._balance = self._initial_balance
        self._positions.clear()
        self._orders.clear()
        self._order_history.clear()
        self._current_prices.clear()
        logger.info("PaperBrokerリセット完了")
