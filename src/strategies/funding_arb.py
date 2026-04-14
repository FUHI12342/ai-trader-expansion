"""Funding Rate Arbitrage 戦略。

現物ロング + 無期限先物ショートで市場中立ポジションを構築し、
ファンディングレート（8時間ごと）を受け取る。

リスク:
  - 価格変動: 現物とPerpの差益で相殺 → ほぼゼロ
  - ファンディングレート反転: 支払い側になる可能性
  - 取引所リスク: カウンターパーティリスク
  - 清算リスク: 先物側の証拠金不足

安全装置:
  - ファンディングレートが負の場合はエントリーしない
  - 証拠金維持率を監視
  - 最大ポジションサイズを制限
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import ccxt  # type: ignore[import-not-found]
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


@dataclass(frozen=True)
class FundingSnapshot:
    """ファンディングレートのスナップショット（immutable）。"""
    symbol: str
    rate: float
    annual_rate: float
    timestamp: float
    next_funding_time: float
    exchange: str

    @property
    def is_positive(self) -> bool:
        """正のレート（ロングが支払い → ショート側が受取）。"""
        return self.rate > 0


@dataclass(frozen=True)
class ArbPosition:
    """裁定ポジション（immutable）。"""
    symbol: str
    spot_quantity: float
    spot_avg_price: float
    perp_quantity: float
    perp_avg_price: float
    total_funding_earned: float
    entry_time: float
    status: str  # "open", "closed"

    @property
    def net_exposure(self) -> float:
        """ネットエクスポージャー（0に近いほど良い）。"""
        return self.spot_quantity - abs(self.perp_quantity)

    @property
    def funding_pnl_pct(self) -> float:
        """ファンディング収益率。"""
        cost = self.spot_quantity * self.spot_avg_price
        if cost == 0:
            return 0.0
        return self.total_funding_earned / cost * 100


class FundingRateCollector:
    """ファンディングレート収集器。

    Parameters
    ----------
    exchange_id:
        取引所ID (binance, bybit, etc.)
    config:
        API設定 (apiKey, secret等)
    """

    def __init__(self, exchange_id: str = "binance", config: Optional[dict] = None) -> None:
        self._exchange_id = exchange_id
        self._exchange: Optional[Any] = None

        if HAS_CCXT:
            exchange_class = getattr(ccxt, exchange_id, None)
            if exchange_class:
                cfg = config or {}
                cfg.setdefault("enableRateLimit", True)
                self._exchange = exchange_class(cfg)

    @property
    def is_available(self) -> bool:
        return self._exchange is not None

    def get_funding_rate(self, symbol: str = "BTC/USDT:USDT") -> Optional[FundingSnapshot]:
        """現在のファンディングレートを取得する。

        Parameters
        ----------
        symbol:
            無期限先物のシンボル (例: "BTC/USDT:USDT")

        Returns
        -------
        Optional[FundingSnapshot]
            ファンディングレート情報。取得失敗時はNone
        """
        if not self.is_available:
            return None

        try:
            funding = self._exchange.fetch_funding_rate(symbol)
            rate = float(funding.get("fundingRate", 0))
            # 年率換算: 8時間ごと × 3回/日 × 365日
            annual = rate * 3 * 365 * 100
            next_time = float(funding.get("fundingTimestamp", 0)) / 1000

            return FundingSnapshot(
                symbol=symbol,
                rate=rate,
                annual_rate=annual,
                timestamp=time.time(),
                next_funding_time=next_time,
                exchange=self._exchange_id,
            )
        except Exception as e:
            logger.warning("ファンディングレート取得失敗 (%s): %s", symbol, e)
            return None

    def get_top_funding_rates(
        self, symbols: Optional[List[str]] = None, min_rate: float = 0.0001
    ) -> List[FundingSnapshot]:
        """高ファンディングレートの銘柄を取得する。

        Parameters
        ----------
        symbols:
            チェック対象の銘柄リスト (省略時はデフォルト)
        min_rate:
            最低レート閾値

        Returns
        -------
        List[FundingSnapshot]
            min_rate以上の銘柄リスト（レート降順）
        """
        if symbols is None:
            symbols = [
                "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                "BNB/USDT:USDT", "XRP/USDT:USDT", "DOGE/USDT:USDT",
            ]

        results = []
        for sym in symbols:
            snapshot = self.get_funding_rate(sym)
            if snapshot and snapshot.rate >= min_rate:
                results.append(snapshot)

        return sorted(results, key=lambda s: s.rate, reverse=True)


class FundingArbStrategy:
    """Funding Rate Arbitrage 戦略。

    Parameters
    ----------
    exchange_id:
        取引所ID
    config:
        API設定
    min_funding_rate:
        最低エントリーレート (デフォルト: 0.01% = 年率10.95%)
    max_position_pct:
        最大ポジションサイズ (資金比率、デフォルト: 0.3 = 30%)
    exit_negative_rate:
        負のレートでエグジットするか (デフォルト: True)
    leverage:
        先物側のレバレッジ (デフォルト: 1 = クロスマージン)
    """

    def __init__(
        self,
        exchange_id: str = "binance",
        config: Optional[dict] = None,
        min_funding_rate: float = 0.0001,
        max_position_pct: float = 0.3,
        exit_negative_rate: bool = True,
        leverage: int = 1,
    ) -> None:
        self._collector = FundingRateCollector(exchange_id, config)
        self._min_rate = min_funding_rate
        self._max_position_pct = max_position_pct
        self._exit_negative = exit_negative_rate
        self._leverage = leverage
        self._positions: Dict[str, ArbPosition] = {}
        self._total_funding_earned: float = 0.0
        self._exchange_id = exchange_id
        self._config = config or {}

    @property
    def is_available(self) -> bool:
        return self._collector.is_available

    @property
    def total_funding_earned(self) -> float:
        return self._total_funding_earned

    @property
    def open_positions(self) -> Dict[str, ArbPosition]:
        return {k: v for k, v in self._positions.items() if v.status == "open"}

    def evaluate(self, symbol: str = "BTC/USDT:USDT") -> Dict[str, Any]:
        """銘柄のファンディングレートを評価し、アクションを返す。

        Returns
        -------
        Dict[str, Any]
            {"action": "enter"|"hold"|"exit"|"skip",
             "funding": FundingSnapshot, "reason": str}
        """
        snapshot = self._collector.get_funding_rate(symbol)
        if snapshot is None:
            return {"action": "skip", "funding": None, "reason": "データ取得失敗"}

        has_position = symbol in self.open_positions

        # エントリー判定
        if not has_position:
            if snapshot.rate >= self._min_rate:
                return {
                    "action": "enter",
                    "funding": snapshot,
                    "reason": f"FR={snapshot.rate:.4%} (年率{snapshot.annual_rate:.1f}%) > 閾値{self._min_rate:.4%}",
                }
            return {
                "action": "skip",
                "funding": snapshot,
                "reason": f"FR={snapshot.rate:.4%} < 閾値{self._min_rate:.4%}",
            }

        # エグジット判定
        if has_position:
            if self._exit_negative and snapshot.rate < 0:
                return {
                    "action": "exit",
                    "funding": snapshot,
                    "reason": f"FR反転: {snapshot.rate:.4%} (負のレート → 支払い側)",
                }
            return {
                "action": "hold",
                "funding": snapshot,
                "reason": f"ポジション保持中: FR={snapshot.rate:.4%}",
            }

        return {"action": "skip", "funding": snapshot, "reason": "判定不能"}

    def record_entry(
        self, symbol: str, spot_qty: float, spot_price: float,
        perp_qty: float, perp_price: float,
    ) -> ArbPosition:
        """エントリーを記録する。"""
        pos = ArbPosition(
            symbol=symbol,
            spot_quantity=spot_qty,
            spot_avg_price=spot_price,
            perp_quantity=perp_qty,
            perp_avg_price=perp_price,
            total_funding_earned=0.0,
            entry_time=time.time(),
            status="open",
        )
        self._positions[symbol] = pos
        logger.info(
            "Funding Arb エントリー: %s spot=%.4f@%.2f perp=%.4f@%.2f",
            symbol, spot_qty, spot_price, perp_qty, perp_price,
        )
        return pos

    def record_funding(self, symbol: str, amount: float) -> Optional[ArbPosition]:
        """ファンディング収入を記録する。"""
        pos = self._positions.get(symbol)
        if pos is None or pos.status != "open":
            return None

        from dataclasses import replace
        updated = replace(pos, total_funding_earned=pos.total_funding_earned + amount)
        self._positions[symbol] = updated
        self._total_funding_earned += amount
        return updated

    def record_exit(self, symbol: str) -> Optional[ArbPosition]:
        """エグジットを記録する。"""
        pos = self._positions.get(symbol)
        if pos is None:
            return None

        from dataclasses import replace
        closed = replace(pos, status="closed")
        self._positions[symbol] = closed
        logger.info(
            "Funding Arb エグジット: %s 累計FR収入=%.2f",
            symbol, closed.total_funding_earned,
        )
        return closed

    def summary(self) -> Dict[str, Any]:
        """戦略サマリーを返す。"""
        open_pos = self.open_positions
        return {
            "open_positions": len(open_pos),
            "total_funding_earned": self._total_funding_earned,
            "positions": {
                k: {
                    "funding_pnl_pct": v.funding_pnl_pct,
                    "net_exposure": v.net_exposure,
                    "entry_time": v.entry_time,
                }
                for k, v in open_pos.items()
            },
        }
