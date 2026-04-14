"""Grid Trading 戦略。

一定の価格帯にグリッド状の買い/売り注文を配置し、
価格が上下するたびに利確を繰り返す。レンジ相場で最も効果的。

パラメータ:
  - upper_price: グリッド上限価格
  - lower_price: グリッド下限価格
  - grid_count: グリッド本数
  - total_investment: 投資総額
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GridLevel:
    """グリッドの1レベル（immutable）。"""
    price: float
    quantity: float
    side: str  # "buy" or "sell"
    filled: bool = False
    fill_price: float = 0.0


@dataclass(frozen=True)
class GridTrade:
    """グリッドで確定した取引（immutable）。"""
    buy_price: float
    sell_price: float
    quantity: float
    profit: float
    profit_pct: float
    timestamp: float = 0.0


class GridTradingStrategy:
    """グリッドトレーディング戦略。

    Parameters
    ----------
    upper_price:
        グリッド上限価格
    lower_price:
        グリッド下限価格
    grid_count:
        グリッド本数 (デフォルト: 10)
    total_investment:
        投資総額 (デフォルト: 100,000)
    """

    def __init__(
        self,
        upper_price: float,
        lower_price: float,
        grid_count: int = 10,
        total_investment: float = 100_000,
    ) -> None:
        if upper_price <= lower_price:
            raise ValueError("upper_price は lower_price より大きくしてください")
        if grid_count < 2:
            raise ValueError("grid_count は 2 以上にしてください")

        self._upper = upper_price
        self._lower = lower_price
        self._grid_count = grid_count
        self._total_investment = total_investment

        # グリッドレベルを計算
        self._step = (upper_price - lower_price) / grid_count
        self._quantity_per_grid = total_investment / grid_count / ((upper_price + lower_price) / 2)
        self._levels = self._build_levels()

        # 取引記録
        self._trades: List[GridTrade] = []
        self._total_profit: float = 0.0
        self._buy_fills: Dict[float, float] = {}  # price → quantity

    @property
    def grid_levels(self) -> List[float]:
        """グリッド価格レベルの一覧。"""
        return [self._lower + i * self._step for i in range(self._grid_count + 1)]

    @property
    def total_profit(self) -> float:
        return self._total_profit

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def trades(self) -> List[GridTrade]:
        return list(self._trades)

    def _build_levels(self) -> List[GridLevel]:
        levels = []
        for i in range(self._grid_count + 1):
            price = self._lower + i * self._step
            levels.append(GridLevel(
                price=round(price, 8),
                quantity=self._quantity_per_grid,
                side="buy",
            ))
        return levels

    def on_price_update(self, current_price: float, timestamp: float = 0.0) -> List[GridTrade]:
        """価格更新時の処理。グリッドレベルを超えたら取引を記録する。

        Parameters
        ----------
        current_price:
            現在価格
        timestamp:
            Unix timestamp

        Returns
        -------
        List[GridTrade]
            この更新で確定した取引のリスト
        """
        new_trades: List[GridTrade] = []

        for level_price in self.grid_levels:
            # 価格がグリッドレベルを下回った → 買い
            if current_price <= level_price and level_price not in self._buy_fills:
                self._buy_fills[level_price] = current_price
                logger.debug("Grid BUY at %.2f (level=%.2f)", current_price, level_price)

            # 価格がグリッドレベル + ステップを上回った → 対応する買いの利確
            sell_trigger = level_price + self._step
            if current_price >= sell_trigger and level_price in self._buy_fills:
                buy_price = self._buy_fills.pop(level_price)
                profit = (current_price - buy_price) * self._quantity_per_grid
                profit_pct = (current_price / buy_price - 1) * 100

                trade = GridTrade(
                    buy_price=buy_price,
                    sell_price=current_price,
                    quantity=self._quantity_per_grid,
                    profit=profit,
                    profit_pct=profit_pct,
                    timestamp=timestamp,
                )
                self._trades.append(trade)
                self._total_profit += profit
                new_trades.append(trade)
                logger.debug(
                    "Grid SELL at %.2f (bought=%.2f profit=%.2f)",
                    current_price, buy_price, profit,
                )

        return new_trades

    def backtest(self, prices: List[float], timestamps: Optional[List[float]] = None) -> Dict[str, Any]:
        """価格シリーズでバックテストを実行する。

        Parameters
        ----------
        prices:
            価格の時系列リスト
        timestamps:
            対応するタイムスタンプ（省略時は連番）

        Returns
        -------
        Dict[str, Any]
            バックテスト結果
        """
        if timestamps is None:
            timestamps = list(range(len(prices)))

        self._trades = []
        self._total_profit = 0.0
        self._buy_fills = {}

        for price, ts in zip(prices, timestamps):
            self.on_price_update(price, ts)

        # Buy&Hold 比較
        if len(prices) >= 2:
            bh_return = (prices[-1] / prices[0] - 1) * 100
            bh_pnl = self._total_investment * (prices[-1] / prices[0] - 1)
        else:
            bh_return = 0.0
            bh_pnl = 0.0

        grid_return = self._total_profit / self._total_investment * 100

        return {
            "total_profit": round(self._total_profit, 2),
            "total_trades": self.trade_count,
            "grid_return_pct": round(grid_return, 2),
            "buy_hold_return_pct": round(bh_return, 2),
            "buy_hold_pnl": round(bh_pnl, 2),
            "alpha_pct": round(grid_return - bh_return, 2),
            "avg_profit_per_trade": round(
                self._total_profit / self.trade_count if self.trade_count > 0 else 0, 2
            ),
            "grid_levels": len(self.grid_levels),
            "grid_step": round(self._step, 2),
            "unfilled_buys": len(self._buy_fills),
        }

    def summary(self) -> Dict[str, Any]:
        """現在の状態サマリー。"""
        return {
            "upper": self._upper,
            "lower": self._lower,
            "grid_count": self._grid_count,
            "step": round(self._step, 2),
            "total_investment": self._total_investment,
            "total_profit": round(self._total_profit, 2),
            "trade_count": self.trade_count,
            "open_buys": len(self._buy_fills),
        }
