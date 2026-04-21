"""待機資金マネージャー — アクティブポジション以外を DeFi に回す。

FLAT 比率 (未使用資金/総資金) を監視し、閾値を超えたら Aave に預入、
新規エントリー時に引出のシグナルを生成する。

実ブローカーとの連携は上位層 (AutoTrader / TradingLoop) が担う。
このクラスは判断ロジックのみ提供する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.defi.aave_simulator import AaveSimulator

logger = logging.getLogger(__name__)


class RebalanceActionType(str, Enum):
    """リバランスアクション種別。"""

    DEPOSIT = "deposit"       # DeFiに預入
    WITHDRAW = "withdraw"     # DeFiから引出
    HOLD = "hold"             # 何もしない


@dataclass(frozen=True)
class RebalanceAction:
    """リバランス判断結果（immutable）。"""

    action: RebalanceActionType
    amount: float
    reason: str
    flat_ratio: float


class WaitingCapitalManager:
    """待機資金リバランスマネージャー。

    Parameters
    ----------
    aave:
        AaveSimulator インスタンス
    flat_threshold_deposit:
        この比率以上FLATなら預入 (デフォルト 0.5 = 50%)
    flat_threshold_withdraw:
        この比率未満FLATなら引出 (デフォルト 0.2 = 20%)
    buffer_pct:
        即時エントリー用に手元に残す比率 (デフォルト 0.1 = 10%)
    min_rebalance_amount:
        最小リバランス額 (デフォルト 10.0 USDC)
    """

    def __init__(
        self,
        aave: AaveSimulator,
        flat_threshold_deposit: float = 0.5,
        flat_threshold_withdraw: float = 0.2,
        buffer_pct: float = 0.1,
        min_rebalance_amount: float = 10.0,
    ) -> None:
        if flat_threshold_deposit <= flat_threshold_withdraw:
            raise ValueError(
                "flat_threshold_deposit は flat_threshold_withdraw より大きい必要あり"
            )
        if not (0.0 <= buffer_pct < 1.0):
            raise ValueError(f"buffer_pct は 0.0〜1.0 の範囲、got {buffer_pct}")
        self._aave = aave
        self._deposit_th = flat_threshold_deposit
        self._withdraw_th = flat_threshold_withdraw
        self._buffer_pct = buffer_pct
        self._min_amount = min_rebalance_amount

    @property
    def aave(self) -> AaveSimulator:
        return self._aave

    def decide(
        self,
        cash_balance: float,
        active_position_value: float,
        total_capital: float,
    ) -> RebalanceAction:
        """リバランスすべきかを判断する。

        Parameters
        ----------
        cash_balance:
            ブローカー口座の現金残高
        active_position_value:
            現在のアクティブポジションの評価額
        total_capital:
            総資金 (cash + positions + aave balance)

        Returns
        -------
        RebalanceAction
            アクション (deposit/withdraw/hold) と金額
        """
        if total_capital <= 0:
            return RebalanceAction(
                action=RebalanceActionType.HOLD, amount=0.0,
                reason="total_capital が 0 以下", flat_ratio=0.0,
            )

        flat_ratio = cash_balance / total_capital

        # FLAT 比率が高い → DeFi に預ける
        if flat_ratio >= self._deposit_th:
            # buffer を手元に残す
            buffer = total_capital * self._buffer_pct
            deposit_amount = cash_balance - buffer
            if deposit_amount < self._min_amount:
                return RebalanceAction(
                    action=RebalanceActionType.HOLD, amount=0.0,
                    reason=f"預入額 {deposit_amount:.2f} < 最小 {self._min_amount}",
                    flat_ratio=flat_ratio,
                )
            return RebalanceAction(
                action=RebalanceActionType.DEPOSIT,
                amount=round(deposit_amount, 2),
                reason=f"FLAT比率 {flat_ratio:.1%} >= 閾値 {self._deposit_th:.1%}",
                flat_ratio=flat_ratio,
            )

        # FLAT 比率が低い → DeFi から引き出す
        if flat_ratio < self._withdraw_th:
            aave_balance = self._aave.balance
            if aave_balance < self._min_amount:
                return RebalanceAction(
                    action=RebalanceActionType.HOLD, amount=0.0,
                    reason=f"Aave残高 {aave_balance:.2f} < 最小 {self._min_amount}",
                    flat_ratio=flat_ratio,
                )
            # 目標: flat_ratio が deposit_th 以下になるよう引出
            target_cash = total_capital * self._withdraw_th
            shortfall = target_cash - cash_balance
            withdraw_amount = min(shortfall, aave_balance)
            if withdraw_amount < self._min_amount:
                return RebalanceAction(
                    action=RebalanceActionType.HOLD, amount=0.0,
                    reason=f"引出額 {withdraw_amount:.2f} < 最小 {self._min_amount}",
                    flat_ratio=flat_ratio,
                )
            return RebalanceAction(
                action=RebalanceActionType.WITHDRAW,
                amount=round(withdraw_amount, 2),
                reason=f"FLAT比率 {flat_ratio:.1%} < 閾値 {self._withdraw_th:.1%}",
                flat_ratio=flat_ratio,
            )

        return RebalanceAction(
            action=RebalanceActionType.HOLD, amount=0.0,
            reason=f"FLAT比率 {flat_ratio:.1%} は閾値内",
            flat_ratio=flat_ratio,
        )

    def expected_annual_yield(self, avg_flat_ratio: float, total_capital: float) -> float:
        """期待年間利回り (待機資金 × APY × 運用比率)。

        Parameters
        ----------
        avg_flat_ratio:
            平均FLAT比率 (0.0〜1.0)
        total_capital:
            総資金

        Returns
        -------
        float
            期待年間利息額
        """
        waiting_capital = total_capital * avg_flat_ratio
        deployed = max(0.0, waiting_capital - total_capital * self._buffer_pct)
        return deployed * self._aave.apy
