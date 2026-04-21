"""Aave V3 レンディングシミュレーター。

実接続 (Web3/ウォレット) はせず、指定 APY で日次複利を計算するペーパー実装。
待機資金を USDC で運用する想定。

リアル接続は将来 WalletConnect + Aave SDK で拡張可能。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepositResult:
    """預入結果（immutable）。"""

    amount: float
    new_balance: float
    timestamp: str
    success: bool
    reason: str = ""


@dataclass(frozen=True)
class WithdrawResult:
    """引出結果（immutable）。"""

    amount: float
    new_balance: float
    timestamp: str
    success: bool
    reason: str = ""


@dataclass(frozen=True)
class AavePosition:
    """Aave ポジションスナップショット（immutable）。"""

    principal: float
    accrued_interest: float
    total_balance: float
    apy: float
    last_accrual: str


class AaveSimulator:
    """Aave V3 USDC レンディング模擬。

    日次複利で利息を累積する。実APY 3-7% の範囲で設定可能。

    Parameters
    ----------
    apy:
        年率 (例: 0.05 = 5%)。実Aave V3 USDC: 4-7% (2026年)
    initial_balance:
        初期残高
    compound_daily:
        True なら日次複利、False なら単利
    min_deposit:
        最小預入額 (USDC)
    """

    def __init__(
        self,
        apy: float = 0.05,
        initial_balance: float = 0.0,
        compound_daily: bool = True,
        min_deposit: float = 1.0,
    ) -> None:
        if apy < 0 or apy > 0.5:
            raise ValueError(f"APY must be 0.0〜0.5, got {apy}")
        self._apy = apy
        self._principal = initial_balance
        self._accrued_interest = 0.0
        self._compound_daily = compound_daily
        self._min_deposit = min_deposit
        self._last_accrual: datetime = datetime.now()
        self._deposit_history: List[DepositResult] = []
        self._withdraw_history: List[WithdrawResult] = []

    @property
    def apy(self) -> float:
        return self._apy

    @property
    def balance(self) -> float:
        """元本+累積利息の合計（利息は未accrueでも返す）。"""
        return self._principal + self._accrued_interest

    def deposit(self, amount: float, now: Optional[datetime] = None) -> DepositResult:
        """USDC を預け入れる。

        Parameters
        ----------
        amount:
            預入額 (USDC)
        now:
            現在時刻 (テスト用、省略時は datetime.now())
        """
        now = now or datetime.now()
        ts = now.isoformat()

        if amount < self._min_deposit:
            result = DepositResult(
                amount=0.0, new_balance=self.balance, timestamp=ts,
                success=False, reason=f"最小預入額 {self._min_deposit} 未満",
            )
            self._deposit_history.append(result)
            return result

        # 先に accrue してから預入
        self.accrue_interest(now)
        self._principal += amount
        result = DepositResult(
            amount=amount, new_balance=self.balance, timestamp=ts, success=True,
        )
        self._deposit_history.append(result)
        logger.info(f"Aave 預入: +{amount:.2f} USDC → 残高 {self.balance:.2f}")
        return result

    def withdraw(self, amount: float, now: Optional[datetime] = None) -> WithdrawResult:
        """USDC を引き出す。

        先に利息を accrue し、元本+利息の合計から引き出す。
        利息優先で減らし、足りなければ元本から。
        """
        now = now or datetime.now()
        ts = now.isoformat()

        self.accrue_interest(now)

        if amount <= 0:
            result = WithdrawResult(
                amount=0.0, new_balance=self.balance, timestamp=ts,
                success=False, reason="引出額が0以下",
            )
            self._withdraw_history.append(result)
            return result

        if amount > self.balance:
            result = WithdrawResult(
                amount=0.0, new_balance=self.balance, timestamp=ts,
                success=False, reason=f"残高不足: {amount:.2f} > {self.balance:.2f}",
            )
            self._withdraw_history.append(result)
            return result

        # 利息優先で減らす
        if amount <= self._accrued_interest:
            self._accrued_interest -= amount
        else:
            remainder = amount - self._accrued_interest
            self._accrued_interest = 0.0
            self._principal -= remainder

        result = WithdrawResult(
            amount=amount, new_balance=self.balance, timestamp=ts, success=True,
        )
        self._withdraw_history.append(result)
        logger.info(f"Aave 引出: -{amount:.2f} USDC → 残高 {self.balance:.2f}")
        return result

    def accrue_interest(self, now: Optional[datetime] = None) -> float:
        """最後の accrue からの経過時間に応じて利息を加算する。

        Returns
        -------
        float
            今回累積した利息額
        """
        now = now or datetime.now()
        elapsed_days = (now - self._last_accrual).total_seconds() / 86400.0
        if elapsed_days <= 0:
            return 0.0

        base = self._principal + self._accrued_interest
        if self._compound_daily:
            # 日次複利: balance * ((1 + apy)^(days/365) - 1)
            rate = math.pow(1.0 + self._apy, elapsed_days / 365.0) - 1.0
        else:
            # 単利: balance * apy * days/365
            rate = self._apy * elapsed_days / 365.0

        new_interest = base * rate
        self._accrued_interest += new_interest
        self._last_accrual = now
        return new_interest

    def snapshot(self, now: Optional[datetime] = None) -> AavePosition:
        """現在のポジションスナップショットを返す（accrue込み）。"""
        now = now or datetime.now()
        self.accrue_interest(now)
        return AavePosition(
            principal=round(self._principal, 6),
            accrued_interest=round(self._accrued_interest, 6),
            total_balance=round(self.balance, 6),
            apy=self._apy,
            last_accrual=self._last_accrual.isoformat(),
        )

    def reset(self) -> None:
        """内部状態をリセットする（テスト用）。"""
        self._principal = 0.0
        self._accrued_interest = 0.0
        self._last_accrual = datetime.now()
        self._deposit_history.clear()
        self._withdraw_history.clear()
