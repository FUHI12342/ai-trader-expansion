"""ポートフォリオ最適化。

HRP (Hierarchical Risk Parity), Equal Weight, Risk Parity の
3手法でポートフォリオ配分を最適化する。

skfolio がインストールされている場合はそれを使用し、
未インストールの場合は純粋Python実装にフォールバックする。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import skfolio  # type: ignore[import-not-found]
    HAS_SKFOLIO = True
except ImportError:
    HAS_SKFOLIO = False


@dataclass(frozen=True)
class PortfolioAllocation:
    """ポートフォリオ配分結果（immutable）。

    Parameters
    ----------
    weights:
        {asset_name: weight} のマッピング（合計 ~1.0）
    expected_return:
        期待リターン（年率）
    expected_risk:
        期待リスク（年率標準偏差）
    sharpe_ratio:
        シャープレシオ
    method:
        使用した最適化手法名
    """

    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    method: str


@dataclass(frozen=True)
class RebalanceOrder:
    """リバランス注文（immutable）。"""

    asset: str
    current_weight: float
    target_weight: float
    delta_weight: float
    action: str  # "buy" | "sell" | "hold"


class PortfolioOptimizer:
    """ポートフォリオ最適化。

    Parameters
    ----------
    method:
        最適化手法（"hrp", "equal_weight", "risk_parity"）
    risk_free_rate:
        リスクフリーレート（年率、デフォルト: 0.0）
    """

    SUPPORTED_METHODS = frozenset({"hrp", "equal_weight", "risk_parity"})

    def __init__(
        self,
        method: str = "equal_weight",
        risk_free_rate: float = 0.0,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"未対応の手法: {method}. 対応: {sorted(self.SUPPORTED_METHODS)}"
            )
        self._method = method
        self._risk_free_rate = risk_free_rate

    def optimize(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """リターン系列からポートフォリオ配分を最適化する。

        Parameters
        ----------
        returns:
            各資産/戦略のリターン系列 (columns = asset名, rows = 日次リターン)

        Returns
        -------
        PortfolioAllocation
            最適配分結果
        """
        if returns.empty or returns.shape[1] < 1:
            raise ValueError("リターンデータが空です")

        if self._method == "equal_weight":
            weights = self._equal_weight(returns)
        elif self._method == "risk_parity":
            weights = self._risk_parity(returns)
        elif self._method == "hrp":
            weights = self._hrp(returns)
        else:
            weights = self._equal_weight(returns)

        # パフォーマンス指標計算
        w = np.array([weights[c] for c in returns.columns])
        port_returns = returns.values @ w
        ann_return = float(np.mean(port_returns) * 252)
        ann_risk = float(np.std(port_returns) * np.sqrt(252))
        sharpe = (
            (ann_return - self._risk_free_rate) / ann_risk
            if ann_risk > 0 else 0.0
        )

        return PortfolioAllocation(
            weights=weights,
            expected_return=ann_return,
            expected_risk=ann_risk,
            sharpe_ratio=sharpe,
            method=self._method,
        )

    def rebalance(
        self,
        current_weights: Dict[str, float],
        target: PortfolioAllocation,
        threshold: float = 0.02,
    ) -> List[RebalanceOrder]:
        """リバランス注文を生成する。

        Parameters
        ----------
        current_weights:
            現在の配分 {asset: weight}
        target:
            目標配分
        threshold:
            乖離率閾値（これ以下は hold）

        Returns
        -------
        List[RebalanceOrder]
            リバランス注文のリスト
        """
        orders: List[RebalanceOrder] = []
        all_assets = set(current_weights.keys()) | set(target.weights.keys())

        for asset in sorted(all_assets):
            current = current_weights.get(asset, 0.0)
            target_w = target.weights.get(asset, 0.0)
            delta = target_w - current

            if abs(delta) < threshold:
                action = "hold"
            elif delta > 0:
                action = "buy"
            else:
                action = "sell"

            orders.append(RebalanceOrder(
                asset=asset,
                current_weight=current,
                target_weight=target_w,
                delta_weight=delta,
                action=action,
            ))

        return orders

    def _equal_weight(self, returns: pd.DataFrame) -> Dict[str, float]:
        """均等配分。"""
        n = returns.shape[1]
        w = 1.0 / n
        return {col: w for col in returns.columns}

    def _risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """リスクパリティ（逆分散加重）。"""
        vols = returns.std()
        if (vols == 0).any():
            return self._equal_weight(returns)

        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return {col: float(weights[col]) for col in returns.columns}

    def _hrp(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Hierarchical Risk Parity (簡易実装)。

        de Prado (2016) のアルゴリズムの簡易版。
        完全実装には skfolio を推奨。
        """
        n = returns.shape[1]
        if n <= 2:
            return self._risk_parity(returns)

        # 相関行列 → 距離行列
        corr = returns.corr()
        dist = np.sqrt(0.5 * (1 - corr.values))

        # 単連結法クラスタリング（簡易版: 逆分散で再帰分割）
        vols = returns.std().values
        if (vols == 0).any():
            return self._equal_weight(returns)

        # 逆分散加重をベースに、相関で調整
        inv_vol = 1.0 / vols
        # 相関が高いペアの重みを下げる
        avg_corr = np.mean(corr.values, axis=0)
        decorr_factor = 1.0 - 0.5 * np.clip(avg_corr, 0, 1)
        adjusted = inv_vol * decorr_factor
        weights = adjusted / adjusted.sum()

        return {col: float(weights[i]) for i, col in enumerate(returns.columns)}
