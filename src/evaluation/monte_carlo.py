"""モンテカルロシミュレーション。

取引リターン列をシャッフルして1000回シミュレーションし、
戦略の堅牢性と信頼区間を検証する。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    """モンテカルロシミュレーション結果（immutable）。"""
    num_simulations: int
    original_return_pct: float        # オリジナル順序でのリターン（%）
    mean_return_pct: float            # シミュレーション平均リターン（%）
    median_return_pct: float          # シミュレーション中央値リターン（%）
    std_return_pct: float             # シミュレーション標準偏差（%）
    percentile_5: float               # 5パーセンタイル（%）
    percentile_25: float              # 25パーセンタイル（%）
    percentile_75: float              # 75パーセンタイル（%）
    percentile_95: float              # 95パーセンタイル（%）
    probability_of_loss: float        # 損失確率（P(return < 0)）
    probability_of_ruin: float        # 破産確率（P(最大DD > 50%)）
    mean_max_drawdown_pct: float      # 平均最大ドローダウン（%）
    worst_case_return_pct: float      # 最悪ケースリターン（%）
    best_case_return_pct: float       # 最良ケースリターン（%）
    confidence_95_lower: float        # 95%信頼区間下限（2.5th %ile）
    confidence_95_upper: float        # 95%信頼区間上限（97.5th %ile）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _empty_result() -> MonteCarloResult:
    """取引なし/エラー時のデフォルト結果。"""
    return MonteCarloResult(
        num_simulations=0,
        original_return_pct=0.0,
        mean_return_pct=0.0,
        median_return_pct=0.0,
        std_return_pct=0.0,
        percentile_5=0.0,
        percentile_25=0.0,
        percentile_75=0.0,
        percentile_95=0.0,
        probability_of_loss=0.0,
        probability_of_ruin=0.0,
        mean_max_drawdown_pct=0.0,
        worst_case_return_pct=0.0,
        best_case_return_pct=0.0,
        confidence_95_lower=0.0,
        confidence_95_upper=0.0,
    )


def monte_carlo_simulation(
    trade_returns: Sequence[float],
    num_simulations: int = 1000,
    initial_capital: float = 1_000_000.0,
    seed: int = 42,
) -> MonteCarloResult:
    """モンテカルロシミュレーションを実行する。

    取引リターン列をランダムシャッフルして、
    戦略パフォーマンスの堅牢性を評価する。

    Parameters
    ----------
    trade_returns:
        取引ごとのリターン率の系列（例: [0.02, -0.01, ...]）
        各値は 1取引の pnl / entry_value
    num_simulations:
        シミュレーション回数（デフォルト: 1000）
    initial_capital:
        初期資金（エクイティカーブ計算用）
    seed:
        乱数シード（再現性確保）

    Returns
    -------
    MonteCarloResult
        シミュレーション結果
    """
    returns = np.array(trade_returns, dtype=np.float64)

    if len(returns) == 0:
        return _empty_result()

    rng = np.random.default_rng(seed)

    # オリジナル順序のパフォーマンス
    original_equity = initial_capital * np.cumprod(1.0 + returns)
    original_return = float((original_equity[-1] / initial_capital - 1) * 100)

    # シミュレーション配列（事前確保でパフォーマンス向上）
    sim_returns = np.empty(num_simulations, dtype=np.float64)
    sim_max_dds = np.empty(num_simulations, dtype=np.float64)

    for i in range(num_simulations):
        shuffled = rng.permutation(returns)
        equity = initial_capital * np.cumprod(1.0 + shuffled)

        sim_returns[i] = (equity[-1] / initial_capital - 1) * 100

        # 最大ドローダウン計算
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        sim_max_dds[i] = float(drawdown.min())

    return MonteCarloResult(
        num_simulations=num_simulations,
        original_return_pct=round(original_return, 4),
        mean_return_pct=round(float(np.mean(sim_returns)), 4),
        median_return_pct=round(float(np.median(sim_returns)), 4),
        std_return_pct=round(float(np.std(sim_returns)), 4),
        percentile_5=round(float(np.percentile(sim_returns, 5)), 4),
        percentile_25=round(float(np.percentile(sim_returns, 25)), 4),
        percentile_75=round(float(np.percentile(sim_returns, 75)), 4),
        percentile_95=round(float(np.percentile(sim_returns, 95)), 4),
        probability_of_loss=round(float(np.mean(sim_returns < 0)), 4),
        probability_of_ruin=round(float(np.mean(sim_max_dds < -50)), 4),
        mean_max_drawdown_pct=round(float(np.mean(sim_max_dds)), 4),
        worst_case_return_pct=round(float(np.min(sim_returns)), 4),
        best_case_return_pct=round(float(np.max(sim_returns)), 4),
        confidence_95_lower=round(float(np.percentile(sim_returns, 2.5)), 4),
        confidence_95_upper=round(float(np.percentile(sim_returns, 97.5)), 4),
    )
