"""統計的有意性テストモジュール。

t検定（帰無仮説: mean return = 0）とブートストラップ信頼区間を提供する。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticsResult:
    """統計検定結果（immutable）。"""
    # t検定結果
    t_statistic: float
    p_value_ttest: float
    is_significant_ttest: bool        # p < 0.05 かつ mean > 0

    # ブートストラップ信頼区間
    bootstrap_mean: float
    bootstrap_std: float
    bootstrap_ci_lower_95: float     # 95%信頼区間下限
    bootstrap_ci_upper_95: float     # 95%信頼区間上限
    bootstrap_ci_lower_99: float     # 99%信頼区間下限
    bootstrap_ci_upper_99: float     # 99%信頼区間上限

    # 基本統計量
    sample_mean: float
    sample_std: float
    sample_size: int
    positive_rate: float              # プラスリターンの割合

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _empty_statistics_result() -> StatisticsResult:
    """空データ時のデフォルト結果。"""
    return StatisticsResult(
        t_statistic=0.0,
        p_value_ttest=1.0,
        is_significant_ttest=False,
        bootstrap_mean=0.0,
        bootstrap_std=0.0,
        bootstrap_ci_lower_95=0.0,
        bootstrap_ci_upper_95=0.0,
        bootstrap_ci_lower_99=0.0,
        bootstrap_ci_upper_99=0.0,
        sample_mean=0.0,
        sample_std=0.0,
        sample_size=0,
        positive_rate=0.0,
    )


def run_statistical_tests(
    returns: Sequence[float],
    num_bootstrap: int = 10000,
    seed: int = 42,
) -> StatisticsResult:
    """統計的有意性テストを実行する。

    Parameters
    ----------
    returns:
        取引リターン率または日次リターン系列
    num_bootstrap:
        ブートストラップサンプル数（デフォルト: 10000）
    seed:
        乱数シード

    Returns
    -------
    StatisticsResult
        t検定とブートストラップの結果
    """
    arr = np.array(returns, dtype=np.float64)

    if len(arr) < 2:
        return _empty_statistics_result()

    # ------------------------------------------------------------------
    # t検定（帰無仮説: mean = 0）
    # ------------------------------------------------------------------
    t_stat, p_value = stats.ttest_1samp(arr, 0)
    is_significant = bool(p_value < 0.05 and float(arr.mean()) > 0)

    # ------------------------------------------------------------------
    # ブートストラップ信頼区間
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_bootstrap, dtype=np.float64)

    for i in range(num_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means[i] = sample.mean()

    return StatisticsResult(
        t_statistic=round(float(t_stat), 6),
        p_value_ttest=round(float(p_value), 6),
        is_significant_ttest=is_significant,
        bootstrap_mean=round(float(np.mean(bootstrap_means)), 6),
        bootstrap_std=round(float(np.std(bootstrap_means)), 6),
        bootstrap_ci_lower_95=round(float(np.percentile(bootstrap_means, 2.5)), 6),
        bootstrap_ci_upper_95=round(float(np.percentile(bootstrap_means, 97.5)), 6),
        bootstrap_ci_lower_99=round(float(np.percentile(bootstrap_means, 0.5)), 6),
        bootstrap_ci_upper_99=round(float(np.percentile(bootstrap_means, 99.5)), 6),
        sample_mean=round(float(arr.mean()), 6),
        sample_std=round(float(arr.std()), 6),
        sample_size=len(arr),
        positive_rate=round(float(np.mean(arr > 0)), 4),
    )
