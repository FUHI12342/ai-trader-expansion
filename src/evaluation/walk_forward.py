"""Walk-Forward Analysis（ウォークフォワード分析）。

in-sample 504日 / out-of-sample 126日のローリングウィンドウで
戦略の汎化性能を検証する。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .backtester import run_backtest, BacktestResult
from ..strategies.base import BaseStrategy


@dataclass(frozen=True)
class WalkForwardResult:
    """ウォークフォワード分析結果（immutable）。"""
    num_walks: int
    in_sample_avg_return: float       # IS平均リターン（%）
    out_of_sample_avg_return: float   # OOS平均リターン（%）
    oos_sharpe_avg: float             # OOS平均シャープレシオ
    oos_max_dd_avg: float             # OOS平均最大ドローダウン（%）
    consistency_ratio: float          # OOSでプラスリターンの割合（0〜1）
    degradation_ratio: float          # IS/OOSリターン比（1に近いほど良い）
    is_statistically_significant: bool
    p_value: float
    walks: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def walk_forward_analysis(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    symbol: str = "UNKNOWN",
    in_sample_days: int = 504,
    out_of_sample_days: int = 126,
    step_days: Optional[int] = None,
    initial_capital: float = 1_000_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    min_walks: int = 3,
) -> WalkForwardResult:
    """ウォークフォワード分析を実行する。

    Parameters
    ----------
    strategy:
        検証する戦略インスタンス
    data:
        フルOHLCV DataFrame（DatetimeIndex）
    symbol:
        銘柄コード/名称
    in_sample_days:
        インサンプル期間（日数、デフォルト: 504 ≈ 2年）
    out_of_sample_days:
        アウトオブサンプル期間（日数、デフォルト: 126 ≈ 半年）
    step_days:
        ウィンドウスライド幅（省略時はout_of_sample_daysと同じ）
    initial_capital:
        初期資金
    fee_rate:
        片道手数料率
    slippage_rate:
        スリッページ率
    min_walks:
        最低ウォーク数（足りない場合はValueError）

    Returns
    -------
    WalkForwardResult
    """
    if step_days is None:
        step_days = out_of_sample_days

    total_len = len(data)
    min_required = in_sample_days + out_of_sample_days
    if total_len < min_required:
        raise ValueError(
            f"データが不足しています。必要: {min_required}日, 実際: {total_len}日"
        )

    walks: List[Dict[str, Any]] = []
    start_idx = 0
    walk_num = 0

    while start_idx + in_sample_days + out_of_sample_days <= total_len:
        is_end = start_idx + in_sample_days
        oos_end = min(is_end + out_of_sample_days, total_len)

        is_data = data.iloc[start_idx:is_end].copy()
        oos_data = data.iloc[is_end:oos_end].copy()

        # ISバックテスト
        try:
            is_result = run_backtest(
                strategy=strategy,
                data=is_data,
                symbol=symbol,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
        except Exception as e:
            is_result = None

        # OOSバックテスト
        try:
            oos_result = run_backtest(
                strategy=strategy,
                data=oos_data,
                symbol=symbol,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
        except Exception as e:
            oos_result = None

        walk_entry: Dict[str, Any] = {
            "walk": walk_num,
            "is_start": str(is_data.index[0]),
            "is_end": str(is_data.index[-1]),
            "oos_start": str(oos_data.index[0]),
            "oos_end": str(oos_data.index[-1]),
            "is_return_pct": is_result.metrics.total_return_pct if is_result else 0.0,
            "oos_return_pct": oos_result.metrics.total_return_pct if oos_result else 0.0,
            "oos_sharpe": oos_result.metrics.sharpe_ratio if oos_result else 0.0,
            "oos_max_dd": oos_result.metrics.max_drawdown_pct if oos_result else 0.0,
        }
        walks.append(walk_entry)

        start_idx += step_days
        walk_num += 1

    if len(walks) < min_walks:
        raise ValueError(
            f"ウォーク数が不足しています。必要: {min_walks}, 実際: {len(walks)}"
        )

    # 統計集計
    is_returns = [w["is_return_pct"] for w in walks]
    oos_returns = [w["oos_return_pct"] for w in walks]
    oos_sharpes = [w["oos_sharpe"] for w in walks]
    oos_max_dds = [w["oos_max_dd"] for w in walks]

    is_avg = float(np.mean(is_returns))
    oos_avg = float(np.mean(oos_returns))
    consistency = float(np.mean([r > 0 for r in oos_returns]))
    degradation = float(oos_avg / is_avg) if is_avg != 0 else 0.0

    # t検定（帰無仮説: OOSリターン平均 = 0）
    if len(oos_returns) >= 2:
        t_stat, p_value = stats.ttest_1samp(oos_returns, 0)
        is_significant = bool(p_value < 0.05 and oos_avg > 0)
    else:
        p_value = 1.0
        is_significant = False

    return WalkForwardResult(
        num_walks=len(walks),
        in_sample_avg_return=round(is_avg, 4),
        out_of_sample_avg_return=round(oos_avg, 4),
        oos_sharpe_avg=round(float(np.mean(oos_sharpes)), 4),
        oos_max_dd_avg=round(float(np.mean(oos_max_dds)), 4),
        consistency_ratio=round(consistency, 4),
        degradation_ratio=round(degradation, 4),
        is_statistically_significant=is_significant,
        p_value=round(float(p_value), 6),
        walks=walks,
    )
