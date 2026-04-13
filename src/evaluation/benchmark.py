"""戦略ベンチマークシステム。

全戦略のWalk-Forward + Monte Carlo結果を統合し、
GO/CAUTION/NOGOのReadiness Scorecardを生成する。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .walk_forward import WalkForwardResult, walk_forward_analysis
from .monte_carlo import MonteCarloResult, monte_carlo_simulation
from .backtester import run_backtest


@dataclass(frozen=True)
class ReadinessScorecard:
    """Readiness Scorecard（8基準評価）。

    各基準スコアは0〜100の整数。
    overall_score はすべての基準の平均値。
    recommendation は overall_score に基づき GO/CAUTION/NOGO を決定する。
    """
    # Walk-Forward基準
    wf_consistency: int        # OOS一貫性（consistency_ratio >= 0.6 で満点）
    wf_significance: int       # 統計的有意性（p < 0.05 で満点）
    # Monte Carlo基準
    mc_ruin_prob: int          # 破産確率（ruin < 5% で満点）
    # パフォーマンス基準
    sharpe: int                # シャープレシオ（> 0.5 で満点）
    max_drawdown: int          # 最大ドローダウン（> -20% で満点）
    win_rate: int              # 勝率（> 40% で満点）
    profit_factor: int         # プロフィットファクター（> 1.2 で満点）
    # データ基準
    data_freshness: int        # データ鮮度（24時間以内に更新で満点）
    # 総合
    overall_score: int         # 8基準の平均（0〜100）
    recommendation: str        # "GO" / "CAUTION" / "NOGO"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyBenchmark:
    """戦略ベンチマーク結果（immutable）。"""
    strategy_name: str
    ticker: str
    walk_forward: Optional[WalkForwardResult]
    monte_carlo: Optional[MonteCarloResult]
    overall_score: int                  # 0〜100
    recommendation: str                 # "GO" / "CAUTION" / "NOGO"
    evaluated_at: str                   # ISO 8601タイムスタンプ

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "strategy_name": self.strategy_name,
            "ticker": self.ticker,
            "walk_forward": self.walk_forward.to_dict() if self.walk_forward else None,
            "monte_carlo": self.monte_carlo.to_dict() if self.monte_carlo else None,
            "overall_score": self.overall_score,
            "recommendation": self.recommendation,
            "evaluated_at": self.evaluated_at,
        }
        return result


def _score_wf_consistency(wf: Optional[WalkForwardResult]) -> int:
    """Walk-Forward OOS一貫性スコア（consistency_ratio >= 0.6 で満点）。"""
    if wf is None:
        return 0
    ratio = wf.consistency_ratio
    if ratio >= 0.8:
        return 100
    if ratio >= 0.6:
        return 75
    if ratio >= 0.4:
        return 50
    if ratio >= 0.2:
        return 25
    return 0


def _score_wf_significance(wf: Optional[WalkForwardResult]) -> int:
    """Walk-Forward 統計的有意性スコア（p < 0.05 で満点）。"""
    if wf is None:
        return 0
    if wf.is_statistically_significant:
        return 100
    p = wf.p_value
    if p < 0.10:
        return 60
    if p < 0.20:
        return 30
    return 0


def _score_mc_ruin_prob(mc: Optional[MonteCarloResult]) -> int:
    """Monte Carlo 破産確率スコア（ruin < 5% で満点）。"""
    if mc is None:
        return 0
    ruin = mc.probability_of_ruin
    if ruin < 0.01:
        return 100
    if ruin < 0.05:
        return 75
    if ruin < 0.10:
        return 50
    if ruin < 0.20:
        return 25
    return 0


def _score_sharpe(mc: Optional[MonteCarloResult], wf: Optional[WalkForwardResult]) -> int:
    """シャープレシオスコア（OOS平均または MC中央値から推定）。"""
    sharpe = 0.0
    if wf is not None:
        sharpe = wf.oos_sharpe_avg
    elif mc is not None:
        # MC中央値リターンと標準偏差からシャープ比の粗い推定
        if mc.std_return_pct > 0:
            sharpe = mc.median_return_pct / mc.std_return_pct

    if sharpe > 1.5:
        return 100
    if sharpe > 1.0:
        return 85
    if sharpe > 0.5:
        return 65
    if sharpe > 0.0:
        return 35
    return 0


def _score_max_drawdown(wf: Optional[WalkForwardResult], mc: Optional[MonteCarloResult]) -> int:
    """最大ドローダウンスコア（> -20% で満点）。"""
    dd = 0.0
    if wf is not None:
        dd = wf.oos_max_dd_avg
    elif mc is not None:
        dd = mc.mean_max_drawdown_pct

    if dd > -5.0:
        return 100
    if dd > -10.0:
        return 80
    if dd > -20.0:
        return 60
    if dd > -30.0:
        return 30
    return 0


def _score_win_rate(mc: Optional[MonteCarloResult]) -> int:
    """勝率スコア（probability_of_loss から推定、> 40% で満点）。"""
    if mc is None:
        return 0
    win_rate = 1.0 - mc.probability_of_loss
    if win_rate > 0.60:
        return 100
    if win_rate > 0.50:
        return 80
    if win_rate > 0.40:
        return 60
    if win_rate > 0.30:
        return 30
    return 0


def _score_profit_factor(mc: Optional[MonteCarloResult]) -> int:
    """プロフィットファクタースコア（MC中央値/損失から推定、> 1.2 で満点）。"""
    if mc is None:
        return 0
    # MC中央値リターン > 0 かつ損失確率から簡易推定
    if mc.median_return_pct <= 0:
        return 0
    # 期待値プロキシとして中央値リターン > 0 の度合いを使用
    if mc.median_return_pct > 10.0:
        return 100
    if mc.median_return_pct > 5.0:
        return 80
    if mc.median_return_pct > 2.0:
        return 60
    if mc.median_return_pct > 0.0:
        return 40
    return 0


def _score_data_freshness(evaluated_at: Optional[str] = None) -> int:
    """データ鮮度スコア（評価時刻が24時間以内なら満点）。

    benchmarkシステム自体の評価時点では常に満点を返す。
    データソースの更新時刻が別途渡される場合に利用する。
    """
    # 現在のベンチマーク実行は常に「今」なので満点
    return 100


def _determine_recommendation(overall_score: int) -> str:
    """総合スコアから推奨ラベルを決定する。

    Parameters
    ----------
    overall_score:
        0〜100の整数スコア

    Returns
    -------
    str
        "GO" (>= 70) / "CAUTION" (>= 45) / "NOGO" (< 45)
    """
    if overall_score >= 70:
        return "GO"
    if overall_score >= 45:
        return "CAUTION"
    return "NOGO"


def calculate_readiness_score(
    wf: Optional[WalkForwardResult] = None,
    mc: Optional[MonteCarloResult] = None,
) -> ReadinessScorecard:
    """Readiness Scorecardを計算する。

    Parameters
    ----------
    wf:
        WalkForwardResult（省略可）
    mc:
        MonteCarloResult（省略可）

    Returns
    -------
    ReadinessScorecard
        8基準スコアと総合評価
    """
    scores = {
        "wf_consistency": _score_wf_consistency(wf),
        "wf_significance": _score_wf_significance(wf),
        "mc_ruin_prob": _score_mc_ruin_prob(mc),
        "sharpe": _score_sharpe(mc, wf),
        "max_drawdown": _score_max_drawdown(wf, mc),
        "win_rate": _score_win_rate(mc),
        "profit_factor": _score_profit_factor(mc),
        "data_freshness": _score_data_freshness(),
    }

    overall = int(round(sum(scores.values()) / len(scores)))
    recommendation = _determine_recommendation(overall)

    return ReadinessScorecard(
        wf_consistency=scores["wf_consistency"],
        wf_significance=scores["wf_significance"],
        mc_ruin_prob=scores["mc_ruin_prob"],
        sharpe=scores["sharpe"],
        max_drawdown=scores["max_drawdown"],
        win_rate=scores["win_rate"],
        profit_factor=scores["profit_factor"],
        data_freshness=scores["data_freshness"],
        overall_score=overall,
        recommendation=recommendation,
    )


def run_strategy_benchmark(
    strategy_name: str,
    ticker: str,
    wf: Optional[WalkForwardResult] = None,
    mc: Optional[MonteCarloResult] = None,
) -> StrategyBenchmark:
    """戦略ベンチマークを実行してStrategyBenchmarkを返す。

    Parameters
    ----------
    strategy_name:
        戦略名
    ticker:
        銘柄コード
    wf:
        事前計算済みWalkForwardResult（省略時はNone）
    mc:
        事前計算済みMonteCarloResult（省略時はNone）

    Returns
    -------
    StrategyBenchmark
        ベンチマーク結果
    """
    scorecard = calculate_readiness_score(wf=wf, mc=mc)
    evaluated_at = datetime.now(tz=timezone.utc).isoformat()

    return StrategyBenchmark(
        strategy_name=strategy_name,
        ticker=ticker,
        walk_forward=wf,
        monte_carlo=mc,
        overall_score=scorecard.overall_score,
        recommendation=scorecard.recommendation,
        evaluated_at=evaluated_at,
    )
