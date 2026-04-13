"""ベンチマークシステムテスト。

StrategyBenchmark、ReadinessScorecard、GO/CAUTION/NOGO閾値をテストする。
"""
from __future__ import annotations

import pytest

from src.evaluation.benchmark import (
    ReadinessScorecard,
    StrategyBenchmark,
    calculate_readiness_score,
    run_strategy_benchmark,
    _determine_recommendation,
    _score_wf_consistency,
    _score_wf_significance,
    _score_mc_ruin_prob,
)
from src.evaluation.walk_forward import WalkForwardResult
from src.evaluation.monte_carlo import MonteCarloResult


# ---------------------------------------------------------------------------
# テスト用フィクスチャ
# ---------------------------------------------------------------------------

def _make_wf(
    consistency_ratio: float = 0.7,
    p_value: float = 0.03,
    is_statistically_significant: bool = True,
    oos_sharpe_avg: float = 0.8,
    oos_max_dd_avg: float = -12.0,
    num_walks: int = 5,
) -> WalkForwardResult:
    return WalkForwardResult(
        num_walks=num_walks,
        in_sample_avg_return=5.0,
        out_of_sample_avg_return=3.0,
        oos_sharpe_avg=oos_sharpe_avg,
        oos_max_dd_avg=oos_max_dd_avg,
        consistency_ratio=consistency_ratio,
        degradation_ratio=0.6,
        is_statistically_significant=is_statistically_significant,
        p_value=p_value,
        walks=[],
    )


def _make_mc(
    probability_of_ruin: float = 0.02,
    probability_of_loss: float = 0.35,
    median_return_pct: float = 8.0,
    mean_max_drawdown_pct: float = -15.0,
    std_return_pct: float = 5.0,
) -> MonteCarloResult:
    return MonteCarloResult(
        num_simulations=1000,
        original_return_pct=10.0,
        mean_return_pct=8.0,
        median_return_pct=median_return_pct,
        std_return_pct=std_return_pct,
        percentile_5=-5.0,
        percentile_25=2.0,
        percentile_75=12.0,
        percentile_95=20.0,
        probability_of_loss=probability_of_loss,
        probability_of_ruin=probability_of_ruin,
        mean_max_drawdown_pct=mean_max_drawdown_pct,
        worst_case_return_pct=-30.0,
        best_case_return_pct=40.0,
        confidence_95_lower=-8.0,
        confidence_95_upper=25.0,
    )


# ---------------------------------------------------------------------------
# _determine_recommendation テスト
# ---------------------------------------------------------------------------

class TestDetermineRecommendation:
    def test_go_threshold(self) -> None:
        assert _determine_recommendation(70) == "GO"
        assert _determine_recommendation(100) == "GO"
        assert _determine_recommendation(85) == "GO"

    def test_caution_threshold(self) -> None:
        assert _determine_recommendation(45) == "CAUTION"
        assert _determine_recommendation(69) == "CAUTION"
        assert _determine_recommendation(55) == "CAUTION"

    def test_nogo_threshold(self) -> None:
        assert _determine_recommendation(44) == "NOGO"
        assert _determine_recommendation(0) == "NOGO"
        assert _determine_recommendation(10) == "NOGO"


# ---------------------------------------------------------------------------
# 個別スコアリング関数テスト
# ---------------------------------------------------------------------------

class TestScoreFunctions:
    def test_wf_consistency_none_returns_zero(self) -> None:
        assert _score_wf_consistency(None) == 0

    def test_wf_consistency_high_ratio(self) -> None:
        wf = _make_wf(consistency_ratio=0.85)
        assert _score_wf_consistency(wf) == 100

    def test_wf_consistency_medium_ratio(self) -> None:
        wf = _make_wf(consistency_ratio=0.65)
        assert _score_wf_consistency(wf) == 75

    def test_wf_consistency_low_ratio(self) -> None:
        wf = _make_wf(consistency_ratio=0.15)
        assert _score_wf_consistency(wf) == 0

    def test_wf_significance_none_returns_zero(self) -> None:
        assert _score_wf_significance(None) == 0

    def test_wf_significance_significant(self) -> None:
        wf = _make_wf(is_statistically_significant=True, p_value=0.02)
        assert _score_wf_significance(wf) == 100

    def test_wf_significance_not_significant(self) -> None:
        wf = _make_wf(is_statistically_significant=False, p_value=0.50)
        assert _score_wf_significance(wf) == 0

    def test_mc_ruin_prob_none_returns_zero(self) -> None:
        assert _score_mc_ruin_prob(None) == 0

    def test_mc_ruin_prob_very_low(self) -> None:
        mc = _make_mc(probability_of_ruin=0.005)
        assert _score_mc_ruin_prob(mc) == 100

    def test_mc_ruin_prob_high(self) -> None:
        mc = _make_mc(probability_of_ruin=0.25)
        assert _score_mc_ruin_prob(mc) == 0


# ---------------------------------------------------------------------------
# calculate_readiness_score テスト
# ---------------------------------------------------------------------------

class TestCalculateReadinessScore:
    def test_no_data_returns_scorecard(self) -> None:
        """WF/MCなしでもScorecardを返す（data_freshnessのみ満点）。"""
        scorecard = calculate_readiness_score()
        assert isinstance(scorecard, ReadinessScorecard)
        assert scorecard.data_freshness == 100

    def test_all_data_produces_high_score(self) -> None:
        """良好なWF+MCで高スコアを返す。"""
        wf = _make_wf(consistency_ratio=0.9, p_value=0.01, oos_sharpe_avg=1.2, oos_max_dd_avg=-5.0)
        mc = _make_mc(probability_of_ruin=0.005, probability_of_loss=0.30, median_return_pct=15.0)
        scorecard = calculate_readiness_score(wf=wf, mc=mc)
        assert scorecard.overall_score >= 70
        assert scorecard.recommendation == "GO"

    def test_poor_data_produces_low_score(self) -> None:
        """低品質なWF+MCで低スコアを返す。"""
        wf = _make_wf(consistency_ratio=0.1, p_value=0.80, is_statistically_significant=False, oos_sharpe_avg=-0.5)
        mc = _make_mc(probability_of_ruin=0.30, probability_of_loss=0.70, median_return_pct=-5.0)
        scorecard = calculate_readiness_score(wf=wf, mc=mc)
        assert scorecard.overall_score < 45
        assert scorecard.recommendation == "NOGO"

    def test_scorecard_has_eight_criteria(self) -> None:
        """Scorecardに8つの基準スコアがある。"""
        scorecard = calculate_readiness_score()
        criteria = [
            scorecard.wf_consistency,
            scorecard.wf_significance,
            scorecard.mc_ruin_prob,
            scorecard.sharpe,
            scorecard.max_drawdown,
            scorecard.win_rate,
            scorecard.profit_factor,
            scorecard.data_freshness,
        ]
        assert len(criteria) == 8
        assert all(0 <= c <= 100 for c in criteria)

    def test_overall_score_is_average_of_criteria(self) -> None:
        """overall_scoreは8基準の平均値である。"""
        wf = _make_wf()
        mc = _make_mc()
        scorecard = calculate_readiness_score(wf=wf, mc=mc)
        criteria_sum = (
            scorecard.wf_consistency
            + scorecard.wf_significance
            + scorecard.mc_ruin_prob
            + scorecard.sharpe
            + scorecard.max_drawdown
            + scorecard.win_rate
            + scorecard.profit_factor
            + scorecard.data_freshness
        )
        expected = round(criteria_sum / 8)
        assert scorecard.overall_score == expected


# ---------------------------------------------------------------------------
# run_strategy_benchmark テスト
# ---------------------------------------------------------------------------

class TestRunStrategyBenchmark:
    def test_returns_strategy_benchmark(self) -> None:
        """run_strategy_benchmark()がStrategyBenchmarkを返す。"""
        result = run_strategy_benchmark(
            strategy_name="momentum",
            ticker="7203",
        )
        assert isinstance(result, StrategyBenchmark)

    def test_strategy_name_and_ticker_preserved(self) -> None:
        """strategy_nameとtickerが結果に保存される。"""
        result = run_strategy_benchmark(
            strategy_name="mean_reversion",
            ticker="6758",
        )
        assert result.strategy_name == "mean_reversion"
        assert result.ticker == "6758"

    def test_evaluated_at_is_iso_timestamp(self) -> None:
        """evaluated_atがISO 8601形式のタイムスタンプである。"""
        from datetime import datetime
        result = run_strategy_benchmark(strategy_name="test", ticker="0000")
        # ISO 8601パース可能であることを確認
        parsed = datetime.fromisoformat(result.evaluated_at)
        assert parsed is not None

    def test_with_wf_and_mc(self) -> None:
        """WF+MCを渡した場合に結果に格納される。"""
        wf = _make_wf()
        mc = _make_mc()
        result = run_strategy_benchmark(
            strategy_name="breakout",
            ticker="9984",
            wf=wf,
            mc=mc,
        )
        assert result.walk_forward is wf
        assert result.monte_carlo is mc
        assert 0 <= result.overall_score <= 100
        assert result.recommendation in {"GO", "CAUTION", "NOGO"}

    def test_to_dict_serializable(self) -> None:
        """to_dict()がJSONシリアライズ可能な辞書を返す。"""
        import json
        wf = _make_wf()
        mc = _make_mc()
        result = run_strategy_benchmark("momentum", "7203", wf=wf, mc=mc)
        d = result.to_dict()
        # JSON化できることを確認（例外が出ないこと）
        serialized = json.dumps(d, ensure_ascii=False)
        assert len(serialized) > 0
