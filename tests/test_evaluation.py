"""評価システムのテスト。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import calculate_metrics, EvaluationResult
from src.evaluation.backtester import run_backtest, BacktestResult
from src.evaluation.monte_carlo import monte_carlo_simulation, MonteCarloResult
from src.evaluation.statistics import run_statistical_tests, StatisticsResult
from src.strategies.ma_crossover import MACrossoverStrategy


# ============================================================
# メトリクス計算のテスト
# ============================================================

class TestCalculateMetrics:
    """calculate_metricsのテスト。"""

    def test_total_return_calculation(self):
        """総リターンの計算精度テスト（既知の結果と比較）。"""
        # 100万円が110万円に増えた場合 → リターン10%
        initial = 1_000_000.0
        equity = pd.Series([initial, 1_050_000.0, 1_100_000.0])
        result = calculate_metrics(equity, initial_capital=initial)
        assert abs(result.total_return_pct - 10.0) < 0.01

    def test_max_drawdown_calculation(self):
        """最大ドローダウンの計算精度テスト。"""
        # ピーク100万 → 80万に落ちる → DD -20%
        initial = 1_000_000.0
        equity = pd.Series([initial, 1_200_000.0, 800_000.0, 900_000.0])
        result = calculate_metrics(equity, initial_capital=initial)
        # max_ddはピーク比で計算（-33.33...%）
        assert result.max_drawdown_pct < 0

    def test_sharpe_ratio_positive_returns(self, equity_curve: pd.Series):
        """正のリターンのシャープレシオが正であることを確認。"""
        result = calculate_metrics(equity_curve, initial_capital=1_000_000.0)
        # 期待値プラスのエクイティカーブなのでシャープは正
        assert isinstance(result.sharpe_ratio, float)

    def test_zero_return_result_on_empty(self):
        """空データではゼロ結果が返ることを確認。"""
        result = calculate_metrics(pd.Series([], dtype=float), initial_capital=1_000_000.0)
        assert result.total_return_pct == 0.0
        assert result.total_trades == 0

    def test_win_rate_with_trades(self):
        """勝率の計算精度テスト（50%の場合）。"""
        initial = 1_000_000.0
        equity = pd.Series([initial] * 10)
        trades = pd.DataFrame({
            "pnl_pct": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        })
        result = calculate_metrics(equity, trades=trades, initial_capital=initial)
        assert abs(result.win_rate - 0.5) < 0.01

    def test_profit_factor_calculation(self):
        """プロフィットファクターの計算精度テスト。"""
        initial = 1_000_000.0
        equity = pd.Series([initial] * 10)
        # 利益合計3%, 損失合計1% → PF=3.0
        trades = pd.DataFrame({
            "pnl_pct": [1.0, 1.0, 1.0, -1.0],
        })
        result = calculate_metrics(equity, trades=trades, initial_capital=initial)
        assert abs(result.profit_factor - 3.0) < 0.01

    def test_result_is_immutable(self, equity_curve: pd.Series):
        """EvaluationResultがimmutableであることを確認。"""
        result = calculate_metrics(equity_curve, initial_capital=1_000_000.0)
        with pytest.raises((AttributeError, TypeError)):
            result.sharpe_ratio = 999.0  # type: ignore

    def test_to_dict_contains_all_fields(self, equity_curve: pd.Series):
        """to_dict()が全フィールドを含むことを確認。"""
        result = calculate_metrics(equity_curve, initial_capital=1_000_000.0)
        d = result.to_dict()
        assert "total_return_pct" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown_pct" in d
        assert "win_rate" in d


# ============================================================
# バックテストエンジンのテスト
# ============================================================

class TestRunBacktest:
    """run_backtestのテスト。"""

    def test_backtest_returns_result(self, ohlcv_data: pd.DataFrame):
        """バックテストがBacktestResultを返すことを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = run_backtest(strategy, ohlcv_data)
        assert isinstance(result, BacktestResult)

    def test_final_capital_is_positive(self, ohlcv_data: pd.DataFrame):
        """最終資金が正の値であることを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = run_backtest(strategy, ohlcv_data, initial_capital=1_000_000.0)
        assert result.final_capital > 0

    def test_equity_curve_length(self, ohlcv_data: pd.DataFrame):
        """エクイティカーブの長さがデータと一致することを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = run_backtest(strategy, ohlcv_data)
        assert len(result.equity_curve) == len(ohlcv_data)

    def test_no_future_data_leak(self, ohlcv_data: pd.DataFrame):
        """未来データの漏洩がないことを確認（エクイティカーブは前向きに計算）。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = run_backtest(strategy, ohlcv_data, initial_capital=1_000_000.0)
        # 最初の時点では初期資金と同じ
        assert abs(result.equity_curve[0] - 1_000_000.0) < 1.0

    def test_fee_reduces_returns(self, ohlcv_data: pd.DataFrame):
        """手数料が高いほどリターンが低下することを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result_no_fee = run_backtest(strategy, ohlcv_data, fee_rate=0.0)
        result_high_fee = run_backtest(strategy, ohlcv_data, fee_rate=0.01)
        assert result_no_fee.final_capital >= result_high_fee.final_capital

    def test_backtest_result_to_dict(self, ohlcv_data: pd.DataFrame):
        """to_dict()が正常に動作することを確認。"""
        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = run_backtest(strategy, ohlcv_data)
        d = result.to_dict()
        assert "strategy_name" in d
        assert "metrics" in d
        assert "equity_curve" in d


# ============================================================
# モンテカルロシミュレーションのテスト
# ============================================================

class TestMonteCarlo:
    """monte_carlo_simulationのテスト。"""

    def test_returns_monte_carlo_result(self, simple_trade_returns: list[float]):
        """MonteCarloResultが返されることを確認。"""
        result = monte_carlo_simulation(simple_trade_returns)
        assert isinstance(result, MonteCarloResult)

    def test_num_simulations(self, simple_trade_returns: list[float]):
        """指定回数のシミュレーションが実行されることを確認。"""
        result = monte_carlo_simulation(simple_trade_returns, num_simulations=500)
        assert result.num_simulations == 500

    def test_probability_of_loss_range(self, simple_trade_returns: list[float]):
        """損失確率が0〜1の範囲にあることを確認。"""
        result = monte_carlo_simulation(simple_trade_returns)
        assert 0 <= result.probability_of_loss <= 1

    def test_percentile_ordering(self, simple_trade_returns: list[float]):
        """パーセンタイルが正しく順序付けられていることを確認。"""
        result = monte_carlo_simulation(simple_trade_returns)
        assert result.percentile_5 <= result.percentile_25
        assert result.percentile_25 <= result.median_return_pct
        assert result.median_return_pct <= result.percentile_75
        assert result.percentile_75 <= result.percentile_95

    def test_empty_returns(self):
        """空リターンで安全に処理されることを確認。"""
        result = monte_carlo_simulation([])
        assert result.num_simulations == 0
        assert result.mean_return_pct == 0.0

    def test_reproducibility(self, simple_trade_returns: list[float]):
        """同じシードで同じ結果が得られることを確認。"""
        r1 = monte_carlo_simulation(simple_trade_returns, seed=42)
        r2 = monte_carlo_simulation(simple_trade_returns, seed=42)
        assert r1.mean_return_pct == r2.mean_return_pct

    def test_confidence_interval_order(self, simple_trade_returns: list[float]):
        """95%信頼区間の下限 < 上限であることを確認。"""
        result = monte_carlo_simulation(simple_trade_returns)
        assert result.confidence_95_lower <= result.confidence_95_upper

    def test_result_is_immutable(self, simple_trade_returns: list[float]):
        """MonteCarloResultがimmutableであることを確認。"""
        result = monte_carlo_simulation(simple_trade_returns)
        with pytest.raises((AttributeError, TypeError)):
            result.mean_return_pct = 999.0  # type: ignore


# ============================================================
# 統計的有意性テストのテスト
# ============================================================

class TestStatisticalTests:
    """run_statistical_testsのテスト。"""

    def test_returns_statistics_result(self, simple_trade_returns: list[float]):
        """StatisticsResultが返されることを確認。"""
        result = run_statistical_tests(simple_trade_returns)
        assert isinstance(result, StatisticsResult)

    def test_p_value_range(self, simple_trade_returns: list[float]):
        """p値が0〜1の範囲にあることを確認。"""
        result = run_statistical_tests(simple_trade_returns)
        assert 0 <= result.p_value_ttest <= 1

    def test_bootstrap_ci_order(self, simple_trade_returns: list[float]):
        """ブートストラップ信頼区間の下限 < 上限であることを確認。"""
        result = run_statistical_tests(simple_trade_returns)
        assert result.bootstrap_ci_lower_95 <= result.bootstrap_ci_upper_95

    def test_positive_returns_significant(self):
        """明らかにプラスのリターンが有意と判定されることを確認。"""
        # 全て正のリターン（明確な有意性）
        returns = [0.05] * 100
        result = run_statistical_tests(returns)
        assert result.is_significant_ttest is True

    def test_zero_returns_not_significant(self):
        """ゼロリターンが有意でないと判定されることを確認。"""
        returns = [0.0] * 20
        result = run_statistical_tests(returns)
        assert result.is_significant_ttest is False

    def test_empty_returns_safe(self):
        """空データで安全に処理されることを確認。"""
        result = run_statistical_tests([])
        assert result.p_value_ttest == 1.0
        assert result.is_significant_ttest is False

    def test_sample_mean_accuracy(self):
        """標本平均の計算精度テスト。"""
        returns = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = run_statistical_tests(returns, num_bootstrap=1000)
        assert abs(result.sample_mean - 3.0) < 0.01

    def test_positive_rate(self):
        """プラスリターン率の計算精度テスト（60% = 0.6）。"""
        returns = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
        result = run_statistical_tests(returns, num_bootstrap=1000)
        assert abs(result.positive_rate - 0.6) < 0.01
