"""Phase 2: 戦略最適化エンジンのテスト。"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.optimization.purged_cv import PurgedWalkForwardCV, CVFold


# ============================================================
# テストデータ生成ヘルパー
# ============================================================

def _make_ohlcv(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """テスト用の OHLCV データを生成する。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 1000 + np.cumsum(rng.randn(n) * 10)
    close = np.maximum(close, 100)  # 最低100円
    return pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.01, 0.01, n)),
        "high": close * (1 + rng.uniform(0, 0.02, n)),
        "low": close * (1 - rng.uniform(0, 0.02, n)),
        "close": close,
        "volume": rng.randint(100_000, 1_000_000, n).astype(float),
    }, index=dates)


# ============================================================
# PurgedWalkForwardCV テスト
# ============================================================

class TestPurgedWalkForwardCV:
    """PurgedWalkForwardCV のテスト。"""

    def test_init_valid(self) -> None:
        cv = PurgedWalkForwardCV(n_splits=5, train_ratio=0.6)
        assert cv.n_splits == 5

    def test_init_invalid_n_splits(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            PurgedWalkForwardCV(n_splits=1)

    def test_init_invalid_train_ratio(self) -> None:
        with pytest.raises(ValueError, match="train_ratio"):
            PurgedWalkForwardCV(n_splits=3, train_ratio=0.05)

    def test_split_returns_correct_number_of_folds(self) -> None:
        data = _make_ohlcv(1000)
        cv = PurgedWalkForwardCV(n_splits=5, train_ratio=0.6)
        folds = cv.split(data)
        assert len(folds) == 5

    def test_split_folds_are_cvfold_instances(self) -> None:
        data = _make_ohlcv(500)
        cv = PurgedWalkForwardCV(n_splits=3, train_ratio=0.6)
        folds = cv.split(data)
        for fold in folds:
            assert isinstance(fold, CVFold)

    def test_split_no_overlap_between_train_and_test(self) -> None:
        data = _make_ohlcv(1000)
        cv = PurgedWalkForwardCV(n_splits=5, train_ratio=0.6)
        folds = cv.split(data)
        for fold in folds:
            # embargo があるので train_end < test_start
            assert fold.train_end < fold.test_start
            assert fold.embargo_days >= 1

    def test_split_expanding_window(self) -> None:
        data = _make_ohlcv(1000)
        cv = PurgedWalkForwardCV(n_splits=3, train_ratio=0.6, expanding=True)
        folds = cv.split(data)
        # expanding の場合、全 fold で train_start == 0
        for fold in folds:
            assert fold.train_start == 0

    def test_split_rolling_window(self) -> None:
        data = _make_ohlcv(1000)
        cv = PurgedWalkForwardCV(n_splits=3, train_ratio=0.6, expanding=False)
        folds = cv.split(data)
        # rolling の場合、train_start が増加していく
        starts = [f.train_start for f in folds]
        assert starts == sorted(starts)
        assert starts[0] == 0
        if len(starts) > 1:
            assert starts[1] > 0

    def test_split_embargo_prevents_leakage(self) -> None:
        data = _make_ohlcv(1000)
        cv = PurgedWalkForwardCV(n_splits=3, train_ratio=0.6, embargo_pct=0.05)
        folds = cv.split(data)
        for fold in folds:
            gap = fold.test_start - fold.train_end
            assert gap == fold.embargo_days
            assert gap >= 1

    def test_get_train_test_pairs(self) -> None:
        data = _make_ohlcv(500)
        cv = PurgedWalkForwardCV(n_splits=3, train_ratio=0.6)
        pairs = cv.get_train_test_pairs(data)
        assert len(pairs) == 3
        for train, test in pairs:
            assert isinstance(train, pd.DataFrame)
            assert isinstance(test, pd.DataFrame)
            assert len(train) > 0
            assert len(test) > 0

    def test_get_train_test_pairs_are_copies(self) -> None:
        """train/test DataFrame は元データの copy であること。"""
        data = _make_ohlcv(500)
        cv = PurgedWalkForwardCV(n_splits=2, train_ratio=0.6)
        pairs = cv.get_train_test_pairs(data)
        train, test = pairs[0]
        # 変更しても元データに影響しない
        original_val = data.iloc[0]["close"]
        train.iloc[0, train.columns.get_loc("close")] = -999
        assert data.iloc[0]["close"] == original_val

    def test_split_insufficient_data(self) -> None:
        data = _make_ohlcv(10)
        cv = PurgedWalkForwardCV(n_splits=5, train_ratio=0.6)
        with pytest.raises(ValueError, match="データが不足"):
            cv.split(data)

    def test_fold_is_frozen(self) -> None:
        fold = CVFold(
            fold_index=0, train_start=0, train_end=100,
            test_start=105, test_end=200, embargo_days=5,
        )
        with pytest.raises(AttributeError):
            fold.fold_index = 99  # type: ignore[misc]


# ============================================================
# OptunaOptimizer テスト（Optuna がある場合のみ）
# ============================================================

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestOptunaOptimizer:
    """OptunaOptimizer のテスト。"""

    def test_init_valid(self) -> None:
        from src.strategies.ma_crossover import MACrossoverStrategy
        from src.optimization.optuna_optimizer import OptunaOptimizer

        strategy = MACrossoverStrategy()
        optimizer = OptunaOptimizer(strategy=strategy, n_trials=5)
        assert optimizer is not None

    def test_init_no_parameter_space(self) -> None:
        from src.strategies.base import BaseStrategy
        from src.optimization.optuna_optimizer import OptunaOptimizer

        class NoParamStrategy(BaseStrategy):
            name = "no_param"
            def generate_signals(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(0, index=data.index)

        with pytest.raises(ValueError, match="parameter_space"):
            OptunaOptimizer(strategy=NoParamStrategy(), n_trials=5)

    def test_init_invalid_metric(self) -> None:
        from src.strategies.ma_crossover import MACrossoverStrategy
        from src.optimization.optuna_optimizer import OptunaOptimizer

        with pytest.raises(ValueError, match="無効な objective_metric"):
            OptunaOptimizer(
                strategy=MACrossoverStrategy(),
                n_trials=5,
                objective_metric="invalid",
            )

    def test_suggest_param_int(self) -> None:
        from src.optimization.optuna_optimizer import _suggest_param

        study = optuna.create_study()
        trial = study.ask()
        val = _suggest_param(trial, "test_int", (int, 1, 100))
        assert isinstance(val, int)
        assert 1 <= val <= 100

    def test_suggest_param_float(self) -> None:
        from src.optimization.optuna_optimizer import _suggest_param

        study = optuna.create_study()
        trial = study.ask()
        val = _suggest_param(trial, "test_float", (float, 0.01, 1.0))
        assert isinstance(val, float)
        assert 0.01 <= val <= 1.0

    def test_optimize_minimal(self) -> None:
        """最小限の最適化テスト（2 trials、小データ）。"""
        from src.strategies.ma_crossover import MACrossoverStrategy
        from src.optimization.optuna_optimizer import OptunaOptimizer

        data = _make_ohlcv(800)
        strategy = MACrossoverStrategy()
        optimizer = OptunaOptimizer(
            strategy=strategy,
            n_trials=2,
            objective_metric="consistency_ratio",
            seed=42,
        )

        result = optimizer.optimize(
            data=data,
            symbol="TEST",
            in_sample_days=300,
            out_of_sample_days=100,
        )

        assert result.strategy_name == "ma_crossover"
        assert result.n_trials >= 1
        assert result.study_name == "optimize_ma_crossover"
        assert result.objective_metric == "consistency_ratio"
        assert isinstance(result.best_params, dict)

    def test_optimize_combined_score(self) -> None:
        """combined_score メトリックでの最適化テスト。"""
        from src.strategies.ma_crossover import MACrossoverStrategy
        from src.optimization.optuna_optimizer import OptunaOptimizer

        data = _make_ohlcv(800)
        optimizer = OptunaOptimizer(
            strategy=MACrossoverStrategy(),
            n_trials=2,
            objective_metric="combined_score",
            seed=42,
        )

        result = optimizer.optimize(
            data=data, symbol="TEST",
            in_sample_days=300, out_of_sample_days=100,
        )

        assert result.objective_metric == "combined_score"

    def test_optimization_result_is_frozen(self) -> None:
        """OptimizationResult は frozen dataclass であること。"""
        from src.optimization.optuna_optimizer import OptimizationResult

        result = OptimizationResult(
            strategy_name="test", best_params={}, best_value=0.5,
            n_trials=10, n_complete=8, n_pruned=2,
            study_name="test_study", objective_metric="consistency_ratio",
        )
        with pytest.raises(AttributeError):
            result.best_value = 0.9  # type: ignore[misc]

    def test_param_mapping_ma_crossover(self) -> None:
        """MA Crossover のパラメータマッピングテスト。"""
        from src.strategies.ma_crossover import MACrossoverStrategy
        from src.optimization.optuna_optimizer import OptunaOptimizer

        strategy = MACrossoverStrategy()
        optimizer = OptunaOptimizer(strategy=strategy, n_trials=1)

        mapped = optimizer._map_params({"fast_period": 10, "slow_period": 50})
        assert mapped == {"short_window": 10, "long_window": 50}

    def test_param_mapping_unknown_strategy(self) -> None:
        """マッピングが定義されていない戦略はパススルー。"""
        from src.strategies.ma_crossover import MACrossoverStrategy
        from src.optimization.optuna_optimizer import OptunaOptimizer

        strategy = MACrossoverStrategy()
        optimizer = OptunaOptimizer(strategy=strategy, n_trials=1)
        # 存在しないキーはそのまま返す
        mapped = optimizer._map_params({"unknown_key": 42})
        assert mapped == {"unknown_key": 42}


# ============================================================
# 統合テスト
# ============================================================

@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
class TestOptimizationIntegration:
    """PurgedCV + OptunaOptimizer の統合テスト。"""

    def test_cv_fold_indices_within_data_bounds(self) -> None:
        data = _make_ohlcv(1000)
        cv = PurgedWalkForwardCV(n_splits=5, train_ratio=0.6)
        folds = cv.split(data)
        for fold in folds:
            assert fold.train_start >= 0
            assert fold.train_end <= len(data)
            assert fold.test_start >= 0
            assert fold.test_end <= len(data)

    def test_multiple_strategies_have_parameter_space(self) -> None:
        """コア戦略すべてが parameter_space() を持つこと。"""
        from src.strategies.ma_crossover import MACrossoverStrategy
        from src.strategies.dual_momentum import DualMomentumStrategy
        from src.strategies.macd_rsi import MACDRSIStrategy
        from src.strategies.bollinger_rsi_adx import BollingerRSIADXStrategy

        for cls in [
            MACrossoverStrategy,
            DualMomentumStrategy,
            MACDRSIStrategy,
            BollingerRSIADXStrategy,
        ]:
            strategy = cls()
            space = strategy.parameter_space()
            assert len(space) > 0, f"{cls.__name__} has empty parameter_space"
            for name, spec in space.items():
                assert len(spec) == 3, f"{cls.__name__}.{name}: spec must be (type, min, max)"
                assert spec[0] in (int, float), f"{cls.__name__}.{name}: type must be int or float"
