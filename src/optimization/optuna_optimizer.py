"""Optuna 戦略最適化エンジン。

Walk-Forward OOS consistency_ratio を objective に使用し、
過学習を防止しながらハイパーパラメータを最適化する。

重要: 財務指標 (Sharpe ratio 等) を直接 objective にすると
TPE がノイズ面の最適解を見つけてしまうため、OOS の統計的一貫性を使用する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ..strategies.base import BaseStrategy
from ..evaluation.backtester import run_backtest
from ..evaluation.walk_forward import WalkForwardResult, walk_forward_analysis
from .purged_cv import PurgedWalkForwardCV

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@dataclass(frozen=True)
class OptimizationResult:
    """最適化結果（immutable）。"""

    strategy_name: str
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    n_complete: int
    n_pruned: int
    study_name: str
    objective_metric: str
    walk_forward_result: Optional[Dict[str, Any]] = None
    all_trials: List[Dict[str, Any]] = field(default_factory=list)


def _suggest_param(
    trial: "optuna.Trial", name: str, spec: tuple
) -> Any:
    """parameter_space() の定義から Optuna の suggest を呼ぶ。

    Parameters
    ----------
    trial:
        Optuna trial
    name:
        パラメータ名
    spec:
        (type, min_value, max_value) のタプル

    Returns
    -------
    Any
        suggest された値
    """
    param_type, low, high = spec
    if param_type is int or param_type == int:
        return trial.suggest_int(name, int(low), int(high))
    elif param_type is float or param_type == float:
        return trial.suggest_float(name, float(low), float(high))
    else:
        return trial.suggest_float(name, float(low), float(high))


class OptunaOptimizer:
    """Optuna ベースの戦略パラメータ最適化。

    Walk-Forward 分析の OOS consistency_ratio を最大化する。

    Parameters
    ----------
    strategy:
        最適化対象の戦略インスタンス
    n_trials:
        試行回数（デフォルト: 50）
    timeout:
        タイムアウト（秒、デフォルト: None = 無制限）
    objective_metric:
        最適化する指標（デフォルト: "consistency_ratio"）
    seed:
        乱数シード（再現性用）
    """

    VALID_METRICS = frozenset({
        "consistency_ratio",
        "oos_sharpe_avg",
        "degradation_ratio",
        "combined_score",
    })

    def __init__(
        self,
        strategy: BaseStrategy,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        objective_metric: str = "consistency_ratio",
        seed: Optional[int] = None,
    ) -> None:
        if not HAS_OPTUNA:
            raise ImportError(
                "optuna が必要です: pip install optuna"
            )
        if objective_metric not in self.VALID_METRICS:
            raise ValueError(
                f"無効な objective_metric: {objective_metric}. "
                f"有効な値: {sorted(self.VALID_METRICS)}"
            )

        self._strategy = strategy
        self._n_trials = n_trials
        self._timeout = timeout
        self._objective_metric = objective_metric
        self._seed = seed

        space = strategy.parameter_space()
        if not space:
            raise ValueError(
                f"戦略 '{strategy.name}' は parameter_space() が空です。"
                "最適化するパラメータがありません。"
            )
        self._param_space = space

    def optimize(
        self,
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        in_sample_days: int = 504,
        out_of_sample_days: int = 126,
        initial_capital: float = 1_000_000.0,
        fee_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> OptimizationResult:
        """最適化を実行する。

        Parameters
        ----------
        data:
            OHLCV DataFrame（十分な長さが必要）
        symbol:
            銘柄コード
        in_sample_days:
            Walk-Forward のインサンプル期間
        out_of_sample_days:
            Walk-Forward のアウトオブサンプル期間
        initial_capital:
            バックテスト初期資金
        fee_rate:
            手数料率
        slippage_rate:
            スリッページ率
        study_name:
            Optuna study 名（省略時は自動生成）
        storage:
            Optuna storage URL（省略時はインメモリ）

        Returns
        -------
        OptimizationResult
            最適化結果
        """
        if study_name is None:
            study_name = f"optimize_{self._strategy.name}"

        sampler = TPESampler(
            seed=self._seed,
            multivariate=True,
            n_startup_trials=10,
        )
        pruner = HyperbandPruner(
            min_resource=1,
            max_resource=self._n_trials,
            reduction_factor=3,
        )

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial) -> float:
            return self._objective(
                trial=trial,
                data=data,
                symbol=symbol,
                in_sample_days=in_sample_days,
                out_of_sample_days=out_of_sample_days,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )

        # ログ抑制（大量の trial ログを防ぐ）
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
        )

        # ベストパラメータで最終 Walk-Forward 実行
        best_wf_result = None
        if study.best_trial is not None:
            try:
                best_strategy = self._strategy.set_parameters(
                    **self._map_params(study.best_params)
                )
                wf = walk_forward_analysis(
                    strategy=best_strategy,
                    data=data,
                    symbol=symbol,
                    in_sample_days=in_sample_days,
                    out_of_sample_days=out_of_sample_days,
                    initial_capital=initial_capital,
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    min_walks=2,
                )
                best_wf_result = wf.to_dict()
            except Exception as e:
                logger.warning(f"ベストパラメータでの WF 分析失敗: {e}")

        # 全 trial の記録
        all_trials = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                all_trials.append({
                    "number": t.number,
                    "value": t.value,
                    "params": dict(t.params),
                    "duration_seconds": (
                        (t.datetime_complete - t.datetime_start).total_seconds()
                        if t.datetime_complete and t.datetime_start
                        else None
                    ),
                })

        return OptimizationResult(
            strategy_name=self._strategy.name,
            best_params=dict(study.best_params) if study.best_trial else {},
            best_value=study.best_value if study.best_trial else 0.0,
            n_trials=len(study.trials),
            n_complete=len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]),
            n_pruned=len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ]),
            study_name=study_name,
            objective_metric=self._objective_metric,
            walk_forward_result=best_wf_result,
            all_trials=all_trials,
        )

    def _objective(
        self,
        trial: "optuna.Trial",
        data: pd.DataFrame,
        symbol: str,
        in_sample_days: int,
        out_of_sample_days: int,
        initial_capital: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> float:
        """Optuna objective 関数。

        parameter_space() からパラメータを suggest し、
        Walk-Forward 分析で OOS パフォーマンスを評価する。
        """
        # パラメータを suggest
        suggested = {}
        for name, spec in self._param_space.items():
            suggested[name] = _suggest_param(trial, name, spec)

        # 戦略インスタンスを生成（immutable パターン）
        try:
            strategy = self._strategy.set_parameters(
                **self._map_params(suggested)
            )
        except (AttributeError, ValueError) as e:
            logger.debug(f"パラメータ設定失敗: {e}")
            return 0.0

        # Walk-Forward 分析
        try:
            wf_result = walk_forward_analysis(
                strategy=strategy,
                data=data,
                symbol=symbol,
                in_sample_days=in_sample_days,
                out_of_sample_days=out_of_sample_days,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                min_walks=2,
            )
        except (ValueError, Exception) as e:
            logger.debug(f"WF分析失敗 (trial={trial.number}): {e}")
            return 0.0

        # Pruning: 中間結果を報告
        trial.report(wf_result.consistency_ratio, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return self._compute_score(wf_result)

    def _compute_score(self, wf: WalkForwardResult) -> float:
        """WalkForwardResult から最適化スコアを計算する。"""
        if self._objective_metric == "consistency_ratio":
            return wf.consistency_ratio

        elif self._objective_metric == "oos_sharpe_avg":
            return wf.oos_sharpe_avg

        elif self._objective_metric == "degradation_ratio":
            # degradation_ratio は 1.0 に近いほど良い
            return 1.0 - abs(1.0 - wf.degradation_ratio)

        elif self._objective_metric == "combined_score":
            # 複合スコア: consistency (60%) + sharpe (20%) + degradation (20%)
            consistency_score = wf.consistency_ratio
            sharpe_score = max(0.0, min(1.0, (wf.oos_sharpe_avg + 1.0) / 3.0))
            degrad_score = 1.0 - abs(1.0 - wf.degradation_ratio)
            return (
                0.6 * consistency_score
                + 0.2 * sharpe_score
                + 0.2 * max(0.0, degrad_score)
            )

        return 0.0

    def _map_params(self, suggested: Dict[str, Any]) -> Dict[str, Any]:
        """suggest されたパラメータを戦略のコンストラクタ引数にマッピングする。

        parameter_space() のキー名と __init__ のキー名が異なる場合の
        マッピングを行う。
        """
        # 既知のマッピング (parameter_space key → __init__ kwarg)
        KNOWN_MAPPINGS: Dict[str, Dict[str, str]] = {
            "ma_crossover": {
                "fast_period": "short_window",
                "slow_period": "long_window",
            },
            "macd_rsi": {
                "fast_period": "macd_fast",
                "slow_period": "macd_slow",
                "signal_period": "macd_signal",
                "rsi_period": "rsi_period",
                "rsi_upper": "rsi_overbought",
                "rsi_lower": "rsi_oversold",
            },
            "bollinger_rsi_adx": {
                "bb_period": "bb_period",
                "rsi_period": "rsi_period",
                "adx_period": "adx_period",
                "adx_threshold": "adx_threshold",
            },
            "dual_momentum": {
                "lookback_period": "lookback_period",
                "rebalance_period": "rebalance_freq",
            },
            "lgbm_predictor": {
                "n_estimators": "n_estimators",
                "learning_rate": "learning_rate",
                "train_window": "train_window",
            },
        }

        mapping = KNOWN_MAPPINGS.get(self._strategy.name, {})
        mapped: Dict[str, Any] = {}

        for key, value in suggested.items():
            mapped_key = mapping.get(key, key)
            mapped[mapped_key] = value

        return mapped
