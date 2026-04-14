"""自動強化学習モジュール。

ペーパートレード中にバックグラウンドで以下を自動実行:
1. 市場データの定期蓄積 (日次OHLCV)
2. 戦略パラメータの定期再最適化 (Optuna)
3. 新しい市場パターンの検出 (ドリフト/レジーム)
4. 戦略パフォーマンスのフィードバック記録
5. 改善候補の自動A/Bテスト
6. 結果のスコアボード管理

このモジュールは AutoTrader から定期呼び出しされる。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImprovementCandidate:
    """改善候補（immutable）。"""
    strategy_name: str
    symbol: str
    old_params: Dict[str, Any]
    new_params: Dict[str, Any]
    old_sharpe: float
    new_sharpe: float
    improvement_pct: float
    timestamp: str
    status: str  # "pending", "testing", "adopted", "rejected"


@dataclass
class PerformanceRecord:
    """戦略パフォーマンス記録。"""
    strategy_name: str
    symbol: str
    date: str
    signal: int
    pnl: float
    cumulative_pnl: float
    win_count: int
    loss_count: int
    sharpe_30d: float


class SelfImprover:
    """自動強化学習エンジン。

    Parameters
    ----------
    data_dir:
        データ保存ディレクトリ
    optimize_interval_days:
        再最適化の間隔（日、デフォルト: 7）
    min_data_days:
        最適化に必要な最低データ日数（デフォルト: 90）
    improvement_threshold:
        採用する最低改善率（デフォルト: 0.1 = 10%）
    """

    def __init__(
        self,
        data_dir: str = "data/self_improve",
        optimize_interval_days: int = 7,
        min_data_days: int = 90,
        improvement_threshold: float = 0.1,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._optimize_interval = optimize_interval_days
        self._min_data_days = min_data_days
        self._improvement_threshold = improvement_threshold

        # 状態
        self._performance_log: List[PerformanceRecord] = []
        self._candidates: List[ImprovementCandidate] = []
        self._last_optimize: Optional[str] = None
        self._last_data_collect: Optional[str] = None
        self._scoreboard: Dict[str, Dict[str, float]] = {}

        # 永続化からロード
        self._load_state()

    @property
    def total_records(self) -> int:
        return len(self._performance_log)

    @property
    def candidates(self) -> List[ImprovementCandidate]:
        return list(self._candidates)

    # ------------------------------------------------------------------
    # 1. 市場データ自動蓄積
    # ------------------------------------------------------------------

    def collect_market_data(
        self, dm: Any, symbols: List[str]
    ) -> Dict[str, int]:
        """市場データを取得しローカルにキャッシュする。

        Returns
        -------
        Dict[str, int]
            {symbol: 取得行数}
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_data_collect == today:
            return {}

        results = {}
        start = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

        for symbol in symbols:
            try:
                data = dm.fetch_ohlcv(symbol, start, today)
                if not data.empty:
                    path = self._data_dir / f"{symbol.replace('/', '_')}_daily.parquet"
                    data.to_parquet(str(path))
                    results[symbol] = len(data)
            except Exception as e:
                logger.debug("データ収集失敗 %s: %s", symbol, e)

        self._last_data_collect = today
        self._save_state()
        logger.info("市場データ収集完了: %s", results)
        return results

    # ------------------------------------------------------------------
    # 2. パフォーマンス記録
    # ------------------------------------------------------------------

    def record_performance(
        self,
        strategy_name: str,
        symbol: str,
        signal: int,
        pnl: float,
    ) -> PerformanceRecord:
        """取引結果を記録する。"""
        today = datetime.now().strftime("%Y-%m-%d")

        # 累積計算
        prev_records = [
            r for r in self._performance_log
            if r.strategy_name == strategy_name and r.symbol == symbol
        ]
        cumulative = sum(r.pnl for r in prev_records) + pnl
        wins = sum(1 for r in prev_records if r.pnl > 0) + (1 if pnl > 0 else 0)
        losses = sum(1 for r in prev_records if r.pnl < 0) + (1 if pnl < 0 else 0)

        # 30日Sharpe
        recent_pnls = [r.pnl for r in prev_records[-29:]] + [pnl]
        arr = np.array(recent_pnls)
        sharpe_30d = float(np.mean(arr) / np.std(arr) * np.sqrt(252)) if len(arr) > 1 and np.std(arr) > 0 else 0.0

        record = PerformanceRecord(
            strategy_name=strategy_name,
            symbol=symbol,
            date=today,
            signal=signal,
            pnl=pnl,
            cumulative_pnl=cumulative,
            win_count=wins,
            loss_count=losses,
            sharpe_30d=sharpe_30d,
        )
        self._performance_log.append(record)

        # スコアボード更新
        key = f"{strategy_name}/{symbol}"
        self._scoreboard[key] = {
            "cumulative_pnl": cumulative,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) * 100 if (wins + losses) > 0 else 0,
            "sharpe_30d": sharpe_30d,
            "total_trades": wins + losses,
        }

        self._save_state()
        return record

    # ------------------------------------------------------------------
    # 3. 自動再最適化
    # ------------------------------------------------------------------

    def maybe_reoptimize(
        self,
        strategies: Dict[str, Any],
        dm: Any,
        symbols: List[str],
        n_trials: int = 20,
    ) -> List[ImprovementCandidate]:
        """条件が揃った場合に戦略パラメータを再最適化する。

        Returns
        -------
        List[ImprovementCandidate]
            改善候補のリスト
        """
        today = datetime.now().strftime("%Y-%m-%d")

        if self._last_optimize:
            days_since = (datetime.now() - datetime.strptime(self._last_optimize, "%Y-%m-%d")).days
            if days_since < self._optimize_interval:
                return []

        try:
            from src.optimization.optuna_optimizer import OptunaOptimizer
        except ImportError:
            return []

        new_candidates = []
        end_date = today
        start_date = (datetime.now() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

        for symbol in symbols[:2]:  # メモリ節約: 上位2銘柄のみ
            for strat_name, strategy in strategies.items():
                if not strategy.parameter_space():
                    continue

                try:
                    data = dm.fetch_ohlcv(symbol, start_date, end_date)
                    if data.empty or len(data) < self._min_data_days:
                        continue

                    # 現在のパフォーマンス
                    from src.evaluation.backtester import run_backtest
                    old_result = run_backtest(strategy, data, symbol=symbol, initial_capital=200_000)
                    old_sharpe = old_result.metrics.sharpe_ratio

                    # Optuna再最適化
                    optimizer = OptunaOptimizer(
                        strategy=strategy,
                        n_trials=n_trials,
                        objective_metric="combined_score",
                        seed=int(time.time()) % 10000,
                    )
                    opt_result = optimizer.optimize(data=data, symbol=symbol)

                    if not opt_result.best_params:
                        continue

                    # 最適化後パラメータでバックテスト
                    new_strategy = strategy.set_parameters(
                        **self._map_params(strat_name, opt_result.best_params)
                    )
                    new_result = run_backtest(new_strategy, data, symbol=symbol, initial_capital=200_000)
                    new_sharpe = new_result.metrics.sharpe_ratio

                    improvement = (new_sharpe - old_sharpe) / max(abs(old_sharpe), 0.01)

                    candidate = ImprovementCandidate(
                        strategy_name=strat_name,
                        symbol=symbol,
                        old_params=strategy.get_parameters(),
                        new_params=opt_result.best_params,
                        old_sharpe=round(old_sharpe, 4),
                        new_sharpe=round(new_sharpe, 4),
                        improvement_pct=round(improvement * 100, 1),
                        timestamp=today,
                        status="pending" if improvement > self._improvement_threshold else "rejected",
                    )
                    new_candidates.append(candidate)
                    self._candidates.append(candidate)

                    logger.info(
                        "再最適化 %s/%s: Sharpe %.3f→%.3f (%+.1f%%) → %s",
                        symbol, strat_name, old_sharpe, new_sharpe,
                        improvement * 100, candidate.status,
                    )
                except Exception as e:
                    logger.warning("再最適化失敗 %s/%s: %s", symbol, strat_name, e)

        self._last_optimize = today
        self._save_state()
        return new_candidates

    def _map_params(self, strategy_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optunaパラメータを戦略引数にマッピング。"""
        mappings = {
            "ma_crossover": {"fast_period": "short_window", "slow_period": "long_window"},
            "bollinger_rsi_adx": {},
        }
        mapping = mappings.get(strategy_name, {})
        return {mapping.get(k, k): v for k, v in params.items()}

    # ------------------------------------------------------------------
    # 4. スコアボード
    # ------------------------------------------------------------------

    def get_scoreboard(self) -> Dict[str, Dict[str, float]]:
        """戦略×銘柄のスコアボードを返す。"""
        return dict(self._scoreboard)

    def get_best_strategy(self) -> Optional[str]:
        """最も高い Sharpe の戦略/銘柄を返す。"""
        if not self._scoreboard:
            return None
        return max(self._scoreboard.items(), key=lambda x: x[1].get("sharpe_30d", 0))[0]

    def get_improvement_summary(self) -> Dict[str, Any]:
        """改善活動のサマリーを返す。"""
        adopted = [c for c in self._candidates if c.status == "adopted"]
        pending = [c for c in self._candidates if c.status == "pending"]
        rejected = [c for c in self._candidates if c.status == "rejected"]
        return {
            "total_candidates": len(self._candidates),
            "adopted": len(adopted),
            "pending": len(pending),
            "rejected": len(rejected),
            "total_performance_records": len(self._performance_log),
            "last_optimize": self._last_optimize,
            "best_strategy": self.get_best_strategy(),
        }

    # ------------------------------------------------------------------
    # 5. 永続化
    # ------------------------------------------------------------------

    def _save_state(self) -> None:
        """状態をJSONに保存する。"""
        state = {
            "last_optimize": self._last_optimize,
            "last_data_collect": self._last_data_collect,
            "scoreboard": self._scoreboard,
            "performance_count": len(self._performance_log),
            "candidates_count": len(self._candidates),
        }
        path = self._data_dir / "self_improve_state.json"
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _load_state(self) -> None:
        """状態をJSONからロードする。"""
        path = self._data_dir / "self_improve_state.json"
        if path.exists():
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                self._last_optimize = state.get("last_optimize")
                self._last_data_collect = state.get("last_data_collect")
                self._scoreboard = state.get("scoreboard", {})
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 6. 統合実行（AutoTraderから呼ばれる）
    # ------------------------------------------------------------------

    def run_cycle(
        self,
        dm: Any,
        strategies: Dict[str, Any],
        symbols: List[str],
    ) -> Dict[str, Any]:
        """1回の自動強化サイクルを実行する。

        Returns
        -------
        Dict[str, Any]
            サイクル結果サマリー
        """
        result: Dict[str, Any] = {"timestamp": datetime.now().isoformat()}

        # 1. データ収集
        collected = self.collect_market_data(dm, symbols)
        result["data_collected"] = collected

        # 2. 再最適化（条件が揃った場合のみ）
        candidates = self.maybe_reoptimize(strategies, dm, symbols)
        result["new_candidates"] = len(candidates)
        result["pending_improvements"] = [
            {"strategy": c.strategy_name, "symbol": c.symbol,
             "improvement": f"{c.improvement_pct:+.1f}%"}
            for c in candidates if c.status == "pending"
        ]

        # 3. サマリー
        result["summary"] = self.get_improvement_summary()

        logger.info("自動強化サイクル完了: %s", result)
        return result
