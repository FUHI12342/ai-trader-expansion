"""全戦略ベンチマーク実行スクリプト。

Walk-Forward + Monte Carlo を各戦略×銘柄の組み合わせで実行し、
results/benchmark_{date}.json に結果を保存する。

使用方法:
    python scripts/benchmark_all.py
    python scripts/benchmark_all.py --strategies momentum mean_reversion
    python scripts/benchmark_all.py --tickers 7203 6758
    python scripts/benchmark_all.py --strategies momentum --tickers 7203 6758
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

# プロジェクトルートをパスに追加
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.benchmark import run_strategy_benchmark, StrategyBenchmark  # noqa: E402
from src.evaluation.walk_forward import WalkForwardResult  # noqa: E402
from src.evaluation.monte_carlo import MonteCarloResult  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# デフォルト戦略リスト
DEFAULT_STRATEGIES: List[str] = [
    "momentum",
    "mean_reversion",
    "breakout",
    "trend_following",
    "ml_lgbm",
]

# デフォルト銘柄リスト
DEFAULT_TICKERS: List[str] = [
    "7203",  # トヨタ自動車
    "6758",  # ソニーグループ
    "9984",  # ソフトバンクグループ
]

# 結果ディレクトリ
_RESULTS_DIR = _PROJECT_ROOT / "results"


def _load_strategy(strategy_name: str) -> object | None:
    """戦略インスタンスをロードする。ロード失敗時はNoneを返す。"""
    try:
        from src.strategies import get_strategy  # type: ignore[import]
        return get_strategy(strategy_name)
    except Exception as exc:
        logger.warning(f"戦略 '{strategy_name}' のロードに失敗しました: {exc}")
        return None


def _load_data(ticker: str) -> object | None:
    """銘柄データをロードする。ロード失敗時はNoneを返す。"""
    try:
        from src.data.data_manager import DataManager
        with DataManager() as dm:
            df = dm.fetch_ohlcv(ticker, start="2018-01-01")
        return df
    except Exception as exc:
        logger.warning(f"銘柄 '{ticker}' のデータ取得に失敗しました: {exc}")
        return None


def _run_single_benchmark(
    strategy_name: str,
    ticker: str,
) -> StrategyBenchmark:
    """単一の戦略×銘柄ベンチマークを実行する。"""
    import pandas as pd

    wf: WalkForwardResult | None = None
    mc: MonteCarloResult | None = None

    strategy = _load_strategy(strategy_name)
    data = _load_data(ticker)

    if strategy is not None and data is not None and isinstance(data, pd.DataFrame) and len(data) >= 630:
        # Walk-Forward分析
        try:
            from src.evaluation.walk_forward import walk_forward_analysis
            wf = walk_forward_analysis(strategy=strategy, data=data, symbol=ticker)  # type: ignore[arg-type]
            logger.info(f"  WF完了: {ticker} × {strategy_name} (walks={wf.num_walks})")
        except Exception as exc:
            logger.warning(f"  WF失敗: {ticker} × {strategy_name}: {exc}")

        # Monte Carloシミュレーション（WF OOSリターンを使用）
        if wf is not None:
            try:
                from src.evaluation.monte_carlo import monte_carlo_simulation
                oos_returns = [w["oos_return_pct"] / 100.0 for w in wf.walks]
                if oos_returns:
                    mc = monte_carlo_simulation(oos_returns)
                    logger.info(f"  MC完了: {ticker} × {strategy_name} (ruin={mc.probability_of_ruin:.2%})")
            except Exception as exc:
                logger.warning(f"  MC失敗: {ticker} × {strategy_name}: {exc}")
    else:
        logger.warning(
            f"  スキップ: {ticker} × {strategy_name} "
            f"(strategy={strategy is not None}, data={'OK' if data is not None else 'NG'})"
        )

    return run_strategy_benchmark(
        strategy_name=strategy_name,
        ticker=ticker,
        wf=wf,
        mc=mc,
    )


def _print_summary_table(benchmarks: List[StrategyBenchmark]) -> None:
    """サマリーテーブルをstdoutに出力する。"""
    header = f"{'戦略':<20} {'銘柄':<8} {'スコア':>6} {'推奨':>8}"
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)
    for b in sorted(benchmarks, key=lambda x: x.overall_score, reverse=True):
        print(f"{b.strategy_name:<20} {b.ticker:<8} {b.overall_score:>6} {b.recommendation:>8}")
    print(separator + "\n")


def main() -> None:
    """CLIエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description="全戦略×銘柄ベンチマークを実行し結果をJSONに保存する",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="実行する戦略名リスト（スペース区切り）",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="対象銘柄コードリスト（スペース区切り）",
    )
    args = parser.parse_args()

    strategies: List[str] = args.strategies
    tickers: List[str] = args.tickers

    logger.info(f"ベンチマーク開始: 戦略={strategies}, 銘柄={tickers}")

    benchmarks: List[StrategyBenchmark] = []

    for strategy_name in strategies:
        for ticker in tickers:
            logger.info(f"実行中: {strategy_name} × {ticker}")
            try:
                result = _run_single_benchmark(strategy_name, ticker)
                benchmarks.append(result)
            except Exception as exc:
                logger.error(f"ベンチマーク失敗: {strategy_name} × {ticker}: {exc}")

    # 結果を保存
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _RESULTS_DIR / f"benchmark_{date.today().isoformat()}.json"

    output: Dict[str, Any] = {
        "date": date.today().isoformat(),
        "strategies": strategies,
        "tickers": tickers,
        "results": [b.to_dict() for b in benchmarks],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"結果を保存しました: {output_path}")

    _print_summary_table(benchmarks)


if __name__ == "__main__":
    main()
