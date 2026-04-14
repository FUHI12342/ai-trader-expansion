"""戦略パラメータ最適化 CLI。

使用例:
    python scripts/optimize.py --strategy ma_crossover --symbol 7203.T --n-trials 50
    python scripts/optimize.py --strategy macd_rsi --symbol AAPL --start 2020-01-01
    python scripts/optimize.py --strategy bollinger_rsi_adx --metric combined_score
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートを PYTHONPATH に追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.data_manager import DataManager
from src.optimization.optuna_optimizer import OptunaOptimizer

logger = logging.getLogger(__name__)

# 戦略名 → クラスのマッピング
STRATEGY_MAP = {}


def _load_strategies() -> None:
    """利用可能な戦略を遅延ロードする。"""
    from src.strategies.ma_crossover import MACrossoverStrategy
    from src.strategies.dual_momentum import DualMomentumStrategy
    from src.strategies.macd_rsi import MACDRSIStrategy
    from src.strategies.bollinger_rsi_adx import BollingerRSIADXStrategy

    STRATEGY_MAP.update({
        "ma_crossover": MACrossoverStrategy,
        "dual_momentum": DualMomentumStrategy,
        "macd_rsi": MACDRSIStrategy,
        "bollinger_rsi_adx": BollingerRSIADXStrategy,
    })

    try:
        from src.strategies.lgbm_predictor import LGBMPredictorStrategy
        STRATEGY_MAP["lgbm_predictor"] = LGBMPredictorStrategy
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Trader 戦略パラメータ最適化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python scripts/optimize.py --strategy ma_crossover --symbol 7203.T
  python scripts/optimize.py --strategy macd_rsi --n-trials 100 --metric combined_score
  python scripts/optimize.py --strategy bollinger_rsi_adx --start 2018-01-01 --seed 42
        """,
    )
    parser.add_argument(
        "--strategy", required=True,
        help="戦略名 (ma_crossover, dual_momentum, macd_rsi, bollinger_rsi_adx, lgbm_predictor)",
    )
    parser.add_argument("--symbol", default="7203.T", help="銘柄コード (デフォルト: 7203.T)")
    parser.add_argument("--start", default=None, help="開始日 (YYYY-MM-DD, デフォルト: 5年前)")
    parser.add_argument("--end", default=None, help="終了日 (YYYY-MM-DD, デフォルト: 今日)")
    parser.add_argument("--n-trials", type=int, default=50, help="試行回数 (デフォルト: 50)")
    parser.add_argument("--timeout", type=float, default=None, help="タイムアウト秒数")
    parser.add_argument(
        "--metric", default="consistency_ratio",
        choices=["consistency_ratio", "oos_sharpe_avg", "degradation_ratio", "combined_score"],
        help="最適化指標 (デフォルト: consistency_ratio)",
    )
    parser.add_argument("--is-days", type=int, default=504, help="インサンプル日数 (デフォルト: 504)")
    parser.add_argument("--oos-days", type=int, default=126, help="アウトオブサンプル日数 (デフォルト: 126)")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="初期資金")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument("--output", default=None, help="結果出力ファイル (JSON)")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ出力")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _load_strategies()

    if args.strategy not in STRATEGY_MAP:
        print(f"エラー: 未知の戦略 '{args.strategy}'")
        print(f"利用可能: {sorted(STRATEGY_MAP.keys())}")
        sys.exit(1)

    # データ取得
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start or (datetime.now() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    print(f"データ取得中: {args.symbol} ({start_date} ~ {end_date})")
    dm = DataManager()
    data = dm.fetch_ohlcv(args.symbol, start_date, end_date)

    if data.empty:
        print("エラー: データが取得できませんでした。")
        sys.exit(1)

    print(f"データ取得完了: {len(data)} 行")

    # 戦略インスタンス生成
    strategy = STRATEGY_MAP[args.strategy]()
    print(f"戦略: {strategy.name}")
    print(f"パラメータ空間: {strategy.parameter_space()}")

    # 最適化実行
    optimizer = OptunaOptimizer(
        strategy=strategy,
        n_trials=args.n_trials,
        timeout=args.timeout,
        objective_metric=args.metric,
        seed=args.seed,
    )

    print(f"\n最適化開始 (n_trials={args.n_trials}, metric={args.metric})")
    print("=" * 60)

    result = optimizer.optimize(
        data=data,
        symbol=args.symbol,
        in_sample_days=args.is_days,
        out_of_sample_days=args.oos_days,
        initial_capital=args.capital,
    )

    # 結果表示
    print("=" * 60)
    print(f"\n最適化完了: {result.strategy_name}")
    print(f"  試行数: {result.n_trials} (完了: {result.n_complete}, 枝刈り: {result.n_pruned})")
    print(f"  最良スコア: {result.best_value:.4f} ({result.objective_metric})")
    print(f"  最良パラメータ: {result.best_params}")

    if result.walk_forward_result:
        wf = result.walk_forward_result
        print(f"\n  Walk-Forward 検証:")
        print(f"    ウォーク数: {wf.get('num_walks', 'N/A')}")
        print(f"    OOS平均リターン: {wf.get('out_of_sample_avg_return', 0):.2f}%")
        print(f"    OOS平均Sharpe: {wf.get('oos_sharpe_avg', 0):.4f}")
        print(f"    Consistency: {wf.get('consistency_ratio', 0):.2f}")
        print(f"    統計的有意性: {wf.get('is_statistically_significant', False)}")

    # JSON 出力
    if args.output:
        output_data = {
            "strategy_name": result.strategy_name,
            "best_params": result.best_params,
            "best_value": result.best_value,
            "n_trials": result.n_trials,
            "n_complete": result.n_complete,
            "n_pruned": result.n_pruned,
            "objective_metric": result.objective_metric,
            "walk_forward_result": result.walk_forward_result,
            "all_trials": result.all_trials,
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
        print(f"\n結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()
