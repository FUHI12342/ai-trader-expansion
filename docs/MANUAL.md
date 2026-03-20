# AI Trader 株式拡張 — 使い方マニュアル (MANUAL.md)

最終更新: 2026-03-19

---

## 1. インストール

### 前提条件

- Python 3.10 以上
- pip
- (推奨) 仮想環境 (venv / conda)

### セットアップ手順

```bash
# 1. リポジトリのクローン
cd /c/_dev/repos
git clone <repository-url> ai-trader-expansion
cd ai-trader-expansion

# 2. 仮想環境の作成と有効化
python -m venv .venv
# Windows (Git Bash)
source .venv/Scripts/activate
# Linux/Mac
source .venv/bin/activate

# 3. 依存パッケージのインストール
pip install -r requirements.txt

# 4. 開発用パッケージのインストール（テスト実行に必要）
pip install -e ".[dev]"
```

---

## 2. 環境変数の設定

秘密情報は環境変数で管理します。`.env` ファイルは **コミットしないこと**。

```bash
# J-Quants API（日本株データ取得）
export JQUANTS_EMAIL="your.email@example.com"
export JQUANTS_PASSWORD="your_password"

# EDINET API（開示書類取得）
export EDINET_API_KEY="your_api_key"

# kabuステーション API（実取引）
export KABU_API_PASSWORD="your_kabu_password"
export KABU_API_HOST="localhost"
export KABU_API_PORT="18080"

# バックテスト設定
export TRADER_INITIAL_CAPITAL="1000000"
export TRADER_FEE_RATE="0.001"
export TRADER_SLIPPAGE_RATE="0.0005"

# APIサーバー設定
export TRADER_API_PORT="8765"
```

---

## 3. テストの実行

```bash
cd /c/_dev/repos/ai-trader-expansion

# 全テスト実行（カバレッジレポート付き）
pytest

# 特定のテストファイル
pytest tests/test_strategies.py -v

# 特定のテスト関数
pytest tests/test_evaluation.py::TestCalculateMetrics::test_total_return_calculation

# カバレッジなしで高速実行
pytest --no-cov
```

### テスト結果の確認

```bash
# HTMLカバレッジレポート
open htmlcov/index.html   # Mac/Linux
start htmlcov/index.html  # Windows
```

---

## 4. APIサーバーの起動

```bash
# サーバー起動（デフォルト: http://localhost:8765）
python -m src.api.server

# またはuvicornで直接起動
uvicorn src.api.server:app --host 0.0.0.0 --port 8765 --reload

# APIドキュメント確認
# ブラウザで http://localhost:8765/docs を開く
```

---

## 5. バックテストの実行例

### Python スクリプトから

```python
from src.data.data_manager import DataManager
from src.strategies.macd_rsi import MACDRSIStrategy
from src.evaluation.backtester import run_backtest

# データ取得
dm = DataManager(source="yfinance")
data = dm.fetch_ohlcv("7203.T", "2020-01-01", "2024-12-31")

# 戦略の初期化
strategy = MACDRSIStrategy(
    macd_fast=12,
    macd_slow=26,
    rsi_period=14,
)

# バックテスト実行
result = run_backtest(
    strategy=strategy,
    data=data,
    symbol="7203.T",
    initial_capital=1_000_000,
    fee_rate=0.001,
)

# 結果確認
metrics = result.metrics
print(f"総リターン: {metrics.total_return_pct:.2f}%")
print(f"シャープレシオ: {metrics.sharpe_ratio:.2f}")
print(f"最大DD: {metrics.max_drawdown_pct:.2f}%")
print(f"勝率: {metrics.win_rate:.1%}")
print(f"総取引数: {metrics.total_trades}")
```

### API経由で実行

```bash
curl -X POST http://localhost:8765/api/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_name": "macd_rsi",
    "symbol": "7203.T",
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 1000000
  }'
```

---

## 6. Walk-Forward 分析の実行例

```python
from src.evaluation.walk_forward import walk_forward_analysis
from src.strategies.bollinger_rsi_adx import BollingerRSIADXStrategy

strategy = BollingerRSIADXStrategy()
result = walk_forward_analysis(
    strategy=strategy,
    data=data,
    symbol="7203.T",
    in_sample_days=504,
    out_of_sample_days=126,
)

print(f"ウォーク数: {result.num_walks}")
print(f"IS平均リターン: {result.in_sample_avg_return:.2f}%")
print(f"OOS平均リターン: {result.out_of_sample_avg_return:.2f}%")
print(f"一貫性: {result.consistency_ratio:.1%}")
print(f"統計的有意: {result.is_statistically_significant}")
print(f"p値: {result.p_value:.4f}")
```

---

## 7. モンテカルロシミュレーション

```python
from src.evaluation.monte_carlo import monte_carlo_simulation

# バックテスト結果から取引リターンを抽出
trade_returns = [t["pnl_pct"] / 100 for t in result.trades]

mc_result = monte_carlo_simulation(
    trade_returns=trade_returns,
    num_simulations=1000,
    initial_capital=1_000_000,
)

print(f"平均リターン: {mc_result.mean_return_pct:.2f}%")
print(f"95%CI: [{mc_result.confidence_95_lower:.2f}%, {mc_result.confidence_95_upper:.2f}%]")
print(f"損失確率: {mc_result.probability_of_loss:.1%}")
print(f"破産確率: {mc_result.probability_of_ruin:.1%}")
```

---

## 8. ペーパートレード

```python
from src.brokers.paper_broker import PaperBroker
from src.brokers.base import OrderSide, OrderType

broker = PaperBroker(initial_balance=1_000_000)

# 現在価格を更新
broker.update_price("7203.T", 2500.0)

# 買い注文
order = broker.place_order(
    "7203.T",
    OrderSide.BUY,
    OrderType.MARKET,
    quantity=100,
)
print(f"注文ID: {order.order_id}, 約定価格: {order.avg_fill_price:.0f}円")

# ポジション確認
pos = broker.get_positions()["7203.T"]
print(f"保有数: {pos.quantity}株, 含み損益: {pos.unrealized_pnl:.0f}円")

# 残高確認
print(f"残高: {broker.get_balance():.0f}円")
print(f"総資産: {broker.get_equity():.0f}円")
```

---

## 9. デスクトップショートカットの作成

```bash
python scripts/create_shortcut.py
```

---

## 10. よくある質問 (FAQ)

**Q: J-Quantsが認証エラーになる**
A: `JQUANTS_EMAIL` と `JQUANTS_PASSWORD` が正しく設定されているか確認してください。
J-Quantsの無料プランでは取得できるデータに制限があります。

**Q: kabuステーションに接続できない**
A: kabuステーション® が起動しているか、ポート18080で待ち受けているか確認してください。
APIパスワードは kabuステーション® の設定画面で確認できます。

**Q: テストが失敗する（ImportError: No module named 'lightgbm'）**
A: `pip install lightgbm` でインストールしてください。
LGBMPredictorStrategyのテストをスキップする場合: `pytest -k "not lgbm"`

**Q: バックテストの結果が毎回異なる**
A: シード値が固定されているか確認してください。LGBMは内部でシードを設定しています。
ランダム性が問題の場合は `lgbm_params={"random_state": 42}` を渡してください。
