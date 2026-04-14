# AI Trader v2.0 ロードマップ (Phase 2-7)

最終更新: 2026-04-14
ステータス: Phase 1 完了 / Phase 2-7 計画策定済み

---

## 前提条件

- **Phase 1 完了** (commit d851260): AssetClass基盤 + マルチアセット対応
- **テスト**: 413パス, カバレッジ 84.43%
- **Python**: 3.10+
- **設計原則**: immutable dataclass, DI, 純粋関数優先
- **既存基盤**: 10戦略, 5データソース, 3ブローカー, LearningPipeline, TradingLoop

---

## Phase 2/7: 戦略最適化エンジン

### 目的
Optuna統合によるハイパーパラメータ自動最適化。既存の `parameter_space()` と `WalkForward` を活用し、
過学習を防止しながら戦略パラメータを体系的にチューニングする。

### 技術選定

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| optuna | >=4.0 | ベイズ最適化フレームワーク |
| optuna-dashboard | >=0.16 | 最適化結果の可視化 |

### 設計方針

**重要**: Optunaの objective 関数には **統計的指標** (cross-validated accuracy, log-loss) を使用する。
Sharpe ratio や equity curve ベースの財務指標を直接最適化すると、
TPEが効率的にノイズ面のグローバル最適解を見つけてしまい過学習となる。

代わりに以下のアプローチを採用:
1. Walk-Forward の OOS 期間で `consistency_ratio` を計算
2. Purged K-Fold Cross-Validation で統計的有意性を検証
3. 最終評価のみ財務指標 (Sharpe, Max DD) を使用

### 実装タスク

#### 2-1. OptunaOptimizer クラス (`src/optimization/optuna_optimizer.py`)
```python
@dataclass(frozen=True)
class OptimizationResult:
    strategy_name: str
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study_name: str
    walk_forward_results: List[WalkForwardResult]

class OptunaOptimizer:
    def __init__(self, strategy: BaseStrategy, sampler: str = "tpe",
                 pruner: str = "hyperband", n_trials: int = 50)
    def optimize(self, data: pd.DataFrame, ...) -> OptimizationResult
    def _objective(self, trial: optuna.Trial, data: pd.DataFrame) -> float
```

#### 2-2. Purged K-Fold CV (`src/optimization/purged_cv.py`)
- 金融データ用のクロスバリデーション (時系列の自己相関を考慮)
- embargo 期間設定で情報漏洩防止
- Walk-Forward の各ウォーク結果を fold として扱う

#### 2-3. parameter_space() 統合
- 既存5コア戦略の `parameter_space()` を Optuna の `suggest_*` に変換
- `(type, min, max)` → `trial.suggest_int()` / `trial.suggest_float()`

#### 2-4. CLI コマンド (`scripts/optimize.py`)
```bash
python scripts/optimize.py --strategy ma_crossover --symbol 7203.T \
  --start 2020-01-01 --n-trials 100
```

#### 2-5. API エンドポイント追加
- `POST /api/optimize` — 非同期最適化実行
- `GET /api/optimize/{study_name}` — 最適化結果取得

### テスト要件
- [ ] OptunaOptimizer の基本動作 (mock objective)
- [ ] parameter_space → suggest 変換の正確性
- [ ] PurgedCV の fold 分割 + embargo 検証
- [ ] WalkForward 統合テスト (小規模データ)
- [ ] API エンドポイント テスト

### 推定規模
- 新規ファイル: 4-5
- 変更ファイル: 3-4 (base.py, walk_forward.py, server.py, requirements.txt)
- 推定行数: +800-1,000

### リスク
- Optuna の trial 数増加で計算時間爆発 → n_trials 上限 + timeout 設定
- TPE の過学習 → OOS consistency_ratio を objective に使用して緩和

---

## Phase 3/7: リアルタイムデータ + WebSocket 強化

### 目的
ccxt.pro WebSocket と kabuステーション PUSH API を統合し、
リアルタイム価格フィード・板情報・約定情報を処理する。

### 技術選定

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| ccxt | >=4.0 (ccxt.pro 統合済み) | 暗号資産 WebSocket |
| orjson | >=3.10 | 高速 JSON パース (ccxt 推奨) |
| websockets | >=12.0 | kabuステーション PUSH API 用 |

### 実装タスク

#### 3-1. WebSocketFeed 本格化 (`src/data/ws_feed.py` 改修)
- 既存の `WebSocketFeed` クラスを拡張
- ccxt.pro の `watch_order_book()`, `watch_trades()`, `watch_ohlcv()` 対応
- 再接続ロジック (exponential backoff)
- メッセージバッファリング + バックプレッシャー制御

#### 3-2. KabuStation PUSH API クライアント (`src/data/kabu_push.py`)
- kabuステーション WebSocket (localhost:18080) 接続
- 銘柄登録 → リアルタイム板情報受信
- TickData への変換 + コールバック発火

#### 3-3. DataManager WebSocket 統合
- `DataManager` に `subscribe_realtime(symbol, callback)` 追加
- ソース自動切替: 株式 → kabuStation PUSH, 暗号 → ccxt.pro
- キャッシュ (SQLite) への定期保存 (1分OHLCV集約)

#### 3-4. 板情報データモデル (`src/models/orderbook.py`)
```python
@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: float

@dataclass(frozen=True)
class OrderBookSnapshot:
    symbol: str
    timestamp: float
    bids: tuple[OrderBookLevel, ...]  # immutable
    asks: tuple[OrderBookLevel, ...]
    exchange: str
```

#### 3-5. リアルタイムシグナル生成
- `BaseStrategy` に `generate_signal_realtime(tick: TickData)` 追加 (optional)
- OHLCV バーの動的構築 (TickData → 1分足集約)

### テスト要件
- [ ] WebSocketFeed 再接続テスト (mock exchange)
- [ ] TickData → OHLCV 集約の正確性
- [ ] OrderBookSnapshot の immutability
- [ ] DataManager subscribe/unsubscribe ライフサイクル
- [ ] kabu PUSH API パーサーテスト

### 推定規模
- 新規ファイル: 3-4
- 変更ファイル: 3 (ws_feed.py, data_manager.py, models/)
- 推定行数: +600-800

### 依存
- Phase 1 の `resolve_instrument()` でソース自動判別

---

## Phase 4/7: ブローカー実取引接続

### 目的
ペーパートレーディングから本番取引への移行。
暗号資産 (ccxt)、日本株 (kabuステーション)、米国株+先物 (Interactive Brokers) の
3系統のブローカー接続を本番化する。

### 技術選定

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| ccxt | >=4.0 | 暗号資産取引所 (binance, bitflyer, gmo) |
| ib_async | >=1.0 | Interactive Brokers TWS API (ib_insync後継) |
| kabuステーション API | REST v1 | auカブコム証券 日本株注文 |

**重要**: `ib_insync` の原作者 Ewald de Wit 氏は 2024年初頭に逝去。
後継プロジェクト `ib_async` (github.com/ib-api-reloaded/ib_async) を使用する。

### 実装タスク

#### 4-1. CCXTBroker 本番化 (`src/brokers/ccxt_broker.py` 改修)
- 既存コードは sandbox モード中心 → 本番モード対応
- レート制限ハンドリング (429 Too Many Requests)
- 注文状態の非同期ポーリング + WebSocket 監視
- 部分約定 (partial fill) の正確なトラッキング

#### 4-2. IBBroker 本番化 (`src/brokers/ib_broker.py` 改修)
- `ib_insync` → `ib_async` への移行
- 株式 (Stock)、先物 (Future)、ETF の注文サポート
- TWS/IB Gateway 接続管理 (自動再接続)
- マーケットデータのストリーミング受信

#### 4-3. KabuStationBroker 強化 (`src/brokers/kabu_station.py` 改修)
- 注文種別拡張: 逆指値、OCO、IFD
- ポジション取得の本格化 (残高照会 API)
- 信用取引対応 (制度信用/一般信用)

#### 4-4. OMS (Order Management System) (`src/trading/oms.py`)
```python
@dataclass(frozen=True)
class ManagedOrder:
    order: Order
    strategy_name: str
    target_quantity: float
    filled_quantity: float
    status: OMSStatus  # PENDING, PARTIAL, FILLED, CANCELLED, ERROR
    retries: int
    last_updated: str

class OrderManagementSystem:
    def submit(self, decision: TradeDecision, broker: BrokerBase) -> ManagedOrder
    def monitor(self) -> List[ManagedOrder]  # 未約定注文の状態更新
    def cancel_stale(self, max_age_seconds: int) -> List[ManagedOrder]
    def reconcile(self, broker: BrokerBase) -> ReconciliationResult
```

#### 4-5. BrokerFactory 拡張
- `create_broker(asset_class, config)` で適切なブローカー自動選択
- STOCK → KabuStation or IB, CRYPTO → CCXT, FUTURES → IB

### テスト要件
- [ ] CCXTBroker sandbox モードでの E2E テスト
- [ ] OMS 注文状態遷移テスト (全パス)
- [ ] BrokerFactory のルーティングテスト
- [ ] 部分約定シナリオテスト
- [ ] 再接続テスト (mock)

### 推定規模
- 新規ファイル: 2-3 (oms.py, etc.)
- 変更ファイル: 5 (ccxt_broker.py, ib_broker.py, kabu_station.py, broker_factory.py, trading_loop.py)
- 推定行数: +1,000-1,200

### リスク
- 実取引では注文失敗・ネットワーク障害が不可避 → リトライ + フォールバック必須
- kabuステーション API のレスポンス遅延 → タイムアウト設定
- ib_async の互換性問題 → ib_insync からの段階的移行

---

## Phase 5/7: 継続学習システム

### 目的
市場レジーム変化に自動適応する継続学習 (online learning) システム。
既存の `LearningPipeline` + `RegimeDetector` + `ThompsonBandit` を強化し、
concept drift 検出と自動モデル更新を実現する。

### 技術選定

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| river | >=0.21 | オンライン学習 + ドリフト検出 |
| frouros | >=0.8 | ドリフト検出 (補助) |

**River** は creme + scikit-multiflow の後継統合プロジェクト。
ストリーミングデータの機械学習に特化しており、DDM/ADWIN/KSWIN 等の
ドリフト検出アルゴリズムを網羅する。

### 実装タスク

#### 5-1. ConceptDriftDetector (`src/learning/drift_detector.py`)
```python
class DriftDetector:
    """複数のドリフト検出アルゴリズムを統合する。"""
    def __init__(self, methods: List[str] = ["adwin", "ddm", "kswin"])
    def update(self, value: float) -> DriftResult
    def reset(self) -> None

@dataclass(frozen=True)
class DriftResult:
    detected: bool
    method: str         # 検出したアルゴリズム名
    severity: str       # "warning" | "drift"
    timestamp: str
    details: Dict[str, float]
```

#### 5-2. OnlineLearner 統合 (`src/learning/online_learner.py`)
- River の `HoeffdingTreeClassifier` / `AdaptiveRandomForestClassifier` ラッパー
- `partial_fit(X, y)` でインクリメンタル学習
- 特徴量の動的追加・削除

#### 5-3. LearningPipeline 強化 (`src/learning/pipeline.py` 改修)
- `run_cycle()` にドリフト検出ステップを追加
- ドリフト検出時の自動アクション:
  - WARNING → 再訓練頻度を2倍に
  - DRIFT → 即時再訓練 + モデルA/Bテスト開始
- ModelRegistry に `promote()` / `rollback()` 追加

#### 5-4. A/B テストフレームワーク (`src/learning/ab_test.py`)
```python
@dataclass(frozen=True)
class ABTestResult:
    champion_id: str
    challenger_id: str
    champion_sharpe: float
    challenger_sharpe: float
    p_value: float
    winner: str  # "champion" | "challenger" | "inconclusive"

class ABTestManager:
    def start_test(self, champion: str, challenger: str, ...) -> str
    def update(self, test_id: str, champion_pnl: float, challenger_pnl: float)
    def evaluate(self, test_id: str, min_samples: int = 30) -> ABTestResult
```

#### 5-5. RegimeDetector 強化 (`src/learning/regime_detector.py` 改修)
- 現在の HMM ベースに加え、River の ADWIN による適応型レジーム検出
- レジーム遷移時のイベント発火

### テスト要件
- [ ] DriftDetector の各アルゴリズムでの検出精度テスト
- [ ] OnlineLearner の partial_fit 正確性テスト
- [ ] A/B テストの統計的有意性判定テスト
- [ ] LearningPipeline ドリフト→再訓練フローの統合テスト
- [ ] RegimeDetector の遷移イベント発火テスト

### 推定規模
- 新規ファイル: 3-4
- 変更ファイル: 3 (pipeline.py, regime_detector.py, model_registry.py)
- 推定行数: +900-1,100

### 依存
- Phase 2 (最適化) の結果をモデル更新時に利用
- Phase 3 (リアルタイムデータ) のストリーミング入力

---

## Phase 6/7: ポートフォリオ管理 + リスク管理強化

### 目的
マルチアセットポートフォリオの最適配分と動的リバランス。
既存の `CapitalController` (段階的資金投入) を強化し、
相関ベースのリスク分散と高度なリスク管理を実現する。

### 技術選定

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| skfolio | >=0.5 | ポートフォリオ最適化 (scikit-learn API) |
| riskfolio-lib | >=6.0 | 13リスク指標 + CVaR最適化 (補助) |

**skfolio** を主軸に採用。理由:
- scikit-learn API 準拠 → `fit()` / `predict()` で統一的に扱える
- Hierarchical Risk Parity (HRP) 対応
- クロスバリデーションによる過学習防止が組み込み

### 実装タスク

#### 6-1. PortfolioOptimizer (`src/risk/portfolio_optimizer.py`)
```python
@dataclass(frozen=True)
class PortfolioAllocation:
    weights: Dict[str, float]        # {strategy_or_asset: weight}
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    method: str                      # "hrp", "mean_variance", "risk_parity"
    timestamp: str

class PortfolioOptimizer:
    def __init__(self, method: str = "hrp", risk_measure: str = "cvar")
    def optimize(self, returns: pd.DataFrame) -> PortfolioAllocation
    def rebalance(self, current: Dict[str, float],
                  target: PortfolioAllocation) -> List[RebalanceOrder]
```

#### 6-2. CapitalController 強化 (`src/risk/capital_controller.py` 改修)
- PortfolioOptimizer の結果を `set_strategy_weights()` に反映
- 動的リバランス: 定期 (月次) + 乖離率トリガー (>5%)
- アセットクラス間の配分制約 (例: 株式 50-70%, 暗号 10-20%, 債券ETF 10-30%)

#### 6-3. CorrelationMonitor (`src/risk/correlation_monitor.py`)
- 戦略間・アセット間の相関行列の定期計算
- 相関急上昇 (>0.8) のアラート → 分散効果低下の警告
- 相関行列のローリング推定 (指数加重移動共分散)

#### 6-4. VaRGuard 強化 (`src/risk/var_guard.py` 改修)
- ヒストリカル VaR に加え、Parametric VaR + Monte Carlo VaR
- Expected Shortfall (CVaR) の計算
- ポートフォリオレベルの VaR (個別資産 → 全体)

#### 6-5. DrawdownController (`src/risk/drawdown_controller.py`)
```python
class DrawdownController:
    def __init__(self, max_drawdown: float = 0.10,
                 trailing_stop_pct: float = 0.05)
    def check(self, equity_curve: pd.Series) -> DrawdownAction
    # DrawdownAction: NORMAL, REDUCE_EXPOSURE, HALT_TRADING, EMERGENCY_EXIT
```

### テスト要件
- [ ] PortfolioOptimizer HRP/MeanVariance/RiskParity の各手法テスト
- [ ] リバランス注文生成の正確性テスト
- [ ] CorrelationMonitor のアラート閾値テスト
- [ ] VaR/CVaR 計算の精度テスト (既知データ)
- [ ] DrawdownController の全状態遷移テスト

### 推定規模
- 新規ファイル: 3-4
- 変更ファイル: 3 (capital_controller.py, var_guard.py, trading_loop.py)
- 推定行数: +800-1,000

### 依存
- Phase 4 (ブローカー) のリバランス注文実行
- Phase 5 (継続学習) のレジーム情報によるリスク調整

---

## Phase 7/7: 本番稼働 + モニタリング

### 目的
本番運用に必要なダッシュボード、通知、モニタリング、
自動レポート生成、Reconciliation の自動化を実装する。

### 技術選定

| ライブラリ | バージョン | 用途 |
|-----------|----------|------|
| plotly | >=6.0 | チャート描画 |
| jinja2 | >=3.1 | HTMLテンプレート |
| htmx | 2.x (CDN) | リアルタイム UI 更新 |
| line-bot-sdk | >=3.0 | LINE Messaging API 通知 |
| slack-sdk | >=3.30 | Slack 通知 |

**重要**: LINE Notify は 2025年3月31日終了済み。
LINE Messaging API (Official Account) に移行する。

### 実装タスク

#### 7-1. ダッシュボード UI (`src/api/dashboard/`)
- FastAPI + Jinja2 + HTMX でサーバーサイドレンダリング
- Plotly で描画するチャート:
  - エクイティカーブ (リアルタイム更新)
  - ポジション一覧 (PnL ヒートマップ)
  - 戦略別パフォーマンス (レーダーチャート)
  - リスク指標 (VaR, DD, 相関行列)
  - 最適化履歴 (Optuna パラメータ重要度)
- WebSocket で 5秒間隔のリアルタイム価格更新

#### 7-2. 通知システム (`src/notifications/`)
```python
class NotificationRouter:
    """通知先を一元管理する。"""
    def __init__(self, channels: List[NotificationChannel])
    async def send(self, event: TradingEvent) -> None

class LINEMessagingChannel(NotificationChannel):
    """LINE Messaging API (Official Account)。"""
    # LINE Notify 終了済み → Messaging API を使用

class SlackChannel(NotificationChannel):
    """Slack Incoming Webhook。"""

class DiscordChannel(NotificationChannel):
    """Discord Webhook (オプション)。"""
```

通知対象イベント:
- 注文約定 / 注文失敗
- サーキットブレーカー発動
- ドリフト検出
- ステージ昇格/降格
- 日次サマリー

#### 7-3. 自動レポート生成 (`src/reports/`)
- 日次レポート: PnL, 勝率, 取引数, ドローダウン
- 週次レポート: 戦略別パフォーマンス比較, 相関変化
- 月次レポート: 最適化結果サマリー, ステージ昇格進捗
- HTML + PDF (WeasyPrint) 出力

#### 7-4. Reconciliation 自動化 (`src/trading/reconciliation.py` 改修)
- ブローカー残高 vs 内部計算の自動突合
- 不一致検出時のアラート + 自動修正提案
- 日次バッチ実行 (スケジューラー連携)

#### 7-5. ヘルスチェック + 死活監視
```python
@app.get("/api/health")
async def health_check() -> HealthStatus:
    return HealthStatus(
        api=True,
        broker_connections={...},
        data_feeds={...},
        last_trade_age_seconds=...,
        circuit_breaker_state=...,
        memory_usage_mb=...,
    )
```

#### 7-6. 運用スクリプト (`scripts/`)
- `start_trading.py` — 全コンポーネント起動 (watchdog 付き)
- `daily_report.py` — 日次レポート生成 + 通知送信
- `reconcile.py` — Reconciliation バッチ実行

### テスト要件
- [ ] ダッシュボード各ページのレンダリングテスト
- [ ] WebSocket 接続・切断テスト
- [ ] 通知チャネル送信テスト (mock)
- [ ] レポート生成の出力フォーマットテスト
- [ ] Reconciliation 不一致検出テスト
- [ ] ヘルスチェック全項目テスト

### 推定規模
- 新規ファイル: 10-15 (dashboard templates, notification channels, reports)
- 変更ファイル: 5 (server.py, reconciliation.py, trading_loop.py, etc.)
- 推定行数: +1,500-2,000

---

## 全体スケジュール + 依存関係

```
Phase 2 (最適化)          ──────┐
                               ├──→ Phase 5 (継続学習)
Phase 3 (リアルタイム)    ──────┤
                               ├──→ Phase 6 (ポートフォリオ)
Phase 4 (ブローカー)      ──────┤
                               └──→ Phase 7 (本番稼働)
```

### 実装順序 (推奨)

| 順序 | Phase | 理由 |
|------|-------|------|
| 1st | Phase 2 | 他に依存なし。既存 parameter_space() + WalkForward を活用 |
| 2nd | Phase 3 | Phase 2 と独立。リアルタイムデータは P4, P5, P7 の前提 |
| 3rd | Phase 4 | Phase 3 のデータフィードが必要。本番取引の基盤 |
| 4th | Phase 5 | Phase 2 (最適化), Phase 3 (ストリーミング) が前提 |
| 5th | Phase 6 | Phase 4 (注文実行), Phase 5 (レジーム) が前提 |
| 6th | Phase 7 | 全 Phase の統合。最後に実装 |

### 全体推定規模

| Phase | 新規ファイル | 変更ファイル | 推定行数 | テスト数 |
|-------|------------|------------|---------|---------|
| P2 | 4-5 | 3-4 | +800-1,000 | ~25 |
| P3 | 3-4 | 3 | +600-800 | ~20 |
| P4 | 2-3 | 5 | +1,000-1,200 | ~30 |
| P5 | 3-4 | 3 | +900-1,100 | ~25 |
| P6 | 3-4 | 3 | +800-1,000 | ~25 |
| P7 | 10-15 | 5 | +1,500-2,000 | ~35 |
| **合計** | **~30** | **~20** | **+5,600-7,100** | **~160** |

---

## 依存パッケージ追加 (requirements.txt 更新予定)

```
# Phase 2
optuna>=4.0
optuna-dashboard>=0.16

# Phase 3
orjson>=3.10
websockets>=12.0

# Phase 4
ib_async>=1.0

# Phase 5
river>=0.21
frouros>=0.8

# Phase 6
skfolio>=0.5
riskfolio-lib>=6.0

# Phase 7
plotly>=6.0
htmx  # CDN, no pip install
line-bot-sdk>=3.0
slack-sdk>=3.30
weasyprint>=62.0
```

---

## 品質基準 (全 Phase 共通)

- テストカバレッジ: 80% 以上を維持
- 型アノテーション: 全関数に必須
- Immutable パターン: `@dataclass(frozen=True)` + DataFrame copy()
- セキュリティ: APIキーは環境変数、secrets.compare_digest 使用
- エラーハンドリング: 外部API呼び出しはリトライ + タイムアウト + ログ

---

## リサーチソース

- [Optuna TPE Sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Walk-Forward Optimization (PyQuant News)](https://www.pyquantnews.com/free-python-resources/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis)
- [Bayesian Hyperparameter Optimization with Purged CV (MQL5)](https://www.mql5.com/en/articles/20117)
- [CCXT Pro WebSocket Manual](https://docs.ccxt.com/ccxt.pro.manual)
- [ib_async (ib_insync successor)](https://github.com/ib-api-reloaded/ib_async)
- [kabuステーション API リファレンス](https://kabucom.github.io/kabusapi/reference/index.html)
- [River: Online ML in Python](https://github.com/online-ml/river)
- [Frouros: Drift Detection](https://www.researchgate.net/publication/379824915_Frouros_An_open-source_Python_library_for_drift_detection_in_machine_learning_systems)
- [skfolio: Portfolio Optimization](https://skfolio.org/)
- [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib)
- [LINE Notify 終了通知 → Messaging API 移行](https://ke2b.com/en/line-notify-closing-alt/)
- [FastAPI + HTMX + Plotly Dashboard](https://medium.com/codex/building-real-time-dashboards-with-fastapi-htmx-plotly-python-the-pure-python-charts-edition-2c29e77da953)
