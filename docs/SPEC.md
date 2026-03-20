# AI Trader 株式拡張 — 仕様書 (SPEC.md)

最終更新: 2026-03-19

---

## 概要

本プロジェクトは株式市場向けのAIトレーダー拡張システムです。
複数の取引戦略、評価フレームワーク、データ取得クライアント、
ブローカー接続、FastAPI REST APIを統合したMVPです。

---

## システムコンポーネント仕様

### 1. 設定管理 (`config/settings.py`)

| 設定クラス | 環境変数 | デフォルト値 |
|-----------|---------|------------|
| JQuantsSettings.email | JQUANTS_EMAIL | "" |
| JQuantsSettings.password | JQUANTS_PASSWORD | "" |
| EdinetSettings.api_key | EDINET_API_KEY | "" |
| KabuStationSettings.api_password | KABU_API_PASSWORD | "" |
| Settings.initial_capital | TRADER_INITIAL_CAPITAL | 1,000,000 |
| Settings.fee_rate | TRADER_FEE_RATE | 0.001 |
| Settings.slippage_rate | TRADER_SLIPPAGE_RATE | 0.0005 |

全設定はfrozen=TrueのDataclassで実装（immutableパターン）。
`get_settings()` はlru_cacheでシングルトン提供。

---

### 2. 戦略仕様 (`src/strategies/`)

#### 2.1 BaseStrategy (base.py)

**抽象メソッド:**
- `generate_signals(data: pd.DataFrame) -> pd.Series`
  - 入力: OHLCV DataFrame（必須カラム: open, high, low, close, volume）
  - 出力: SignalType Series（1=BUY, -1=SELL, 0=FLAT）

**共通メソッド:**
- `validate_data(data)` — 入力バリデーション
- `get_parameters() -> dict` — 現在パラメータを返す
- `set_parameters(**kwargs) -> BaseStrategy` — 新しいインスタンスを返す（immutable）

---

#### 2.2 MACrossoverStrategy (ma_crossover.py)

| パラメータ | 型 | デフォルト | 説明 |
|-----------|---|-----------|-----|
| short_window | int | 20 | 短期MA期間 |
| long_window | int | 100 | 長期MA期間 |
| price_col | str | "close" | 使用価格カラム |

**ロジック:**
- ゴールデンクロス（short > long かつ 前日 short ≤ long）→ BUY
- デッドクロス（short < long かつ 前日 short ≥ long）→ SELL
- その他 → FLAT

---

#### 2.3 DualMomentumStrategy (dual_momentum.py)

| パラメータ | 型 | デフォルト | 説明 |
|-----------|---|-----------|-----|
| lookback_period | int | 252 | モメンタム計算期間（日） |
| threshold | float | 0.0 | BUY閾値リターン率 |
| rebalance_freq | str | "ME" | リバランス頻度 |

**ロジック:**
- lookback_period日前比リターン > threshold → BUY
- リバランス日（月末）のみシグナル更新
- 前回BUYで今回条件不成立 → SELL

---

#### 2.4 MACDRSIStrategy (macd_rsi.py)

| パラメータ | 型 | デフォルト | 説明 |
|-----------|---|-----------|-----|
| macd_fast | int | 12 | MACD短期EMA |
| macd_slow | int | 26 | MACD長期EMA |
| macd_signal | int | 9 | MACDシグナル線EMA |
| rsi_period | int | 14 | RSI期間 |
| rsi_overbought | float | 70 | 過買水準 |
| rsi_oversold | float | 30 | 過売水準 |

**ロジック:**
- MACDゴールデンクロス かつ RSI < rsi_overbought → BUY
- MACDデッドクロス かつ RSI > rsi_oversold → SELL
- 両条件が一致した場合のみエントリー（AND条件）

---

#### 2.5 BollingerRSIADXStrategy (bollinger_rsi_adx.py)

| パラメータ | 型 | デフォルト | 説明 |
|-----------|---|-----------|-----|
| bb_period | int | 20 | BB期間 |
| bb_std | float | 2.0 | BB標準偏差倍数 |
| rsi_period | int | 14 | RSI期間 |
| adx_period | int | 14 | ADX期間 |
| adx_threshold | float | 25 | ADXトレンド閾値 |

**ロジック:**
- 終値 ≤ 下バンド かつ RSI < 50 かつ ADX > threshold → BUY
- 終値 ≥ 上バンド かつ RSI > 50 かつ ADX > threshold → SELL

---

#### 2.6 LGBMPredictorStrategy (lgbm_predictor.py)

| パラメータ | 型 | デフォルト | 説明 |
|-----------|---|-----------|-----|
| train_window | int | 504 | 学習期間（日） |
| predict_window | int | 126 | 予測期間（日） |
| prob_threshold | float | 0.55 | BUY確率閾値 |

**特徴量:** SMA(5,20,60), RSI(14), MACD, BB位置, 出来高変化率, 曜日, 過去リターン(1,5,20日)

**ラベル:** 翌日リターン > 0 → 1, else 0

**方式:** Walk-Forward（未来データ漏洩防止）

---

### 3. 評価システム仕様 (`src/evaluation/`)

#### 3.1 メトリクス (metrics.py)

`calculate_metrics()` が返す `EvaluationResult` フィールド:

| フィールド | 説明 |
|-----------|-----|
| total_return_pct | 総リターン（%） |
| annualized_return_pct | 年率リターン（%） |
| max_drawdown_pct | 最大ドローダウン（%、負値） |
| sharpe_ratio | シャープレシオ（年率） |
| sortino_ratio | ソルティノレシオ（年率） |
| calmar_ratio | カルマーレシオ |
| win_rate | 勝率（0〜1） |
| profit_factor | プロフィットファクター |
| total_trades | 総取引数 |
| volatility_pct | 年率ボラティリティ（%） |

---

#### 3.2 バックテスター (backtester.py)

**手数料・スリッページ:**
- 買い実行価格: close × (1 + slippage_rate)
- 売り実行価格: close × (1 - slippage_rate)
- 手数料: 実行価格 × 株数 × fee_rate（往復）

**制約:**
- DataFrameはすべてcopy()でimmutableパターン
- ロングオンリー（allow_short=Falseデフォルト）

---

#### 3.3 Walk-Forward (walk_forward.py)

| パラメータ | デフォルト | 説明 |
|-----------|-----------|-----|
| in_sample_days | 504 | IS期間（≈ 2年） |
| out_of_sample_days | 126 | OOS期間（≈ 半年） |
| step_days | None | スライド幅（デフォルト=OOSと同じ） |

`WalkForwardResult` フィールド:
- `consistency_ratio`: OOSでプラスのウォーク割合
- `degradation_ratio`: IS/OOSリターン比（1に近いほど良い）
- `is_statistically_significant`: p < 0.05 かつ OOS平均 > 0

---

#### 3.4 モンテカルロ (monte_carlo.py)

- 取引リターン列を1000回ランダムシャッフル
- 各シミュレーションでエクイティカーブと最大DDを計算
- `probability_of_ruin`: 最大DD > -50%となる確率

---

#### 3.5 統計検定 (statistics.py)

- **t検定**: 帰無仮説「平均リターン = 0」の両側検定
- **ブートストラップ信頼区間**: 10000回リサンプリングで95%/99%CI計算

---

### 4. データクライアント仕様 (`src/data/`)

#### 4.1 YFinanceClient

- `fetch_ohlcv(symbol, start, end, interval)` → OHLCV DataFrame
- `fetch_multiple(symbols, ...)` → {symbol: DataFrame}
- メモリキャッシュ対応（cache_enabled=True時）

#### 4.2 JQuantsClient

**認証フロー:**
1. `POST /token/auth_user` (email + password) → refreshToken
2. `POST /token/auth_refresh` (refreshToken) → idToken
3. 以降のリクエストに `Authorization: Bearer {idToken}` ヘッダー

**主要メソッド:**
- `authenticate()` — 認証実行
- `fetch_stock_prices(code, date_from, date_to)` → OHLCV DataFrame
- `fetch_listed_info()` → 上場銘柄一覧 DataFrame

#### 4.3 EdinetClient

**主要メソッド:**
- `fetch_document_list(date)` → 書類一覧 DataFrame
- `fetch_document(doc_id, file_type)` → bytes（PDF/XBRL）
- `search_filings(edinetcode, doc_type_code, date_from, date_to)` → DataFrame

#### 4.4 DataManager

- ソース切替: "yfinance", "jquants", "auto"（autoはJ-Quants優先）
- SQLiteキャッシュ（デフォルト1日有効期限）
- `force_refresh=True` でキャッシュ無視して再取得

---

### 5. ブローカー仕様 (`src/brokers/`)

#### 5.1 BrokerBase (ABC)

**必須実装メソッド:**
- `get_balance() -> float`
- `get_positions() -> Dict[str, Position]`
- `place_order(symbol, side, order_type, quantity, price) -> Order`
- `cancel_order(order_id) -> bool`
- `get_order(order_id) -> Optional[Order]`
- `get_open_orders(symbol) -> List[Order]`

#### 5.2 PaperBroker

- 即時約定（成行・指値ともに現在価格で約定）
- スリッページ適用（buy: +slippage, sell: -slippage）
- 手数料適用（片道 fee_rate）
- `reset()` でリセット可能（テスト用）

#### 5.3 KabuStationBroker

- kabuステーション® REST API接続（localhost:18080）
- `KABU_API_PASSWORD` 環境変数必須
- トークン認証 → `X-API-KEY` ヘッダー
- 指数バックオフ3回リトライ

---

### 6. FastAPI サーバー仕様 (`src/api/server.py`)

| エンドポイント | メソッド | 説明 |
|--------------|---------|-----|
| / | GET | ヘルスチェック |
| /api/status | GET | 全戦略のGo/Nogoステータス |
| /api/positions | GET | 現在ポジション一覧 |
| /api/performance | GET | パフォーマンスメトリクス |
| /api/backtest | POST | オンデマンドバックテスト |

**デフォルトポート:** 8765（環境変数 TRADER_API_PORT で変更可）

**CORSオリジン:** localhost:3000, localhost:5173（SHANON Desktop）
