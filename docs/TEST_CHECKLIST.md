# AI Trader v2.0 テストチェックリスト

最終更新: 2026-04-14

---

## 1. テスト種別と目的

| 種別 | 目的 | ツール | 頻度 |
|------|------|--------|------|
| **ユニットテスト** | 個別モジュールの正確性 | pytest | コミット毎 |
| **バックテスト** | 戦略の過去パフォーマンス | run_backtest() | 戦略変更時 |
| **Walk-Forward** | OOS (未知データ) での堅牢性 | walk_forward_analysis() | 最適化後 |
| **ストレステスト** | 極端な市場条件での耐性 | 手動シナリオ | リリース前 |
| **統合テスト** | コンポーネント間連携 | pytest | 週次 |
| **ペーパートレード** | リアルタイム動作確認 | auto_trader.py --mode paper | 常時 |

---

## 2. ユニットテスト チェックリスト

### 2-1. 戦略 (src/strategies/)
```
[x] MA Crossover — シグナル生成、parameter_space()
[x] MACD RSI — シグナル生成、パラメータ検証
[x] Bollinger RSI ADX — シグナル生成、高選択性
[x] Dual Momentum — エラーハンドリング
[x] BaseStrategy — set_parameters(), generate_signal_realtime()
[x] Funding Rate Arb — FR取得、エントリー/エグジット記録
[x] Grid Trading — レベル生成、レンジバックテスト
[x] Pairs Trading — ヘッジ比率、共和分検定、Z-score
```

### 2-2. リスク管理 (src/risk/)
```
[x] PositionSizer — Kelly計算、Half/Quarter Kelly
[x] PositionSizer — 日次損失制限ブロック
[x] PositionSizer — VIX連動調整 (4段階)
[x] PositionSizer — 最大/最小ポジション制限
[x] DrawdownController — 4段階アクション (NORMAL→EMERGENCY)
[x] CorrelationMonitor — 高相関アラート
[x] PortfolioOptimizer — HRP/EqualWeight/RiskParity
[x] PortfolioOptimizer — リバランス注文生成
[x] CircuitBreaker — 連続失敗でOPEN
[x] VaRGuard — VaR計算
```

### 2-3. 注文管理 (src/trading/)
```
[x] OMS — 成行注文即約定 (FILLED)
[x] OMS — 指値注文 (SUBMITTED)
[x] OMS — 部分約定検出 (PARTIAL)
[x] OMS — ステイル注文キャンセル
[x] OMS — リトライ (最大回数制限)
[x] OMS — 発注失敗 → ERROR
[x] BrokerFactory — アセットクラス別ルーティング
```

### 2-4. データ (src/data/)
```
[x] WebSocketFeed — subscribe/unsubscribe
[x] WebSocketFeed — バックプレッシャー制御
[x] WebSocketFeed — exponential backoff再接続
[x] BarBuilder — TickData → 1分足集約
[x] OrderBookSnapshot — immutable、スプレッド計算
[x] DataManager — subscribe_realtime()
```

### 2-5. 学習 (src/learning/)
```
[x] DriftDetector — フォールバック検出器
[x] DriftDetector — 安定データでドリフトなし
[x] DriftDetector — レジーム変化で検出
[x] ABTestManager — テスト開始/更新/評価
[x] ABTestManager — チャレンジャー勝利判定
[x] ABTestManager — サンプル不足 → inconclusive
[x] Welch t-test — 同一分布/異なる分布
```

### 2-6. 最適化 (src/optimization/)
```
[x] PurgedWalkForwardCV — fold分割、embargo
[x] PurgedWalkForwardCV — expanding/rolling window
[x] OptunaOptimizer — parameter_space → suggest変換
[x] OptunaOptimizer — combined_score計算
[x] OptunaOptimizer — パラメータマッピング
```

### 2-7. 通知 (src/notifications/)
```
[x] TradingEvent — immutable、自動timestamp
[x] LogChannel — 送信成功
[x] NotificationRouter — 複数チャネル配信
[x] NotificationRouter — 失敗チャネルのエラーカウント
[x] NotificationRouter — イベントフィルタ
```

### 2-8. LLMアドバイザー (src/advisors/)
```
[x] LLMAdvice — signal変換 (BUY→1, SELL→-1)
[x] LLMAdvisor — API不可時フォールバック
[x] LLMAdvisor — JSONパース (正常/異常)
[x] Ensemble — 両方BUY → 実行
[x] Ensemble — 矛盾 → FLAT
[x] Ensemble — LLM高確信FLAT → テクニカル拒否
[x] Ensemble — テクニカルFLAT + LLM高確信BUY → 実行
```

---

## 3. バックテスト チェックリスト

### 3-1. 必須バックテスト
```
[ ] SPY 5年 (2021-2026) — 全戦略
[ ] QQQ 5年 — 全戦略
[ ] 7203.T 5年 — 日本株代表
[ ] BTC/USDT 3年 — 暗号資産 (ccxtインストール後)
[ ] 期間別比較: 5Y/3Y/1Y/6M/3M — 同一戦略
```

### 3-2. バックテスト品質基準
```
[ ] Sharpe > 0.5 (リスク調整後リターンが有意)
[ ] MaxDD < 25% (資金管理上の許容範囲)
[ ] 取引数 > 20 (統計的信頼性)
[ ] Profit Factor > 1.5 (利益が損失の1.5倍以上)
[ ] Buy&Hold比較 — 上回っているか記録
```

### 3-3. 過学習検出
```
[ ] Walk-Forward OOS リターン > 0%
[ ] IS vs OOS の degradation_ratio < 0.5
[ ] Consistency ratio > 0.3
[ ] 異なるシード (seed=42,123,777) で結果が安定
```

---

## 4. ストレステスト チェックリスト

### 4-1. 市場ストレス
```
[ ] フラッシュクラッシュ: 1日で-10%下落 → DD制御が発動するか
[ ] ボラティリティ急騰: VIX 40超 → ポジション縮小するか
[ ] レンジ相場: 6ヶ月横ばい → MA Crossoverが取引ゼロでも損失なしか
[ ] ギャップアップ/ダウン: 前日比+5%/-5%の寄付 → スリッページ処理
[ ] 流動性枯渇: 出来高が通常の10%以下 → 注文拒否されるか
```

### 4-2. システムストレス
```
[ ] API切断 5分 → 自動再接続
[ ] API切断 30分 → ポジション安全か
[ ] プロセスクラッシュ → 再起動後にポジション状態復元
[ ] メモリ不足 (RSS > 1GB) → 検知可能か
[ ] ディスク満杯 → ログローテーション動作
```

### 4-3. 注文ストレス
```
[ ] 部分約定 (50%のみ約定) → OMS追跡
[ ] 注文拒否 → リトライ + 最大回数
[ ] ステイル注文 (5分未約定) → 自動キャンセル
[ ] 二重注文防止 → 同一銘柄の重複チェック
```

---

## 5. 統合テスト チェックリスト

### 5-1. エンドツーエンドフロー
```
[ ] データ取得 → シグナル → PositionSizer → OMS → ブローカー
[ ] ドリフト検出 → 通知送信 → 再最適化トリガー
[ ] DD超過 → DrawdownController → 全注文停止
[ ] 日次レポート → 通知ルーター → ログ出力
```

### 5-2. コンポーネント間連携
```
[ ] DataManager → BaseStrategy.generate_signals()
[ ] OptunaOptimizer → WalkForward → BacktestResult
[ ] LLMAdvisor → ensemble() → OMS.submit()
[ ] CorrelationMonitor → PortfolioOptimizer.rebalance()
[ ] DriftDetector → ABTestManager → モデル切替
```

---

## 6. ペーパートレード チェックリスト

### 6-1. 起動前
```
[ ] config JSONの設定値確認 (capital, strategies, symbols)
[ ] DD閾値が適切か (元本の3%/5%/8%)
[ ] APIキー不要（ペーパーモード）
[ ] ログディレクトリ存在確認
```

### 6-2. 起動直後 (5分以内)
```
[ ] プロセスが存在するか (tasklist | grep python)
[ ] ログにエラーがないか
[ ] データ取得成功のログ確認
[ ] シグナル生成ログ (FLAT でも正常)
```

### 6-3. 24時間後
```
[ ] プロセスがまだ存在するか
[ ] メモリ使用量が異常増加していないか
[ ] ログファイルサイズが適切か (< 100MB)
[ ] 1回以上のシグナル評価が行われたか
```

### 6-4. 1週間後
```
[ ] 取引が1件以上発生したか (Bollingerは低頻度)
[ ] DD制御が閾値内に収まっているか
[ ] エラーログの頻度 (1日1件以下なら正常)
```

---

## 7. リリース前 最終チェックリスト

### 7-1. コード品質
```
[ ] 全ユニットテスト PASS
[ ] カバレッジ 80% 以上 (対象モジュール)
[ ] 型アノテーション付き (mypy --strict)
[ ] ハードコード値なし (設定ファイル or 環境変数)
```

### 7-2. セキュリティ
```
[ ] APIキーがコードにハードコードされていない
[ ] .env が .gitignore に含まれている
[ ] API通信がHTTPS
[ ] ブローカーAPIのレート制限対応
```

### 7-3. 運用準備
```
[ ] OPERATIONS_GUIDE.md の手順に従って環境構築完了
[ ] 通知チャネル (Slack/Discord) テスト送信成功
[ ] バックアップ手順確認 (trade_journal.db)
[ ] 緊急停止手順の確認 (Ctrl+C / kill)
```

---

## 8. テスト実行コマンド

```bash
# 全ユニットテスト
python -m pytest tests/ --no-cov -v

# 特定テストファイル
python -m pytest tests/test_position_sizer.py -v --no-cov

# カバレッジ付き
python -m pytest tests/ --cov=src --cov-report=html

# 高速実行 (カバレッジなし、verbose なし)
python -m pytest tests/ --no-cov -q

# バックテスト (ペーパー検証)
python scripts/auto_trader.py --mode paper --symbols SPY,QQQ

# 最適化
python scripts/auto_trader.py --mode optimize --symbols SPY
```

---

## 9. 現在のテスト状況

| テストファイル | テスト数 | 状態 |
|--------------|---------|------|
| test_optimization.py | 25 | PASS |
| test_ws_feed.py | 32 | PASS |
| test_oms.py | 23 | PASS |
| test_drift_and_ab.py | 28 | PASS |
| test_portfolio_risk.py | 24 | PASS |
| test_notifications.py | 14 | PASS |
| test_llm_advisor.py | 19 | PASS |
| test_auto_trader.py | 8 | PASS |
| test_new_strategies.py | 20 | PASS |
| test_position_sizer.py | 19 | PASS |
| **合計** | **212** | **ALL PASS** |
