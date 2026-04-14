# AI Trader v2.0 運用ガイドブック

最終更新: 2026-04-14
対象: ペーパー検証 → 実運用 → スケールアップ → 緊急対応

---

## 目次

1. [実運用開始前の準備チェックリスト](#1-実運用開始前の準備チェックリスト)
2. [ペーパー検証フェーズ](#2-ペーパー検証フェーズ)
3. [実運用移行判定基準](#3-実運用移行判定基準)
4. [実運用開始手順](#4-実運用開始手順)
5. [日常運用ルーティン](#5-日常運用ルーティン)
6. [利益確定・出金ルール](#6-利益確定出金ルール)
7. [スケールアップ基準](#7-スケールアップ基準)
8. [緊急時対応マニュアル](#8-緊急時対応マニュアル)
9. [障害対応フローチャート](#9-障害対応フローチャート)
10. [税務・記録管理](#10-税務記録管理)

---

## 1. 実運用開始前の準備チェックリスト

### 1-1. 環境準備

```
[ ] Python 3.12+ インストール済み
[ ] 必須パッケージ: pip install -r requirements.txt
[ ] GPU確認: RTX 4070 Ti SUPER + CUDA 12.6 (PyTorch 2.10)
[ ] ディスク空き: 10GB以上 (ログ・データキャッシュ用)
[ ] ネットワーク: 安定した回線 (切断 → 自動再接続あり)
```

### 1-2. 証券口座開設

| 対象 | 証券会社 | 口座タイプ | 所要期間 |
|------|---------|----------|---------|
| 日本株 | auカブコム証券 | kabuSTATION API対応口座 | 1〜2週間 |
| 暗号資産 | bitFlyer/GMOコイン/Binance | API取引対応口座 | 1〜3日 |
| 米国株/ETF | Interactive Brokers | IB口座 (個人) | 2〜4週間 |

### 1-3. API キー取得・設定

```bash
# .env ファイルを作成 (プロジェクトルート)
# !! 絶対に git にコミットしないこと !!

# 日本株 (kabuSTATION)
KABU_API_PASSWORD=your_password_here

# 暗号資産 (ccxt)
CCXT_EXCHANGE=bitflyer      # or binance, gmo
CCXT_API_KEY=your_api_key
CCXT_SECRET=your_secret

# 米国株 (Interactive Brokers)
IB_HOST=127.0.0.1
IB_PORT=7497                 # TWS: 7497, IB Gateway: 4001

# 通知
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx
```

### 1-4. 資金計画

| 項目 | 推奨値 | 備考 |
|------|-------|------|
| 最低運用資金 | 100万円 | 手数料負けしない最低ライン |
| 初期投入比率 | 総資産の5%以下 | DrawdownController で自動管理 |
| 追加投入タイミング | Sharpe 0.3以上を30日維持後 | 自動スケールアップ |
| 最大投入比率 | 総資産の50% | auto_trader設定の上限 |
| 生活防衛資金 | 最低6ヶ月分 | 投資資金と完全分離 |

---

## 2. ペーパー検証フェーズ

### 2-1. 開始コマンド

```bash
cd C:/_dev/repos/ai-trader-expansion

# Step 1: バックテスト (過去データで検証)
python scripts/auto_trader.py --mode paper \
  --symbols SPY,QQQ,7203.T \
  --strategies bollinger_rsi_adx,ma_crossover

# Step 2: 最適化
python scripts/auto_trader.py --mode optimize \
  --symbols SPY,QQQ,7203.T

# Step 3: 自動ペーパートレード (24時間稼働)
python scripts/auto_trader.py --mode auto \
  --config config/auto_trader_production.json
```

### 2-2. 検証期間と判定

| 検証項目 | 最低期間 | 合格基準 |
|---------|---------|---------|
| バックテスト (過去5年) | 即日 | Sharpe > 0.3, DD < 25% |
| ペーパートレード | 2週間 | 実シグナルが出て正常動作 |
| Walk-Forward OOS | 即日 | Consistency > 0.3 |

### 2-3. ペーパー検証の確認ポイント

```
[ ] シグナルが正しく発生するか (ログ確認)
[ ] OMS が注文を正しく管理するか
[ ] DrawdownController が閾値で停止するか
[ ] 日次レポートが送信されるか
[ ] ドリフト検出が動作するか
[ ] データ取得がエラーなく動くか (yfinance API制限)
```

---

## 3. 実運用移行判定基準

### 3-1. 必須条件 (ALL PASS で移行可)

| # | 条件 | 根拠 |
|---|------|------|
| 1 | バックテスト Sharpe > 0.3 (3年以上) | 統計的に有意なエッジ |
| 2 | Walk-Forward OOS リターン > 0% | 過学習でないことの証明 |
| 3 | MaxDD < 25% | 資金管理上の許容範囲 |
| 4 | ペーパートレード 2週間正常稼働 | システム安定性 |
| 5 | API接続テスト成功 (実ブローカー) | 注文系の動作確認 |

### 3-2. 推奨条件 (あれば望ましい)

| 条件 | 効果 |
|------|------|
| Optuna最適化後のOOS改善 | パラメータの堅牢性 |
| 3銘柄以上の分散投資 | リスク分散 |
| 通知チャネル設定済み | 即時アラート受信 |
| 自動再起動設定済み | 24時間無人稼働 |

---

## 4. 実運用開始手順

### 4-1. 移行チェックリスト

```
[ ] Section 3 の必須条件を全て確認
[ ] .env にAPIキーを設定
[ ] config/auto_trader_production.json の mode を "live" に変更
[ ] 初期資金を証券口座に入金
[ ] 通知先 (Slack/Discord) を設定・テスト送信
[ ] PCの自動スリープを無効化
[ ] UPS (無停電電源) があれば接続
```

### 4-2. 起動コマンド

```bash
# ライブモード開始
python scripts/auto_trader.py --mode live \
  --config config/auto_trader_production.json \
  --capital 1000000 --capital-pct 5

# バックグラウンド実行 (Windows)
# NSSM でサービス化するか、タスクスケジューラで起動時実行
start /B python scripts/auto_trader.py --mode auto \
  --config config/auto_trader_production.json > logs/auto_trader.log 2>&1
```

### 4-3. 起動直後の確認 (最初の30分)

```
[ ] ログにエラーがないか: tail -f logs/auto_trader.log
[ ] ヘルスチェック: curl http://127.0.0.1:8000/api/health
[ ] 証券口座の残高がシステムと一致するか
[ ] 最初のシグナル生成まで待機 (FLATでも正常)
```

---

## 5. 日常運用ルーティン

### 5-1. 毎日 (自動)

| 時刻 | アクション | 担当 |
|------|----------|------|
| 毎時 | シグナル生成 + 注文実行 | AutoTrader (自動) |
| 毎時 | OMS 監視 + ステイルキャンセル | AutoTrader (自動) |
| 毎時 | ドリフト検出 | DriftDetector (自動) |
| 18:00 | 日次レポート送信 | NotificationRouter (自動) |

### 5-2. 毎日 (手動確認 — 5分)

```
[ ] Slack/Discord の日次レポートを確認
[ ] PnL がマイナスの場合: 原因を確認 (市場全体の下落 or 戦略の問題)
[ ] エラーログがないか確認
[ ] 証券口座の残高とシステムの残高が一致するか
```

### 5-3. 毎週 (手動確認 — 15分)

```
[ ] 週次パフォーマンスサマリーを確認
[ ] 戦略別の勝率・Sharpe を確認
[ ] CorrelationMonitor のアラート有無
[ ] DrawdownController の状態 (NORMAL であるか)
[ ] ディスク・メモリ使用量の確認
```

### 5-4. 毎月 (手動 — 30分)

```
[ ] 月次パフォーマンスレビュー
[ ] Optuna 再最適化の実行 (auto_optimize_days=30 で自動)
[ ] Walk-Forward 再検証
[ ] 取引ジャーナルのバックアップ
[ ] 税務用の取引記録エクスポート
```

---

## 6. 利益確定・出金ルール

### 6-1. 利益確定の基準

| ルール | 条件 | アクション |
|--------|------|----------|
| 定期出金 | 月次リターン > +5% | 利益の50%を出金 |
| 目標達成 | 累積リターン > +30% | 元本の30%を出金 |
| リスク削減 | DD が 15%に到達 | 投入資金を50%に削減 |

### 6-2. 出金手順

```
1. AutoTrader を一時停止: Ctrl+C (SIGINT)
2. 全ポジションを手動決済 (必要な場合)
3. 証券口座から出金
4. config の initial_capital を更新
5. AutoTrader を再起動
```

### 6-3. 再投資ルール

- 出金後の残高で capital_pct を再計算
- Sharpe が維持されている場合は自動スケールアップが再開
- 大幅なDD後の再スタートは capital_pct を初期値 (5%) にリセット

---

## 7. スケールアップ基準

### 7-1. 自動スケールアップ条件

```python
# auto_trader.py の _maybe_scale_up() が自動判定
条件:
  1. 直近30日の日次PnLデータが30件以上
  2. 30日Sharpe > scale_up_sharpe_threshold (デフォルト: 0.3)
  3. 現在の capital_pct < scale_up_max_capital_pct (デフォルト: 50%)

アクション:
  capital_pct × 1.5 (上限50%)
  例: 5% → 7.5% → 11.25% → 16.9% → 25.3% → 37.9% → 50%
```

### 7-2. 手動スケールアップ判断

| 段階 | 条件 | 投入比率 |
|------|------|---------|
| Stage 1 (検証) | ペーパーで勝てた | 5% |
| Stage 2 (初期) | 1ヶ月連続プラス | 10% |
| Stage 3 (成長) | 3ヶ月 Sharpe > 0.5 | 20% |
| Stage 4 (安定) | 6ヶ月 DD < 15% | 30% |
| Stage 5 (成熟) | 1年 年率 > 10% | 50% |

### 7-3. スケールダウン条件

| 条件 | アクション |
|------|----------|
| 月次リターン < -5% | capital_pct を 50% に削減 |
| DD > 15% | capital_pct を初期値 (5%) にリセット |
| DD > 20% | 全ポジション決済 + システム停止 |
| 連続3ヶ月マイナス | 戦略見直し + 再最適化 |

---

## 8. 緊急時対応マニュアル

### 8-1. 緊急レベル定義

| レベル | 状態 | 対応時間 |
|--------|------|---------|
| L1 CRITICAL | DD > 12%, システム障害, API断 | 即時 (5分以内) |
| L2 HIGH | DD > 8%, 異常な連敗, ドリフト検出 | 1時間以内 |
| L3 MEDIUM | DD > 5%, 勝率低下, 相関急上昇 | 当日中 |
| L4 LOW | パフォーマンス低下傾向 | 週次レビューで対応 |

### 8-2. L1 CRITICAL 対応手順

```
■ システム障害
  1. logs/auto_trader.log を確認
  2. Ctrl+C で停止
  3. python scripts/auto_trader.py --mode paper で動作確認
  4. 原因特定後に再起動

■ DD > 12% (EMERGENCY_EXIT 自動発動)
  → DrawdownController が自動で全ポジション決済指示
  → 通知が Slack/Discord に送信される
  → 手動確認: 証券口座で全ポジションが決済されたか確認
  → 原因分析後、capital_pct を 5% にリセットして再開

■ API接続断
  → WebSocketFeed の自動再接続 (exponential backoff: 1s→2s→4s...60s)
  → 5分以上復旧しない場合:
    1. 証券会社のシステム状況を確認
    2. ネットワーク接続を確認
    3. AutoTrader を再起動

■ ブローカー注文拒否
  → OMS が自動リトライ (最大3回)
  → 全リトライ失敗:
    1. 証券口座の残高・余力を確認
    2. 銘柄の取引制限 (値幅制限, 売買停止) を確認
    3. API キーの有効期限を確認
```

### 8-3. L2 HIGH 対応手順

```
■ DD > 8% (HALT_TRADING 自動発動)
  → AutoTrader が新規注文を自動停止
  → 既存ポジションは保持
  → 対応:
    1. 市場全体の下落か、戦略固有の問題か判断
    2. 市場全体 → 放置 (回復待ち)
    3. 戦略固有 → 該当戦略を config から除外して再起動

■ ドリフト検出
  → DriftDetector が自動検知 + 通知送信
  → 対応:
    1. auto_optimize_days を短縮 (30日 → 7日)
    2. Optuna 再最適化を手動実行
    3. 最適化後のパラメータで A/B テスト

■ 異常な連敗 (5連敗以上)
  → 対応:
    1. 直近の取引ログを確認
    2. 市場レジームが変化していないか確認
    3. DrawdownController の閾値に達するまで自動運用継続
    4. 閾値到達で自動停止
```

### 8-4. 全ポジション緊急決済コマンド

```bash
# 緊急停止 (全ポジション決済は手動)
# AutoTrader を停止
kill -SIGINT $(pgrep -f auto_trader)

# 証券口座にログインして手動で全決済
# !! システム経由の決済は通信障害時に信頼できない !!
```

### 8-5. フラッシュクラッシュ対応

```
1. サーキットブレーカー (CircuitBreaker) が自動発動
   → 連続API失敗で OPEN 状態 → 全取引停止
2. DrawdownController が EMERGENCY_EXIT
   → exposure_ratio = 0.0 → 新規注文停止
3. 手動確認:
   → 証券口座にログインして損益確認
   → 市場が安定するまで再起動しない
   → 最低24時間は冷却期間
```

---

## 9. 障害対応フローチャート

```
異常検知
  │
  ├─ DD > 12% ──→ EMERGENCY_EXIT (自動)
  │                  └→ 全注文停止 → 通知 → 手動確認 → リセット
  │
  ├─ DD > 8% ───→ HALT_TRADING (自動)
  │                  └→ 新規注文停止 → 回復待ち
  │
  ├─ DD > 5% ───→ REDUCE_EXPOSURE (自動)
  │                  └→ エクスポージャー段階削減
  │
  ├─ ドリフト ──→ 通知 → 再最適化検討
  │
  ├─ API障害 ──→ 自動再接続 (backoff)
  │                  └→ 5分超 → 手動確認
  │
  ├─ PC障害 ───→ NSSM/タスクスケジューラで自動再起動
  │                  └→ 再起動失敗 → 手動対応
  │
  └─ 全て正常 ─→ 通常運用継続
```

---

## 10. 税務・記録管理

### 10-1. 取引記録の保存

- `trade_journal.db` (SQLite): 全取引が自動記録される
- バックアップ: 月次で `trade_journal_YYYY-MM.db` にコピー
- 保存期間: 最低7年 (税務調査対応)

### 10-2. 確定申告の準備

| 所得区分 | 対象 | 税率 |
|---------|------|------|
| 株式譲渡所得 | 日本株・米国ETF | 20.315% (分離課税) |
| 雑所得 | 暗号資産 | 累進課税 (5〜45%) |

### 10-3. 年次作業

```
[ ] 1月: 前年の取引記録を集計
[ ] 2月: 確定申告書の作成
[ ] 3月15日: 確定申告期限
[ ] 通年: 損益通算・繰越控除の確認
```

### 10-4. 記録エクスポート

```bash
# 取引履歴をCSV出力 (将来実装予定)
python scripts/export_trades.py --year 2026 --format csv
```

---

## 付録: 重要な設定ファイル

| ファイル | 用途 |
|---------|------|
| `config/auto_trader_production.json` | 本番設定 |
| `.env` | APIキー (gitignore対象) |
| `config/settings.py` | システム全体設定 |
| `trade_journal.db` | 取引記録DB |
| `data_cache.db` | 価格データキャッシュ |
| `logs/auto_trader.log` | 実行ログ |

---

## 付録: クイックリファレンス

```bash
# ペーパーバックテスト
python scripts/auto_trader.py --mode paper --symbols SPY,QQQ,7203.T

# パラメータ最適化
python scripts/auto_trader.py --mode optimize --symbols SPY

# ペーパー自動取引 (24時間)
python scripts/auto_trader.py --mode auto --config config/auto_trader_production.json

# ライブ取引
python scripts/auto_trader.py --mode live --config config/auto_trader_production.json

# ヘルスチェック
curl http://127.0.0.1:8000/api/health

# 緊急停止
Ctrl+C  (or)  kill -SIGINT $(pgrep -f auto_trader)
```
