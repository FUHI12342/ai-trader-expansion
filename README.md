# AI Trader 株式拡張

日本株・米国株向けAIトレーダー拡張システム。
複数の取引戦略、評価フレームワーク、SHANON AIアシスタントとのREST API連携を提供します。

## 機能

### 取引戦略（5種類）
- **MA Crossover** — 移動平均ゴールデン/デッドクロス
- **Dual Momentum** — Antonacci方式絶対・相対モメンタム
- **MACD + RSI** — MACDクロス + RSIフィルタ複合
- **Bollinger + RSI + ADX** — ボリンジャー + RSI + ADXトリプルフィルタ
- **LightGBM Predictor** — Walk-Forward機械学習予測

### 評価システム
- **バックテスト** — イベントドリブン、手数料・スリッページ考慮
- **Walk-Forward分析** — IS/OOS分離で過学習検証
- **モンテカルロシミュレーション** — 1000回シャッフルで堅牢性検証
- **統計的有意性検定** — t検定 + ブートストラップ信頼区間

### データ取得
- **yfinance** — 米国・日本株日次データ
- **J-Quants API** — 日本株公式データ（認証付き）
- **EDINET API v2** — 有価証券報告書・開示書類
- **SQLiteキャッシュ** — APIコールを最小化

### ブローカー接続
- **ペーパートレード** — 仮想注文・ポジション管理
- **kabuステーション** — auカブコム証券 REST API

### REST API（SHANON連携）
- `GET /api/status` — 全戦略のGo/Nogoステータス
- `GET /api/positions` — 現在ポジション
- `GET /api/performance` — パフォーマンスメトリクス
- `POST /api/backtest` — オンデマンドバックテスト

## クイックスタート

```bash
pip install -r requirements.txt
pytest  # 全テスト実行
python -m src.api.server  # APIサーバー起動 (http://localhost:8765)
```

## 設計原則

- **Immutableパターン** — DataFrameは常にcopy()、結果はfrozen dataclass
- **型ヒント必須** — 全関数に型アノテーション
- **秘密情報は環境変数** — APIキーのハードコード禁止
- **エラーハンドリング** — API失敗時のリトライ + グレースフルデグレデーション
- **テストカバレッジ80%以上** — pytest + pytest-cov

## ドキュメント

- [仕様書 (SPEC.md)](docs/SPEC.md)
- [使い方マニュアル (MANUAL.md)](docs/MANUAL.md)
- [テスト一覧 (TEST_LIST.md)](docs/TEST_LIST.md)
- [システム構成図 (ARCHITECTURE.mmd)](docs/ARCHITECTURE.mmd)
