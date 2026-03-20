# テスト一覧 (TEST_LIST.md)

最終更新: 2026-03-19

---

## テスト実行コマンド

```bash
pytest                          # 全テスト（カバレッジ付き）
pytest -v --no-cov             # カバレッジなし（高速）
pytest tests/test_strategies.py  # 戦略テストのみ
```

---

## tests/test_strategies.py — 戦略テスト

### TestBaseStrategy（基底クラス）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_validate_data_empty | 空データでValueError |
| test_validate_data_missing_columns | 必須カラム不足でValueError |
| test_validate_data_insufficient_rows | 行数不足でValueError |
| test_get_parameters | パラメータ辞書取得 |
| test_set_parameters_immutable | set_parametersが元のインスタンスを変更しない |
| test_set_parameters_invalid_key | 不正パラメータでAttributeError |

### TestMACrossoverStrategy（MAクロス）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_signal_type_values | シグナル値が-1,0,1のみ |
| test_signal_length_matches_data | シグナル長さがデータと一致 |
| test_signal_index_matches_data | シグナルインデックスがデータと一致 |
| test_buy_signal_on_golden_cross | ゴールデンクロスでBUY発生 |
| test_no_signal_before_warmup | ウォームアップ期間中はFLAT |
| test_data_not_mutated | 元データが変更されない（immutable） |

### TestDualMomentumStrategy（デュアルモメンタム）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_generate_signals_returns_series | pd.Seriesを返す |
| test_signal_values_valid | シグナル値が-1,0,1のみ |
| test_buy_on_uptrend | 上昇トレンドでBUYが多い |
| test_flat_when_insufficient_data | データ不足でValueError |
| test_threshold_effect | 閾値が高いほどBUYが少ない |

### TestMACDRSIStrategy（MACD+RSI）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_ema_length | EMAの長さが入力と一致 |
| test_rsi_range | RSIが0〜100の範囲 |
| test_generate_signals_valid | シグナル生成が正常 |
| test_buy_requires_both_conditions | BUYにはMACDとRSI両方が必要 |
| test_get_indicators | 指標DataFrame取得 |
| test_signal_no_lookahead | 未来データ参照なし |

### TestBollingerRSIADXStrategy（BB+RSI+ADX）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_bollinger_bands_shape | BBの上下関係（upper≥mid≥lower） |
| test_adx_range | ADXが0以上 |
| test_generate_signals_valid | シグナル生成が正常 |
| test_adx_filter_reduces_signals | ADX閾値が高いほどシグナルが少ない |
| test_get_indicators | 指標DataFrame取得 |

---

## tests/test_evaluation.py — 評価システムテスト

### TestCalculateMetrics（メトリクス計算）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_total_return_calculation | 総リターン計算精度（+10%） |
| test_max_drawdown_calculation | 最大DD計算（負値確認） |
| test_sharpe_ratio_positive_returns | シャープレシオの型確認 |
| test_zero_return_result_on_empty | 空データでゼロ結果 |
| test_win_rate_with_trades | 勝率計算精度（50%） |
| test_profit_factor_calculation | PF計算精度（3.0） |
| test_result_is_immutable | EvaluationResultがimmutable |
| test_to_dict_contains_all_fields | to_dict()の完全性 |

### TestRunBacktest（バックテストエンジン）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_backtest_returns_result | BacktestResultを返す |
| test_final_capital_is_positive | 最終資金が正 |
| test_equity_curve_length | エクイティカーブの長さ |
| test_no_future_data_leak | 初期時点で初期資金と同じ |
| test_fee_reduces_returns | 手数料でリターン低下 |
| test_backtest_result_to_dict | to_dict()の動作 |

### TestMonteCarlo（モンテカルロ）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_returns_monte_carlo_result | MonteCarloResultを返す |
| test_num_simulations | 指定回数のシミュレーション |
| test_probability_of_loss_range | 損失確率が0〜1 |
| test_percentile_ordering | パーセンタイルの順序 |
| test_empty_returns | 空リターンで安全処理 |
| test_reproducibility | 同シードで同結果 |
| test_confidence_interval_order | 信頼区間下限 < 上限 |
| test_result_is_immutable | MonteCarloResultがimmutable |

### TestStatisticalTests（統計検定）

| テスト関数 | 検証内容 |
|-----------|---------|
| test_returns_statistics_result | StatisticsResultを返す |
| test_p_value_range | p値が0〜1 |
| test_bootstrap_ci_order | CIの順序 |
| test_positive_returns_significant | 正リターンが有意と判定 |
| test_zero_returns_not_significant | ゼロリターンが非有意 |
| test_empty_returns_safe | 空データで安全処理 |
| test_sample_mean_accuracy | 標本平均の計算精度 |
| test_positive_rate | プラスリターン率の精度 |

---

## tests/test_data_clients.py — データクライアントテスト

### TestNormalizeColumns

| テスト関数 | 検証内容 |
|-----------|---------|
| test_lowercase_columns | カラム名の小文字変換 |
| test_missing_required_column | 必須カラム不足でValueError |
| test_adj_close_rename | adj closeのリネーム |

### TestYFinanceClient

| テスト関数 | 検証内容 |
|-----------|---------|
| test_fetch_ohlcv_returns_dataframe | DataFrameを返す（モック） |
| test_fetch_ohlcv_empty_data_raises | 空データでValueError |
| test_memory_cache | メモリキャッシュの動作 |
| test_clear_cache | キャッシュクリアの動作 |

### TestDataManager

| テスト関数 | 検証内容 |
|-----------|---------|
| test_init_creates_db | DB初期化 |
| test_save_and_load_cache | キャッシュ保存・読み込み |
| test_cache_miss_returns_none | キャッシュミス時のNone |
| test_fetch_ohlcv_with_yfinance_mock | yfinanceモックから取得 |
| test_context_manager | withブロックでリソース解放 |
| test_clear_cache_all | 全キャッシュクリア |

---

## tests/test_brokers.py — ブローカーテスト

### TestPaperBroker

| テスト関数 | 検証内容 |
|-----------|---------|
| test_initial_balance | 初期残高の確認 |
| test_buy_order_reduces_balance | 買い注文で残高減少 |
| test_buy_creates_position | 買い注文でポジション作成 |
| test_sell_closes_position | 売り注文でポジションクローズ |
| test_profit_on_price_increase | 価格上昇で利益発生 |
| test_insufficient_balance_raises | 残高不足でエラー |
| test_oversell_raises | 保有数超の売りでエラー |
| test_no_price_set_raises | 価格未設定でエラー |
| test_average_price_on_multiple_buys | 複数買いの平均取得価格 |
| test_get_equity_includes_positions | get_equity()の計算 |
| test_reset_clears_state | reset()の動作 |
| test_order_is_returned | 注文情報の返却 |
| test_get_open_orders_empty | オープン注文が常に空 |
| test_order_history | 注文履歴の記録 |

### TestOrder

| テスト関数 | 検証内容 |
|-----------|---------|
| test_order_immutable | Orderがimmutable |
| test_to_dict | to_dict()の動作 |

---

## tests/test_api.py — APIエンドポイントテスト

### TestHealthCheck

| テスト関数 | 検証内容 |
|-----------|---------|
| test_root_returns_ok | GET / が200を返す |

### TestStatusEndpoint

| テスト関数 | 検証内容 |
|-----------|---------|
| test_status_returns_list | リストを返す |
| test_status_fields | 必要フィールドの確認 |
| test_status_on_data_fetch_failure | 失敗時にNOGOを返す |

### TestPositionsEndpoint

| テスト関数 | 検証内容 |
|-----------|---------|
| test_positions_returns_list | リストを返す |
| test_positions_empty_by_default | デフォルトで空リスト |

### TestPerformanceEndpoint

| テスト関数 | 検証内容 |
|-----------|---------|
| test_performance_returns_metrics | メトリクスを返す |
| test_initial_return_is_zero | 初期リターンの確認 |

### TestBacktestEndpoint

| テスト関数 | 検証内容 |
|-----------|---------|
| test_backtest_valid_request | 有効リクエストで成功 |
| test_backtest_invalid_strategy | 不正戦略名で400エラー |
| test_backtest_data_fetch_failure | データ取得失敗で422 |
| test_backtest_result_fields | レスポンスフィールドの確認 |

---

## テスト合計

| ファイル | テスト数 |
|---------|---------|
| test_strategies.py | 25 |
| test_evaluation.py | 26 |
| test_data_clients.py | 13 |
| test_brokers.py | 16 |
| test_api.py | 10 |
| **合計** | **90** |
