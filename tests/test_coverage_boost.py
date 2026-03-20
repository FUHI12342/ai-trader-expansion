"""カバレッジ向上のための追加テスト。

walk_forward, lgbm_predictor, edinet, jquants, kabu_station の
カバレッジを補完するテスト群。
"""
from __future__ import annotations

import sqlite3
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ============================================================
# Walk-Forward Analysis のテスト
# ============================================================

class TestWalkForwardAnalysis:
    """walk_forward_analysisのテスト。"""

    def test_walk_forward_returns_result(self, ohlcv_long: pd.DataFrame):
        """Walk-Forward分析がWalkForwardResultを返すことを確認。"""
        from src.evaluation.walk_forward import walk_forward_analysis, WalkForwardResult
        from src.strategies.ma_crossover import MACrossoverStrategy

        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = walk_forward_analysis(
            strategy=strategy,
            data=ohlcv_long,
            in_sample_days=100,
            out_of_sample_days=50,
            step_days=50,
            min_walks=2,
        )
        assert isinstance(result, WalkForwardResult)

    def test_walk_forward_num_walks(self, ohlcv_long: pd.DataFrame):
        """ウォーク数が正しく計算されることを確認。"""
        from src.evaluation.walk_forward import walk_forward_analysis
        from src.strategies.ma_crossover import MACrossoverStrategy

        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = walk_forward_analysis(
            strategy=strategy,
            data=ohlcv_long,
            in_sample_days=100,
            out_of_sample_days=50,
            step_days=100,
            min_walks=2,
        )
        assert result.num_walks >= 2

    def test_walk_forward_insufficient_data(self, ohlcv_data: pd.DataFrame):
        """データ不足でValueErrorが発生することを確認。"""
        from src.evaluation.walk_forward import walk_forward_analysis
        from src.strategies.ma_crossover import MACrossoverStrategy

        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        with pytest.raises(ValueError, match="データが不足"):
            walk_forward_analysis(
                strategy=strategy,
                data=ohlcv_data.iloc[:50],
                in_sample_days=200,
                out_of_sample_days=100,
                min_walks=2,
            )

    def test_walk_forward_too_few_walks(self, ohlcv_long: pd.DataFrame):
        """ウォーク数不足でValueErrorが発生することを確認。"""
        from src.evaluation.walk_forward import walk_forward_analysis
        from src.strategies.ma_crossover import MACrossoverStrategy

        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        with pytest.raises(ValueError, match="ウォーク数が不足"):
            walk_forward_analysis(
                strategy=strategy,
                data=ohlcv_long,
                in_sample_days=300,
                out_of_sample_days=200,
                step_days=300,
                min_walks=10,  # 達成できない高い要求
            )

    def test_walk_forward_consistency_ratio(self, ohlcv_long: pd.DataFrame):
        """一貫性比率が0〜1の範囲であることを確認。"""
        from src.evaluation.walk_forward import walk_forward_analysis
        from src.strategies.ma_crossover import MACrossoverStrategy

        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = walk_forward_analysis(
            strategy=strategy,
            data=ohlcv_long,
            in_sample_days=100,
            out_of_sample_days=50,
            step_days=50,
            min_walks=2,
        )
        assert 0.0 <= result.consistency_ratio <= 1.0

    def test_walk_forward_result_to_dict(self, ohlcv_long: pd.DataFrame):
        """to_dict()が正常に動作することを確認。"""
        from src.evaluation.walk_forward import walk_forward_analysis
        from src.strategies.ma_crossover import MACrossoverStrategy

        strategy = MACrossoverStrategy(short_window=5, long_window=20)
        result = walk_forward_analysis(
            strategy=strategy,
            data=ohlcv_long,
            in_sample_days=100,
            out_of_sample_days=50,
            step_days=50,
            min_walks=2,
        )
        d = result.to_dict()
        assert "num_walks" in d
        assert "is_statistically_significant" in d
        assert "walks" in d


# ============================================================
# LGBMPredictor 戦略のテスト
# ============================================================

class TestLGBMPredictor:
    """LGBMPredictorStrategyのテスト。"""

    def test_lgbm_available(self):
        """LightGBMがインストールされているか確認。"""
        try:
            import lightgbm
            HAS_LGB = True
        except ImportError:
            HAS_LGB = False
        # インストール有無に応じて動作確認
        if HAS_LGB:
            from src.strategies.lgbm_predictor import LGBMPredictorStrategy
            strategy = LGBMPredictorStrategy(train_window=100, predict_window=30)
            assert strategy.name == "lgbm_predictor"
        else:
            with pytest.raises(ImportError):
                from src.strategies.lgbm_predictor import LGBMPredictorStrategy
                LGBMPredictorStrategy()

    def test_build_features(self, ohlcv_long: pd.DataFrame):
        """特徴量ビルダーのテスト。"""
        from src.strategies.lgbm_predictor import _build_features
        features = _build_features(ohlcv_long)
        assert "sma_5" in features.columns
        assert "rsi" in features.columns
        assert "macd" in features.columns
        assert "bb_position" in features.columns
        assert len(features) == len(ohlcv_long)

    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('lightgbm'),
        reason="LightGBM未インストール"
    )
    def test_lgbm_generate_signals(self, ohlcv_long: pd.DataFrame):
        """LGBMシグナル生成のテスト（LightGBMが必要）。"""
        from src.strategies.lgbm_predictor import LGBMPredictorStrategy
        strategy = LGBMPredictorStrategy(
            train_window=150,
            predict_window=50,
            min_train_samples=50,
        )
        signals = strategy.generate_signals(ohlcv_long)
        assert len(signals) == len(ohlcv_long)
        assert set(signals.unique()).issubset({-1, 0, 1})


# ============================================================
# DataManager の追加テスト
# ============================================================

class TestDataManagerAdditional:
    """DataManagerの追加テスト。"""

    def _make_settings_with_tempdb(self):
        """一時DBパスを使った設定を作成する。"""
        import tempfile
        from config.settings import Settings, DatabaseSettings
        tmp = tempfile.mktemp(suffix=".db")
        return Settings(database=DatabaseSettings(cache_db_path=tmp))

    def test_fallback_to_yfinance_when_jquants_fails(self, ohlcv_data: pd.DataFrame):
        """J-Quants失敗時にyfinanceにフォールバックすることを確認。"""
        from src.data.data_manager import DataManager
        settings = self._make_settings_with_tempdb()
        dm = DataManager(settings=settings, source="auto")

        with patch.object(dm, "_fetch_jquants", side_effect=Exception("J-Quants失敗")):
            with patch.object(dm, "_fetch_yfinance", return_value=ohlcv_data):
                result = dm.fetch_ohlcv("7203.T", "2022-01-01", "2022-12-31", force_refresh=True)
                assert not result.empty
        dm.close()

    def test_clear_cache_specific_symbol(self, ohlcv_data: pd.DataFrame):
        """特定銘柄のキャッシュクリアが動作することを確認。"""
        from src.data.data_manager import DataManager
        settings = self._make_settings_with_tempdb()
        dm = DataManager(settings=settings)

        dm._save_to_cache("7203.T", ohlcv_data, "yfinance")
        dm._save_to_cache("9984.T", ohlcv_data, "yfinance")

        dm.clear_cache(symbol="7203.T")  # 7203のみクリア

        # 9984はまだキャッシュにある
        start = str(ohlcv_data.index[0])[:10]
        end = str(ohlcv_data.index[-1])[:10]
        remaining = dm._load_from_cache("9984.T", start, end, "yfinance")
        assert remaining is not None

        dm.close()

    def test_jquants_source_fetch(self, ohlcv_data: pd.DataFrame):
        """jquantsソースで正常に取得できることを確認。"""
        from src.data.data_manager import DataManager
        settings = self._make_settings_with_tempdb()
        dm = DataManager(settings=settings, source="jquants")

        with patch.object(dm, "_fetch_jquants", return_value=ohlcv_data):
            result = dm.fetch_ohlcv("7203", "2022-01-01", "2022-12-31", force_refresh=True)
            assert not result.empty
        dm.close()


# ============================================================
# EDINET Client のテスト
# ============================================================

class TestEdinetClient:
    """EdinetClientのモックテスト。"""

    def test_fetch_document_list_success(self):
        """書類一覧取得のモックテスト。"""
        from src.data.edinet_client import EdinetClient
        from config.settings import EdinetSettings

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "docID": "S100TEST",
                    "filerName": "テスト株式会社",
                    "docTypeCode": "120",
                    "edinetCode": "E00001",
                },
            ]
        }

        client = EdinetClient(EdinetSettings(api_key="test_key"))
        with patch.object(client._session, "request", return_value=mock_response):
            result = client.fetch_document_list("2023-03-31")
        assert not result.empty
        assert "docID" in result.columns

    def test_fetch_document_list_empty(self):
        """書類なし日付での処理確認。"""
        from src.data.edinet_client import EdinetClient
        from config.settings import EdinetSettings

        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}

        client = EdinetClient(EdinetSettings(api_key="test_key"))
        with patch.object(client._session, "request", return_value=mock_response):
            result = client.fetch_document_list("2023-01-01")
        assert result.empty

    def test_fetch_document_no_results_raises(self):
        """resultsキーなし時にEdinetErrorが発生することを確認。"""
        from src.data.edinet_client import EdinetClient, EdinetError
        from config.settings import EdinetSettings

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "error"}

        client = EdinetClient(EdinetSettings(api_key="test_key"))
        with patch.object(client._session, "request", return_value=mock_response):
            with pytest.raises(EdinetError, match="書類一覧取得失敗"):
                client.fetch_document_list("2023-01-01")

    def test_search_filings_requires_dates(self):
        """date_from/date_to未指定でValueErrorが発生することを確認。"""
        from src.data.edinet_client import EdinetClient
        from config.settings import EdinetSettings

        client = EdinetClient(EdinetSettings())
        with pytest.raises(ValueError, match="date_from と date_to"):
            client.search_filings(edinetcode="E00001")

    def test_close(self):
        """closeが正常に動作することを確認。"""
        from src.data.edinet_client import EdinetClient
        from config.settings import EdinetSettings
        client = EdinetClient(EdinetSettings())
        client.close()  # エラーなく完了


# ============================================================
# J-Quants Client のテスト
# ============================================================

class TestJQuantsClient:
    """JQuantsClientのモックテスト。"""

    def test_authenticate_success(self):
        """認証成功のモックテスト。"""
        from src.data.jquants_client import JQuantsClient
        from config.settings import JQuantsSettings

        client = JQuantsClient(JQuantsSettings(
            email="test@example.com",
            password="password123",
        ))

        refresh_response = MagicMock()
        refresh_response.json.return_value = {"refreshToken": "test_refresh_token"}

        id_response = MagicMock()
        id_response.json.return_value = {"idToken": "test_id_token"}

        with patch.object(client._session, "request", side_effect=[refresh_response, id_response]):
            client.authenticate()

        assert client._id_token == "test_id_token"
        assert client._refresh_token == "test_refresh_token"

    def test_authenticate_missing_credentials(self):
        """認証情報未設定時にJQuantsErrorが発生することを確認。"""
        from src.data.jquants_client import JQuantsClient, JQuantsError
        from config.settings import JQuantsSettings

        client = JQuantsClient(JQuantsSettings())  # 空の設定
        with pytest.raises(JQuantsError, match="認証情報が設定されていません"):
            client.authenticate()

    def test_fetch_stock_prices_success(self):
        """株価取得のモックテスト。"""
        from src.data.jquants_client import JQuantsClient
        from config.settings import JQuantsSettings

        client = JQuantsClient(JQuantsSettings(
            email="test@example.com",
            password="test",
        ))
        client._id_token = "dummy_token"  # 認証済みをシミュレート

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "daily_quotes": [
                {
                    "Date": "2023-01-04", "Code": "72030",
                    "Open": 2000.0, "High": 2100.0, "Low": 1950.0,
                    "Close": 2050.0, "Volume": 5000000.0,
                },
                {
                    "Date": "2023-01-05", "Code": "72030",
                    "Open": 2050.0, "High": 2150.0, "Low": 2000.0,
                    "Close": 2100.0, "Volume": 4500000.0,
                },
            ]
        }

        with patch.object(client._session, "request", return_value=mock_response):
            result = client.fetch_stock_prices("7203", "2023-01-01", "2023-01-31")

        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        assert len(result) == 2

    def test_fetch_stock_prices_no_data(self):
        """データなし時に空DataFrameが返ることを確認。"""
        from src.data.jquants_client import JQuantsClient
        from config.settings import JQuantsSettings

        client = JQuantsClient(JQuantsSettings())
        client._id_token = "dummy_token"

        mock_response = MagicMock()
        mock_response.json.return_value = {"daily_quotes": []}

        with patch.object(client._session, "request", return_value=mock_response):
            result = client.fetch_stock_prices("9999", "2023-01-01")

        assert result.empty

    def test_close(self):
        """closeが正常に動作することを確認。"""
        from src.data.jquants_client import JQuantsClient
        from config.settings import JQuantsSettings
        client = JQuantsClient(JQuantsSettings())
        client.close()  # エラーなく完了


# ============================================================
# KabuStation Broker のテスト
# ============================================================

class TestKabuStationBroker:
    """KabuStationBrokerのスタブテスト。"""

    def test_broker_init(self):
        """初期化が正常に動作することを確認。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings(
            api_password="test_pass",
            host="localhost",
            port=18080,
        ))
        assert broker._base_url == "http://localhost:18080/kabusapi"

    def test_authenticate_success(self):
        """認証成功のモックテスト。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings
        import json

        broker = KabuStationBroker(KabuStationSettings(api_password="test"))

        with patch.object(broker, "_request", return_value={"Token": "test_token"}):
            token = broker.authenticate()

        assert token == "test_token"
        assert broker._token == "test_token"

    def test_get_balance(self):
        """残高取得のモックテスト。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings(api_password="test"))
        broker._token = "test_token"  # 認証済みシミュレート

        with patch.object(broker, "_request", return_value={"StockAccountWallet": 1_000_000}):
            balance = broker.get_balance()

        assert balance == 1_000_000.0

    def test_get_positions_empty(self):
        """ポジションなし時の処理確認。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings(api_password="test"))
        broker._token = "test_token"

        with patch.object(broker, "_request", return_value=[]):
            positions = broker.get_positions()

        assert positions == {}

    def test_parse_order_status(self):
        """注文状態のパース確認。"""
        from src.brokers.kabu_station import KabuStationBroker

        assert KabuStationBroker._parse_order_status(1) == "open"
        assert KabuStationBroker._parse_order_status(3) == "filled"
        assert KabuStationBroker._parse_order_status(5) == "cancelled"
        assert KabuStationBroker._parse_order_status(99) == "unknown"

    def test_place_order(self):
        """注文発注のモックテスト。"""
        from src.brokers.kabu_station import KabuStationBroker
        from src.brokers.base import OrderSide, OrderType
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings(api_password="test"))
        broker._token = "test_token"

        with patch.object(broker, "_request", return_value={"OrderId": "ORDER001"}):
            order = broker.place_order(
                "7203", OrderSide.BUY, OrderType.MARKET, 100
            )

        assert order.order_id == "ORDER001"
        assert order.status == "open"

    def test_cancel_order_success(self):
        """注文キャンセルのモックテスト。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings(api_password="test"))
        broker._token = "test_token"

        with patch.object(broker, "_request", return_value={}):
            result = broker.cancel_order("ORDER001")

        assert result is True

    def test_cancel_order_failure(self):
        """注文キャンセル失敗時にFalseが返ることを確認。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings(api_password="test"))
        broker._token = "test_token"

        with patch.object(broker, "_request", side_effect=RuntimeError("キャンセル失敗")):
            result = broker.cancel_order("INVALID_ORDER")

        assert result is False

    def test_is_connected_true(self):
        """接続確認のモックテスト。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings(api_password="test"))
        broker._token = "test_token"

        with patch.object(broker, "_request", return_value={}):
            assert broker.is_connected() is True

    def test_is_connected_false(self):
        """接続失敗時にFalseが返ることを確認。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings())

        with patch.object(broker, "_request", side_effect=Exception("接続失敗")):
            assert broker.is_connected() is False

    def test_get_open_orders(self):
        """オープン注文取得のモックテスト。"""
        from src.brokers.kabu_station import KabuStationBroker
        from config.settings import KabuStationSettings

        broker = KabuStationBroker(KabuStationSettings(api_password="test"))
        broker._token = "test_token"

        # 空のオープン注文
        with patch.object(broker, "_request", return_value=[]):
            orders = broker.get_open_orders()
        assert orders == []


# ============================================================
# 設定クラスのテスト
# ============================================================

class TestSettings:
    """設定クラスのテスト。"""

    def test_settings_from_env(self):
        """環境変数から設定が読み込まれることを確認。"""
        import os
        from config.settings import Settings

        with patch.dict(os.environ, {
            "TRADER_INITIAL_CAPITAL": "2000000",
            "TRADER_FEE_RATE": "0.0005",
        }):
            settings = Settings.from_env()
        assert settings.initial_capital == 2_000_000.0
        assert settings.fee_rate == 0.0005

    def test_jquants_settings_from_env(self):
        """J-Quants設定が環境変数から読み込まれることを確認。"""
        import os
        from config.settings import JQuantsSettings

        with patch.dict(os.environ, {
            "JQUANTS_EMAIL": "test@example.com",
            "JQUANTS_PASSWORD": "secret123",
        }):
            settings = JQuantsSettings.from_env()
        assert settings.email == "test@example.com"
        assert settings.password == "secret123"

    def test_settings_are_frozen(self):
        """設定がimmutable（frozen）であることを確認。"""
        from config.settings import Settings
        settings = Settings()
        with pytest.raises((AttributeError, TypeError)):
            settings.initial_capital = 9999999  # type: ignore
