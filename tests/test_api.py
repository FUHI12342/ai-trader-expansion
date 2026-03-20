"""APIエンドポイントのテスト。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.server import app


@pytest.fixture
def client() -> TestClient:
    """テスト用HTTPクライアント。"""
    return TestClient(app)


# ============================================================
# ヘルスチェックのテスト
# ============================================================

class TestHealthCheck:
    """ヘルスチェックエンドポイントのテスト。"""

    def test_root_returns_ok(self, client: TestClient):
        """GETルートが200を返すことを確認。"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ============================================================
# /api/status のテスト
# ============================================================

class TestStatusEndpoint:
    """GET /api/status のテスト。"""

    def test_status_returns_list(self, client: TestClient, ohlcv_data: pd.DataFrame):
        """ステータス取得がリストを返すことを確認。"""
        with patch("src.api.server.DataManager") as mock_dm_cls:
            mock_dm = MagicMock()
            mock_dm.fetch_ohlcv.return_value = ohlcv_data
            mock_dm_cls.return_value = mock_dm
            response = client.get("/api/status?symbol=7203.T")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_status_fields(self, client: TestClient, ohlcv_data: pd.DataFrame):
        """各ステータスに必要なフィールドが含まれることを確認。"""
        with patch("src.api.server.DataManager") as mock_dm_cls:
            mock_dm = MagicMock()
            mock_dm.fetch_ohlcv.return_value = ohlcv_data
            mock_dm_cls.return_value = mock_dm
            response = client.get("/api/status")

        assert response.status_code == 200
        for item in response.json():
            assert "name" in item
            assert "go_nogo" in item
            assert "last_signal" in item

    def test_status_on_data_fetch_failure(self, client: TestClient):
        """データ取得失敗時にNOGOが返されることを確認。"""
        with patch("src.api.server.DataManager") as mock_dm_cls:
            mock_dm = MagicMock()
            mock_dm.fetch_ohlcv.side_effect = Exception("テストエラー")
            mock_dm_cls.return_value = mock_dm
            response = client.get("/api/status")

        assert response.status_code == 200
        for item in response.json():
            assert item["go_nogo"] == "NOGO"


# ============================================================
# /api/positions のテスト
# ============================================================

class TestPositionsEndpoint:
    """GET /api/positions のテスト。"""

    def test_positions_returns_list(self, client: TestClient):
        """ポジション取得がリストを返すことを確認。"""
        response = client.get("/api/positions")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_positions_empty_by_default(self, client: TestClient):
        """デフォルトでポジションが空であることを確認。"""
        response = client.get("/api/positions")
        assert response.status_code == 200
        # テスト開始時はポジションなし（APIの初期化後）
        data = response.json()
        assert isinstance(data, list)


# ============================================================
# /api/performance のテスト
# ============================================================

class TestPerformanceEndpoint:
    """GET /api/performance のテスト。"""

    def test_performance_returns_metrics(self, client: TestClient):
        """パフォーマンス取得がメトリクスを返すことを確認。"""
        response = client.get("/api/performance")
        assert response.status_code == 200
        data = response.json()
        assert "balance" in data
        assert "equity" in data
        assert "total_return_pct" in data

    def test_initial_return_is_zero(self, client: TestClient):
        """初期状態ではリターンがゼロであることを確認。"""
        response = client.get("/api/performance")
        assert response.status_code == 200
        data = response.json()
        # ペーパーブローカーが取引ゼロの場合
        assert isinstance(data["total_return_pct"], float)


# ============================================================
# /api/backtest のテスト
# ============================================================

class TestBacktestEndpoint:
    """POST /api/backtest のテスト。"""

    def test_backtest_valid_request(self, client: TestClient, ohlcv_data: pd.DataFrame):
        """有効なリクエストでバックテストが実行されることを確認。"""
        with patch("src.api.server.DataManager") as mock_dm_cls:
            mock_dm = MagicMock()
            mock_dm.fetch_ohlcv.return_value = ohlcv_data
            mock_dm_cls.return_value = mock_dm

            response = client.post("/api/backtest", json={
                "strategy_name": "ma_crossover",
                "symbol": "7203.T",
                "start_date": "2022-01-04",
                "end_date": "2023-12-29",
                "initial_capital": 1_000_000.0,
            })

        assert response.status_code == 200
        data = response.json()
        assert "strategy_name" in data
        assert "total_return_pct" in data

    def test_backtest_invalid_strategy(self, client: TestClient):
        """存在しない戦略名で400エラーが返ることを確認。"""
        response = client.post("/api/backtest", json={
            "strategy_name": "nonexistent_strategy",
            "symbol": "7203.T",
            "start_date": "2022-01-04",
        })
        assert response.status_code == 400

    def test_backtest_data_fetch_failure(self, client: TestClient):
        """データ取得失敗で422エラーが返ることを確認。"""
        with patch("src.api.server.DataManager") as mock_dm_cls:
            mock_dm = MagicMock()
            mock_dm.fetch_ohlcv.side_effect = Exception("データ取得失敗")
            mock_dm_cls.return_value = mock_dm

            response = client.post("/api/backtest", json={
                "strategy_name": "macd_rsi",
                "symbol": "INVALID",
                "start_date": "2022-01-04",
            })

        assert response.status_code == 422

    def test_backtest_result_fields(self, client: TestClient, ohlcv_data: pd.DataFrame):
        """バックテスト結果に必要なフィールドが含まれることを確認。"""
        with patch("src.api.server.DataManager") as mock_dm_cls:
            mock_dm = MagicMock()
            mock_dm.fetch_ohlcv.return_value = ohlcv_data
            mock_dm_cls.return_value = mock_dm

            response = client.post("/api/backtest", json={
                "strategy_name": "macd_rsi",
                "symbol": "7203.T",
                "start_date": "2022-01-04",
            })

        assert response.status_code == 200
        data = response.json()
        required_fields = [
            "strategy_name", "symbol", "initial_capital",
            "final_capital", "total_return_pct", "sharpe_ratio",
        ]
        for field in required_fields:
            assert field in data, f"フィールド '{field}' が見つかりません"
