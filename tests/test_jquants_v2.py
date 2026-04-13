"""J-Quants V2 API対応テスト。

V2 APIキー認証、V1後方互換、公式クライアントアダプタをテストする。
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from config.settings import JQuantsSettings
from src.data.jquants_client import JQuantsClient, JQuantsError


# ---------------------------------------------------------------------------
# JQuantsSettings V2 設定テスト
# ---------------------------------------------------------------------------

class TestJQuantsSettingsV2:
    """JQuantsSettings V2フィールドのテスト。"""

    def test_default_has_api_key_field(self) -> None:
        """デフォルト設定にapi_keyフィールドが存在する。"""
        s = JQuantsSettings()
        assert hasattr(s, "api_key")
        assert s.api_key == ""

    def test_default_base_url_is_v2(self) -> None:
        """デフォルトbase_urlがV2エンドポイントを指している。"""
        s = JQuantsSettings()
        assert "v2" in s.base_url

    def test_v2_api_key_can_be_set(self) -> None:
        """V2 APIキーを設定できる。"""
        s = JQuantsSettings(api_key="test-api-key-123")
        assert s.api_key == "test-api-key-123"

    def test_v1_fields_still_present(self) -> None:
        """V1フィールド（email, password）が後方互換として残っている。"""
        s = JQuantsSettings(email="user@example.com", password="secret")
        assert s.email == "user@example.com"
        assert s.password == "secret"

    def test_use_official_client_default_false(self) -> None:
        """use_official_clientのデフォルト値はFalse。"""
        s = JQuantsSettings()
        assert s.use_official_client is False

    def test_from_env_reads_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_env()でJQUANTS_API_KEY環境変数を読み込む。"""
        monkeypatch.setenv("JQUANTS_API_KEY", "env-api-key")
        s = JQuantsSettings.from_env()
        assert s.api_key == "env-api-key"

    def test_from_env_reads_v1_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_env()でV1フィールドも読み込む（後方互換）。"""
        monkeypatch.setenv("JQUANTS_EMAIL", "test@example.com")
        monkeypatch.setenv("JQUANTS_PASSWORD", "pass123")
        monkeypatch.delenv("JQUANTS_API_KEY", raising=False)
        s = JQuantsSettings.from_env()
        assert s.email == "test@example.com"
        assert s.password == "pass123"

    def test_from_env_use_official_client_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """JQUANTS_USE_OFFICIAL_CLIENT=1でuse_official_clientがTrueになる。"""
        monkeypatch.setenv("JQUANTS_USE_OFFICIAL_CLIENT", "1")
        s = JQuantsSettings.from_env()
        assert s.use_official_client is True

    def test_from_env_use_official_client_true_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """JQUANTS_USE_OFFICIAL_CLIENT=trueでuse_official_clientがTrueになる。"""
        monkeypatch.setenv("JQUANTS_USE_OFFICIAL_CLIENT", "true")
        s = JQuantsSettings.from_env()
        assert s.use_official_client is True


# ---------------------------------------------------------------------------
# JQuantsClient V2/V1 認証テスト
# ---------------------------------------------------------------------------

class TestJQuantsClientAuth:
    """JQuantsClient認証ロジックのテスト。"""

    def test_no_credentials_raises_error(self) -> None:
        """認証情報が設定されていない場合JQuantsErrorを発生させる。"""
        settings = JQuantsSettings(api_key="", email="", password="")
        client = JQuantsClient(settings)
        with pytest.raises(JQuantsError, match="JQUANTS_API_KEY"):
            client._get_refresh_token()

    def test_v2_api_key_sends_apikey_payload(self) -> None:
        """V2認証でapikey payloadをPOSTする。"""
        settings = JQuantsSettings(api_key="my-v2-key", base_url="https://api.jquants.com/v2")
        client = JQuantsClient(settings)

        mock_response = MagicMock()
        mock_response.json.return_value = {"refreshToken": "rt-v2-token"}
        mock_response.raise_for_status.return_value = None

        with patch.object(client._session, "request", return_value=mock_response) as mock_req:
            token = client._get_refresh_token()

        assert token == "rt-v2-token"
        call_kwargs = mock_req.call_args
        assert call_kwargs.kwargs.get("json") == {"apikey": "my-v2-key"} or \
               (len(call_kwargs.args) >= 3 and call_kwargs.args[2] == {"apikey": "my-v2-key"}) or \
               call_kwargs[1].get("json") == {"apikey": "my-v2-key"} or \
               any(v == {"apikey": "my-v2-key"} for v in call_kwargs.kwargs.values())

    def test_v1_email_password_emits_deprecation_warning(self) -> None:
        """V1認証（メール/パスワード）でDeprecationWarningを発生させる。"""
        settings = JQuantsSettings(
            api_key="",
            email="user@example.com",
            password="secret",
            base_url="https://api.jquants.com/v2",
        )
        client = JQuantsClient(settings)

        mock_response = MagicMock()
        mock_response.json.return_value = {"refreshToken": "rt-v1-token"}
        mock_response.raise_for_status.return_value = None

        with patch.object(client._session, "request", return_value=mock_response):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                token = client._get_refresh_token()

        assert token == "rt-v1-token"
        deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "V1" in str(deprecation_warnings[0].message) or "非推奨" in str(deprecation_warnings[0].message)


# ---------------------------------------------------------------------------
# 公式クライアントアダプタ テスト（モック）
# ---------------------------------------------------------------------------

class TestJQuantsOfficialClient:
    """JQuantsOfficialClientアダプタのテスト（公式SDKをモック）。"""

    def test_import_error_when_package_missing(self) -> None:
        """jquants-api-clientが未インストール時にImportErrorを発生させる。"""
        with patch.dict("sys.modules", {"jquantsapi": None}):
            from src.data.jquants_official_client import JQuantsOfficialClient
            with pytest.raises(ImportError, match="jquants-api-client"):
                JQuantsOfficialClient()

    def test_fetch_stock_prices_maps_columns(self) -> None:
        """fetch_stock_prices()が標準OHLCVカラム名にマッピングする。"""
        import pandas as pd

        mock_client = MagicMock()
        mock_df = pd.DataFrame({
            "Date": ["2024-01-05", "2024-01-06"],
            "Code": ["7203", "7203"],
            "Open": [2500.0, 2510.0],
            "High": [2550.0, 2560.0],
            "Low": [2480.0, 2490.0],
            "Close": [2520.0, 2530.0],
            "Volume": [1000000.0, 1100000.0],
        })
        mock_client.get_price_range.return_value = mock_df

        mock_jquantsapi = MagicMock()
        mock_jquantsapi.Client.return_value = mock_client

        with patch.dict("sys.modules", {"jquantsapi": mock_jquantsapi}):
            from src.data.jquants_official_client import JQuantsOfficialClient
            settings = JQuantsSettings(api_key="test-key")
            adapter = JQuantsOfficialClient.__new__(JQuantsOfficialClient)
            adapter._settings = settings
            adapter._client = mock_client

            result = adapter.fetch_stock_prices("7203", "2024-01-05", "2024-01-06")

        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns
