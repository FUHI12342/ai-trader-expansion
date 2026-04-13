"""J-Quants API クライアント。

認証フロー（V2）: APIキー → リフレッシュトークン → IDトークン → 株価取得
認証フロー（V1、非推奨）: メールアドレス+パスワード → リフレッシュトークン → IDトークン → 株価取得
全APIキー情報は環境変数から取得する（ハードコード禁止）。

API仕様: https://jpx-jquants.com/
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

from config.settings import JQuantsSettings

logger = logging.getLogger(__name__)

# リトライ設定
_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 2.0


class JQuantsError(Exception):
    """J-Quants API エラー。"""
    pass


class JQuantsClient:
    """J-Quants API クライアント。

    V2 APIキー認証（推奨）またはV1メール/パスワード認証（非推奨）をサポートする。

    Parameters
    ----------
    settings:
        J-Quants設定（省略時は環境変数から読み込み）
    """

    def __init__(self, settings: Optional[JQuantsSettings] = None) -> None:
        self._settings = settings or JQuantsSettings.from_env()
        self._refresh_token: Optional[str] = None
        self._id_token: Optional[str] = None
        self._session = requests.Session()

    def _get_refresh_token(self) -> str:
        """リフレッシュトークンを取得する。

        V2 APIキーが設定されている場合はV2認証を使用する。
        V1メール/パスワードが設定されている場合はV1認証（非推奨）を使用する。
        どちらも設定されていない場合はエラーを発生させる。
        """
        if self._settings.api_key:
            return self._get_refresh_token_v2()
        elif self._settings.email and self._settings.password:
            return self._get_refresh_token_v1()
        else:
            raise JQuantsError(
                "J-Quants認証情報が設定されていません。"
                "環境変数 JQUANTS_API_KEY（推奨）または"
                " JQUANTS_EMAIL と JQUANTS_PASSWORD を設定してください。"
            )

    def _get_refresh_token_v2(self) -> str:
        """V2 APIキーを使ってリフレッシュトークンを取得する。"""
        url = f"{self._settings.base_url}/token/auth_user"
        payload = {"apikey": self._settings.api_key}

        response = self._request_with_retry("POST", url, json=payload, auth_required=False)
        data = response.json()

        if "refreshToken" not in data:
            raise JQuantsError(f"リフレッシュトークン取得失敗（V2）: {data}")

        return str(data["refreshToken"])

    def _get_refresh_token_v1(self) -> str:
        """V1メール/パスワードを使ってリフレッシュトークンを取得する（非推奨）。"""
        warnings.warn(
            "J-Quants V1認証（メール/パスワード）は非推奨です。"
            "JQUANTS_API_KEY 環境変数を使用したV2認証に移行してください。",
            DeprecationWarning,
            stacklevel=3,
        )

        url = f"{self._settings.base_url}/token/auth_user"
        payload = {
            "mailaddress": self._settings.email,
            "password": self._settings.password,
        }

        response = self._request_with_retry("POST", url, json=payload, auth_required=False)
        data = response.json()

        if "refreshToken" not in data:
            raise JQuantsError(f"リフレッシュトークン取得失敗（V1）: {data}")

        return str(data["refreshToken"])

    def _get_id_token(self, refresh_token: str) -> str:
        """IDトークンを取得する（リフレッシュトークン使用）。"""
        url = f"{self._settings.base_url}/token/auth_refresh"
        params = {"refreshtoken": refresh_token}

        response = self._request_with_retry("POST", url, params=params, auth_required=False)
        data = response.json()

        if "idToken" not in data:
            raise JQuantsError(f"IDトークン取得失敗: {data}")

        return str(data["idToken"])

    def authenticate(self) -> None:
        """認証を実行してIDトークンを取得する。"""
        logger.info("J-Quants API 認証中...")
        self._refresh_token = self._get_refresh_token()
        self._id_token = self._get_id_token(self._refresh_token)
        logger.info("J-Quants API 認証成功")

    def _ensure_authenticated(self) -> None:
        """認証済みでない場合は認証を実行する。"""
        if not self._id_token:
            self.authenticate()

    def _request_with_retry(
        self,
        method: str,
        url: str,
        auth_required: bool = True,
        **kwargs: Any,
    ) -> requests.Response:
        """指数バックオフでリトライするHTTPリクエスト。

        Parameters
        ----------
        method:
            HTTPメソッド（"GET", "POST"等）
        url:
            リクエストURL
        auth_required:
            認証ヘッダーが必要かどうか

        Returns
        -------
        requests.Response
        """
        headers = kwargs.pop("headers", {})
        if auth_required and self._id_token:
            headers["Authorization"] = f"Bearer {self._id_token}"

        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.request(
                    method, url, headers=headers, timeout=30, **kwargs
                )

                # 認証エラーの場合はトークンを更新してリトライ
                if resp.status_code == 401 and auth_required and attempt < _MAX_RETRIES - 1:
                    logger.warning("認証エラー: トークンを更新します")
                    self.authenticate()
                    headers["Authorization"] = f"Bearer {self._id_token}"
                    continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.HTTPError as e:
                if e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code != 401:
                    raise JQuantsError(f"J-Quants API HTTPエラー {e.response.status_code}: {e.response.text}") from e
                last_error = e
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e

            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAY_SEC * (2 ** attempt)
                logger.warning(f"リトライ {attempt + 1}/{_MAX_RETRIES} ({wait}秒後)")
                time.sleep(wait)

        raise JQuantsError(f"J-Quants API リクエスト失敗（{_MAX_RETRIES}回試行）: {last_error}") from last_error

    def fetch_stock_prices(
        self,
        code: str,
        date_from: str,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """日次株価データを取得する。

        Parameters
        ----------
        code:
            銘柄コード（例: "7203"）
        date_from:
            開始日（"YYYY-MM-DD"形式）
        date_to:
            終了日（省略時は今日）

        Returns
        -------
        pd.DataFrame
            OHLCV DataFrame（DatetimeIndex）
        """
        self._ensure_authenticated()

        params: Dict[str, str] = {"code": code, "dateFrom": date_from}
        if date_to:
            params["dateTo"] = date_to

        url = f"{self._settings.base_url}/prices/daily_quotes"
        response = self._request_with_retry("GET", url, params=params)
        data = response.json()

        if "daily_quotes" not in data:
            raise JQuantsError(f"株価データ取得失敗: {data}")

        quotes: List[Dict[str, Any]] = data["daily_quotes"]

        if not quotes:
            return pd.DataFrame()

        df = pd.DataFrame(quotes)

        # カラム名の統一（J-Quants形式 → OHLCV標準）
        column_map = {
            "Date": "date",
            "Code": "code",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "TurnoverValue": "turnover_value",
            "AdjustmentFactor": "adj_factor",
            "AdjustmentOpen": "adj_open",
            "AdjustmentHigh": "adj_high",
            "AdjustmentLow": "adj_low",
            "AdjustmentClose": "adj_close",
            "AdjustmentVolume": "adj_volume",
        }
        df = df.rename(columns=column_map)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        return df.copy()

    def fetch_listed_info(self) -> pd.DataFrame:
        """上場銘柄一覧を取得する。

        Returns
        -------
        pd.DataFrame
            上場銘柄DataFrame（code, name, sector等）
        """
        self._ensure_authenticated()

        url = f"{self._settings.base_url}/listed/info"
        response = self._request_with_retry("GET", url)
        data = response.json()

        if "info" not in data:
            raise JQuantsError(f"上場銘柄取得失敗: {data}")

        return pd.DataFrame(data["info"]).copy()

    def close(self) -> None:
        """セッションをクローズする。"""
        self._session.close()
