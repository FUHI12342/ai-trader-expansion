"""公式 jquants-api-client パッケージのアダプタ。

JQuantsClientと同じインターフェースで公式SDKをラップする。
jquants-api-clientがインストールされていない場合はImportErrorを発生させる。
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.settings import JQuantsSettings

logger = logging.getLogger(__name__)


class JQuantsOfficialClientError(Exception):
    """公式クライアントアダプタエラー。"""
    pass


class JQuantsOfficialClient:
    """公式 jquants-api-client SDK のアダプタ。

    JQuantsClient と同じインターフェースを提供するため、
    DataManager から透過的に切り替えられる。

    Parameters
    ----------
    settings:
        J-Quants設定（省略時は環境変数から読み込み）

    Raises
    ------
    ImportError
        jquants-api-client パッケージがインストールされていない場合。
    """

    def __init__(self, settings: Optional[JQuantsSettings] = None) -> None:
        try:
            import jquantsapi  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "jquants-api-client パッケージがインストールされていません。\n"
                "pip install 'ai-trader-expansion[jquants]' または\n"
                "pip install jquants-api-client>=2.0.0 を実行してください。"
            ) from exc

        self._settings = settings or JQuantsSettings.from_env()
        self._client = self._create_client()

    def _create_client(self) -> object:
        """公式クライアントを初期化する。"""
        import jquantsapi

        if self._settings.api_key:
            return jquantsapi.Client(apikey=self._settings.api_key)  # type: ignore[attr-defined]
        elif self._settings.email and self._settings.password:
            return jquantsapi.Client(  # type: ignore[attr-defined]
                mail_address=self._settings.email,
                password=self._settings.password,
            )
        else:
            raise JQuantsOfficialClientError(
                "J-Quants認証情報が設定されていません。"
                "環境変数 JQUANTS_API_KEY（推奨）または"
                " JQUANTS_EMAIL と JQUANTS_PASSWORD を設定してください。"
            )

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
        import jquantsapi

        client: jquantsapi.Client = self._client  # type: ignore[assignment]

        df: pd.DataFrame = client.get_price_range(  # type: ignore[attr-defined]
            code=code,
            start_dt=date_from,
            end_dt=date_to or "",
        )

        if df.empty:
            return df

        # 公式SDK形式 → OHLCV標準カラムへマッピング
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
        import jquantsapi

        client: jquantsapi.Client = self._client  # type: ignore[assignment]
        df: pd.DataFrame = client.get_listed_info()  # type: ignore[attr-defined]
        return df.copy()

    def close(self) -> None:
        """セッションをクローズする（公式SDKには不要だがインターフェース互換）。"""
        pass
