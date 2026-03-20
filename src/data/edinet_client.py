"""EDINET API v2 クライアント。

金融庁の EDINET（電子開示システム）API v2 を使用して
有価証券報告書等の開示書類を取得する。

API仕様: https://disclosure2.edinet-fsa.go.jp/weee0020.aspx
全APIキー情報は環境変数から取得する（ハードコード禁止）。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

from config.settings import EdinetSettings

logger = logging.getLogger(__name__)

# リトライ設定
_MAX_RETRIES = 3
_RETRY_DELAY_SEC = 2.0


class EdinetError(Exception):
    """EDINET API エラー。"""
    pass


# 書類種別コード
DOC_TYPE_ANNUAL_REPORT = "120"          # 有価証券報告書
DOC_TYPE_QUARTERLY_REPORT = "140"       # 四半期報告書
DOC_TYPE_SHORT_REPORT = "180"           # 臨時報告書

# ファイル種別
FILE_TYPE_XBRL = 1                      # XBRL一式
FILE_TYPE_PDF = 2                       # PDF


class EdinetClient:
    """EDINET API v2 クライアント。

    Parameters
    ----------
    settings:
        EDINET設定（省略時は環境変数から読み込み）
    """

    def __init__(self, settings: Optional[EdinetSettings] = None) -> None:
        self._settings = settings or EdinetSettings.from_env()
        self._session = requests.Session()

    def _make_params(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """APIキーを含むリクエストパラメータを構築する。"""
        params: Dict[str, str] = {}
        if self._settings.api_key:
            params["Subscription-Key"] = self._settings.api_key
        if extra:
            params.update(extra)
        return params

    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """指数バックオフでリトライするHTTPリクエスト。"""
        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.request(method, url, timeout=60, **kwargs)

                if resp.status_code == 429:
                    # レートリミット: 少し待つ
                    wait = 10.0 * (attempt + 1)
                    logger.warning(f"EDINET レートリミット: {wait}秒待機")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.HTTPError as e:
                if e.response is not None and 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise EdinetError(f"EDINET API HTTPエラー {e.response.status_code}: {e.response.text}") from e
                last_error = e
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e

            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_DELAY_SEC * (2 ** attempt)
                logger.warning(f"EDINET リトライ {attempt + 1}/{_MAX_RETRIES} ({wait}秒後)")
                time.sleep(wait)

        raise EdinetError(f"EDINET API リクエスト失敗（{_MAX_RETRIES}回試行）: {last_error}") from last_error

    def fetch_document_list(
        self,
        date: str,
        doc_type: int = 2,
    ) -> pd.DataFrame:
        """指定日の開示書類一覧を取得する。

        Parameters
        ----------
        date:
            対象日（"YYYY-MM-DD"形式）
        doc_type:
            書類種別 (1=メタデータのみ, 2=メタデータ+書類一覧)

        Returns
        -------
        pd.DataFrame
            書類一覧DataFrame（docID, filerName, docDescription等）
        """
        url = f"{self._settings.base_url}/documents.json"
        params = self._make_params({"date": date, "type": str(doc_type)})

        response = self._request_with_retry("GET", url, params=params)
        data = response.json()

        if "results" not in data:
            raise EdinetError(f"書類一覧取得失敗: {data}")

        docs: List[Dict[str, Any]] = data["results"]
        if not docs:
            return pd.DataFrame()

        return pd.DataFrame(docs).copy()

    def fetch_document(
        self,
        doc_id: str,
        file_type: int = FILE_TYPE_PDF,
    ) -> bytes:
        """書類ファイルを取得する。

        Parameters
        ----------
        doc_id:
            書類管理番号（例: "S100TEST"）
        file_type:
            ファイル種別 (1=XBRL, 2=PDF)

        Returns
        -------
        bytes
            書類ファイルのバイナリデータ
        """
        url = f"{self._settings.base_url}/documents/{doc_id}"
        params = self._make_params({"type": str(file_type)})

        response = self._request_with_retry("GET", url, params=params)
        return response.content

    def search_filings(
        self,
        edinetcode: Optional[str] = None,
        doc_type_code: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        """開示書類を条件検索する。

        Parameters
        ----------
        edinetcode:
            EDINETコード（例: "E00001"）
        doc_type_code:
            書類種別コード（例: "120" = 有価証券報告書）
        date_from:
            開始日（"YYYY-MM-DD"形式）
        date_to:
            終了日（"YYYY-MM-DD"形式）

        Returns
        -------
        pd.DataFrame
            条件に合致する書類一覧
        """
        # 日付範囲を1日ずつ取得してフィルタリング（EDINET APIは日付単位）
        if not date_from or not date_to:
            raise ValueError("date_from と date_to を指定してください。")

        all_docs: List[Dict[str, Any]] = []
        dates = pd.date_range(start=date_from, end=date_to, freq="B")  # 営業日のみ

        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            try:
                day_df = self.fetch_document_list(date_str)
                if not day_df.empty:
                    all_docs.extend(day_df.to_dict(orient="records"))
            except EdinetError as e:
                logger.warning(f"{date_str} の書類取得失敗: {e}")
                continue

        if not all_docs:
            return pd.DataFrame()

        result_df = pd.DataFrame(all_docs)

        # フィルタリング
        if edinetcode and "edinetCode" in result_df.columns:
            result_df = result_df[result_df["edinetCode"] == edinetcode]
        if doc_type_code and "docTypeCode" in result_df.columns:
            result_df = result_df[result_df["docTypeCode"] == doc_type_code]

        return result_df.copy()

    def close(self) -> None:
        """セッションをクローズする。"""
        self._session.close()
