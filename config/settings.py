"""設定管理モジュール。

環境変数からAPIキーや接続情報を読み込む。
秘密情報は一切ハードコードしない。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class JQuantsSettings:
    """J-Quants API設定。"""
    email: str = ""
    password: str = ""
    base_url: str = "https://api.jquants.com/v1"

    @classmethod
    def from_env(cls) -> "JQuantsSettings":
        return cls(
            email=os.environ.get("JQUANTS_EMAIL", ""),
            password=os.environ.get("JQUANTS_PASSWORD", ""),
            base_url=os.environ.get("JQUANTS_BASE_URL", "https://api.jquants.com/v1"),
        )


@dataclass(frozen=True)
class EdinetSettings:
    """EDINET API設定。"""
    api_key: str = ""
    base_url: str = "https://disclosure.edinet-fsa.go.jp/api/v2"

    @classmethod
    def from_env(cls) -> "EdinetSettings":
        return cls(
            api_key=os.environ.get("EDINET_API_KEY", ""),
            base_url=os.environ.get("EDINET_BASE_URL", "https://disclosure.edinet-fsa.go.jp/api/v2"),
        )


@dataclass(frozen=True)
class KabuStationSettings:
    """kabuステーション API設定。"""
    api_password: str = ""
    host: str = "localhost"
    port: int = 18080
    timeout_sec: int = 15

    @classmethod
    def from_env(cls) -> "KabuStationSettings":
        return cls(
            api_password=os.environ.get("KABU_API_PASSWORD", ""),
            host=os.environ.get("KABU_API_HOST", "localhost"),
            port=int(os.environ.get("KABU_API_PORT", "18080")),
            timeout_sec=int(os.environ.get("KABU_API_TIMEOUT_SEC", "15")),
        )


@dataclass(frozen=True)
class DatabaseSettings:
    """データベース設定。"""
    cache_db_path: str = "./data_cache.db"

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        return cls(
            cache_db_path=os.environ.get("TRADER_CACHE_DB_PATH", "./data_cache.db"),
        )


@dataclass(frozen=True)
class ApiServerSettings:
    """FastAPIサーバー設定。"""
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "info"

    @classmethod
    def from_env(cls) -> "ApiServerSettings":
        return cls(
            host=os.environ.get("TRADER_API_HOST", "127.0.0.1"),
            port=int(os.environ.get("TRADER_API_PORT", "8765")),
            log_level=os.environ.get("TRADER_API_LOG_LEVEL", "info"),
        )


@dataclass(frozen=True)
class Settings:
    """全設定の集約クラス（immutable）。"""
    jquants: JQuantsSettings = field(default_factory=JQuantsSettings)
    edinet: EdinetSettings = field(default_factory=EdinetSettings)
    kabu_station: KabuStationSettings = field(default_factory=KabuStationSettings)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    api_server: ApiServerSettings = field(default_factory=ApiServerSettings)

    # バックテスト共通設定
    initial_capital: float = 1_000_000.0   # 初期資金（1百万円）
    fee_rate: float = 0.001                 # 片道手数料率 0.1%
    slippage_rate: float = 0.0005           # スリッページ率 0.05%

    @classmethod
    def from_env(cls) -> "Settings":
        """環境変数から設定を読み込む。"""
        return cls(
            jquants=JQuantsSettings.from_env(),
            edinet=EdinetSettings.from_env(),
            kabu_station=KabuStationSettings.from_env(),
            database=DatabaseSettings.from_env(),
            api_server=ApiServerSettings.from_env(),
            initial_capital=float(os.environ.get("TRADER_INITIAL_CAPITAL", "1000000")),
            fee_rate=float(os.environ.get("TRADER_FEE_RATE", "0.001")),
            slippage_rate=float(os.environ.get("TRADER_SLIPPAGE_RATE", "0.0005")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """キャッシュ付き設定取得（シングルトン）。"""
    return Settings.from_env()
