"""ブローカーファクトリー — アセットクラスに基づくブローカー自動選択。"""
from __future__ import annotations
import logging
import os
from typing import Dict, List, Optional
from src.models.instrument import AssetClass
from .base import BrokerBase
from .paper_broker import PaperBroker

logger = logging.getLogger(__name__)


class BrokerFactory:
    """ブローカーファクトリー。

    アセットクラスと設定に基づいて適切なブローカーを生成・管理する。
    """

    def __init__(self) -> None:
        self._brokers: Dict[str, BrokerBase] = {}
        self._asset_class_map: Dict[AssetClass, str] = {}

    def register(self, name: str, broker: BrokerBase,
                 asset_classes: Optional[List[AssetClass]] = None) -> None:
        """ブローカーを登録する。

        Parameters
        ----------
        name:
            ブローカー識別名
        broker:
            BrokerBaseを実装したブローカーインスタンス
        asset_classes:
            このブローカーが担当するアセットクラス一覧
        """
        self._brokers[name] = broker
        if asset_classes:
            for ac in asset_classes:
                self._asset_class_map[ac] = name

    def get_broker(self, name: str) -> Optional[BrokerBase]:
        """ブローカー名で取得する。"""
        return self._brokers.get(name)

    def get_broker_for_asset(self, asset_class: AssetClass) -> Optional[BrokerBase]:
        """アセットクラスに対応するブローカーを取得する。"""
        name = self._asset_class_map.get(asset_class)
        if name:
            return self._brokers.get(name)
        return None

    def list_brokers(self) -> Dict[str, str]:
        """登録ブローカー一覧を返す（名前→クラス名のマッピング）。"""
        return {name: type(broker).__name__ for name, broker in self._brokers.items()}

    @classmethod
    def create_default(cls, paper_mode: bool = True,
                       initial_capital: float = 1_000_000) -> "BrokerFactory":
        """デフォルト構成のファクトリーを作成する。

        Parameters
        ----------
        paper_mode:
            Trueの場合はペーパーブローカーのみ使用（デフォルト: True）
        initial_capital:
            初期資本額（円）

        Returns
        -------
        BrokerFactory
            設定済みのファクトリーインスタンス
        """
        factory = cls()

        if paper_mode:
            paper = PaperBroker(initial_balance=initial_capital)
            factory.register("paper", paper, [
                AssetClass.STOCK, AssetClass.CRYPTO,
                AssetClass.FUTURES, AssetClass.BOND_ETF,
            ])
        else:
            # ライブモード: 環境変数から設定を読んでブローカーを初期化
            # kabuSTATION（日本株+先物+債券ETF）
            try:
                from .kabu_station import KabuStationBroker
                from config.settings import KabuStationSettings
                kabu_password = os.environ.get("KABU_API_PASSWORD", "")
                if kabu_password:
                    kabu = KabuStationBroker(settings=KabuStationSettings.from_env())
                    factory.register("kabu_station", kabu, [
                        AssetClass.STOCK, AssetClass.FUTURES, AssetClass.BOND_ETF,
                    ])
            except (ImportError, Exception) as e:
                logger.warning(f"kabuSTATIONブローカー初期化失敗: {e}")

            # CCXT（暗号資産）
            try:
                from .ccxt_broker import CCXTBroker
                exchange_id = os.environ.get("CCXT_EXCHANGE", "")
                api_key = os.environ.get("CCXT_API_KEY", "")
                secret = os.environ.get("CCXT_SECRET", "")
                if exchange_id and api_key:
                    ccxt_broker = CCXTBroker(
                        exchange_id=exchange_id, api_key=api_key,
                        secret=secret, sandbox=False,
                    )
                    factory.register("ccxt", ccxt_broker, [AssetClass.CRYPTO])
            except (ImportError, Exception) as e:
                logger.warning(f"CCXTブローカー初期化失敗: {e}")

            # IB（海外市場）
            try:
                from .ib_broker import IBBroker
                ib_host = os.environ.get("IB_HOST", "")
                if ib_host:
                    ib = IBBroker(host=ib_host)
                    if ib.connect():
                        factory.register("ib", ib)
            except (ImportError, Exception) as e:
                logger.warning(f"IBブローカー初期化失敗: {e}")

            # フォールバック: 未割り当てアセットクラスにはペーパーブローカー
            paper_fallback = PaperBroker(initial_balance=initial_capital)
            for ac in AssetClass:
                if ac not in factory._asset_class_map:
                    factory.register("paper_fallback", paper_fallback, [ac])

        return factory
