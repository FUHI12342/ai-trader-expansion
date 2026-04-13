"""金融商品モデル。

アセットクラス、銘柄情報、既知銘柄レジストリを提供する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class AssetClass(str, Enum):
    """アセットクラス。"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FUTURES = "futures"
    BOND_ETF = "bond_etf"


@dataclass(frozen=True)
class Instrument:
    """金融商品の定義。

    Parameters
    ----------
    symbol:
        銘柄シンボル（例: "7203.T", "BTC/JPY", "NK225F"）
    asset_class:
        アセットクラス（AssetClass enum）
    exchange:
        取引所コード（例: "TSE", "GMO", "bitFlyer", "OSE", "NYSE"）
    currency:
        決済通貨（デフォルト: "JPY"）
    tick_size:
        最小呼値（デフォルト: 1.0）
    lot_size:
        最小取引単位（日本株: 100, BTC: 0.0001）
    periods_per_year:
        年間取引日数（株式/先物: 252, 暗号資産: 365）
    margin_required:
        証拠金取引の場合は True（デフォルト: False）
    default_leverage:
        デフォルトレバレッジ（デフォルト: 1.0）
    fee_rate:
        手数料率（アセットクラスごとのデフォルト）
    """
    symbol: str
    asset_class: AssetClass
    exchange: str = ""
    currency: str = "JPY"
    tick_size: float = 1.0
    lot_size: float = 100.0
    periods_per_year: int = 252
    margin_required: bool = False
    default_leverage: float = 1.0
    fee_rate: float = 0.001


# 既知銘柄レジストリ
KNOWN_INSTRUMENTS: Dict[str, Instrument] = {
    # 日本株（東証）
    "7203.T": Instrument(
        symbol="7203.T",
        asset_class=AssetClass.STOCK,
        exchange="TSE",
        currency="JPY",
        tick_size=1.0,
        lot_size=100.0,
        periods_per_year=252,
        fee_rate=0.001,
    ),
    "9984.T": Instrument(
        symbol="9984.T",
        asset_class=AssetClass.STOCK,
        exchange="TSE",
        currency="JPY",
        tick_size=1.0,
        lot_size=100.0,
        periods_per_year=252,
        fee_rate=0.001,
    ),
    # 暗号資産（円建て）
    "BTC/JPY": Instrument(
        symbol="BTC/JPY",
        asset_class=AssetClass.CRYPTO,
        exchange="bitFlyer",
        currency="JPY",
        tick_size=1.0,
        lot_size=0.0001,
        periods_per_year=365,
        fee_rate=0.0005,
    ),
    "ETH/JPY": Instrument(
        symbol="ETH/JPY",
        asset_class=AssetClass.CRYPTO,
        exchange="bitFlyer",
        currency="JPY",
        tick_size=1.0,
        lot_size=0.001,
        periods_per_year=365,
        fee_rate=0.0005,
    ),
    # 暗号資産（USD建て）
    "BTCUSDT": Instrument(
        symbol="BTCUSDT",
        asset_class=AssetClass.CRYPTO,
        exchange="",
        currency="USDT",
        tick_size=0.01,
        lot_size=0.0001,
        periods_per_year=365,
        fee_rate=0.0005,
    ),
    "ETHUSDT": Instrument(
        symbol="ETHUSDT",
        asset_class=AssetClass.CRYPTO,
        exchange="",
        currency="USDT",
        tick_size=0.01,
        lot_size=0.001,
        periods_per_year=365,
        fee_rate=0.0005,
    ),
    # 先物
    "NK225F": Instrument(
        symbol="NK225F",
        asset_class=AssetClass.FUTURES,
        exchange="OSE",
        currency="JPY",
        tick_size=5.0,
        lot_size=1.0,
        periods_per_year=252,
        margin_required=True,
        fee_rate=0.001,
    ),
    # 債券ETF
    "2510": Instrument(
        symbol="2510",
        asset_class=AssetClass.BOND_ETF,
        exchange="TSE",
        currency="JPY",
        tick_size=1.0,
        lot_size=1.0,
        periods_per_year=252,
        fee_rate=0.001,
    ),
    "2511": Instrument(
        symbol="2511",
        asset_class=AssetClass.BOND_ETF,
        exchange="TSE",
        currency="JPY",
        tick_size=1.0,
        lot_size=1.0,
        periods_per_year=252,
        fee_rate=0.001,
    ),
}


def resolve_instrument(symbol: str) -> Instrument:
    """銘柄シンボルから Instrument を解決する。

    Parameters
    ----------
    symbol:
        銘柄シンボル（例: "7203.T", "BTC/JPY", "NK225F"）

    Returns
    -------
    Instrument
        KNOWN_INSTRUMENTS に登録済みの場合はその定義を返す。
        未登録の場合は自動判定でデフォルト値を持つ Instrument を返す。
    """
    # 既知銘柄レジストリを優先検索
    if symbol in KNOWN_INSTRUMENTS:
        return KNOWN_INSTRUMENTS[symbol]

    # 自動判定: "/" を含む → 暗号資産
    if "/" in symbol:
        return Instrument(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            exchange="",
            currency="JPY",
            tick_size=1.0,
            lot_size=0.0001,
            periods_per_year=365,
            fee_rate=0.0005,
        )

    # 自動判定: ".T" サフィックス → 日本株（東証）
    if symbol.endswith(".T"):
        return Instrument(
            symbol=symbol,
            asset_class=AssetClass.STOCK,
            exchange="TSE",
            currency="JPY",
            tick_size=1.0,
            lot_size=100.0,
            periods_per_year=252,
            fee_rate=0.001,
        )

    # 自動判定: "F" サフィックス → 先物
    if symbol.endswith("F"):
        return Instrument(
            symbol=symbol,
            asset_class=AssetClass.FUTURES,
            exchange="",
            currency="JPY",
            tick_size=1.0,
            lot_size=1.0,
            periods_per_year=252,
            margin_required=True,
            fee_rate=0.001,
        )

    # 不明シンボル: STOCKとして扱い合理的なデフォルトを返す
    return Instrument(
        symbol=symbol,
        asset_class=AssetClass.STOCK,
        exchange="",
        currency="JPY",
        tick_size=1.0,
        lot_size=100.0,
        periods_per_year=252,
        fee_rate=0.001,
    )
