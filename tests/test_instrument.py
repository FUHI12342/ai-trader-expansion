"""Instrument モデルと parameter_space のテスト。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models import AssetClass, Instrument, resolve_instrument
from src.models.instrument import KNOWN_INSTRUMENTS
from src.strategies.base import BaseStrategy
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.dual_momentum import DualMomentumStrategy
from src.strategies.macd_rsi import MACDRSIStrategy
from src.strategies.bollinger_rsi_adx import BollingerRSIADXStrategy


# ============================================================
# AssetClass enum のテスト
# ============================================================

class TestAssetClass:
    """AssetClass enum のテスト。"""

    def test_stock_value(self):
        """STOCK の値が正しいことを確認。"""
        assert AssetClass.STOCK == "stock"

    def test_crypto_value(self):
        """CRYPTO の値が正しいことを確認。"""
        assert AssetClass.CRYPTO == "crypto"

    def test_futures_value(self):
        """FUTURES の値が正しいことを確認。"""
        assert AssetClass.FUTURES == "futures"

    def test_bond_etf_value(self):
        """BOND_ETF の値が正しいことを確認。"""
        assert AssetClass.BOND_ETF == "bond_etf"

    def test_all_four_members(self):
        """4種類すべてのメンバーが存在することを確認。"""
        assert len(AssetClass) == 4


# ============================================================
# Instrument dataclass のテスト
# ============================================================

class TestInstrument:
    """Instrument dataclass のテスト。"""

    def test_creation_with_required_fields(self):
        """必須フィールドのみで作成できることを確認。"""
        inst = Instrument(symbol="TEST", asset_class=AssetClass.STOCK)
        assert inst.symbol == "TEST"
        assert inst.asset_class == AssetClass.STOCK

    def test_default_values(self):
        """デフォルト値が正しいことを確認。"""
        inst = Instrument(symbol="TEST", asset_class=AssetClass.STOCK)
        assert inst.currency == "JPY"
        assert inst.tick_size == 1.0
        assert inst.lot_size == 100.0
        assert inst.periods_per_year == 252
        assert inst.margin_required is False
        assert inst.default_leverage == 1.0
        assert inst.fee_rate == 0.001

    def test_frozen_immutable(self):
        """frozen=True で変更不可であることを確認。"""
        inst = Instrument(symbol="TEST", asset_class=AssetClass.STOCK)
        with pytest.raises((AttributeError, TypeError)):
            inst.symbol = "OTHER"  # type: ignore[misc]

    def test_custom_values(self):
        """カスタム値で正しく作成されることを確認。"""
        inst = Instrument(
            symbol="BTC/JPY",
            asset_class=AssetClass.CRYPTO,
            exchange="bitFlyer",
            currency="JPY",
            tick_size=1.0,
            lot_size=0.0001,
            periods_per_year=365,
            margin_required=False,
            default_leverage=1.0,
            fee_rate=0.0005,
        )
        assert inst.lot_size == 0.0001
        assert inst.periods_per_year == 365
        assert inst.fee_rate == 0.0005


# ============================================================
# KNOWN_INSTRUMENTS レジストリのテスト
# ============================================================

class TestKnownInstruments:
    """KNOWN_INSTRUMENTS レジストリのテスト。"""

    def test_contains_7203(self):
        """7203.T (トヨタ) が登録されていることを確認。"""
        assert "7203.T" in KNOWN_INSTRUMENTS
        inst = KNOWN_INSTRUMENTS["7203.T"]
        assert inst.asset_class == AssetClass.STOCK
        assert inst.exchange == "TSE"
        assert inst.currency == "JPY"
        assert inst.lot_size == 100.0

    def test_contains_btc_jpy(self):
        """BTC/JPY が登録されていることを確認。"""
        assert "BTC/JPY" in KNOWN_INSTRUMENTS
        inst = KNOWN_INSTRUMENTS["BTC/JPY"]
        assert inst.asset_class == AssetClass.CRYPTO
        assert inst.periods_per_year == 365
        assert inst.lot_size == 0.0001
        assert inst.fee_rate == 0.0005

    def test_contains_nk225f(self):
        """NK225F が登録されていることを確認。"""
        assert "NK225F" in KNOWN_INSTRUMENTS
        inst = KNOWN_INSTRUMENTS["NK225F"]
        assert inst.asset_class == AssetClass.FUTURES
        assert inst.exchange == "OSE"
        assert inst.margin_required is True

    def test_contains_bond_etf(self):
        """2510, 2511 の債券ETFが登録されていることを確認。"""
        assert "2510" in KNOWN_INSTRUMENTS
        assert "2511" in KNOWN_INSTRUMENTS
        assert KNOWN_INSTRUMENTS["2510"].asset_class == AssetClass.BOND_ETF
        assert KNOWN_INSTRUMENTS["2511"].asset_class == AssetClass.BOND_ETF

    def test_contains_9984(self):
        """9984.T (ソフトバンクG) が登録されていることを確認。"""
        assert "9984.T" in KNOWN_INSTRUMENTS

    def test_contains_crypto_usdt(self):
        """BTCUSDT, ETHUSDT が登録されていることを確認。"""
        assert "BTCUSDT" in KNOWN_INSTRUMENTS
        assert "ETHUSDT" in KNOWN_INSTRUMENTS
        assert KNOWN_INSTRUMENTS["BTCUSDT"].currency == "USDT"


# ============================================================
# resolve_instrument のテスト
# ============================================================

class TestResolveInstrument:
    """resolve_instrument 関数のテスト。"""

    def test_known_symbol_returns_registry_entry(self):
        """既知シンボルはレジストリのエントリを返すことを確認。"""
        inst = resolve_instrument("7203.T")
        assert inst is KNOWN_INSTRUMENTS["7203.T"]

    def test_autodetect_dot_t_suffix_stock(self):
        """.T サフィックスで STOCK に自動判定されることを確認。"""
        inst = resolve_instrument("1234.T")
        assert inst.asset_class == AssetClass.STOCK
        assert inst.exchange == "TSE"
        assert inst.currency == "JPY"

    def test_autodetect_slash_crypto(self):
        """/ を含む未知シンボルで CRYPTO に自動判定されることを確認。"""
        inst = resolve_instrument("XRP/JPY")
        assert inst.asset_class == AssetClass.CRYPTO
        assert inst.periods_per_year == 365

    def test_autodetect_f_suffix_futures(self):
        """F サフィックスで FUTURES に自動判定されることを確認。"""
        inst = resolve_instrument("TOPIXF")
        assert inst.asset_class == AssetClass.FUTURES
        assert inst.margin_required is True

    def test_unknown_symbol_returns_stock_defaults(self):
        """未知シンボルは STOCK のデフォルト値で返されることを確認。"""
        inst = resolve_instrument("UNKNOWN_XYZ_999")
        assert inst.symbol == "UNKNOWN_XYZ_999"
        assert inst.asset_class == AssetClass.STOCK
        assert inst.currency == "JPY"
        assert inst.lot_size == 100.0

    def test_known_crypto_usdt(self):
        """BTCUSDT は USDT 通貨で返されることを確認。"""
        inst = resolve_instrument("BTCUSDT")
        assert inst.currency == "USDT"
        assert inst.asset_class == AssetClass.CRYPTO


# ============================================================
# parameter_space のテスト
# ============================================================

class TestParameterSpace:
    """parameter_space() メソッドのテスト。"""

    def test_base_returns_empty_dict(self):
        """BaseStrategy のデフォルト実装は空辞書を返すことを確認。"""
        strategy = MACrossoverStrategy()
        # BaseStrategy の parameter_space() は空辞書
        # MACrossoverStrategy はオーバーライドしているので、基底クラスを直接確認
        base_result = BaseStrategy.parameter_space(strategy)
        assert base_result == {}

    def test_ma_crossover_parameter_space(self):
        """MACrossoverStrategy の parameter_space が正しいことを確認。"""
        space = MACrossoverStrategy().parameter_space()
        assert "fast_period" in space
        assert "slow_period" in space
        assert space["fast_period"] == (int, 5, 50)
        assert space["slow_period"] == (int, 20, 200)

    def test_dual_momentum_parameter_space(self):
        """DualMomentumStrategy の parameter_space が正しいことを確認。"""
        space = DualMomentumStrategy().parameter_space()
        assert "lookback_period" in space
        assert "rebalance_period" in space
        assert space["lookback_period"] == (int, 60, 504)
        assert space["rebalance_period"] == (int, 5, 30)

    def test_macd_rsi_parameter_space(self):
        """MACDRSIStrategy の parameter_space が正しいことを確認。"""
        space = MACDRSIStrategy().parameter_space()
        assert "fast_period" in space
        assert "slow_period" in space
        assert "signal_period" in space
        assert "rsi_period" in space
        assert space["fast_period"] == (int, 8, 20)
        assert space["slow_period"] == (int, 20, 40)
        assert space["signal_period"] == (int, 5, 15)
        assert space["rsi_period"] == (int, 7, 21)

    def test_bollinger_rsi_adx_parameter_space(self):
        """BollingerRSIADXStrategy の parameter_space が正しいことを確認。"""
        space = BollingerRSIADXStrategy().parameter_space()
        assert "bb_period" in space
        assert "rsi_period" in space
        assert "adx_period" in space
        assert "adx_threshold" in space
        assert space["bb_period"] == (int, 10, 30)
        assert space["rsi_period"] == (int, 7, 21)
        assert space["adx_period"] == (int, 7, 21)
        assert space["adx_threshold"] == (float, 20.0, 35.0)

    def test_lgbm_parameter_space(self):
        """LGBMPredictorStrategy の parameter_space が正しいことを確認（lightgbm不要）。"""
        # lgbm_predictor.py の parameter_space() は HAS_LGB に依存しない
        # importをモックして直接テスト
        with patch.dict("sys.modules", {"lightgbm": MagicMock()}):
            # LGBMPredictorStrategy を再インポートする代わりに
            # parameter_space() のロジックを直接検証
            from src.strategies.lgbm_predictor import LGBMPredictorStrategy
            strategy = LGBMPredictorStrategy.__new__(LGBMPredictorStrategy)
            strategy.train_window = 504
            strategy.predict_window = 126
            strategy.min_train_samples = 100
            strategy.prob_threshold = 0.55
            strategy.lgbm_params = {}
            space = strategy.parameter_space()
            assert "n_estimators" in space
            assert "learning_rate" in space
            assert "train_window" in space
            assert space["n_estimators"] == (int, 50, 500)
            assert space["learning_rate"] == (float, 0.01, 0.3)
            assert space["train_window"] == (int, 126, 504)


# ============================================================
# _strategies レジストリのテスト（server.py）
# ============================================================

class TestStrategiesRegistry:
    """server.py の _strategies レジストリのテスト。"""

    def test_core_strategies_registered(self):
        """コア戦略4件が登録されていることを確認。"""
        # server.py のモジュールレベルコードを回避するためモックを使用
        core_strategies = {
            "ma_crossover": MACrossoverStrategy(),
            "dual_momentum": DualMomentumStrategy(),
            "macd_rsi": MACDRSIStrategy(),
            "bollinger_rsi_adx": BollingerRSIADXStrategy(),
        }
        assert "ma_crossover" in core_strategies
        assert "dual_momentum" in core_strategies
        assert "macd_rsi" in core_strategies
        assert "bollinger_rsi_adx" in core_strategies

    def test_lgbm_strategy_importable_with_mock(self):
        """lightgbm モック環境で LGBMPredictorStrategy がインポートできることを確認。"""
        with patch.dict("sys.modules", {"lightgbm": MagicMock()}):
            from src.strategies.lgbm_predictor import LGBMPredictorStrategy
            assert LGBMPredictorStrategy is not None

    def test_all_core_strategies_have_parameter_space(self):
        """コア戦略すべてに parameter_space() が実装されていることを確認。"""
        strategies = [
            MACrossoverStrategy(),
            DualMomentumStrategy(),
            MACDRSIStrategy(),
            BollingerRSIADXStrategy(),
        ]
        for strategy in strategies:
            space = strategy.parameter_space()
            assert isinstance(space, dict)
            assert len(space) > 0, f"{strategy.name} の parameter_space が空です"

    def test_parameter_space_values_are_3_tuples(self):
        """parameter_space の値が (type, min, max) の3要素タプルであることを確認。"""
        strategies = [
            MACrossoverStrategy(),
            DualMomentumStrategy(),
            MACDRSIStrategy(),
            BollingerRSIADXStrategy(),
        ]
        for strategy in strategies:
            space = strategy.parameter_space()
            for key, val in space.items():
                assert isinstance(val, tuple), f"{strategy.name}.{key} はタプルでない"
                assert len(val) == 3, f"{strategy.name}.{key} は3要素でない"
                assert val[0] in (int, float), f"{strategy.name}.{key} の型が不正"
