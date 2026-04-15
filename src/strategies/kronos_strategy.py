"""Kronos + VaR ベース戦略。

Kronos（金融市場特化基盤モデル、AAAI 2026）でOHLCV K-lineシーケンスから
次期価格方向を予測し、VaRベースのポジションサイジングを行う。

transformers/torch が未インストールの場合は全シグナルFLATにフォールバック。
"""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

from .base import BaseStrategy, SignalType
from ..forecasters.kronos_forecaster import _KRONOS_AVAILABLE
from ..risk.var_calculator import VaRCalculator

if _KRONOS_AVAILABLE:
    from ..forecasters.kronos_forecaster import KronosForecaster


class KronosStrategy(BaseStrategy):
    """Kronos + VaR ベース戦略。

    1. Kronos で OHLCV K-line シーケンスから horizon 日先を予測。
    2. 予測方向の多数決でトレンドを判定 → BUY/SELL。
    3. VaR ベースのポジションサイジング情報を付加。

    transformers/torch が未インストールの場合は全シグナルFLAT。

    Parameters
    ----------
    model_name:
        Kronos モデル名（デフォルト: "NeoQuasar/Kronos-base"）
    horizon:
        予測ステップ数（デフォルト: 5）
    var_confidence:
        VaR 算出の信頼水準（デフォルト: 0.95）
    capital:
        VaR ベースポジションサイジング用の想定資本（デフォルト: 1_000_000）
    var_limit:
        資本に対する VaR 上限率（デフォルト: 0.02 = 2%）
    revision:
        モデルリビジョン（破壊的更新対策にピン可能）
    """

    name = "kronos"

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-base",
        horizon: int = 5,
        var_confidence: float = 0.95,
        capital: float = 1_000_000.0,
        var_limit: float = 0.02,
        revision: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.horizon = horizon
        self.var_confidence = var_confidence
        self.capital = capital
        self.var_limit = var_limit
        self.revision = revision
        self._forecaster: Optional["KronosForecaster"] = None
        self._var_calc = VaRCalculator()

    def _min_bars(self) -> int:
        """最低限必要なデータ行数。"""
        return max(30, self.horizon * 3)

    def _get_forecaster(self) -> "KronosForecaster":
        """Kronos フォーキャスターをオンデマンドで初期化（遅延初期化）。"""
        if self._forecaster is None:
            self._forecaster = KronosForecaster(
                model_name=self.model_name,
                revision=self.revision,
            )
        return self._forecaster

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Kronos 予測に基づくシグナルを生成する。

        transformers/torch が未インストールの場合は全シグナルFLAT。

        Parameters
        ----------
        data:
            OHLCV DataFrame（index: DatetimeIndex）

        Returns
        -------
        pd.Series
            シグナル系列（1=BUY, -1=SELL, 0=FLAT）
        """
        self.validate_data(data)

        if not _KRONOS_AVAILABLE:
            warnings.warn(
                "transformers/torch が未インストールのため全シグナルFLATです。"
                " pip install transformers torch でインストールしてください。",
                RuntimeWarning,
                stacklevel=2,
            )
            return pd.Series(
                SignalType.FLAT,
                index=data.index,
                dtype=int,
            )

        df = data.copy()
        close = df["close"].astype(float)
        returns = close.pct_change().dropna()

        signals = pd.Series(SignalType.FLAT, index=df.index, dtype=int)
        position_sizes: list[float] = [0.0] * len(df)

        forecaster = self._get_forecaster()
        var_calc = self._var_calc
        min_context = self._min_bars()

        for i in range(min_context, len(df)):
            context = df.iloc[:i]

            try:
                direction = forecaster.predict_direction(
                    data=context,
                    horizon=self.horizon,
                )
            except Exception:  # noqa: BLE001
                continue

            if direction > 0:
                signal = SignalType.BUY
            elif direction < 0:
                signal = SignalType.SELL
            else:
                signal = SignalType.FLAT

            signals.iloc[i] = int(signal)

            # VaR ベースポジションサイジング
            if signal != SignalType.FLAT and len(returns) >= 2:
                try:
                    var_value = var_calc.calculate_var(
                        returns=returns.iloc[:i],
                        confidence=self.var_confidence,
                        method="historical",
                    )
                    pos_size = var_calc.position_size_by_var(
                        capital=self.capital,
                        var_limit=self.var_limit,
                        var_per_unit=var_value if var_value > 0 else 1.0,
                    )
                    position_sizes[i] = pos_size
                except Exception:  # noqa: BLE001
                    position_sizes[i] = 0.0

        self._last_position_sizes = pd.Series(
            position_sizes,
            index=df.index,
            name="position_size",
        )

        return signals
