"""デュアルモメンタム戦略（Antonacci スタイル）。

- 絶対モメンタム: アセットの lookback_period 日リターン > threshold → BUY条件
- 相対モメンタム: 複数アセット対比（単一アセット版では省略）
- 月次リバランス想定だが日次データにも対応

参考: Gary Antonacci「Dual Momentum Investing」
"""
from __future__ import annotations

import pandas as pd

from .base import BaseStrategy, SignalType


class DualMomentumStrategy(BaseStrategy):
    """デュアルモメンタム戦略。

    Parameters
    ----------
    lookback_period:
        モメンタム計算期間（日数、デフォルト: 252営業日 ≈ 1年）
    threshold:
        BUYシグナルを発生させるリターン閾値（デフォルト: 0.0 = プラスリターン）
    rebalance_freq:
        リバランス頻度 ('D'=毎日, 'W'=週次, 'ME'=月次, デフォルト: 'ME')
    """

    name = "dual_momentum"

    def __init__(
        self,
        lookback_period: int = 252,
        threshold: float = 0.0,
        rebalance_freq: str = "ME",
    ) -> None:
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.rebalance_freq = rebalance_freq

    def _min_bars(self) -> int:
        return self.lookback_period + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """デュアルモメンタムシグナルを生成する。

        Parameters
        ----------
        data:
            OHLCV DataFrame（index: DatetimeIndex）

        Returns
        -------
        pd.Series
            シグナル系列（1=BUY, 0=FLAT）
        """
        self.validate_data(data)
        df = data.copy()

        close = df["close"].astype(float)

        # lookback_period 日前の価格と比較してリターンを計算
        past_price = close.shift(self.lookback_period)
        momentum_return = (close / past_price) - 1.0

        # 絶対モメンタム: リターンが threshold を超えた場合 BUY
        raw_signal = (momentum_return > self.threshold).astype(int)

        # リバランス頻度でフィルタリング（月末のみシグナル更新）
        if self.rebalance_freq in ("ME", "M", "W", "W-FRI"):
            signals = self._apply_rebalance_filter(raw_signal, df.index)
        else:
            # 'D' の場合はフィルタなし
            signals = raw_signal

        # SignalType変換
        result = pd.Series(SignalType.FLAT, index=df.index, dtype=int)
        result[signals == 1] = SignalType.BUY
        # 前回BUYで今回シグナルが消えた場合はSELL
        prev_buy = (signals.shift(1) == 1)
        current_flat = (signals == 0)
        result[prev_buy & current_flat] = SignalType.SELL

        return result

    def _apply_rebalance_filter(
        self, signal: pd.Series, index: pd.DatetimeIndex
    ) -> pd.Series:
        """リバランス日のみシグナルを更新する。

        リバランス日以外は前回シグナルを維持する。
        """
        if not isinstance(index, pd.DatetimeIndex):
            # DatetimeIndexでない場合はそのまま返す
            return signal

        # リバランス日のマスク（各期間の最終日）
        rebalance_dates = (
            pd.Series(index).resample(self.rebalance_freq).last()
        )
        rebalance_set = set(rebalance_dates.values)

        # リバランス日以外は前回値を引き継ぐ
        result = signal.copy()
        last_value = 0
        for i, (date, val) in enumerate(signal.items()):
            if date in rebalance_set:
                last_value = val
            result.iloc[i] = last_value

        return result
