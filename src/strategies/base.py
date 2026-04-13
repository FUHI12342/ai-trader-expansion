"""戦略基底クラス（ABC）。

全戦略はこのクラスを継承してimplementする。
DataFrameは常にcopyを使用し、immutableパターンを維持する。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Any, Dict

import pandas as pd


class SignalType(IntEnum):
    """シグナル種別。"""
    SELL = -1   # 売り/ショート
    FLAT = 0    # ノーポジション
    BUY = 1     # 買い/ロング


# 必須カラム定義
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


class BaseStrategy(ABC):
    """全戦略の基底クラス。

    サブクラスは generate_signals() を実装すること。
    DataFrameは常にcopy()してimmutableパターンを守ること。
    """

    #: 戦略名（サブクラスで上書き必須）
    name: str = "base"

    def validate_data(self, data: pd.DataFrame) -> None:
        """入力データのバリデーション。

        Parameters
        ----------
        data:
            OHLCV DataFrame（必須カラム: open, high, low, close, volume）

        Raises
        ------
        ValueError
            必須カラムが欠けている、またはデータが空の場合。
        """
        if data.empty:
            raise ValueError("入力データが空です。")
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValueError(f"必須カラムが不足しています: {missing}")
        if len(data) < self._min_bars():
            raise ValueError(
                f"データ行数が不足しています。必要: {self._min_bars()}, 実際: {len(data)}"
            )

    def _min_bars(self) -> int:
        """最低限必要なデータ行数（サブクラスでオーバーライド可）。"""
        return 1

    def get_parameters(self) -> Dict[str, Any]:
        """現在のパラメータを辞書として返す。"""
        params = {}
        for key, val in vars(self).items():
            if not key.startswith("_"):
                params[key] = val
        return params

    def set_parameters(self, **kwargs: Any) -> "BaseStrategy":
        """パラメータを更新した新しいインスタンスを返す（immutableパターン）。

        Parameters
        ----------
        **kwargs:
            更新するパラメータ名=値

        Returns
        -------
        BaseStrategy
            新しいインスタンス（元は変更しない）
        """
        import copy
        new_instance = copy.copy(self)
        for key, val in kwargs.items():
            if not hasattr(new_instance, key):
                raise AttributeError(f"パラメータ '{key}' は存在しません。")
            object.__setattr__(new_instance, key, val)
        return new_instance

    def parameter_space(self) -> Dict[str, tuple]:
        """最適化可能パラメータとその範囲を返す。

        Returns
        -------
        Dict[str, tuple]
            {param_name: (type, min_value, max_value)}
            空辞書の場合はパラメータ最適化非対応。
        """
        return {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """シグナルを生成する（実装必須）。

        Parameters
        ----------
        data:
            OHLCV DataFrame（index: DatetimeIndex）

        Returns
        -------
        pd.Series
            SignalType値のSeries（1=BUY, -1=SELL, 0=FLAT）
            index は data.index と同じ
        """
        ...
