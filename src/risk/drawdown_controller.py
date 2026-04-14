"""ドローダウンコントローラー。

エクイティカーブの最大ドローダウンを監視し、
段階的にエクスポージャーを削減する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DrawdownAction(str, Enum):
    """ドローダウンに対するアクション。"""
    NORMAL = "normal"
    REDUCE_EXPOSURE = "reduce_exposure"
    HALT_TRADING = "halt_trading"
    EMERGENCY_EXIT = "emergency_exit"


@dataclass(frozen=True)
class DrawdownStatus:
    """ドローダウン状態（immutable）。

    Parameters
    ----------
    current_drawdown:
        現在のドローダウン（0.0〜1.0）
    max_drawdown:
        最大ドローダウン（0.0〜1.0）
    peak_equity:
        エクイティのピーク値
    current_equity:
        現在のエクイティ
    action:
        推奨アクション
    exposure_ratio:
        推奨エクスポージャー比率（0.0〜1.0）
    """

    current_drawdown: float
    max_drawdown: float
    peak_equity: float
    current_equity: float
    action: DrawdownAction
    exposure_ratio: float


class DrawdownController:
    """ドローダウンコントローラー。

    Parameters
    ----------
    max_drawdown:
        最大許容ドローダウン（デフォルト: 0.10 = 10%）
    reduce_threshold:
        エクスポージャー削減開始の閾値（デフォルト: 0.05 = 5%）
    halt_threshold:
        取引停止の閾値（デフォルト: 0.08 = 8%）
    emergency_threshold:
        緊急退出の閾値（デフォルト: 0.12 = 12%）
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,
        reduce_threshold: float = 0.05,
        halt_threshold: float = 0.08,
        emergency_threshold: float = 0.12,
    ) -> None:
        if not (0 < reduce_threshold < halt_threshold < emergency_threshold):
            raise ValueError(
                "閾値は reduce < halt < emergency の順に設定してください"
            )
        self._max_drawdown = max_drawdown
        self._reduce_threshold = reduce_threshold
        self._halt_threshold = halt_threshold
        self._emergency_threshold = emergency_threshold

    def check(self, equity_curve: pd.Series) -> DrawdownStatus:
        """エクイティカーブからドローダウン状態を判定する。

        Parameters
        ----------
        equity_curve:
            エクイティカーブ（時系列）

        Returns
        -------
        DrawdownStatus
            現在のドローダウン状態とアクション
        """
        if equity_curve.empty:
            return DrawdownStatus(
                current_drawdown=0.0, max_drawdown=0.0,
                peak_equity=0.0, current_equity=0.0,
                action=DrawdownAction.NORMAL, exposure_ratio=1.0,
            )

        values = equity_curve.values.astype(float)
        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / np.where(peak > 0, peak, 1.0)

        current_dd = float(drawdowns[-1])
        max_dd = float(np.max(drawdowns))
        peak_equity = float(peak[-1])
        current_equity = float(values[-1])

        action, exposure = self._determine_action(current_dd)

        return DrawdownStatus(
            current_drawdown=current_dd,
            max_drawdown=max_dd,
            peak_equity=peak_equity,
            current_equity=current_equity,
            action=action,
            exposure_ratio=exposure,
        )

    def _determine_action(self, dd: float) -> tuple[DrawdownAction, float]:
        """ドローダウンに基づいてアクションとエクスポージャー比率を決定する。"""
        if dd >= self._emergency_threshold:
            return DrawdownAction.EMERGENCY_EXIT, 0.0
        elif dd >= self._halt_threshold:
            return DrawdownAction.HALT_TRADING, 0.0
        elif dd >= self._reduce_threshold:
            # 線形にエクスポージャーを削減
            ratio = 1.0 - (dd - self._reduce_threshold) / (
                self._halt_threshold - self._reduce_threshold
            )
            return DrawdownAction.REDUCE_EXPOSURE, max(0.1, ratio)
        else:
            return DrawdownAction.NORMAL, 1.0
