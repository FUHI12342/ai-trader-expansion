"""相関モニター。

戦略間・アセット間の相関行列を定期的に計算し、
相関急上昇時にアラートを発する。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrelationAlert:
    """相関アラート（immutable）。"""

    pair: tuple[str, str]
    correlation: float
    threshold: float
    message: str


@dataclass(frozen=True)
class CorrelationSnapshot:
    """相関行列のスナップショット（immutable）。

    Parameters
    ----------
    matrix:
        相関行列（DataFrame）
    alerts:
        閾値超過のアラートリスト
    avg_correlation:
        平均相関（対角要素除く）
    max_correlation:
        最大相関（対角要素除く）
    """

    matrix: pd.DataFrame
    alerts: tuple[CorrelationAlert, ...]
    avg_correlation: float
    max_correlation: float


class CorrelationMonitor:
    """相関モニター。

    Parameters
    ----------
    alert_threshold:
        アラート発火の相関閾値（デフォルト: 0.8）
    window:
        ローリングウィンドウ日数（デフォルト: 60）
    ewm_span:
        指数加重移動平均のスパン（デフォルト: None = ローリング使用）
    """

    def __init__(
        self,
        alert_threshold: float = 0.8,
        window: int = 60,
        ewm_span: Optional[int] = None,
    ) -> None:
        self._alert_threshold = alert_threshold
        self._window = window
        self._ewm_span = ewm_span
        self._history: List[CorrelationSnapshot] = []

    @property
    def alert_threshold(self) -> float:
        return self._alert_threshold

    @property
    def snapshot_count(self) -> int:
        return len(self._history)

    def compute(self, returns: pd.DataFrame) -> CorrelationSnapshot:
        """リターン系列から相関行列を計算する。

        Parameters
        ----------
        returns:
            各資産/戦略のリターン系列

        Returns
        -------
        CorrelationSnapshot
            相関行列とアラート
        """
        if returns.empty or returns.shape[1] < 2:
            empty_matrix = pd.DataFrame()
            return CorrelationSnapshot(
                matrix=empty_matrix, alerts=(),
                avg_correlation=0.0, max_correlation=0.0,
            )

        # 直近 window 期間で計算
        data = returns.tail(self._window)

        if self._ewm_span is not None:
            corr = data.ewm(span=self._ewm_span).corr().iloc[-len(data.columns):]
            corr.index = data.columns
        else:
            corr = data.corr()

        # 上三角行列（対角を除く）からアラートを生成
        alerts = []
        n = len(corr.columns)
        corr_values = []

        for i in range(n):
            for j in range(i + 1, n):
                c = float(corr.iloc[i, j])
                corr_values.append(abs(c))
                if abs(c) >= self._alert_threshold:
                    pair = (str(corr.columns[i]), str(corr.columns[j]))
                    alerts.append(CorrelationAlert(
                        pair=pair,
                        correlation=c,
                        threshold=self._alert_threshold,
                        message=f"高相関検出: {pair[0]} / {pair[1]} = {c:.3f}",
                    ))

        avg_corr = float(np.mean(corr_values)) if corr_values else 0.0
        max_corr = float(np.max(corr_values)) if corr_values else 0.0

        snapshot = CorrelationSnapshot(
            matrix=corr.copy(),
            alerts=tuple(alerts),
            avg_correlation=avg_corr,
            max_correlation=max_corr,
        )
        self._history.append(snapshot)

        if alerts:
            for a in alerts:
                logger.warning(a.message)

        return snapshot

    def get_latest(self) -> Optional[CorrelationSnapshot]:
        """最新のスナップショットを返す。"""
        return self._history[-1] if self._history else None
