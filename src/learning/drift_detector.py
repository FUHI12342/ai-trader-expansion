"""Concept Drift Detector。

複数のドリフト検出アルゴリズムを統合し、
市場レジーム変化を自動検出する。

River がインストールされている場合は ADWIN/DDM/KSWIN を使用。
未インストールの場合は統計的フォールバック（移動平均乖離）を使用。
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from river.drift import ADWIN, DDM, KSWIN  # type: ignore[import-not-found]
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False
    logger.info("river未インストール: 統計的フォールバックで動作")


class DriftSeverity(str, Enum):
    """ドリフトの深刻度。"""
    NONE = "none"
    WARNING = "warning"
    DRIFT = "drift"


@dataclass(frozen=True)
class DriftResult:
    """ドリフト検出結果（immutable）。"""

    detected: bool
    method: str
    severity: DriftSeverity
    details: Dict[str, float]

    @property
    def is_warning(self) -> bool:
        return self.severity == DriftSeverity.WARNING

    @property
    def is_drift(self) -> bool:
        return self.severity == DriftSeverity.DRIFT


class _FallbackDetector:
    """River未インストール時の統計的フォールバック検出器。

    短期移動平均と長期移動平均の乖離率でドリフトを検出する。
    """

    def __init__(self, short_window: int = 20, long_window: int = 100,
                 warning_threshold: float = 2.0, drift_threshold: float = 3.0) -> None:
        self._short: deque[float] = deque(maxlen=short_window)
        self._long: deque[float] = deque(maxlen=long_window)
        self._warning_threshold = warning_threshold
        self._drift_threshold = drift_threshold

    def update(self, value: float) -> DriftResult:
        self._short.append(value)
        self._long.append(value)

        if len(self._long) < self._long.maxlen:  # type: ignore[arg-type]
            return DriftResult(
                detected=False, method="fallback",
                severity=DriftSeverity.NONE, details={},
            )

        short_mean = sum(self._short) / len(self._short)
        long_mean = sum(self._long) / len(self._long)
        long_std = (sum((x - long_mean) ** 2 for x in self._long)
                    / len(self._long)) ** 0.5

        if long_std == 0:
            z_score = 0.0
        else:
            z_score = abs(short_mean - long_mean) / long_std

        details = {
            "short_mean": short_mean,
            "long_mean": long_mean,
            "z_score": z_score,
        }

        if z_score >= self._drift_threshold:
            return DriftResult(
                detected=True, method="fallback",
                severity=DriftSeverity.DRIFT, details=details,
            )
        elif z_score >= self._warning_threshold:
            return DriftResult(
                detected=True, method="fallback",
                severity=DriftSeverity.WARNING, details=details,
            )
        return DriftResult(
            detected=False, method="fallback",
            severity=DriftSeverity.NONE, details=details,
        )

    def reset(self) -> None:
        self._short.clear()
        self._long.clear()


class DriftDetector:
    """複数のドリフト検出アルゴリズムを統合する。

    River が利用可能な場合は ADWIN/DDM/KSWIN を使用。
    いずれか1つでもドリフトを検出したら結果を返す。

    Parameters
    ----------
    methods:
        使用するアルゴリズム名のリスト（デフォルト: ["adwin"]）
    """

    SUPPORTED_METHODS = frozenset({"adwin", "ddm", "kswin", "fallback"})

    def __init__(self, methods: Optional[List[str]] = None) -> None:
        if methods is None:
            methods = ["adwin"] if HAS_RIVER else ["fallback"]

        self._detectors: Dict[str, Any] = {}

        for m in methods:
            if m not in self.SUPPORTED_METHODS:
                raise ValueError(f"未対応のメソッド: {m}. 対応: {sorted(self.SUPPORTED_METHODS)}")

            if m == "fallback":
                self._detectors[m] = _FallbackDetector()
            elif not HAS_RIVER:
                logger.warning("river未インストール: %s をfallbackに置換", m)
                self._detectors["fallback"] = _FallbackDetector()
            elif m == "adwin":
                self._detectors[m] = ADWIN()
            elif m == "ddm":
                self._detectors[m] = DDM()
            elif m == "kswin":
                self._detectors[m] = KSWIN(alpha=0.005, window_size=100, stat_size=30)

        self._update_count: int = 0

    @property
    def update_count(self) -> int:
        """update() の呼び出し回数。"""
        return self._update_count

    @property
    def methods(self) -> List[str]:
        """使用中のメソッド名リスト。"""
        return list(self._detectors.keys())

    def update(self, value: float) -> DriftResult:
        """値を更新し、ドリフトを検出する。

        Parameters
        ----------
        value:
            新しい観測値（予測誤差、損益率など）

        Returns
        -------
        DriftResult
            最も深刻なドリフト結果
        """
        self._update_count += 1
        worst = DriftResult(
            detected=False, method="none",
            severity=DriftSeverity.NONE, details={},
        )

        for name, detector in self._detectors.items():
            result = self._check_detector(name, detector, value)
            if result.severity.value > worst.severity.value:
                worst = result
            # DRIFT が見つかったら即返す
            if worst.is_drift:
                return worst

        return worst

    def _check_detector(self, name: str, detector: Any, value: float) -> DriftResult:
        """個別の検出器を更新してチェックする。"""
        if isinstance(detector, _FallbackDetector):
            return detector.update(value)

        # River の検出器
        detector.update(value)

        in_drift = getattr(detector, "drift_detected", False)
        in_warning = getattr(detector, "warning_detected", False)

        if in_drift:
            return DriftResult(
                detected=True, method=name,
                severity=DriftSeverity.DRIFT,
                details={"update_count": self._update_count},
            )
        elif in_warning:
            return DriftResult(
                detected=True, method=name,
                severity=DriftSeverity.WARNING,
                details={"update_count": self._update_count},
            )

        return DriftResult(
            detected=False, method=name,
            severity=DriftSeverity.NONE,
            details={"update_count": self._update_count},
        )

    def reset(self) -> None:
        """全検出器をリセットする。"""
        self._update_count = 0
        for name, detector in self._detectors.items():
            if isinstance(detector, _FallbackDetector):
                detector.reset()
            elif HAS_RIVER:
                # River の検出器は新規インスタンスで置き換え
                if name == "adwin":
                    self._detectors[name] = ADWIN()
                elif name == "ddm":
                    self._detectors[name] = DDM()
                elif name == "kswin":
                    self._detectors[name] = KSWIN(alpha=0.005, window_size=100, stat_size=30)
