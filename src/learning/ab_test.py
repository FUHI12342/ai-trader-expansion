"""A/B テストフレームワーク。

チャンピオン（現行モデル）とチャレンジャー（新モデル）の
パフォーマンスを統計的に比較し、promote/rollback を判断する。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ABTestResult:
    """A/Bテスト結果（immutable）。

    Parameters
    ----------
    test_id:
        テスト識別子
    champion_id:
        チャンピオンモデルID
    challenger_id:
        チャレンジャーモデルID
    champion_mean:
        チャンピオンの平均PnL
    challenger_mean:
        チャレンジャーの平均PnL
    p_value:
        統計的検定のp値
    n_samples:
        サンプル数
    winner:
        勝者（"champion" / "challenger" / "inconclusive"）
    """

    test_id: str
    champion_id: str
    challenger_id: str
    champion_mean: float
    challenger_mean: float
    p_value: float
    n_samples: int
    winner: str  # "champion" | "challenger" | "inconclusive"


@dataclass
class _TestState:
    """A/Bテストの内部状態。"""

    test_id: str
    champion_id: str
    challenger_id: str
    champion_pnls: List[float] = field(default_factory=list)
    challenger_pnls: List[float] = field(default_factory=list)
    concluded: bool = False
    result: Optional[ABTestResult] = None


class ABTestManager:
    """A/B テスト管理。

    Parameters
    ----------
    significance_level:
        統計的有意水準（デフォルト: 0.05）
    min_samples:
        最小サンプル数（デフォルト: 30）
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_samples: int = 30,
    ) -> None:
        self._significance_level = significance_level
        self._min_samples = min_samples
        self._tests: Dict[str, _TestState] = {}
        self._test_counter: int = 0

    def start_test(
        self,
        champion_id: str,
        challenger_id: str,
        test_id: Optional[str] = None,
    ) -> str:
        """A/Bテストを開始する。

        Parameters
        ----------
        champion_id:
            チャンピオンモデルID
        challenger_id:
            チャレンジャーモデルID
        test_id:
            テストID（省略時は自動生成）

        Returns
        -------
        str
            テストID
        """
        if test_id is None:
            self._test_counter += 1
            test_id = f"ab_{self._test_counter}"

        self._tests[test_id] = _TestState(
            test_id=test_id,
            champion_id=champion_id,
            challenger_id=challenger_id,
        )
        logger.info(
            "A/Bテスト開始: %s (champion=%s, challenger=%s)",
            test_id, champion_id, challenger_id,
        )
        return test_id

    def update(
        self,
        test_id: str,
        champion_pnl: float,
        challenger_pnl: float,
    ) -> None:
        """テストにサンプルを追加する。

        Parameters
        ----------
        test_id:
            テストID
        champion_pnl:
            チャンピオンのPnL
        challenger_pnl:
            チャレンジャーのPnL

        Raises
        ------
        KeyError
            テストIDが存在しない場合
        ValueError
            テストが既に終了している場合
        """
        state = self._tests.get(test_id)
        if state is None:
            raise KeyError(f"テスト '{test_id}' が見つかりません")
        if state.concluded:
            raise ValueError(f"テスト '{test_id}' は既に終了しています")

        state.champion_pnls.append(champion_pnl)
        state.challenger_pnls.append(challenger_pnl)

    def evaluate(self, test_id: str) -> ABTestResult:
        """テスト結果を評価する。

        Parameters
        ----------
        test_id:
            テストID

        Returns
        -------
        ABTestResult
            テスト結果

        Raises
        ------
        KeyError
            テストIDが存在しない場合
        """
        state = self._tests.get(test_id)
        if state is None:
            raise KeyError(f"テスト '{test_id}' が見つかりません")

        if state.result is not None:
            return state.result

        n = min(len(state.champion_pnls), len(state.challenger_pnls))

        if n < self._min_samples:
            return ABTestResult(
                test_id=test_id,
                champion_id=state.champion_id,
                challenger_id=state.challenger_id,
                champion_mean=_safe_mean(state.champion_pnls),
                challenger_mean=_safe_mean(state.challenger_pnls),
                p_value=1.0,
                n_samples=n,
                winner="inconclusive",
            )

        # Welch's t-test
        p_value = _welch_t_test(state.champion_pnls[:n], state.challenger_pnls[:n])

        champion_mean = _safe_mean(state.champion_pnls[:n])
        challenger_mean = _safe_mean(state.challenger_pnls[:n])

        if p_value < self._significance_level:
            winner = "challenger" if challenger_mean > champion_mean else "champion"
        else:
            winner = "inconclusive"

        result = ABTestResult(
            test_id=test_id,
            champion_id=state.champion_id,
            challenger_id=state.challenger_id,
            champion_mean=champion_mean,
            challenger_mean=challenger_mean,
            p_value=p_value,
            n_samples=n,
            winner=winner,
        )
        return result

    def conclude(self, test_id: str) -> ABTestResult:
        """テストを終了し、結果を確定する。"""
        result = self.evaluate(test_id)
        state = self._tests[test_id]
        state.concluded = True
        state.result = result
        logger.info("A/Bテスト終了: %s → winner=%s (p=%.4f)", test_id, result.winner, result.p_value)
        return result

    def get_active_tests(self) -> List[str]:
        """実行中のテストID一覧を返す。"""
        return [tid for tid, s in self._tests.items() if not s.concluded]

    @property
    def test_count(self) -> int:
        """テスト総数。"""
        return len(self._tests)


def _safe_mean(values: List[float]) -> float:
    """安全な平均計算。空リストの場合は0.0を返す。"""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _welch_t_test(a: List[float], b: List[float]) -> float:
    """Welch's t-test の p 値を計算する（外部依存なし）。

    scipy がなくても動作するよう、t分布の近似を使用する。
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 1.0

    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    se = (var_a / n_a + var_b / n_b) ** 0.5
    if se == 0:
        return 1.0

    t_stat = abs(mean_a - mean_b) / se

    # Welch-Satterthwaite の自由度近似
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    if denom == 0:
        return 1.0
    df = numerator / denom

    # t分布CDF の近似 (Abramowitz & Stegun)
    p_value = _t_cdf_approx(t_stat, df)
    return 2.0 * (1.0 - p_value)  # 両側検定


def _t_cdf_approx(t: float, df: float) -> float:
    """t分布CDFの近似。正確さは ~0.01 程度。"""
    x = df / (df + t * t)
    # 正則化不完全ベータ関数の近似
    # 大きな df では正規分布に収束
    if df > 100:
        from math import erfc
        return 1.0 - 0.5 * erfc(t / math.sqrt(2))

    # 小さな df ではシンプルな近似
    a = df / 2.0
    b = 0.5
    # Incomplete beta の簡易近似
    if t <= 0:
        return 0.5
    p = 1.0 - 0.5 * x ** a
    return max(0.5, min(1.0, p))
