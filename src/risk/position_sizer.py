"""ポジションサイジング — Kelly Criterion + VIX連動。

最適な1取引あたりのポジションサイズを計算する。
Full Kelly は DD が大きすぎるため、Half/Quarter Kelly を使用。

安全装置:
  - 日次損失制限 (daily_loss_limit)
  - 最大ポジションサイズ制限 (max_position_pct)
  - VIX連動でボラ高時にポジション縮小
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizingResult:
    """ポジションサイジング結果（immutable）。

    Parameters
    ----------
    position_pct:
        推奨ポジションサイズ（資金に対する割合 0.0〜1.0）
    position_amount:
        推奨ポジション金額
    kelly_raw:
        Full Kelly の計算値
    kelly_fraction:
        使用した Kelly 倍率
    reason:
        サイジング根拠
    blocked:
        取引がブロックされた場合 True
    block_reason:
        ブロック理由
    """

    position_pct: float
    position_amount: float
    kelly_raw: float
    kelly_fraction: float
    reason: str
    blocked: bool = False
    block_reason: str = ""


class PositionSizer:
    """Kelly Criterion ベースのポジションサイザー。

    Parameters
    ----------
    kelly_fraction:
        Kelly 倍率 (0.5 = Half Kelly, 0.25 = Quarter Kelly)
    max_position_pct:
        最大ポジションサイズ (資金比率、デフォルト: 0.3 = 30%)
    daily_loss_limit_pct:
        日次損失制限 (資金比率、デフォルト: 0.02 = 2%)
    min_position_pct:
        最小ポジションサイズ (これ以下はスキップ、デフォルト: 0.01 = 1%)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        max_position_pct: float = 0.30,
        daily_loss_limit_pct: float = 0.02,
        min_position_pct: float = 0.01,
    ) -> None:
        self._kelly_fraction = kelly_fraction
        self._max_position_pct = max_position_pct
        self._daily_loss_limit_pct = daily_loss_limit_pct
        self._min_position_pct = min_position_pct
        self._daily_pnl: float = 0.0
        self._daily_trade_count: int = 0
        self._current_date: str = ""

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def daily_loss_remaining(self) -> float:
        """日次損失余力（正の値 = まだ余裕あり）。"""
        return self._daily_loss_limit_pct + (self._daily_pnl if self._daily_pnl < 0 else 0)

    def reset_daily(self, date: str = "") -> None:
        """日次カウンターをリセットする。"""
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._current_date = date

    def record_trade_pnl(self, pnl_pct: float) -> None:
        """取引結果を記録する（日次損失計算用）。"""
        self._daily_pnl += pnl_pct
        self._daily_trade_count += 1

    def calculate(
        self,
        capital: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        vix: Optional[float] = None,
    ) -> SizingResult:
        """最適ポジションサイズを計算する。

        Parameters
        ----------
        capital:
            現在の資金
        win_rate:
            勝率 (0.0〜1.0)
        avg_win:
            平均勝ち金額 (正の値)
        avg_loss:
            平均負け金額 (正の値)
        vix:
            現在のVIX値（省略時はVIX調整なし）

        Returns
        -------
        SizingResult
            ポジションサイジング結果
        """
        # 日次損失制限チェック
        if self._daily_pnl < -self._daily_loss_limit_pct:
            return SizingResult(
                position_pct=0.0, position_amount=0.0,
                kelly_raw=0.0, kelly_fraction=self._kelly_fraction,
                reason="日次損失制限超過",
                blocked=True,
                block_reason=f"日次損失 {self._daily_pnl:.2%} > 制限 {self._daily_loss_limit_pct:.2%}",
            )

        # Kelly 計算
        kelly_raw = self._kelly(win_rate, avg_win, avg_loss)

        if kelly_raw <= 0:
            return SizingResult(
                position_pct=0.0, position_amount=0.0,
                kelly_raw=kelly_raw, kelly_fraction=self._kelly_fraction,
                reason="Kelly <= 0 (エッジなし)",
                blocked=True,
                block_reason="期待値が負のため取引しない",
            )

        # Fractional Kelly
        position_pct = kelly_raw * self._kelly_fraction

        # VIX 調整
        vix_multiplier = 1.0
        if vix is not None:
            vix_multiplier = self._vix_adjustment(vix)
            position_pct *= vix_multiplier

        # 上限・下限
        position_pct = min(position_pct, self._max_position_pct)

        if position_pct < self._min_position_pct:
            return SizingResult(
                position_pct=0.0, position_amount=0.0,
                kelly_raw=kelly_raw, kelly_fraction=self._kelly_fraction,
                reason=f"ポジション {position_pct:.2%} < 最小 {self._min_position_pct:.2%}",
                blocked=True,
                block_reason="ポジションが小さすぎる",
            )

        position_amount = capital * position_pct

        reason_parts = [f"Kelly={kelly_raw:.3f}×{self._kelly_fraction}={kelly_raw*self._kelly_fraction:.3f}"]
        if vix is not None:
            reason_parts.append(f"VIX={vix:.1f}(×{vix_multiplier:.2f})")
        reason_parts.append(f"→{position_pct:.2%}")

        return SizingResult(
            position_pct=position_pct,
            position_amount=round(position_amount, 2),
            kelly_raw=kelly_raw,
            kelly_fraction=self._kelly_fraction,
            reason=" ".join(reason_parts),
        )

    def _kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Kelly Criterion: K% = W - (1-W)/R"""
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        r = avg_win / avg_loss
        return win_rate - (1 - win_rate) / r

    def _vix_adjustment(self, vix: float) -> float:
        """VIX連動のポジション調整倍率。"""
        if vix < 15:
            return 1.0
        elif vix < 25:
            return 0.7
        elif vix < 35:
            return 0.4
        else:
            return 0.1
