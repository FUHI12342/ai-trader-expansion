"""Purged Walk-Forward Cross-Validation。

金融時系列データ用のクロスバリデーション。
時系列の自己相関を考慮し、IS/OOS間に embargo 期間を設けて
情報漏洩を防止する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class CVFold:
    """CV fold の情報（immutable）。"""

    fold_index: int
    train_start: int      # iloc index
    train_end: int
    test_start: int
    test_end: int
    embargo_days: int


class PurgedWalkForwardCV:
    """Purged Walk-Forward Cross-Validation。

    Walk-Forward 方式で train/test を分割し、
    train の末尾と test の先頭の間に embargo 期間を設けて
    自己相関による情報漏洩を防ぐ。

    Parameters
    ----------
    n_splits:
        fold 数（最低 3）
    train_ratio:
        データ全体に対する最初の train の割合（デフォルト: 0.6）
    embargo_pct:
        train サイズに対する embargo 割合（デフォルト: 0.01 = 1%）
    expanding:
        True の場合は train ウィンドウを拡張する（False はローリング）
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        embargo_pct: float = 0.01,
        expanding: bool = True,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits は 2 以上が必要です。")
        if not 0.1 <= train_ratio <= 0.9:
            raise ValueError("train_ratio は 0.1〜0.9 の範囲で指定してください。")
        self._n_splits = n_splits
        self._train_ratio = train_ratio
        self._embargo_pct = embargo_pct
        self._expanding = expanding

    @property
    def n_splits(self) -> int:
        return self._n_splits

    def split(self, data: pd.DataFrame) -> List[CVFold]:
        """データを train/test fold に分割する。

        Parameters
        ----------
        data:
            分割対象の DataFrame (行数でインデックス)

        Returns
        -------
        List[CVFold]
            各 fold の情報
        """
        n = len(data)
        initial_train_size = int(n * self._train_ratio)
        remaining = n - initial_train_size
        test_size = remaining // self._n_splits

        if test_size < 1:
            raise ValueError(
                f"データが不足しています。n={n}, train={initial_train_size}, "
                f"test_size={test_size}"
            )

        folds: List[CVFold] = []

        for i in range(self._n_splits):
            if self._expanding:
                train_start = 0
            else:
                train_start = i * test_size

            train_end = initial_train_size + i * test_size
            embargo_days = max(1, int((train_end - train_start) * self._embargo_pct))

            test_start = train_end + embargo_days
            test_end = min(test_start + test_size, n)

            if test_start >= n or test_end <= test_start:
                break

            folds.append(CVFold(
                fold_index=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=embargo_days,
            ))

        return folds

    def get_train_test_pairs(
        self, data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """DataFrame の train/test ペアを返す。

        Parameters
        ----------
        data:
            分割対象の DataFrame

        Returns
        -------
        List[Tuple[pd.DataFrame, pd.DataFrame]]
            (train_df, test_df) のリスト
        """
        folds = self.split(data)
        pairs: List[Tuple[pd.DataFrame, pd.DataFrame]] = []

        for fold in folds:
            train = data.iloc[fold.train_start:fold.train_end].copy()
            test = data.iloc[fold.test_start:fold.test_end].copy()
            pairs.append((train, test))

        return pairs
