"""SelfImprover のテスト。"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.learning.self_improver import SelfImprover, PerformanceRecord


class TestSelfImprover:

    def _make_improver(self, tmp_path: Path) -> SelfImprover:
        return SelfImprover(data_dir=str(tmp_path / "improve"), optimize_interval_days=0)

    def test_init(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        assert si.total_records == 0
        assert len(si.candidates) == 0

    def test_record_performance(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        r = si.record_performance("bollinger", "SPY", 1, 5000.0)
        assert r.pnl == 5000.0
        assert r.win_count == 1
        assert si.total_records == 1

    def test_record_multiple(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        si.record_performance("bollinger", "SPY", 1, 5000.0)
        si.record_performance("bollinger", "SPY", -1, -3000.0)
        si.record_performance("bollinger", "SPY", 1, 2000.0)

        sb = si.get_scoreboard()
        assert "bollinger/SPY" in sb
        assert sb["bollinger/SPY"]["wins"] == 2
        assert sb["bollinger/SPY"]["losses"] == 1
        assert sb["bollinger/SPY"]["cumulative_pnl"] == 4000.0

    def test_scoreboard_win_rate(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        for _ in range(3):
            si.record_performance("test", "SPY", 1, 1000.0)
        si.record_performance("test", "SPY", -1, -500.0)

        sb = si.get_scoreboard()
        assert sb["test/SPY"]["win_rate"] == 75.0

    def test_get_best_strategy(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        for _ in range(10):
            si.record_performance("good", "SPY", 1, 1000.0)
        for _ in range(10):
            si.record_performance("bad", "SPY", -1, -1000.0)

        best = si.get_best_strategy()
        assert best == "good/SPY"

    def test_get_best_strategy_empty(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        assert si.get_best_strategy() is None

    def test_improvement_summary(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        s = si.get_improvement_summary()
        assert s["total_candidates"] == 0
        assert s["total_performance_records"] == 0

    def test_collect_market_data(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        mock_dm = MagicMock()
        mock_dm.fetch_ohlcv.return_value = pd.DataFrame({
            "open": [100], "high": [101], "low": [99],
            "close": [100], "volume": [1000],
        })
        result = si.collect_market_data(mock_dm, ["SPY"])
        assert "SPY" in result

    def test_collect_skips_same_day(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        mock_dm = MagicMock()
        mock_dm.fetch_ohlcv.return_value = pd.DataFrame({
            "open": [100], "high": [101], "low": [99],
            "close": [100], "volume": [1000],
        })
        si.collect_market_data(mock_dm, ["SPY"])
        result2 = si.collect_market_data(mock_dm, ["SPY"])
        assert result2 == {}  # 同日2回目はスキップ

    def test_state_persistence(self, tmp_path: Path) -> None:
        si = self._make_improver(tmp_path)
        si.record_performance("test", "SPY", 1, 1000.0)
        si._save_state()

        si2 = SelfImprover(data_dir=str(tmp_path / "improve"))
        sb = si2.get_scoreboard()
        assert "test/SPY" in sb

    def test_run_cycle(self, tmp_path: Path) -> None:
        si = SelfImprover(
            data_dir=str(tmp_path / "improve"),
            optimize_interval_days=0,
            min_data_days=5,
        )
        mock_dm = MagicMock()
        mock_dm.fetch_ohlcv.return_value = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )
        result = si.run_cycle(dm=mock_dm, strategies={}, symbols=["SPY"])
        assert "timestamp" in result
        assert "summary" in result
