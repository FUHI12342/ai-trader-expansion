"""DeFi モジュールのテスト (AaveSimulator + WaitingCapitalManager)。"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.defi import (
    AavePosition,
    AaveSimulator,
    DepositResult,
    RebalanceAction,
    WaitingCapitalManager,
    WithdrawResult,
)
from src.defi.waiting_capital_manager import RebalanceActionType


# ---------------------------------------------------------------------------
# AaveSimulator
# ---------------------------------------------------------------------------

def test_aave_init_default():
    """デフォルト初期化で残高0、APY 5%。"""
    aave = AaveSimulator()
    assert aave.apy == 0.05
    assert aave.balance == 0.0


def test_aave_invalid_apy_raises():
    """範囲外APYで ValueError。"""
    with pytest.raises(ValueError):
        AaveSimulator(apy=-0.01)
    with pytest.raises(ValueError):
        AaveSimulator(apy=0.6)


def test_deposit_success():
    """通常の預入が成功する。"""
    aave = AaveSimulator(apy=0.05)
    result = aave.deposit(1000.0)
    assert result.success is True
    assert result.amount == 1000.0
    assert aave.balance == 1000.0


def test_deposit_below_min_fails():
    """最小預入額未満は失敗する。"""
    aave = AaveSimulator(apy=0.05, min_deposit=10.0)
    result = aave.deposit(5.0)
    assert result.success is False
    assert "最小" in result.reason
    assert aave.balance == 0.0


def test_deposit_is_frozen():
    """DepositResult は immutable。"""
    aave = AaveSimulator()
    result = aave.deposit(1000.0)
    with pytest.raises(Exception):
        result.amount = 2000.0  # type: ignore[misc]


def test_withdraw_success():
    """通常の引出が成功する。"""
    aave = AaveSimulator(apy=0.05)
    aave.deposit(1000.0)
    result = aave.withdraw(500.0)
    assert result.success is True
    assert result.amount == 500.0
    assert aave.balance == pytest.approx(500.0, abs=0.01)


def test_withdraw_insufficient_fails():
    """残高不足で失敗する。"""
    aave = AaveSimulator(apy=0.05)
    aave.deposit(100.0)
    result = aave.withdraw(500.0)
    assert result.success is False
    assert "残高不足" in result.reason


def test_withdraw_zero_fails():
    """0引出は失敗する。"""
    aave = AaveSimulator()
    aave.deposit(100.0)
    result = aave.withdraw(0.0)
    assert result.success is False


def test_accrue_interest_daily_compound():
    """1年経過で 5% 複利が累積する。"""
    t0 = datetime(2026, 1, 1)
    t1 = datetime(2027, 1, 1)
    aave = AaveSimulator(apy=0.05, compound_daily=True)
    aave._last_accrual = t0
    aave.deposit(1000.0, now=t0)
    aave.accrue_interest(now=t1)
    # 1年後: 1000 * 1.05 = 1050
    assert aave.balance == pytest.approx(1050.0, rel=0.001)


def test_accrue_interest_simple():
    """単利モードで 1年 5%。"""
    t0 = datetime(2026, 1, 1)
    t1 = datetime(2027, 1, 1)
    aave = AaveSimulator(apy=0.05, compound_daily=False)
    aave._last_accrual = t0
    aave.deposit(1000.0, now=t0)
    aave.accrue_interest(now=t1)
    assert aave.balance == pytest.approx(1050.0, rel=0.001)


def test_accrue_no_time_elapsed():
    """時間経過0なら利息0。"""
    t0 = datetime(2026, 1, 1)
    aave = AaveSimulator(apy=0.05)
    aave._last_accrual = t0
    aave.deposit(1000.0, now=t0)
    accrued = aave.accrue_interest(now=t0)
    assert accrued == 0.0


def test_snapshot_returns_aave_position():
    """snapshot が AavePosition を返す。"""
    aave = AaveSimulator(apy=0.05)
    aave.deposit(1000.0)
    snap = aave.snapshot()
    assert isinstance(snap, AavePosition)
    assert snap.principal == pytest.approx(1000.0, abs=0.01)
    assert snap.apy == 0.05


def test_reset_clears_state():
    """reset で状態がクリアされる。"""
    aave = AaveSimulator(apy=0.05)
    aave.deposit(1000.0)
    aave.reset()
    assert aave.balance == 0.0


# ---------------------------------------------------------------------------
# WaitingCapitalManager
# ---------------------------------------------------------------------------

def test_manager_invalid_thresholds():
    """deposit 閾値 <= withdraw 閾値は ValueError。"""
    aave = AaveSimulator()
    with pytest.raises(ValueError):
        WaitingCapitalManager(aave, flat_threshold_deposit=0.2, flat_threshold_withdraw=0.3)


def test_manager_invalid_buffer():
    """buffer_pct 範囲外は ValueError。"""
    aave = AaveSimulator()
    with pytest.raises(ValueError):
        WaitingCapitalManager(aave, buffer_pct=1.5)
    with pytest.raises(ValueError):
        WaitingCapitalManager(aave, buffer_pct=-0.1)


def test_decide_deposit_when_flat_high():
    """FLAT比率が高いと預入アクション。"""
    aave = AaveSimulator(apy=0.05)
    mgr = WaitingCapitalManager(
        aave, flat_threshold_deposit=0.5, buffer_pct=0.1, min_rebalance_amount=10.0
    )
    # cash 180,000 / total 200,000 = 90% FLAT
    action = mgr.decide(cash_balance=180000.0, active_position_value=20000.0, total_capital=200000.0)
    assert action.action == RebalanceActionType.DEPOSIT
    # buffer = 200000 * 0.1 = 20000, deposit = 180000 - 20000 = 160000
    assert action.amount == 160000.0
    assert action.flat_ratio == 0.9


def test_decide_withdraw_when_flat_low():
    """FLAT比率が低く Aave残高あるなら引出アクション。"""
    aave = AaveSimulator(apy=0.05)
    aave.deposit(50000.0)
    mgr = WaitingCapitalManager(
        aave, flat_threshold_withdraw=0.2, min_rebalance_amount=10.0
    )
    # cash 10,000 / total 200,000 = 5% FLAT (<20%)
    action = mgr.decide(cash_balance=10000.0, active_position_value=190000.0, total_capital=200000.0)
    assert action.action == RebalanceActionType.WITHDRAW
    # target = 200,000 * 0.2 = 40,000, shortfall = 30,000 < aave 50,000+利息
    assert action.amount > 0


def test_decide_hold_when_in_range():
    """閾値内ならHOLD。"""
    aave = AaveSimulator(apy=0.05)
    mgr = WaitingCapitalManager(
        aave, flat_threshold_deposit=0.5, flat_threshold_withdraw=0.2,
    )
    # cash 60,000 / total 200,000 = 30% FLAT (範囲内)
    action = mgr.decide(cash_balance=60000.0, active_position_value=140000.0, total_capital=200000.0)
    assert action.action == RebalanceActionType.HOLD


def test_decide_zero_total_capital():
    """total_capital 0 はHOLD。"""
    aave = AaveSimulator(apy=0.05)
    mgr = WaitingCapitalManager(aave)
    action = mgr.decide(cash_balance=0.0, active_position_value=0.0, total_capital=0.0)
    assert action.action == RebalanceActionType.HOLD


def test_decide_deposit_skip_small_amount():
    """預入額が最小未満ならHOLD。"""
    aave = AaveSimulator(apy=0.05)
    mgr = WaitingCapitalManager(aave, min_rebalance_amount=1000.0, buffer_pct=0.9)
    # total 100, cash 99 → deposit候補 = 99 - 90 = 9 < 1000
    action = mgr.decide(cash_balance=99.0, active_position_value=1.0, total_capital=100.0)
    assert action.action == RebalanceActionType.HOLD


def test_decide_withdraw_skip_no_aave_balance():
    """Aave残高なしの場合、引出条件でもHOLD。"""
    aave = AaveSimulator(apy=0.05)  # 残高ゼロ
    mgr = WaitingCapitalManager(aave, flat_threshold_withdraw=0.2, min_rebalance_amount=10.0)
    action = mgr.decide(cash_balance=10000.0, active_position_value=190000.0, total_capital=200000.0)
    assert action.action == RebalanceActionType.HOLD
    assert "Aave残高" in action.reason


def test_expected_annual_yield():
    """期待年間利回り計算が正しい。"""
    aave = AaveSimulator(apy=0.05)
    mgr = WaitingCapitalManager(aave, buffer_pct=0.1)
    # 総資金20万、平均FLAT 80% → 待機16万、buffer 2万、運用14万 × 5% = 7,000
    yield_amount = mgr.expected_annual_yield(avg_flat_ratio=0.8, total_capital=200000.0)
    assert yield_amount == pytest.approx(7000.0, abs=1.0)


def test_rebalance_action_is_frozen():
    """RebalanceAction は immutable。"""
    aave = AaveSimulator()
    mgr = WaitingCapitalManager(aave)
    action = mgr.decide(cash_balance=100.0, active_position_value=100.0, total_capital=200.0)
    with pytest.raises(Exception):
        action.amount = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 統合シナリオ: 1年運用シミュレーション
# ---------------------------------------------------------------------------

def test_one_year_waiting_capital_yield():
    """20万円、平均FLAT 80%、buffer 10% で年 ~7000円 の利息が得られる。"""
    aave = AaveSimulator(apy=0.05, compound_daily=True)
    mgr = WaitingCapitalManager(aave, buffer_pct=0.1, min_rebalance_amount=10.0)

    t0 = datetime(2026, 1, 1)
    aave._last_accrual = t0
    # 初期: 総資金20万、全額現金、FLAT比率100%
    action = mgr.decide(cash_balance=200000.0, active_position_value=0.0, total_capital=200000.0)
    assert action.action == RebalanceActionType.DEPOSIT
    aave.deposit(action.amount, now=t0)

    # 1年経過
    t1 = t0 + timedelta(days=365)
    snap = aave.snapshot(now=t1)
    # 180,000 * 5% ≈ 9,000 (複利なのでわずかに多め)
    assert snap.accrued_interest > 8500.0
    assert snap.accrued_interest < 10000.0
