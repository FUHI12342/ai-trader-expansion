"""Phase 7: 通知ルーター + ヘルスチェックのテスト。"""
from __future__ import annotations

import dataclasses

import pytest

from src.notifications.router import (
    NotificationRouter, NotificationChannel, TradingEvent,
    EventType, LogChannel,
)


# ---------------------------------------------------------------------------
# テスト用モックチャネル
# ---------------------------------------------------------------------------

class MockChannel(NotificationChannel):
    name = "mock"

    def __init__(self, should_fail: bool = False) -> None:
        self.sent: list[TradingEvent] = []
        self._should_fail = should_fail

    def send(self, event: TradingEvent) -> bool:
        if self._should_fail:
            return False
        self.sent.append(event)
        return True


class UnavailableChannel(NotificationChannel):
    name = "unavailable"

    def send(self, event: TradingEvent) -> bool:
        return True

    def is_available(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# TradingEvent テスト
# ---------------------------------------------------------------------------

class TestTradingEvent:

    def test_is_frozen(self) -> None:
        ev = TradingEvent(
            event_type=EventType.ORDER_FILLED,
            title="Test", message="msg", data={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ev.title = "changed"  # type: ignore[misc]

    def test_timestamp_auto_set(self) -> None:
        ev = TradingEvent(
            event_type=EventType.CUSTOM,
            title="t", message="m", data={},
        )
        assert ev.timestamp != ""

    def test_custom_timestamp(self) -> None:
        ev = TradingEvent(
            event_type=EventType.CUSTOM,
            title="t", message="m", data={},
            timestamp="2026-01-01T00:00:00",
        )
        assert ev.timestamp == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------------
# LogChannel テスト
# ---------------------------------------------------------------------------

class TestLogChannel:

    def test_send_returns_true(self) -> None:
        ch = LogChannel()
        ev = TradingEvent(
            event_type=EventType.DAILY_SUMMARY,
            title="日次サマリー", message="PnL: +10,000", data={},
        )
        assert ch.send(ev) is True

    def test_is_available(self) -> None:
        assert LogChannel().is_available()


# ---------------------------------------------------------------------------
# NotificationRouter テスト
# ---------------------------------------------------------------------------

class TestNotificationRouter:

    def test_init_empty(self) -> None:
        router = NotificationRouter()
        assert router.channel_count == 0
        assert router.sent_count == 0

    def test_add_channel(self) -> None:
        router = NotificationRouter()
        router.add_channel(MockChannel())
        assert router.channel_count == 1

    def test_send_to_single_channel(self) -> None:
        ch = MockChannel()
        router = NotificationRouter([ch])
        ev = TradingEvent(
            event_type=EventType.ORDER_FILLED,
            title="約定", message="BTC 1.0", data={"price": 42000},
        )
        count = router.send(ev)

        assert count == 1
        assert len(ch.sent) == 1
        assert ch.sent[0].title == "約定"
        assert router.sent_count == 1

    def test_send_to_multiple_channels(self) -> None:
        ch1 = MockChannel()
        ch2 = MockChannel()
        router = NotificationRouter([ch1, ch2])
        ev = TradingEvent(
            event_type=EventType.CIRCUIT_BREAKER,
            title="CB発動", message="停止", data={},
        )
        count = router.send(ev)

        assert count == 2
        assert len(ch1.sent) == 1
        assert len(ch2.sent) == 1

    def test_send_with_failed_channel(self) -> None:
        ch_ok = MockChannel()
        ch_fail = MockChannel(should_fail=True)
        router = NotificationRouter([ch_ok, ch_fail])
        ev = TradingEvent(
            event_type=EventType.CUSTOM,
            title="t", message="m", data={},
        )
        count = router.send(ev)

        assert count == 1
        assert router.error_count == 1

    def test_send_skips_unavailable(self) -> None:
        ch_ok = MockChannel()
        ch_down = UnavailableChannel()
        router = NotificationRouter([ch_ok, ch_down])
        ev = TradingEvent(
            event_type=EventType.CUSTOM,
            title="t", message="m", data={},
        )
        count = router.send(ev)

        assert count == 1  # ch_ok のみ

    def test_event_filter(self) -> None:
        ch = MockChannel()
        router = NotificationRouter()
        router.add_channel(ch, event_types=[EventType.ORDER_FILLED])

        # フィルタに含まれるイベント → 送信される
        ev1 = TradingEvent(
            event_type=EventType.ORDER_FILLED,
            title="約定", message="ok", data={},
        )
        router.send(ev1)
        assert len(ch.sent) == 1

        # フィルタに含まれないイベント → スキップ
        ev2 = TradingEvent(
            event_type=EventType.DAILY_SUMMARY,
            title="日次", message="ok", data={},
        )
        router.send(ev2)
        assert len(ch.sent) == 1  # 変わらず

    def test_send_no_channels(self) -> None:
        router = NotificationRouter()
        ev = TradingEvent(
            event_type=EventType.CUSTOM,
            title="t", message="m", data={},
        )
        count = router.send(ev)
        assert count == 0

    def test_init_with_channels(self) -> None:
        router = NotificationRouter([MockChannel(), LogChannel()])
        assert router.channel_count == 2
