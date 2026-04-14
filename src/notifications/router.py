"""通知ルーター — 通知先を一元管理する。

複数の通知チャネル（LINE, Slack, Discord, ログ等）に
取引イベントを並列配信する。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """通知イベント種別。"""
    ORDER_FILLED = "order_filled"
    ORDER_FAILED = "order_failed"
    CIRCUIT_BREAKER = "circuit_breaker"
    DRIFT_DETECTED = "drift_detected"
    STAGE_CHANGE = "stage_change"
    DAILY_SUMMARY = "daily_summary"
    DRAWDOWN_ALERT = "drawdown_alert"
    CORRELATION_ALERT = "correlation_alert"
    HEALTH_CHECK = "health_check"
    CUSTOM = "custom"


@dataclass(frozen=True)
class TradingEvent:
    """取引イベント（immutable）。

    Parameters
    ----------
    event_type:
        イベント種別
    title:
        イベントタイトル
    message:
        イベント本文
    data:
        追加データ
    timestamp:
        イベント発生時刻（ISO format）
    severity:
        深刻度（"info", "warning", "critical"）
    """

    event_type: EventType
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: str = ""
    severity: str = "info"

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(
                self, "timestamp",
                datetime.now().isoformat(timespec="seconds"),
            )


class NotificationChannel(ABC):
    """通知チャネルの基底クラス。"""

    name: str = "base"

    @abstractmethod
    def send(self, event: TradingEvent) -> bool:
        """イベントを送信する。

        Returns
        -------
        bool
            送信成功の場合True
        """
        ...

    def is_available(self) -> bool:
        """チャネルが利用可能か。デフォルト: True。"""
        return True


class LogChannel(NotificationChannel):
    """ログ出力チャネル（常に利用可能）。"""

    name = "log"

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    def send(self, event: TradingEvent) -> bool:
        log_msg = f"[{event.event_type.value}] {event.title}: {event.message}"
        logger.log(self._log_level, log_msg)
        return True


class SlackChannel(NotificationChannel):
    """Slack Incoming Webhook チャネル。"""

    name = "slack"

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    def send(self, event: TradingEvent) -> bool:
        try:
            import urllib.request
            import json

            payload = {
                "text": f"*[{event.severity.upper()}] {event.title}*\n{event.message}",
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error("Slack送信失敗: %s", e)
            return False


class DiscordChannel(NotificationChannel):
    """Discord Webhook チャネル。"""

    name = "discord"

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    def send(self, event: TradingEvent) -> bool:
        try:
            import urllib.request
            import json

            payload = {
                "content": f"**[{event.severity.upper()}] {event.title}**\n{event.message}",
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self._webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 204
        except Exception as e:
            logger.error("Discord送信失敗: %s", e)
            return False


class NotificationRouter:
    """通知ルーター。

    複数のチャネルにイベントを配信する。
    チャネルごとにフィルタ（対象イベント種別）を設定可能。

    Parameters
    ----------
    channels:
        通知チャネルのリスト
    """

    def __init__(self, channels: Optional[List[NotificationChannel]] = None) -> None:
        self._channels: List[NotificationChannel] = channels or []
        self._event_filters: Dict[str, List[EventType]] = {}
        self._sent_count: int = 0
        self._error_count: int = 0

    def add_channel(
        self,
        channel: NotificationChannel,
        event_types: Optional[List[EventType]] = None,
    ) -> None:
        """チャネルを追加する。

        Parameters
        ----------
        channel:
            通知チャネル
        event_types:
            対象イベント種別（Noneの場合は全イベント）
        """
        self._channels.append(channel)
        if event_types:
            self._event_filters[channel.name] = event_types

    def send(self, event: TradingEvent) -> int:
        """全チャネルにイベントを配信する。

        Returns
        -------
        int
            送信成功したチャネル数
        """
        success_count = 0

        for channel in self._channels:
            # フィルタチェック
            allowed = self._event_filters.get(channel.name)
            if allowed and event.event_type not in allowed:
                continue

            if not channel.is_available():
                continue

            try:
                if channel.send(event):
                    success_count += 1
                    self._sent_count += 1
                else:
                    self._error_count += 1
            except Exception as e:
                self._error_count += 1
                logger.error("チャネル %s 送信失敗: %s", channel.name, e)

        return success_count

    @property
    def sent_count(self) -> int:
        """送信成功回数。"""
        return self._sent_count

    @property
    def error_count(self) -> int:
        """送信失敗回数。"""
        return self._error_count

    @property
    def channel_count(self) -> int:
        """登録チャネル数。"""
        return len(self._channels)
