"""AI Trader ウォッチドッグ — プロセス監視 + 自動再起動。

auto_trader.py の稼働を監視し、クラッシュ時に自動再起動する。
メモリ使用量監視、ログローテーション、ヘルスチェックを含む。

使用例:
  python scripts/watchdog.py --config config/paper_20man.json
  python scripts/watchdog.py --check-interval 30 --max-restarts 10

NSSMでサービス化:
  nssm install AITraderWatchdog "C:/Python312/python.exe" "C:/_dev/repos/ai-trader-expansion/scripts/watchdog.py --config config/paper_20man.json"
  nssm set AITraderWatchdog AppDirectory "C:/_dev/repos/ai-trader-expansion"
  nssm start AITraderWatchdog
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

DEFAULT_CHECK_INTERVAL = 30  # 秒
DEFAULT_MAX_MEMORY_MB = 1024
DEFAULT_MAX_RESTARTS = 50
DEFAULT_LOG_MAX_MB = 100


class Watchdog:
    """auto_trader.py のプロセスを監視・再起動する。

    Parameters
    ----------
    config_path:
        auto_trader.py に渡す設定ファイルパス
    check_interval:
        ヘルスチェック間隔（秒）
    max_memory_mb:
        メモリ上限（MB、超過で再起動）
    max_restarts:
        最大再起動回数（超過で停止）
    log_path:
        auto_trader.py のログ出力先
    """

    def __init__(
        self,
        config_path: str = "config/paper_20man.json",
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        max_memory_mb: int = DEFAULT_MAX_MEMORY_MB,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        log_path: str = "logs/auto_trader.log",
    ) -> None:
        self._config_path = config_path
        self._check_interval = check_interval
        self._max_memory_mb = max_memory_mb
        self._max_restarts = max_restarts
        self._log_path = log_path
        self._process: Optional[subprocess.Popen] = None
        self._restart_count = 0
        self._running = False
        self._start_time = time.time()

    @property
    def restart_count(self) -> int:
        return self._restart_count

    @property
    def is_trader_alive(self) -> bool:
        """auto_trader プロセスが生きているか。"""
        if self._process is None:
            return False
        return self._process.poll() is None

    def start(self) -> None:
        """ウォッチドッグを開始する。"""
        self._running = True
        logger.info("Watchdog 開始 (config=%s, interval=%ds)", self._config_path, self._check_interval)

        self._start_trader()

        while self._running:
            try:
                self._check()
            except Exception as e:
                logger.error("Watchdog チェックエラー: %s", e)

            time.sleep(self._check_interval)

    def stop(self) -> None:
        """ウォッチドッグを停止し、子プロセスも停止する。"""
        self._running = False
        if self._process and self._process.poll() is None:
            logger.info("auto_trader プロセス停止中 (PID=%d)", self._process.pid)
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def _start_trader(self) -> None:
        """auto_trader.py を子プロセスとして起動する。"""
        # ログディレクトリ確保
        log_dir = Path(self._log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # ログローテーション
        self._rotate_log()

        cmd = [
            sys.executable,
            "scripts/auto_trader.py",
            "--mode", "auto",
            "--config", self._config_path,
        ]

        log_file = open(self._log_path, "a", encoding="utf-8")
        self._process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        logger.info("auto_trader 起動 (PID=%d, restart=%d)", self._process.pid, self._restart_count)

    def _check(self) -> None:
        """ヘルスチェックを実行する。"""
        # 1. プロセス生存確認
        if not self.is_trader_alive:
            exit_code = self._process.returncode if self._process else -1
            logger.warning("auto_trader 停止検出 (exit=%s)", exit_code)
            self._restart()
            return

        # 2. メモリ使用量チェック
        try:
            import psutil
            proc = psutil.Process(self._process.pid)
            rss_mb = proc.memory_info().rss / (1024 * 1024)

            if rss_mb > self._max_memory_mb:
                logger.warning(
                    "メモリ上限超過: %.0fMB > %dMB → 再起動",
                    rss_mb, self._max_memory_mb,
                )
                self._restart()
                return
        except ImportError:
            pass  # psutil なしでも動作
        except Exception:
            pass  # プロセス消滅時

        # 3. ログサイズチェック
        log_path = Path(self._log_path)
        if log_path.exists():
            size_mb = log_path.stat().st_size / (1024 * 1024)
            if size_mb > DEFAULT_LOG_MAX_MB:
                logger.info("ログローテーション (%.0fMB)", size_mb)
                self._rotate_log()

    def _restart(self) -> None:
        """auto_trader を再起動する。"""
        self._restart_count += 1

        if self._restart_count > self._max_restarts:
            logger.error(
                "最大再起動回数超過 (%d/%d) → Watchdog停止",
                self._restart_count, self._max_restarts,
            )
            self._running = False
            return

        # 既存プロセスを確実に停止
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        # 少し待ってから再起動
        time.sleep(2)
        self._start_trader()

    def _rotate_log(self) -> None:
        """ログファイルをローテーションする。"""
        log_path = Path(self._log_path)
        if log_path.exists() and log_path.stat().st_size > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive = log_path.with_name(f"{log_path.stem}_{timestamp}{log_path.suffix}")
            try:
                log_path.rename(archive)
                logger.info("ログアーカイブ: %s", archive)
            except OSError:
                pass  # 書き込み中の場合はスキップ

    def status(self) -> dict:
        """現在の状態を返す。"""
        return {
            "running": self._running,
            "trader_alive": self.is_trader_alive,
            "trader_pid": self._process.pid if self._process else None,
            "restart_count": self._restart_count,
            "uptime_hours": (time.time() - self._start_time) / 3600,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Trader Watchdog")
    parser.add_argument("--config", default="config/paper_20man.json")
    parser.add_argument("--check-interval", type=int, default=DEFAULT_CHECK_INTERVAL)
    parser.add_argument("--max-memory-mb", type=int, default=DEFAULT_MAX_MEMORY_MB)
    parser.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    parser.add_argument("--log", default="logs/auto_trader.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [WATCHDOG] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/watchdog.log", encoding="utf-8"),
        ],
    )

    wd = Watchdog(
        config_path=args.config,
        check_interval=args.check_interval,
        max_memory_mb=args.max_memory_mb,
        max_restarts=args.max_restarts,
        log_path=args.log,
    )

    def _signal_handler(sig: int, frame) -> None:
        logger.info("シグナル受信 → 停止")
        wd.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    wd.start()


if __name__ == "__main__":
    main()
