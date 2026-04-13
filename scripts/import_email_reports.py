"""メールトレードレポートをインポートするスクリプト。

自動売買レポートメール（noon/evening）を解析し、
trade_journal.db の email_reports テーブルおよび
orders/daily_pnl/snapshots テーブルに書き込む。

使用方法:
    python scripts/import_email_reports.py --dir ~/Downloads/ --db ./trade_journal.db
    python scripts/import_email_reports.py --eml path/to/file.eml
    python scripts/import_email_reports.py --gmail --email user@gmail.com --app-password xxxx
"""
from __future__ import annotations

import argparse
import email
import imaplib
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# スクリプト直実行時に src/ を import パスに追加
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.brokers.base import Order, OrderSide, OrderType  # noqa: E402
from src.brokers.trade_journal import TradeJournal  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

_SUBJECT_RE = re.compile(
    r"Trading Report\s+-\s+(noon|evening)\s+\((\d{8})\)",
    re.IGNORECASE,
)

_SUMMARY_ROW_RE = re.compile(
    r"^\s*(\w+)\s*\|\s*([-+]?\d+\.?\d*)\s*\|\s*([-+]?\d+\.?\d*)\s*"
    r"\|\s*([-+]?\d+\.?\d*)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*$"
)

_DIFF_ROW_RE = re.compile(
    r"^\s*(\w+)\s*\|\s*([+-][\d.]+)\s*\|\s*([+-][\d.]+)\s*"
    r"\|\s*([+-][\d.]+)\s*\|\s*([+-]\d+)\s*$"
)

_CONCLUSION_RE = re.compile(r"Conclusion:\s*(.+)", re.IGNORECASE)

_AI_SIGNAL_RE = re.compile(
    r"ma_cross:\s*(BUY|SELL|FLAT)\s+\([^\)]*?(\d+(?:\.\d+)?)%\)",
    re.IGNORECASE,
)

_SYMBOL_SECTION_RE = re.compile(
    r"(?:##\s*|�\s*)(\w+USDT|\w+BTC|\w+USD|\w+JPY)\s*(?:の状況|�̏�)?",
    re.IGNORECASE,
)

# 日付文字列変換（"20260413" → "2026-04-13"）
def _yyyymmdd_to_iso(raw: str) -> str:
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


# ---------------------------------------------------------------------------
# EML パーサー
# ---------------------------------------------------------------------------

def parse_eml_file(eml_path: str | Path) -> tuple[str, str, str]:
    """EML ファイルを読み込み (date, session, body) を返す。

    Parameters
    ----------
    eml_path:
        EML ファイルのパス。

    Returns
    -------
    tuple[str, str, str]
        (date_iso, session, decoded_body)
        date_iso: "YYYY-MM-DD" 形式
        session: "noon" または "evening"
        decoded_body: UTF-8 デコード済みメール本文

    Raises
    ------
    ValueError
        subject が期待フォーマットでない場合。
    """
    with open(eml_path, "rb") as f:
        msg = email.message_from_bytes(f.read())

    subject: str = msg.get("Subject", "")
    m = _SUBJECT_RE.search(subject)
    if not m:
        raise ValueError(f"Subject が期待フォーマットでない: {subject!r}")

    session = m.group(1).lower()
    date_iso = _yyyymmdd_to_iso(m.group(2))

    body = ""
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            payload = part.get_payload(decode=True)
            if isinstance(payload, bytes):
                charset = part.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace")
            break

    return date_iso, session, body


# ---------------------------------------------------------------------------
# ボディパーサー
# ---------------------------------------------------------------------------

def parse_report_body(body: str, date: str, session: str) -> list[dict]:
    """デコード済みメール本文を解析して構造化レコードリストを返す。

    Parameters
    ----------
    body:
        デコード済みメール本文テキスト。
    date:
        日付文字列（YYYY-MM-DD）。
    session:
        "noon" または "evening"。

    Returns
    -------
    list[dict]
        各シンボルの解析結果レコード。
        キー: date, session, symbol, return_pct, max_dd_pct, sharpe,
               trades, final_value, delta_return, delta_dd, delta_sharpe,
               delta_trades, ai_signal, ai_confidence, conclusion, preset
    """
    records: dict[str, dict] = {}

    # --- Profile ---
    preset = _extract_preset(body)

    # --- Conclusion ---
    conclusion = ""
    m = _CONCLUSION_RE.search(body)
    if m:
        conclusion = m.group(1).strip()

    # --- Summary Table ---
    in_summary = False
    for line in body.splitlines():
        stripped = line.strip()
        if "Multi Symbol Summary Table" in stripped:
            in_summary = True
            continue
        if in_summary:
            if stripped.startswith("---"):
                continue
            if stripped.startswith("Diff") or stripped.startswith("Symbol |"):
                continue
            row_m = _SUMMARY_ROW_RE.match(stripped)
            if row_m:
                sym = row_m.group(1)
                records.setdefault(sym, _empty_record(date, session, preset, conclusion))
                records[sym].update({
                    "return_pct": float(row_m.group(2)),
                    "max_dd_pct": float(row_m.group(3)),
                    "sharpe": float(row_m.group(4)),
                    "trades": int(row_m.group(5)),
                    "final_value": float(row_m.group(6)),
                })
            elif stripped == "" or (stripped and not re.match(r"[\w|]", stripped)):
                in_summary = False

    # --- Diff Summary ---
    in_diff = False
    for line in body.splitlines():
        stripped = line.strip()
        if "Diff Summary" in stripped:
            in_diff = True
            continue
        if in_diff:
            if stripped.startswith("---") or "Symbol |" in stripped:
                continue
            diff_m = _DIFF_ROW_RE.match(stripped)
            if diff_m:
                sym = diff_m.group(1)
                records.setdefault(sym, _empty_record(date, session, preset, conclusion))
                records[sym].update({
                    "delta_return": float(diff_m.group(2)),
                    "delta_dd": float(diff_m.group(3)),
                    "delta_sharpe": float(diff_m.group(4)),
                    "delta_trades": int(diff_m.group(5)),
                })
            elif stripped == "":
                in_diff = False

    # --- AI Signals（シンボルごとのセクションから抽出）---
    _extract_ai_signals(body, records, date, session, preset, conclusion)

    return list(records.values())


def _empty_record(date: str, session: str, preset: str, conclusion: str) -> dict:
    """空レコードのテンプレートを返す。"""
    return {
        "date": date,
        "session": session,
        "symbol": "",
        "return_pct": None,
        "max_dd_pct": None,
        "sharpe": None,
        "trades": None,
        "final_value": None,
        "delta_return": None,
        "delta_dd": None,
        "delta_sharpe": None,
        "delta_trades": None,
        "ai_signal": None,
        "ai_confidence": None,
        "conclusion": conclusion,
        "preset": preset,
    }


def _extract_preset(body: str) -> str:
    """Profile セクションから Preset 値を抽出する。"""
    m = re.search(r"Preset:\s*(\S+)", body)
    return m.group(1).strip() if m else ""


def _extract_ai_signals(
    body: str,
    records: dict[str, dict],
    date: str,
    session: str,
    preset: str,
    conclusion: str,
) -> None:
    """ボディからシンボルごとの AI シグナルを抽出して records を更新する。

    シンボルのセクション境界を探しつつ ma_cross の BUY/SELL/FLAT を取得する。
    テキストは mojibake を含む可能性があるため ASCII パターンのみで照合する。
    """
    lines = body.splitlines()

    # シンボルのブロック開始行インデックスを収集
    # 先に Summary で登録済みのシンボルを優先し、未知シンボルも登録
    symbol_positions: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        # 英字のシンボル名（例: BTCUSDT, ETHUSDT）が行に含まれているか確認
        # セクション区切りの == 行直後の symbol ブロック
        sym_m = re.search(r"\b([A-Z]{3,10}USDT|[A-Z]{3,10}BTC|[A-Z]{3,10}USD)\b", line)
        if sym_m and ("の状況" in line or "current" in line.lower() or "の状" in line or "状況" in line):
            symbol_positions.append((i, sym_m.group(1)))

    # 各シンボルブロックの範囲で AI シグナルを探す
    for idx, (start_line, sym) in enumerate(symbol_positions):
        end_line = symbol_positions[idx + 1][0] if idx + 1 < len(symbol_positions) else len(lines)
        block = "\n".join(lines[start_line:end_line])
        sig_m = _AI_SIGNAL_RE.search(block)
        if sig_m:
            records.setdefault(sym, _empty_record(date, session, preset, conclusion))
            records[sym]["symbol"] = sym
            records[sym]["ai_signal"] = sig_m.group(1).upper()
            records[sym]["ai_confidence"] = float(sig_m.group(2))

    # records 内の各エントリに symbol が設定されているか確認（Summary のみの場合）
    for sym, rec in records.items():
        if not rec["symbol"]:
            rec["symbol"] = sym

    # シンボルブロックが見つからなかった場合のフォールバック：
    # 全体から ma_cross を探してシンボルと紐付ける
    if not symbol_positions:
        for sym, rec in records.items():
            # シンボル名の前後 300 文字を検索範囲とする
            pattern = re.compile(
                rf"\b{re.escape(sym)}\b.{{0,500}}?ma_cross:\s*(BUY|SELL|FLAT)\s+\([^\)]*?(\d+(?:\.\d+)?)%\)",
                re.DOTALL | re.IGNORECASE,
            )
            m = pattern.search(body)
            if m:
                rec["ai_signal"] = m.group(1).upper()
                rec["ai_confidence"] = float(m.group(2))


# ---------------------------------------------------------------------------
# DB インポート
# ---------------------------------------------------------------------------

def ensure_email_reports_table(conn: sqlite3.Connection) -> None:
    """email_reports テーブルが存在しない場合に作成する。"""
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS email_reports (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT    NOT NULL,
                session         TEXT    NOT NULL,
                symbol          TEXT    NOT NULL,
                return_pct      REAL,
                max_dd_pct      REAL,
                sharpe          REAL,
                trades          INTEGER,
                final_value     REAL,
                delta_return    REAL,
                delta_dd        REAL,
                delta_sharpe    REAL,
                delta_trades    INTEGER,
                ai_signal       TEXT,
                ai_confidence   REAL,
                conclusion      TEXT,
                preset          TEXT,
                UNIQUE(date, session, symbol)
            )
            """
        )


def import_records_to_db(
    records: list[dict],
    journal: TradeJournal,
) -> int:
    """解析済みレコードを trade_journal.db にインポートする。

    - email_reports テーブルへの UPSERT
    - orders テーブルへの AI シグナルベースの合成注文挿入
    - daily_pnl テーブルへの return/final 値の記録
    - snapshots テーブルへのポートフォリオスナップショット挿入

    Parameters
    ----------
    records:
        parse_report_body() が返すレコードリスト。
    journal:
        TradeJournal インスタンス。

    Returns
    -------
    int
        インポートされたレコード数。
    """
    conn = journal._conn
    ensure_email_reports_table(conn)

    imported = 0
    for rec in records:
        sym = rec.get("symbol", "")
        if not sym:
            continue

        # --- email_reports への UPSERT ---
        with conn:
            conn.execute(
                """
                INSERT INTO email_reports
                    (date, session, symbol, return_pct, max_dd_pct, sharpe,
                     trades, final_value, delta_return, delta_dd, delta_sharpe,
                     delta_trades, ai_signal, ai_confidence, conclusion, preset)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date, session, symbol) DO UPDATE SET
                    return_pct     = excluded.return_pct,
                    max_dd_pct     = excluded.max_dd_pct,
                    sharpe         = excluded.sharpe,
                    trades         = excluded.trades,
                    final_value    = excluded.final_value,
                    delta_return   = excluded.delta_return,
                    delta_dd       = excluded.delta_dd,
                    delta_sharpe   = excluded.delta_sharpe,
                    delta_trades   = excluded.delta_trades,
                    ai_signal      = excluded.ai_signal,
                    ai_confidence  = excluded.ai_confidence,
                    conclusion     = excluded.conclusion,
                    preset         = excluded.preset
                """,
                (
                    rec["date"],
                    rec["session"],
                    sym,
                    rec["return_pct"],
                    rec["max_dd_pct"],
                    rec["sharpe"],
                    rec["trades"],
                    rec["final_value"],
                    rec["delta_return"],
                    rec["delta_dd"],
                    rec["delta_sharpe"],
                    rec["delta_trades"],
                    rec["ai_signal"],
                    rec["ai_confidence"],
                    rec["conclusion"],
                    rec["preset"],
                ),
            )

        # --- 合成注文レコード（AI シグナルがある場合のみ）---
        ai_signal = rec.get("ai_signal")
        final_value = rec.get("final_value")
        if ai_signal in ("BUY", "SELL") and final_value is not None:
            side = OrderSide.BUY if ai_signal == "BUY" else OrderSide.SELL
            order_id = f"email_{rec['date']}_{rec['session']}_{sym}"
            created_at = f"{rec['date']} {'12:00:00' if rec['session'] == 'noon' else '17:00:00'}"
            order = Order(
                order_id=order_id,
                symbol=sym,
                side=side,
                order_type=OrderType.MARKET,
                quantity=1.0,
                price=final_value,
                status="filled",
                filled_quantity=1.0,
                avg_fill_price=final_value,
                fee=0.0,
                created_at=created_at,
                updated_at=created_at,
            )
            journal.save_order(order)

        # --- daily_pnl への記録（final_value と return_pct から推算）---
        if final_value is not None and rec.get("return_pct") is not None:
            return_pct: float = rec["return_pct"]
            if return_pct != 0.0:
                open_eq = final_value / (1.0 + return_pct / 100.0)
            else:
                open_eq = final_value
            date_key = f"{rec['date']}_{sym}"
            # daily_pnl の PRIMARY KEY は date のみのため、複数シンボルの場合は
            # シンボル付き日付キーで区別する（テーブル定義に合わせる）
            journal.save_daily_pnl(date_key, open_eq, final_value)

        imported += 1

    # --- ポートフォリオスナップショット ---
    if records:
        total_equity = sum(
            r["final_value"] for r in records if r.get("final_value") is not None
        )
        positions = {
            r["symbol"]: {"final_value": r["final_value"], "return_pct": r["return_pct"]}
            for r in records
            if r.get("symbol") and r.get("final_value") is not None
        }
        journal.save_snapshot(
            balance=total_equity,
            equity=total_equity,
            positions=positions,
        )

    return imported


# ---------------------------------------------------------------------------
# ファイル検索ユーティリティ
# ---------------------------------------------------------------------------

def find_eml_files(directory: str | Path) -> list[Path]:
    """指定ディレクトリから Trading Report の EML ファイルを検索する。"""
    dir_path = Path(directory).expanduser()
    return sorted(dir_path.glob("Trading Report*.eml"))


# ---------------------------------------------------------------------------
# Gmail IMAP 一括取得
# ---------------------------------------------------------------------------

def _find_gmail_all_mail(imap: imaplib.IMAP4_SSL) -> Optional[str]:
    """Gmail の "All Mail" フォルダ名を \\All フラグから自動検出する。"""
    status, folder_list = imap.list()
    if status != "OK" or not folder_list:
        return None
    for entry in folder_list:
        if not isinstance(entry, bytes):
            continue
        decoded = entry.decode("utf-8", errors="replace")
        if "\\All" in decoded:
            # フォルダ名を抽出: 例 '(\\HasNoChildren \\All) "/" "[Gmail]/All Mail"'
            m = re.search(r'"([^"]+)"$', decoded)
            if m:
                return f'"{m.group(1)}"'
    return None


def fetch_reports_from_gmail(
    email_addr: str,
    app_password: str,
    search_query: str = 'SUBJECT "Trading Report"',
    max_results: int = 200,
) -> list[tuple[str, str, str]]:
    """Gmail IMAP 経由でトレードレポートメールを一括取得する。

    Parameters
    ----------
    email_addr:
        Gmail アドレス。
    app_password:
        Google アプリパスワード（16文字、スペースなし）。
    search_query:
        IMAP 検索クエリ（デフォルト: 件名に "Trading Report" を含む）。
    max_results:
        取得する最大メール数。

    Returns
    -------
    list[tuple[str, str, str]]
        (date_iso, session, decoded_body) のリスト。
    """
    logger.info("Gmail IMAP に接続中... (%s)", email_addr)
    imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)

    try:
        imap.login(email_addr, app_password)
        logger.info("Gmail IMAP 認証成功")

        # Gmail "All Mail" フォルダを自動検出（言語非依存）
        all_mail_folder = _find_gmail_all_mail(imap)
        msg_ids: list[bytes] = []
        for folder in [all_mail_folder, "INBOX"] if all_mail_folder else ["INBOX"]:
            status, _ = imap.select(folder)
            if status != "OK":
                logger.debug("フォルダ選択失敗: %s", folder)
                continue
            logger.debug("フォルダ選択成功: %s", folder)
            _status, msg_ids_data = imap.search(None, search_query)
            msg_ids = msg_ids_data[0].split()
            if msg_ids:
                logger.info("フォルダ %s で %d 件のメールを発見", folder, len(msg_ids))
                break

        if not msg_ids:
            logger.warning("該当するメールが見つかりませんでした")
            return []

        # 最新の max_results 件に制限
        msg_ids = msg_ids[-max_results:]
        logger.info("%d 件のメールを取得します", len(msg_ids))

        results: list[tuple[str, str, str]] = []
        for msg_id in msg_ids:
            _status, msg_data = imap.fetch(msg_id, "(RFC822)")
            if not msg_data or not msg_data[0]:
                continue

            raw_email = msg_data[0]
            if isinstance(raw_email, tuple) and len(raw_email) >= 2:
                raw_bytes = raw_email[1]
            else:
                continue

            msg = email.message_from_bytes(raw_bytes)
            subject: str = msg.get("Subject", "")

            m = _SUBJECT_RE.search(subject)
            if not m:
                logger.debug("スキップ（件名不一致）: %s", subject)
                continue

            session = m.group(1).lower()
            date_iso = _yyyymmdd_to_iso(m.group(2))

            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        charset = part.get_content_charset() or "utf-8"
                        body = payload.decode(charset, errors="replace")
                    break

            if body:
                results.append((date_iso, session, body))
                logger.debug("取得: %s %s", date_iso, session)

        logger.info("%d 件のレポートを取得しました", len(results))
        return results

    finally:
        try:
            imap.logout()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI エントリーポイント
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="メールトレードレポートを trade_journal.db にインポートする"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dir",
        metavar="DIRECTORY",
        help="EML ファイルを検索するディレクトリ（例: ~/Downloads/）",
    )
    group.add_argument(
        "--eml",
        metavar="FILE",
        help="単一 EML ファイルのパス",
    )
    group.add_argument(
        "--gmail",
        action="store_true",
        help="Gmail IMAP 経由で一括取得（--email と --app-password が必要）",
    )
    parser.add_argument(
        "--db",
        default="./trade_journal.db",
        metavar="DB_PATH",
        help="SQLite DB ファイルのパス（デフォルト: ./trade_journal.db）",
    )
    parser.add_argument(
        "--email",
        metavar="ADDR",
        help="Gmail アドレス（--gmail 使用時。環境変数 GMAIL_ADDRESS でも可）",
    )
    parser.add_argument(
        "--app-password",
        metavar="PASS",
        help="Google アプリパスワード（--gmail 使用時。環境変数 GMAIL_APP_PASSWORD でも可）",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="Gmail から取得する最大メール数（デフォルト: 200）",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="詳細ログを出力する",
    )
    return parser


def _process_eml(eml_path: Path, journal: TradeJournal) -> int:
    """単一 EML ファイルを解析してインポートし、インポート数を返す。"""
    try:
        date_iso, session, body = parse_eml_file(eml_path)
        records = parse_report_body(body, date_iso, session)
        count = import_records_to_db(records, journal)
        logger.info("%s: %d レコードをインポートしました", eml_path.name, count)
        return count
    except Exception as exc:
        logger.error("%s の処理中にエラーが発生: %s", eml_path, exc)
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI メインエントリーポイント。

    Returns
    -------
    int
        終了コード（0 = 成功）。
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    db_path = args.db
    journal = TradeJournal(db_path=db_path)

    try:
        total = 0

        if args.gmail:
            # Gmail IMAP 一括取得モード
            gmail_addr = args.email or os.environ.get("GMAIL_ADDRESS", "")
            gmail_pass = args.app_password or os.environ.get("GMAIL_APP_PASSWORD", "")
            if not gmail_addr or not gmail_pass:
                print(
                    "エラー: --gmail には --email と --app-password が必要です。\n"
                    "  環境変数 GMAIL_ADDRESS / GMAIL_APP_PASSWORD でも設定可能です。\n\n"
                    "アプリパスワードの取得方法:\n"
                    "  1. https://myaccount.google.com/apppasswords にアクセス\n"
                    "  2. 「アプリ名」に任意の名前を入力して「作成」\n"
                    "  3. 表示された16文字のパスワードを使用\n"
                    "  ※ 2段階認証が有効である必要があります"
                )
                return 1

            reports = fetch_reports_from_gmail(
                gmail_addr, gmail_pass, max_results=args.max_results,
            )
            for date_iso, session, body in reports:
                records = parse_report_body(body, date_iso, session)
                count = import_records_to_db(records, journal)
                logger.info("%s %s: %d レコード", date_iso, session, count)
                total += count

        elif args.eml:
            eml_files = [Path(args.eml)]
            for eml_path in eml_files:
                total += _process_eml(eml_path, journal)

        else:
            eml_files = find_eml_files(args.dir)
            if not eml_files:
                logger.warning("EML ファイルが見つかりませんでした: %s", args.dir)
                return 0
            for eml_path in eml_files:
                total += _process_eml(eml_path, journal)

        print(f"合計 {total} レコードをインポートしました → {db_path}")
        return 0
    finally:
        journal.close()


if __name__ == "__main__":
    sys.exit(main())
