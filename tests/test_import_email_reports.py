"""メールレポートインポートスクリプトのテスト。

実際の EML ファイルを使ったパーサーテスト、
DB インポートのテスト、冪等性テストを含む。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest

# テスト実行時に src/ を参照できるようにする
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.import_email_reports import (
    _yyyymmdd_to_iso,
    ensure_email_reports_table,
    find_eml_files,
    import_records_to_db,
    parse_eml_file,
    parse_report_body,
)
from src.brokers.base import OrderSide
from src.brokers.trade_journal import TradeJournal

# ---------------------------------------------------------------------------
# 定数: テスト用 EML ファイルパス
# ---------------------------------------------------------------------------

_EML_DIR = Path(r"C:/Users/17281/Downloads")
_NOON_EML = _EML_DIR / "Trading Report - noon (20260413).eml"
_EVENING_EML = _EML_DIR / "Trading Report - evening (20260413).eml"

# テストの要否を確認するマーカー
_has_eml_files = _NOON_EML.exists() and _EVENING_EML.exists()
_skip_if_no_eml = pytest.mark.skipif(
    not _has_eml_files,
    reason="EML ファイルが存在しない",
)

# ---------------------------------------------------------------------------
# サンプルボディ（テスト用固定テキスト）
# ---------------------------------------------------------------------------

_SAMPLE_BODY = """\
自動売買トレードレポート - noon

Profile:
  Preset: test_preset_20_100
  Symbols: BTCUSDT, ETHUSDT
  MA Short: 20
  MA Long: 100
  Risk PCT: 0.5
  Fee Rate: 0.0005
  Start/End: None/None

Multi Symbol Summary Table:
Symbol | Return | MaxDD | Sharpe | Trades | Final
-------|--------|-------|--------|--------|------
BTCUSDT | -7.29 | -14.74 | -0.05 | 1 | 9271.05
ETHUSDT | 5.03 | -9.12 | 0.04 | 1 | 10502.57

Diff Summary:
Symbol | \u0394Return | \u0394DD | \u0394Sharpe | \u0394Trades
-------|---------|-----|---------|---------
BTCUSDT | +0.00 | +0.00 | +0.00 | +0
ETHUSDT | +0.00 | +0.00 | +0.00 | +0

Conclusion: [LLM] SKIP: changes are below thresholds

==================================================
Market Report (2026-04-13 12:30)
==================================================

## BTCUSDT の状況
  Current price: $11,751.06

AI Strategy Judgment:
  [Sell] ma_cross: SELL (confidence 100%)

Ensemble:
  0/1 strategies BUY, 1/1 strategies SELL

==================================================
Market Report (2026-04-13 12:30)
==================================================

## ETHUSDT の状況
  Current price: $10,970.47

AI Strategy Judgment:
  [Buy] ma_cross: BUY (confidence 100%)

Ensemble:
  1/1 strategies BUY, 0/1 strategies SELL
==================================================
"""

# partial body（Profile なし、Diff なし）
_PARTIAL_BODY = """\
Multi Symbol Summary Table:
Symbol | Return | MaxDD | Sharpe | Trades | Final
-------|--------|-------|--------|--------|------
XRPUSDT | 2.50 | -5.00 | 0.12 | 3 | 10250.00

Conclusion: [LLM] BUY signal detected
"""


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def journal(tmp_path: Path) -> Generator[TradeJournal, None, None]:
    """テスト用一時 DB ジャーナル。"""
    db_path = str(tmp_path / "test_email.db")
    j = TradeJournal(db_path=db_path)
    yield j
    j.close()


# ---------------------------------------------------------------------------
# 1. ユーティリティ関数テスト
# ---------------------------------------------------------------------------

class TestUtilities:
    """ユーティリティ関数のテスト。"""

    def test_yyyymmdd_to_iso(self) -> None:
        """yyyymmdd → ISO 変換が正しい。"""
        assert _yyyymmdd_to_iso("20260413") == "2026-04-13"
        assert _yyyymmdd_to_iso("20231231") == "2023-12-31"


# ---------------------------------------------------------------------------
# 2. EML ファイルパーサーテスト（実ファイル）
# ---------------------------------------------------------------------------

class TestParseEmlFile:
    """parse_eml_file() のテスト。"""

    @_skip_if_no_eml
    def test_parse_noon_eml_subject(self) -> None:
        """noon EML から日付とセッションが正しく抽出される。"""
        date_iso, session, body = parse_eml_file(_NOON_EML)
        assert date_iso == "2026-04-13"
        assert session == "noon"
        assert len(body) > 100

    @_skip_if_no_eml
    def test_parse_evening_eml_subject(self) -> None:
        """evening EML から日付とセッションが正しく抽出される。"""
        date_iso, session, body = parse_eml_file(_EVENING_EML)
        assert date_iso == "2026-04-13"
        assert session == "evening"

    @_skip_if_no_eml
    def test_parse_eml_body_not_empty(self) -> None:
        """EML ボディが空でない。"""
        _, _, body = parse_eml_file(_NOON_EML)
        assert "Multi Symbol Summary Table" in body

    def test_parse_eml_invalid_subject_raises(self, tmp_path: Path) -> None:
        """期待フォーマット外の subject は ValueError を発生させる。"""
        import email
        from email.mime.text import MIMEText

        msg = MIMEText("test body", "plain", "utf-8")
        msg["Subject"] = "不正なメール件名"
        eml_path = tmp_path / "bad.eml"
        eml_path.write_bytes(msg.as_bytes())

        with pytest.raises(ValueError, match="期待フォーマットでない"):
            parse_eml_file(eml_path)


# ---------------------------------------------------------------------------
# 3. ボディパーサーテスト
# ---------------------------------------------------------------------------

class TestParseReportBody:
    """parse_report_body() のテスト。"""

    def test_returns_two_records(self) -> None:
        """サンプルボディから 2 シンボルのレコードが返る。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        assert len(records) == 2

    def test_summary_table_extraction(self) -> None:
        """Summary Table の値が正しく抽出される。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        btc = next(r for r in records if r["symbol"] == "BTCUSDT")
        assert btc["return_pct"] == pytest.approx(-7.29)
        assert btc["max_dd_pct"] == pytest.approx(-14.74)
        assert btc["sharpe"] == pytest.approx(-0.05)
        assert btc["trades"] == 1
        assert btc["final_value"] == pytest.approx(9271.05)

    def test_eth_summary_extraction(self) -> None:
        """ETHUSDT の Summary Table 値が正しい。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        eth = next(r for r in records if r["symbol"] == "ETHUSDT")
        assert eth["return_pct"] == pytest.approx(5.03)
        assert eth["final_value"] == pytest.approx(10502.57)

    def test_ai_signal_extraction_sell(self) -> None:
        """BTCUSDT の AI シグナルが SELL として抽出される。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        btc = next(r for r in records if r["symbol"] == "BTCUSDT")
        assert btc["ai_signal"] == "SELL"
        assert btc["ai_confidence"] == pytest.approx(100.0)

    def test_ai_signal_extraction_buy(self) -> None:
        """ETHUSDT の AI シグナルが BUY として抽出される。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        eth = next(r for r in records if r["symbol"] == "ETHUSDT")
        assert eth["ai_signal"] == "BUY"
        assert eth["ai_confidence"] == pytest.approx(100.0)

    def test_conclusion_extraction(self) -> None:
        """Conclusion が正しく抽出される。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        for rec in records:
            assert rec["conclusion"] == "[LLM] SKIP: changes are below thresholds"

    def test_preset_extraction(self) -> None:
        """Preset 値が正しく抽出される。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        for rec in records:
            assert rec["preset"] == "test_preset_20_100"

    def test_date_and_session_in_records(self) -> None:
        """date と session がレコードに含まれる。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "evening")
        for rec in records:
            assert rec["date"] == "2026-04-13"
            assert rec["session"] == "evening"

    def test_partial_body_missing_fields(self) -> None:
        """Profile なし・Diff なしの部分的ボディでもクラッシュしない。"""
        records = parse_report_body(_PARTIAL_BODY, "2026-01-01", "noon")
        assert len(records) >= 1
        xrp = next((r for r in records if r["symbol"] == "XRPUSDT"), None)
        assert xrp is not None
        assert xrp["return_pct"] == pytest.approx(2.50)
        assert xrp["final_value"] == pytest.approx(10250.00)
        # AI シグナルはなくても None として扱われる
        assert xrp["ai_signal"] is None

    def test_empty_body_returns_empty_list(self) -> None:
        """空ボディは空リストを返す。"""
        records = parse_report_body("", "2026-01-01", "noon")
        assert records == []


# ---------------------------------------------------------------------------
# 4. 実ファイルからの完全パーステスト
# ---------------------------------------------------------------------------

class TestRealEmlParsing:
    """実 EML ファイルを使った統合パーステスト。"""

    @_skip_if_no_eml
    def test_noon_eml_two_symbols(self) -> None:
        """noon EML から BTCUSDT と ETHUSDT が取得される。"""
        date_iso, session, body = parse_eml_file(_NOON_EML)
        records = parse_report_body(body, date_iso, session)
        symbols = {r["symbol"] for r in records}
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols

    @_skip_if_no_eml
    def test_noon_eml_btcusdt_sell_signal(self) -> None:
        """noon EML の BTCUSDT は SELL シグナル。"""
        date_iso, session, body = parse_eml_file(_NOON_EML)
        records = parse_report_body(body, date_iso, session)
        btc = next(r for r in records if r["symbol"] == "BTCUSDT")
        assert btc["ai_signal"] == "SELL"

    @_skip_if_no_eml
    def test_noon_eml_ethusdt_buy_signal(self) -> None:
        """noon EML の ETHUSDT は BUY シグナル。"""
        date_iso, session, body = parse_eml_file(_NOON_EML)
        records = parse_report_body(body, date_iso, session)
        eth = next(r for r in records if r["symbol"] == "ETHUSDT")
        assert eth["ai_signal"] == "BUY"

    @_skip_if_no_eml
    def test_evening_eml_session_label(self) -> None:
        """evening EML の session が 'evening' になっている。"""
        date_iso, session, body = parse_eml_file(_EVENING_EML)
        records = parse_report_body(body, date_iso, session)
        for rec in records:
            assert rec["session"] == "evening"


# ---------------------------------------------------------------------------
# 5. DB インポートテスト
# ---------------------------------------------------------------------------

class TestDbImport:
    """import_records_to_db() のテスト。"""

    def test_email_reports_table_created(self, journal: TradeJournal) -> None:
        """ensure_email_reports_table で email_reports テーブルが作成される。"""
        ensure_email_reports_table(journal._conn)
        tables = {
            row[0]
            for row in journal._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "email_reports" in tables

    def test_import_sample_records(self, journal: TradeJournal) -> None:
        """サンプルレコードが email_reports テーブルに保存される。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        count = import_records_to_db(records, journal)
        assert count == 2

        rows = journal._conn.execute("SELECT * FROM email_reports").fetchall()
        assert len(rows) == 2

    def test_buy_signal_creates_order(self, journal: TradeJournal) -> None:
        """BUY シグナルのレコードが orders テーブルに合成注文を作成する。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        import_records_to_db(records, journal)

        orders = journal.load_orders()
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buy_orders) >= 1
        assert buy_orders[0].symbol == "ETHUSDT"

    def test_sell_signal_creates_order(self, journal: TradeJournal) -> None:
        """SELL シグナルのレコードが orders テーブルに合成注文を作成する。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        import_records_to_db(records, journal)

        orders = journal.load_orders()
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(sell_orders) >= 1
        assert sell_orders[0].symbol == "BTCUSDT"

    def test_snapshot_created_after_import(self, journal: TradeJournal) -> None:
        """インポート後にスナップショットが作成される。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        import_records_to_db(records, journal)

        snap = journal.load_latest_snapshot()
        assert snap is not None
        assert snap["equity"] > 0

    def test_idempotent_import(self, journal: TradeJournal) -> None:
        """同じレコードを2回インポートしても重複が作成されない。"""
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        import_records_to_db(records, journal)
        import_records_to_db(records, journal)

        rows = journal._conn.execute("SELECT * FROM email_reports").fetchall()
        # UNIQUE(date, session, symbol) により重複なし
        assert len(rows) == 2

    def test_order_idempotent(self, journal: TradeJournal) -> None:
        """同じ EML を2回インポートしても orders が重複しない。

        TradeJournal.save_order は INSERT OR REPLACE のため、
        同一 order_id は上書きされる。
        """
        records = parse_report_body(_SAMPLE_BODY, "2026-04-13", "noon")
        import_records_to_db(records, journal)
        import_records_to_db(records, journal)

        orders = journal.load_orders()
        # email_noon_BTCUSDT + email_noon_ETHUSDT = 2 件（SELL/BUY）
        assert len(orders) == 2


# ---------------------------------------------------------------------------
# 6. find_eml_files テスト
# ---------------------------------------------------------------------------

class TestFindEmlFiles:
    """find_eml_files() のテスト。"""

    def test_finds_trading_report_emls(self, tmp_path: Path) -> None:
        """ディレクトリから Trading Report EML が見つかる。"""
        (tmp_path / "Trading Report - noon (20260413).eml").write_text("dummy")
        (tmp_path / "Trading Report - evening (20260413).eml").write_text("dummy")
        (tmp_path / "other.eml").write_text("dummy")

        found = find_eml_files(tmp_path)
        names = [f.name for f in found]
        assert any("noon" in n for n in names)
        assert any("evening" in n for n in names)
        # "other.eml" は含まれない
        assert "other.eml" not in names

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        """EML ファイルがないディレクトリは空リストを返す。"""
        result = find_eml_files(tmp_path)
        assert result == []

    @_skip_if_no_eml
    def test_finds_real_eml_files(self) -> None:
        """実際のダウンロードディレクトリから EML ファイルを検索できる。"""
        found = find_eml_files(_EML_DIR)
        assert len(found) >= 2
