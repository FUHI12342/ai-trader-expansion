"""FastAPI サーバー（SHANON連携用）。

エンドポイント:
    GET  /api/status      — 全戦略のGo/NoGoステータス
    GET  /api/positions   — 現在ポジション一覧
    GET  /api/performance — パフォーマンスメトリクス
    POST /api/backtest    — オンデマンドバックテスト実行
"""
from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config.settings import get_settings
from src.brokers.paper_broker import PaperBroker
from src.evaluation.backtester import run_backtest
from src.strategies import (
    MACrossoverStrategy,
    DualMomentumStrategy,
    MACDRSIStrategy,
    BollingerRSIADXStrategy,
)
from src.data.data_manager import DataManager

logger = logging.getLogger(__name__)

# FastAPIアプリ初期化
app = FastAPI(
    title="AI Trader 株式拡張 API",
    description="株式取引戦略の評価・管理・SHANON連携API",
    version="0.1.0",
)

# CORSミドルウェア（SHANON Desktopからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bearer = HTTPBearer(auto_error=False)


def _verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """TRADER_API_KEY 環境変数が設定されている場合にAPIキー認証を行う。

    環境変数が未設定の場合は認証をスキップする（開発モード）。
    """
    api_key = os.environ.get("TRADER_API_KEY")
    if not api_key:
        # 環境変数未設定時はローカル開発モードとして認証スキップ
        return
    if credentials is None or not secrets.compare_digest(credentials.credentials, api_key):
        raise HTTPException(status_code=401, detail="認証が必要です")


# グローバルインスタンス（起動時に初期化）
_settings = get_settings()
_paper_broker = PaperBroker(
    initial_balance=_settings.initial_capital,
    fee_rate=_settings.fee_rate,
    slippage_rate=_settings.slippage_rate,
)

# 全戦略の登録
_strategies = {
    "ma_crossover": MACrossoverStrategy(),
    "dual_momentum": DualMomentumStrategy(),
    "macd_rsi": MACDRSIStrategy(),
    "bollinger_rsi_adx": BollingerRSIADXStrategy(),
}


# ============================================================
# リクエスト/レスポンスモデル
# ============================================================

class StrategyStatus(BaseModel):
    """戦略ステータス。"""
    name: str
    go_nogo: str          # "GO" または "NOGO"
    reason: str
    last_signal: int      # 1=BUY, -1=SELL, 0=FLAT
    last_updated: str


class PositionInfo(BaseModel):
    """ポジション情報。"""
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PerformanceMetrics(BaseModel):
    """パフォーマンスメトリクス。"""
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    total_trades: int
    balance: float
    equity: float


class BacktestRequest(BaseModel):
    """バックテストリクエスト。"""
    strategy_name: str = Field(..., description="戦略名 (ma_crossover, macd_rsi, 等)")
    symbol: str = Field("7203.T", description="銘柄コード (例: 7203.T)")
    start_date: str = Field(..., description="開始日 (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="終了日 (YYYY-MM-DD)")
    initial_capital: float = Field(1_000_000.0, description="初期資金（円）")
    fee_rate: float = Field(0.001, description="手数料率 (例: 0.001 = 0.1%)")


class BacktestResponse(BaseModel):
    """バックテストレスポンス。"""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int


# ============================================================
# エンドポイント
# ============================================================

@app.get("/", summary="ヘルスチェック")
async def root() -> Dict[str, str]:
    """APIの稼働確認。"""
    return {"status": "ok", "service": "AI Trader 株式拡張 API"}


@app.get(
    "/api/status",
    response_model=List[StrategyStatus],
    summary="全戦略のGo/Nogoステータス取得",
)
async def get_status(
    symbol: str = Query("7203.T", description="確認する銘柄コード"),
    _: None = Depends(_verify_api_key),
) -> List[StrategyStatus]:
    """全戦略の最新シグナルとGo/Nogoを返す。

    Returns
    -------
    List[StrategyStatus]
        各戦略のステータスリスト
    """
    from datetime import datetime, timedelta
    import pandas as pd

    statuses = []

    # 直近60日のデータを取得
    try:
        dm = DataManager()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
        data = dm.fetch_ohlcv(symbol, start, end)
    except Exception as e:
        logger.warning(f"データ取得失敗 ({symbol}): {e}")
        # データ取得失敗時はすべてNOGO
        for name in _strategies:
            statuses.append(StrategyStatus(
                name=name,
                go_nogo="NOGO",
                reason=f"データ取得失敗: {str(e)[:100]}",
                last_signal=0,
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ))
        return statuses

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for name, strategy in _strategies.items():
        try:
            signals = strategy.generate_signals(data)
            last_signal = int(signals.iloc[-1]) if not signals.empty else 0

            if last_signal == 1:
                go_nogo = "GO"
                reason = "BUYシグナル発生"
            elif last_signal == -1:
                go_nogo = "NOGO"
                reason = "SELLシグナル（EXIT）"
            else:
                go_nogo = "NOGO"
                reason = "シグナルなし（FLAT）"

            statuses.append(StrategyStatus(
                name=name,
                go_nogo=go_nogo,
                reason=reason,
                last_signal=last_signal,
                last_updated=now_str,
            ))
        except Exception as e:
            logger.error(f"戦略 {name} のシグナル生成失敗: {e}")
            statuses.append(StrategyStatus(
                name=name,
                go_nogo="NOGO",
                reason=f"エラー: {str(e)[:100]}",
                last_signal=0,
                last_updated=now_str,
            ))

    return statuses


@app.get(
    "/api/positions",
    response_model=List[PositionInfo],
    summary="現在ポジション一覧",
)
async def get_positions(
    _: None = Depends(_verify_api_key),
) -> List[PositionInfo]:
    """ペーパーブローカーの現在ポジション一覧を返す。"""
    positions = _paper_broker.get_positions()
    return [
        PositionInfo(
            symbol=pos.symbol,
            quantity=pos.quantity,
            avg_entry_price=pos.avg_entry_price,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl,
            unrealized_pnl_pct=pos.unrealized_pnl_pct,
        )
        for pos in positions.values()
    ]


@app.get(
    "/api/performance",
    response_model=PerformanceMetrics,
    summary="パフォーマンスメトリクス",
)
async def get_performance(
    _: None = Depends(_verify_api_key),
) -> PerformanceMetrics:
    """現在のパフォーマンスメトリクスを返す。"""
    balance = _paper_broker.get_balance()
    equity = _paper_broker.get_equity()
    history = _paper_broker.get_order_history()

    # 簡易メトリクス計算
    filled_orders = [o for o in history if o.status == "filled"]
    total_trades = len(filled_orders) // 2  # ペアで1取引

    return PerformanceMetrics(
        total_return_pct=round((equity - _settings.initial_capital) / _settings.initial_capital * 100, 2),
        annualized_return_pct=0.0,    # ライブ運用時に計算
        max_drawdown_pct=0.0,         # ライブ運用時に計算
        sharpe_ratio=0.0,             # ライブ運用時に計算
        sortino_ratio=0.0,            # ライブ運用時に計算
        win_rate=0.0,                 # ライブ運用時に計算
        total_trades=total_trades,
        balance=balance,
        equity=equity,
    )


@app.post(
    "/api/backtest",
    response_model=BacktestResponse,
    summary="オンデマンドバックテスト実行",
)
async def run_backtest_api(
    request: BacktestRequest,
    _: None = Depends(_verify_api_key),
) -> BacktestResponse:
    """指定された戦略・銘柄・期間でバックテストを実行する。

    Parameters
    ----------
    request:
        バックテストリクエスト（戦略名、銘柄、期間等）

    Returns
    -------
    BacktestResponse
        バックテスト結果のサマリー
    """
    strategy = _strategies.get(request.strategy_name)
    if strategy is None:
        raise HTTPException(
            status_code=400,
            detail=f"戦略 '{request.strategy_name}' が見つかりません。"
                   f"利用可能: {list(_strategies.keys())}",
        )

    # データ取得
    try:
        dm = DataManager()
        data = dm.fetch_ohlcv(request.symbol, request.start_date, request.end_date)
    except Exception as e:
        logger.warning("バックテスト: データ取得失敗 symbol=%s: %s", request.symbol, e)
        raise HTTPException(status_code=422, detail="データ取得に失敗しました")

    if data.empty:
        raise HTTPException(status_code=422, detail="指定された銘柄・期間のデータが存在しません")

    # バックテスト実行
    try:
        result = run_backtest(
            strategy=strategy,
            data=data,
            symbol=request.symbol,
            initial_capital=request.initial_capital,
            fee_rate=request.fee_rate,
            slippage_rate=_settings.slippage_rate,
        )
    except Exception as e:
        logger.error("バックテスト実行エラー: %s", e)
        raise HTTPException(status_code=500, detail="バックテスト実行中にエラーが発生しました")

    return BacktestResponse(
        strategy_name=result.strategy_name,
        symbol=result.symbol,
        start_date=result.start_date,
        end_date=result.end_date,
        initial_capital=result.initial_capital,
        final_capital=result.final_capital,
        total_return_pct=result.metrics.total_return_pct,
        max_drawdown_pct=result.metrics.max_drawdown_pct,
        sharpe_ratio=result.metrics.sharpe_ratio,
        win_rate=result.metrics.win_rate,
        total_trades=result.metrics.total_trades,
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    # デフォルトは 127.0.0.1（ローカルのみ）。外部公開が必要な場合は設定で上書きする。
    host = getattr(settings.api_server, "host", "127.0.0.1") or "127.0.0.1"
    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=settings.api_server.port,
        log_level=settings.api_server.log_level,
        reload=False,
    )
