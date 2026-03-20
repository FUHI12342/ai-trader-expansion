"""kabuステーション API ブローカー（スタブ実装）。

kabuステーション® は auカブコム証券のデスクトップアプリが提供するローカルREST API。
このファイルはAPIの構造を示すスタブ実装で、実際のAPIコールは最低限に留める。

前提条件:
    - kabuステーション® が localhost で起動していること
    - 環境変数 KABU_API_PASSWORD が設定されていること

参考:
    https://kabucom.github.io/kabusapi/ptal/index.html
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from config.settings import KabuStationSettings
from .base import BrokerBase, Order, OrderSide, OrderType, Position

logger = logging.getLogger(__name__)

# API エンドポイント
_API_PATH = "/kabusapi"

# 売買区分
_SIDE_BUY = 1
_SIDE_SELL = 2

# 執行条件
_EXEC_CONDITION_MARKET = 1    # 成行
_EXEC_CONDITION_LIMIT = 2     # 指値


def _retry(fn, retry_max: int = 3, retry_base_sec: float = 1.0):
    """指数バックオフでリトライ（4xxはリトライしない）。"""
    last_err: Exception = RuntimeError("No attempts made")
    for attempt in range(retry_max):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                body = e.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"kabuステーション HTTP {e.code}: {body}") from e
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError) as e:
            last_err = e
        if attempt < retry_max - 1:
            time.sleep(retry_base_sec * (2 ** attempt))
    raise RuntimeError(
        f"kabuステーション API {retry_max}回試行後も失敗: {last_err}"
    ) from last_err


class KabuStationBroker(BrokerBase):
    """kabuステーション® API ブローカー（スタブ実装）。

    Parameters
    ----------
    settings:
        kabuステーション設定（省略時は環境変数から読み込み）
    """

    def __init__(self, settings: Optional[KabuStationSettings] = None) -> None:
        self._settings = settings or KabuStationSettings.from_env()
        self._token: Optional[str] = None
        self._base_url = (
            f"http://{self._settings.host}:{self._settings.port}{_API_PATH}"
        )

    @staticmethod
    def _mask_body_for_log(body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """ログ出力用にbody内のパスワードフィールドをマスクする。"""
        if body is None:
            return None
        masked = dict(body)
        for key in list(masked.keys()):
            if key.lower() in ("password", "apipassword"):
                masked[key] = "***"
        return masked

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """HTTP リクエストを実行する。

        Parameters
        ----------
        method:
            HTTPメソッド（"GET", "POST", "DELETE"等）
        path:
            APIパス（例: "/token/session"）
        body:
            リクエストボディ（JSONシリアライズ）

        Returns
        -------
        Any
            レスポンスJSON（パース済み）
        """
        url = f"{self._base_url}{path}"
        headers = {"Content-Type": "application/json"}

        if self._token:
            headers["X-API-KEY"] = self._token

        logger.debug("API request: %s %s body=%s", method, path, self._mask_body_for_log(body))

        data = json.dumps(body).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        def do_request():
            with urllib.request.urlopen(req, timeout=self._settings.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))

        return _retry(do_request)

    def authenticate(self) -> str:
        """APIトークンを取得する（パスワード認証）。

        Returns
        -------
        str
            APIトークン文字列
        """
        if not self._settings.api_password:
            raise RuntimeError(
                "kabuステーション APIパスワードが設定されていません。"
                "環境変数 KABU_API_PASSWORD を設定してください。"
            )

        result = self._request("POST", "/token/session", {"APIPassword": self._settings.api_password})
        self._token = str(result.get("Token", ""))
        if not self._token:
            raise RuntimeError(f"トークン取得失敗: {result}")

        logger.info("kabuステーション API 認証成功")
        return self._token

    def _ensure_authenticated(self) -> None:
        """未認証の場合は認証を実行する。"""
        if not self._token:
            self.authenticate()

    def get_balance(self) -> float:
        """現金残高を取得する。

        Returns
        -------
        float
            現金残高（円）
        """
        self._ensure_authenticated()
        result = self._request("GET", "/wallet/cash")
        # kabuステーション APIレスポンス形式に合わせてパース
        return float(result.get("StockAccountWallet", 0))

    def get_positions(self) -> Dict[str, Position]:
        """ポジション一覧を取得する。"""
        self._ensure_authenticated()
        result = self._request("GET", "/positions")

        positions: Dict[str, Position] = {}
        for item in result or []:
            symbol = str(item.get("Symbol", ""))
            qty = float(item.get("LeavesQty", 0))
            avg_price = float(item.get("Price", 0))
            current_price = float(item.get("CurrentPrice", avg_price))
            pnl = float(item.get("ProfitLoss", 0))
            pnl_pct = (pnl / (avg_price * abs(qty))) * 100 if avg_price > 0 and qty != 0 else 0.0

            positions[symbol] = Position(
                symbol=symbol,
                quantity=qty,
                avg_entry_price=avg_price,
                current_price=current_price,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
            )
        return positions

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
    ) -> Order:
        """注文を発注する。

        Parameters
        ----------
        symbol:
            銘柄コード（4桁数字）
        side:
            注文方向
        order_type:
            注文種別
        quantity:
            株数
        price:
            指値価格

        Returns
        -------
        Order
            発注された注文情報
        """
        self._ensure_authenticated()

        # kabuステーションの注文形式に変換
        kabu_side = _SIDE_BUY if side == OrderSide.BUY else _SIDE_SELL
        kabu_exec = _EXEC_CONDITION_MARKET if order_type == OrderType.MARKET else _EXEC_CONDITION_LIMIT

        body: Dict[str, Any] = {
            "Password": self._settings.api_password,
            "Symbol": symbol,
            "Exchange": 1,         # 東証
            "SecurityType": 1,     # 株式
            "Side": str(kabu_side),
            "CashMargin": 1,       # 現物
            "DelivType": 0,        # 自動
            "AccountType": 2,      # 一般
            "Qty": int(quantity),
            "FrontOrderType": kabu_exec,
        }

        if order_type == OrderType.LIMIT and price:
            body["Price"] = price

        result = self._request("POST", "/sendorder", body)

        order_id = str(result.get("OrderId", ""))
        if not order_id:
            raise RuntimeError(f"注文発注失敗: {result}")

        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="open",
            filled_quantity=0.0,
            avg_fill_price=0.0,
            fee=0.0,
            created_at=now,
            updated_at=now,
        )

    def cancel_order(self, order_id: str) -> bool:
        """注文をキャンセルする。"""
        self._ensure_authenticated()
        try:
            self._request(
                "PUT",
                "/cancelorder",
                {
                    "OrderId": order_id,
                    "Password": self._settings.api_password,
                },
            )
            return True
        except RuntimeError as e:
            logger.warning(f"注文キャンセル失敗 {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """注文状況を取得する。"""
        self._ensure_authenticated()
        try:
            result = self._request("GET", f"/orders/{order_id}")
            if not result:
                return None

            from datetime import datetime
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            side_val = str(result.get("Side", "1"))

            return Order(
                order_id=order_id,
                symbol=str(result.get("Symbol", "")),
                side=OrderSide.BUY if side_val == "1" else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=float(result.get("Qty", 0)),
                price=float(result.get("Price", 0)) or None,
                status=self._parse_order_status(result.get("State", 0)),
                filled_quantity=float(result.get("CumQty", 0)),
                avg_fill_price=float(result.get("Price", 0)),
                fee=0.0,
                created_at=str(result.get("RecvTime", now)),
                updated_at=now,
            )
        except Exception as e:
            logger.warning(f"注文取得失敗 {order_id}: {e}")
            return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """オープン注文一覧を返す。"""
        self._ensure_authenticated()
        try:
            path = "/orders"
            if symbol:
                path += f"?symbol={symbol}"
            result = self._request("GET", path)
            orders = []
            for item in result or []:
                if item.get("State") in (1, 4):  # 1=待機, 4=一部約定
                    order = self.get_order(str(item.get("Id", "")))
                    if order:
                        orders.append(order)
            return orders
        except Exception as e:
            logger.warning(f"オープン注文取得失敗: {e}")
            return []

    def is_connected(self) -> bool:
        """接続状況を確認する。"""
        try:
            self._request("GET", "/board/1301")  # 任意銘柄で疎通確認
            return True
        except Exception:
            return False

    @staticmethod
    def _parse_order_status(state: Any) -> str:
        """kabuステーションの注文状態を文字列に変換する。"""
        state_map = {
            1: "open",      # 待機
            2: "open",      # 未約定
            3: "filled",    # 全約定
            4: "partial",   # 一部約定
            5: "cancelled", # 取消済
            6: "cancelled", # 失効
        }
        return state_map.get(int(state), "unknown")
