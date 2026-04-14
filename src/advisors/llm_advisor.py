"""LLMアドバイザー — Claude APIによる市場分析・取引判断。

テクニカル戦略のシグナルに対してLLMが「確認/拒否」する
ハイブリッドアーキテクチャの中核モジュール。

アーキテクチャ:
  テクニカル戦略 (BUY/SELL/FLAT)
       ↓
  LLMAdvisor.evaluate() — ニュース/センチメント/文脈を分析
       ↓
  アンサンブル投票 (両方一致なら実行、不一致ならFLAT)
       ↓
  リスクゲート (DrawdownController/CircuitBreaker)
       ↓
  OMS → ブローカー
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic  # type: ignore[import-not-found]
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass(frozen=True)
class LLMAdvice:
    """LLMの取引アドバイス（immutable）。

    Parameters
    ----------
    action:
        推奨アクション ("BUY", "SELL", "FLAT")
    confidence:
        確信度 (0.0〜1.0)
    reasoning:
        判断根拠
    model:
        使用モデル名
    raw_response:
        LLM生の応答テキスト
    """

    action: str
    confidence: float
    reasoning: str
    model: str = ""
    raw_response: str = ""

    @property
    def signal(self) -> int:
        """SignalType互換の整数値を返す。"""
        if self.action == "BUY":
            return 1
        elif self.action == "SELL":
            return -1
        return 0


class LLMAdvisor:
    """Claude APIベースの取引アドバイザー。

    テクニカル戦略のシグナルと市場コンテキストを渡し、
    Claude が最終判断を返す。

    Parameters
    ----------
    model:
        使用するClaudeモデル (デフォルト: claude-sonnet-4-6)
    api_key:
        Anthropic APIキー (省略時は環境変数 ANTHROPIC_API_KEY)
    confidence_threshold:
        この確信度以上の場合のみシグナルを発行 (デフォルト: 0.6)
    max_tokens:
        応答の最大トークン数 (デフォルト: 500)
    """

    SYSTEM_PROMPT = """あなたは経験豊富なクオンツトレーダーです。
市場データとテクニカル指標を分析し、取引判断を行います。

## ルール
1. リスク管理を最優先にする
2. 確信度が低い場合は FLAT (様子見) を推奨する
3. 判断根拠を簡潔に説明する
4. ニュースや市場環境を考慮する

## 出力形式
必ず以下のJSON形式で回答してください:
{"action": "BUY"|"SELL"|"FLAT", "confidence": 0.0-1.0, "reasoning": "判断根拠"}
"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        confidence_threshold: float = 0.6,
        max_tokens: int = 500,
    ) -> None:
        self._model = model
        self._confidence_threshold = confidence_threshold
        self._max_tokens = max_tokens
        self._call_count = 0
        self._client: Optional[Any] = None

        if HAS_ANTHROPIC:
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if key:
                self._client = Anthropic(api_key=key)

    @property
    def is_available(self) -> bool:
        """APIクライアントが利用可能か。"""
        return self._client is not None

    @property
    def call_count(self) -> int:
        """API呼び出し回数。"""
        return self._call_count

    def build_context(
        self,
        symbol: str,
        close: float,
        change_pct: float,
        rsi: float = 50.0,
        macd: float = 0.0,
        bb_position: float = 0.5,
        volume_ratio: float = 1.0,
        technical_signal: int = 0,
        portfolio_dd: float = 0.0,
        exposure_pct: float = 0.0,
        news_summary: str = "特になし",
    ) -> str:
        """LLMに渡すコンテキスト文字列を構築する。"""
        sig_label = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(technical_signal, "FLAT")

        return f"""## 市場状況

### {symbol}
- 終値: {close:.2f} (前日比: {change_pct:+.2f}%)
- RSI(14): {rsi:.1f}
- MACD: {macd:.4f}
- ボリンジャーバンド位置: {bb_position:.2f} (0=下限, 1=上限)
- 出来高倍率 (20日平均比): {volume_ratio:.1f}x

### テクニカル戦略シグナル: {sig_label}

### ポートフォリオ
- 現在ドローダウン: {portfolio_dd:.1f}%
- エクスポージャー: {exposure_pct:.0f}%

### ニュース
{news_summary}

上記を分析し、取引判断をJSON形式で回答してください。"""

    def evaluate(
        self,
        context: str,
        technical_signal: int = 0,
    ) -> LLMAdvice:
        """市場コンテキストを分析し、取引アドバイスを返す。

        Parameters
        ----------
        context:
            build_context() で生成した文字列
        technical_signal:
            テクニカル戦略のシグナル (1=BUY, -1=SELL, 0=FLAT)

        Returns
        -------
        LLMAdvice
            LLMの判断結果。API不可の場合はテクニカルシグナルをそのまま返す
        """
        if not self.is_available:
            return self._fallback(technical_signal)

        try:
            self._call_count += 1
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )

            raw = response.content[0].text
            advice = self._parse_response(raw)
            return LLMAdvice(
                action=advice["action"],
                confidence=advice["confidence"],
                reasoning=advice["reasoning"],
                model=self._model,
                raw_response=raw,
            )

        except Exception as e:
            logger.warning("LLM呼び出し失敗: %s — テクニカルにフォールバック", e)
            return self._fallback(technical_signal)

    def ensemble(
        self,
        technical_signal: int,
        llm_advice: LLMAdvice,
    ) -> int:
        """テクニカルシグナルとLLMアドバイスのアンサンブル投票。

        ルール:
        - 両方が同じ方向 → その方向で実行
        - 片方がFLAT → もう片方に従う（確信度が閾値以上の場合）
        - 反対方向 → FLAT (矛盾する場合は見送り)

        Parameters
        ----------
        technical_signal:
            テクニカル戦略シグナル (1/-1/0)
        llm_advice:
            LLMの判断結果

        Returns
        -------
        int
            最終シグナル (1=BUY, -1=SELL, 0=FLAT)
        """
        llm_signal = llm_advice.signal
        confidence = llm_advice.confidence

        # 両方一致 → 実行
        if technical_signal == llm_signal and technical_signal != 0:
            return technical_signal

        # テクニカルがシグナルあり、LLMがFLAT → テクニカルに従う（確信度低い = 拒否しない）
        if technical_signal != 0 and llm_signal == 0:
            if confidence < self._confidence_threshold:
                return technical_signal  # LLMが確信を持ってない → テクニカルを信頼
            return 0  # LLMが確信を持ってFLAT → 見送り

        # テクニカルがFLAT、LLMがシグナル → LLMの確信度が高ければ従う
        if technical_signal == 0 and llm_signal != 0:
            if confidence >= self._confidence_threshold:
                return llm_signal
            return 0

        # 反対方向 → FLAT (安全側)
        if technical_signal != 0 and llm_signal != 0 and technical_signal != llm_signal:
            logger.info(
                "テクニカル(%d) vs LLM(%d) 矛盾 → FLAT",
                technical_signal, llm_signal,
            )
            return 0

        return 0

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """LLM応答からJSON部分を抽出・パースする。"""
        # JSONブロックを探す
        json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                action = str(parsed.get("action", "FLAT")).upper()
                if action not in ("BUY", "SELL", "FLAT"):
                    action = "FLAT"
                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                reasoning = str(parsed.get("reasoning", ""))
                return {"action": action, "confidence": confidence, "reasoning": reasoning}
            except (json.JSONDecodeError, ValueError):
                pass

        return {"action": "FLAT", "confidence": 0.0, "reasoning": "応答パース失敗"}

    def _fallback(self, technical_signal: int) -> LLMAdvice:
        """API不可時のフォールバック: テクニカルシグナルをそのまま返す。"""
        action = {1: "BUY", -1: "SELL", 0: "FLAT"}.get(technical_signal, "FLAT")
        return LLMAdvice(
            action=action,
            confidence=0.5,
            reasoning="LLM不可 — テクニカルシグナルにフォールバック",
            model="fallback",
        )
