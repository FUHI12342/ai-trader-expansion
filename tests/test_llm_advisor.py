"""LLMAdvisor のテスト。"""
from __future__ import annotations

import dataclasses

import pytest

from src.advisors.llm_advisor import LLMAdvisor, LLMAdvice


class TestLLMAdvice:

    def test_is_frozen(self) -> None:
        a = LLMAdvice(action="BUY", confidence=0.8, reasoning="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            a.action = "SELL"  # type: ignore[misc]

    def test_signal_buy(self) -> None:
        assert LLMAdvice(action="BUY", confidence=0.9, reasoning="").signal == 1

    def test_signal_sell(self) -> None:
        assert LLMAdvice(action="SELL", confidence=0.9, reasoning="").signal == -1

    def test_signal_flat(self) -> None:
        assert LLMAdvice(action="FLAT", confidence=0.5, reasoning="").signal == 0


class TestLLMAdvisor:

    def test_init_without_api_key(self) -> None:
        advisor = LLMAdvisor(api_key="")
        assert not advisor.is_available

    def test_fallback_when_unavailable(self) -> None:
        advisor = LLMAdvisor(api_key="")
        result = advisor.evaluate("test context", technical_signal=1)
        assert result.action == "BUY"
        assert result.model == "fallback"

    def test_build_context(self) -> None:
        advisor = LLMAdvisor(api_key="")
        ctx = advisor.build_context(
            symbol="7203.T", close=3300.0, change_pct=1.5,
            rsi=65.0, technical_signal=1,
        )
        assert "7203.T" in ctx
        assert "3300.00" in ctx
        assert "BUY" in ctx

    def test_parse_valid_json(self) -> None:
        advisor = LLMAdvisor(api_key="")
        result = advisor._parse_response(
            '分析結果: {"action": "BUY", "confidence": 0.85, "reasoning": "上昇トレンド"}'
        )
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.85

    def test_parse_invalid_json(self) -> None:
        advisor = LLMAdvisor(api_key="")
        result = advisor._parse_response("これはJSONではない")
        assert result["action"] == "FLAT"
        assert result["confidence"] == 0.0

    def test_parse_clamps_confidence(self) -> None:
        advisor = LLMAdvisor(api_key="")
        result = advisor._parse_response('{"action":"BUY","confidence":1.5,"reasoning":"x"}')
        assert result["confidence"] == 1.0

    def test_parse_normalizes_action(self) -> None:
        advisor = LLMAdvisor(api_key="")
        result = advisor._parse_response('{"action":"buy","confidence":0.7,"reasoning":"x"}')
        assert result["action"] == "BUY"


class TestEnsemble:

    def _advisor(self) -> LLMAdvisor:
        return LLMAdvisor(api_key="", confidence_threshold=0.6)

    def test_both_buy(self) -> None:
        a = self._advisor()
        advice = LLMAdvice(action="BUY", confidence=0.9, reasoning="")
        assert a.ensemble(1, advice) == 1

    def test_both_sell(self) -> None:
        a = self._advisor()
        advice = LLMAdvice(action="SELL", confidence=0.9, reasoning="")
        assert a.ensemble(-1, advice) == -1

    def test_contradiction_becomes_flat(self) -> None:
        a = self._advisor()
        advice = LLMAdvice(action="SELL", confidence=0.9, reasoning="")
        assert a.ensemble(1, advice) == 0  # BUY vs SELL → FLAT

    def test_tech_buy_llm_flat_low_confidence(self) -> None:
        a = self._advisor()
        advice = LLMAdvice(action="FLAT", confidence=0.3, reasoning="")
        assert a.ensemble(1, advice) == 1  # LLM不確実 → テクニカルに従う

    def test_tech_buy_llm_flat_high_confidence(self) -> None:
        a = self._advisor()
        advice = LLMAdvice(action="FLAT", confidence=0.8, reasoning="リスク高い")
        assert a.ensemble(1, advice) == 0  # LLMが確信を持って拒否

    def test_tech_flat_llm_buy_high_confidence(self) -> None:
        a = self._advisor()
        advice = LLMAdvice(action="BUY", confidence=0.9, reasoning="好材料")
        assert a.ensemble(0, advice) == 1  # LLMの独自判断

    def test_tech_flat_llm_buy_low_confidence(self) -> None:
        a = self._advisor()
        advice = LLMAdvice(action="BUY", confidence=0.4, reasoning="微妙")
        assert a.ensemble(0, advice) == 0  # 確信度不足 → FLAT

    def test_both_flat(self) -> None:
        a = self._advisor()
        advice = LLMAdvice(action="FLAT", confidence=0.5, reasoning="")
        assert a.ensemble(0, advice) == 0
