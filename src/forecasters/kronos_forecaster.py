"""Kronos ラッパー: 金融市場特化の基盤モデル（AAAI 2026）。

Kronos は K-line (OHLCV) シーケンスを階層的離散トークンに量子化し、
大規模自己回帰 Transformer で事前学習された金融時系列基盤モデル。
45のグローバル取引所、120億以上のK-lineレコードで学習済み。

未インストールの場合は _KRONOS_AVAILABLE = False となり graceful degradation。

参考:
- Paper: https://arxiv.org/abs/2508.02739
- GitHub: https://github.com/shiyu-coder/Kronos
- HuggingFace: https://huggingface.co/NeoQuasar/Kronos-base
"""
from __future__ import annotations

import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd

# kronos は オプション依存
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
    _KRONOS_AVAILABLE = True
except ImportError:
    _KRONOS_AVAILABLE = False
    warnings.warn(
        "transformers/torch が未インストールです。KronosForecaster は使用できません。"
        " pip install transformers torch でインストールしてください。",
        ImportWarning,
        stacklevel=2,
    )


def _quantize_kline(
    ohlcv: pd.DataFrame,
    n_bins: int = 256,
) -> list[list[int]]:
    """OHLCV を Kronos 形式の階層的離散トークンに量子化する。

    Parameters
    ----------
    ohlcv:
        OHLCV DataFrame（カラム: open, high, low, close, volume）
    n_bins:
        量子化ビン数（デフォルト: 256）

    Returns
    -------
    list[list[int]]
        各行が [open_q, high_q, low_q, close_q, vol_q] のトークンリスト
    """
    tokens: list[list[int]] = []
    for col in ["open", "high", "low", "close"]:
        series = ohlcv[col].astype(float)
        min_val, max_val = series.min(), series.max()
        spread = max_val - min_val
        if spread == 0:
            quantized = [n_bins // 2] * len(series)
        else:
            quantized = [
                min(int((v - min_val) / spread * (n_bins - 1)), n_bins - 1)
                for v in series
            ]
        tokens.append(quantized)

    # volume は対数スケールで量子化
    vol = ohlcv["volume"].astype(float).clip(lower=1.0)
    log_vol = np.log(vol)
    min_lv, max_lv = log_vol.min(), log_vol.max()
    spread_lv = max_lv - min_lv
    if spread_lv == 0:
        vol_q = [n_bins // 2] * len(vol)
    else:
        vol_q = [
            min(int((v - min_lv) / spread_lv * (n_bins - 1)), n_bins - 1)
            for v in log_vol
        ]
    tokens.append(vol_q)

    # 転置: (5, N) → (N, 5)
    return [list(row) for row in zip(*tokens)]


class KronosForecaster:
    """Kronos による金融時系列予測器。

    K-line (OHLCV) データを階層的トークンに変換し、
    事前学習済み Transformer で次期価格を予測する。

    Parameters
    ----------
    model_name:
        HuggingFace モデル名（デフォルト: "NeoQuasar/Kronos-base"）
    device:
        推論デバイス（"cpu", "cuda", "auto"）
    n_bins:
        OHLCV 量子化ビン数（デフォルト: 256）
    revision:
        モデルリビジョン（破壊的更新対策にピン可能）
    """

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-base",
        device: str = "auto",
        n_bins: int = 256,
        revision: Optional[str] = None,
    ) -> None:
        if not _KRONOS_AVAILABLE:
            raise ImportError(
                "transformers/torch が未インストールです: "
                "pip install transformers torch"
            )
        self.model_name = model_name
        self.device = device
        self.n_bins = n_bins
        self.revision = revision
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    def _load_model(self) -> tuple[Any, Any]:
        """モデルとトークナイザーをオンデマンドでロード（遅延初期化）。"""
        if self._model is None:
            device_map = self.device if self.device != "auto" else "auto"
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                device_map=device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        return self._model, self._tokenizer

    def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 5,
    ) -> pd.DataFrame:
        """OHLCV データから将来の close 価格を予測する。

        Parameters
        ----------
        data:
            OHLCV DataFrame（カラム: open, high, low, close, volume）
        horizon:
            予測ステップ数（デフォルト: 5）

        Returns
        -------
        pd.DataFrame
            予測結果 DataFrame（カラム: forecast_close, forecast_direction）
            - forecast_close: 予測終値
            - forecast_direction: 1=上昇予測, -1=下落予測, 0=横ばい

        Raises
        ------
        ImportError
            依存パッケージが未インストールの場合。
        ValueError
            data が空または horizon < 1 の場合。
        """
        if not _KRONOS_AVAILABLE:
            raise ImportError(
                "transformers/torch が未インストールです: "
                "pip install transformers torch"
            )
        if data.empty:
            raise ValueError("入力データが空です。")
        if horizon < 1:
            raise ValueError(f"horizon は1以上が必要です。指定値: {horizon}")

        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"必須カラムが不足しています: {missing}")

        model, tokenizer = self._load_model()

        # OHLCV → トークン化
        kline_tokens = _quantize_kline(data, self.n_bins)

        # トークンシーケンスを平坦化して入力
        flat_tokens: list[int] = []
        for row in kline_tokens:
            flat_tokens.extend(row)

        input_ids = torch.tensor([flat_tokens], dtype=torch.long)
        if hasattr(model, "device"):
            input_ids = input_ids.to(model.device)

        # 自己回帰生成: horizon * 5 トークン（OHLCV 5チャネル × horizon ステップ）
        gen_length = horizon * 5
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=gen_length,
                do_sample=True,
                temperature=0.7,
                top_k=50,
            )

        # 生成されたトークンを OHLCV に逆量子化
        generated = output[0, len(flat_tokens):].cpu().tolist()

        # close チャネル（インデックス3）を抽出
        close_min = float(data["close"].min())
        close_max = float(data["close"].max())
        close_spread = close_max - close_min
        if close_spread == 0:
            close_spread = 1.0

        forecasts: list[float] = []
        directions: list[int] = []
        current_price = float(data["close"].iloc[-1])

        for step in range(horizon):
            token_idx = step * 5 + 3  # close = 4番目のチャネル
            if token_idx < len(generated):
                token_val = generated[token_idx]
                # clamp to valid range
                token_val = max(0, min(token_val, self.n_bins - 1))
                predicted_close = close_min + (token_val / (self.n_bins - 1)) * close_spread
            else:
                predicted_close = current_price

            forecasts.append(predicted_close)

            if predicted_close > current_price * 1.001:
                directions.append(1)
            elif predicted_close < current_price * 0.999:
                directions.append(-1)
            else:
                directions.append(0)

            current_price = predicted_close

        return pd.DataFrame({
            "forecast_close": forecasts,
            "forecast_direction": directions,
        })

    def predict_direction(
        self,
        data: pd.DataFrame,
        horizon: int = 5,
    ) -> int:
        """次期の方向性を予測する（集約）。

        Parameters
        ----------
        data:
            OHLCV DataFrame
        horizon:
            予測ステップ数

        Returns
        -------
        int
            1=上昇, -1=下落, 0=横ばい（多数決）
        """
        result = self.predict(data, horizon)
        directions = result["forecast_direction"]
        up_count = int((directions > 0).sum())
        down_count = int((directions < 0).sum())
        if up_count > down_count:
            return 1
        elif down_count > up_count:
            return -1
        return 0
