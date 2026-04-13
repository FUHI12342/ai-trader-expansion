"""LightGBM 方向予測戦略。

特徴量: SMA(5,20,60), RSI(14), MACD, BB位置, 出来高変化率, 曜日
ラベル: 翌日リターン > 0 → 1, else 0
Walk-Forward方式でtrain/predict分離（未来データ漏洩を防止）
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import BaseStrategy, SignalType
from .macd_rsi import _ema, _rsi

# LightGBMはオプション依存
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    warnings.warn("LightGBMが未インストールです。LGBMPredictorStrategyは使用できません。", ImportWarning)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を構築する。

    Parameters
    ----------
    df:
        OHLCV DataFrame

    Returns
    -------
    pd.DataFrame
        特徴量DataFrame（NaN行あり）
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    feat = pd.DataFrame(index=df.index)

    # SMA（短・中・長）
    feat["sma_5"] = close.rolling(5).mean()
    feat["sma_20"] = close.rolling(20).mean()
    feat["sma_60"] = close.rolling(60).mean()

    # SMA比率（価格の相対位置）
    feat["price_sma5_ratio"] = close / feat["sma_5"].replace(0, np.nan)
    feat["price_sma20_ratio"] = close / feat["sma_20"].replace(0, np.nan)
    feat["price_sma60_ratio"] = close / feat["sma_60"].replace(0, np.nan)

    # RSI
    feat["rsi"] = _rsi(close, 14)

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    feat["macd"] = macd_line
    feat["macd_hist"] = macd_line - signal_line

    # ボリンジャーバンド位置（0〜1: 0=下バンド, 1=上バンド）
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    feat["bb_position"] = (close - bb_lower) / bb_range

    # 出来高変化率
    feat["volume_change"] = volume.pct_change().clip(-5, 5)

    # 曜日（0=月曜, 4=金曜）
    if isinstance(df.index, pd.DatetimeIndex):
        feat["day_of_week"] = df.index.dayofweek.astype(float)
    else:
        feat["day_of_week"] = 0.0

    # 過去リターン（モメンタム特徴量）
    feat["return_1d"] = close.pct_change(1)
    feat["return_5d"] = close.pct_change(5)
    feat["return_20d"] = close.pct_change(20)

    # 高値/安値比率
    feat["hl_ratio"] = (high - low) / close.replace(0, np.nan)

    return feat


class LGBMPredictorStrategy(BaseStrategy):
    """LightGBM方向予測戦略。

    Parameters
    ----------
    train_window:
        学習データ日数（デフォルト: 504営業日 ≈ 2年）
    predict_window:
        予測（OOS）データ日数（デフォルト: 126営業日 ≈ 半年）
    min_train_samples:
        学習に必要な最低サンプル数（デフォルト: 100）
    prob_threshold:
        BUYシグナルを発生させる確率閾値（デフォルト: 0.55）
    lgbm_params:
        LightGBMハイパーパラメータ（省略可）
    """

    name = "lgbm_predictor"

    def __init__(
        self,
        train_window: int = 504,
        predict_window: int = 126,
        min_train_samples: int = 100,
        prob_threshold: float = 0.55,
        lgbm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not HAS_LGB:
            raise ImportError("LightGBMが未インストールです: pip install lightgbm")
        self.train_window = train_window
        self.predict_window = predict_window
        self.min_train_samples = min_train_samples
        self.prob_threshold = prob_threshold
        self.lgbm_params: Dict[str, Any] = lgbm_params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "random_state": 42,
            "verbose": -1,
        }

    def parameter_space(self) -> Dict[str, tuple]:
        """最適化パラメータ空間を返す。"""
        return {
            "n_estimators": (int, 50, 500),
            "learning_rate": (float, 0.01, 0.3),
            "train_window": (int, 126, 504),
        }

    def _min_bars(self) -> int:
        return self.train_window + self.predict_window + 61  # 特徴量のウォームアップ

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Walk-Forward方式でLGBMシグナルを生成する。

        Parameters
        ----------
        data:
            OHLCV DataFrame

        Returns
        -------
        pd.Series
            シグナル系列（1=BUY, -1=SELL, 0=FLAT）
        """
        self.validate_data(data)
        df = data.copy()

        # 特徴量構築
        features = _build_features(df)

        # ラベル: 翌日終値リターン > 0 → 1, else 0
        close = df["close"].astype(float)
        next_return = close.pct_change(1).shift(-1)  # 翌日リターン
        labels = (next_return > 0).astype(int)

        # シグナル初期化
        signals = pd.Series(SignalType.FLAT, index=df.index, dtype=int)

        total_len = len(df)
        start_idx = 0

        while start_idx + self.train_window + self.predict_window <= total_len:
            train_end = start_idx + self.train_window
            pred_end = min(train_end + self.predict_window, total_len)

            # 学習データ
            x_train = features.iloc[start_idx:train_end].copy()
            y_train = labels.iloc[start_idx:train_end].copy()

            # NaN除去
            valid_mask = x_train.notna().all(axis=1) & y_train.notna()
            x_train = x_train[valid_mask]
            y_train = y_train[valid_mask]

            if len(x_train) < self.min_train_samples:
                start_idx += self.predict_window
                continue

            # LightGBMモデル学習
            model = lgb.LGBMClassifier(**self.lgbm_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_train, y_train)

            # 予測データ
            x_pred = features.iloc[train_end:pred_end].copy()
            pred_valid = x_pred.notna().all(axis=1)
            x_pred_clean = x_pred[pred_valid]

            if len(x_pred_clean) > 0:
                proba = model.predict_proba(x_pred_clean)[:, 1]
                pred_idx = x_pred_clean.index

                # 確率閾値でシグナル生成
                for i, (idx, p) in enumerate(zip(pred_idx, proba)):
                    if p >= self.prob_threshold:
                        signals[idx] = SignalType.BUY
                    elif p <= (1 - self.prob_threshold):
                        signals[idx] = SignalType.SELL

            start_idx += self.predict_window

        return signals
