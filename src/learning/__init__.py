"""学習パッケージ — 継続学習・モデル管理・戦略選択モジュール。"""
from __future__ import annotations

from .feature_store import FeatureSet, FeatureStore
from .model_registry import ModelMeta, ModelRegistry
from .pipeline import LearningPipeline, PipelineResult
from .regime_detector import MarketRegime, RegimeDetector, RegimeState
from .retrainer import RetrainResult, RetrainScheduler
from .thompson_bandit import ArmStats, SelectionResult, ThompsonBandit
from .drift_detector import DriftDetector, DriftResult, DriftSeverity
from .ab_test import ABTestManager, ABTestResult

__all__ = [
    # feature_store
    "FeatureSet",
    "FeatureStore",
    # model_registry
    "ModelMeta",
    "ModelRegistry",
    # pipeline
    "LearningPipeline",
    "PipelineResult",
    # regime_detector
    "MarketRegime",
    "RegimeDetector",
    "RegimeState",
    # retrainer
    "RetrainResult",
    "RetrainScheduler",
    # thompson_bandit
    "ArmStats",
    "SelectionResult",
    "ThompsonBandit",
    # drift_detector
    "DriftDetector",
    "DriftResult",
    "DriftSeverity",
    # ab_test
    "ABTestManager",
    "ABTestResult",
]
