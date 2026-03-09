# Make it a proper Python package
from .evaluator import AccuracyEvaluator
from .test_cases import get_all_test_cases
from .metrics_calculator import MetricsCalculator

__all__ = ['AccuracyEvaluator', 'get_all_test_cases', 'MetricsCalculator']