import re
from typing import List, Tuple

class MetricsCalculator:
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Simple implementation - you might want a more robust WER calculation
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        
        errors = 0
        min_len = min(len(ref_words), len(hyp_words))
        
        for i in range(min_len):
            if ref_words[i] != hyp_words[i]:
                errors += 1
        
        errors += abs(len(ref_words) - len(hyp_words))
        return errors / len(ref_words)
    
    @staticmethod
    def calculate_accuracy_score(expected: str, actual: str) -> float:
        """Calculate overall accuracy score (0-100)"""
        wer = MetricsCalculator.calculate_wer(expected, actual)
        return max(0, (1 - wer) * 100)
    
    @staticmethod
    def score_factual_accuracy(expected: str, actual: str) -> int:
        """Manual scoring of factual accuracy (0-10)"""
        # This requires manual evaluation
        # You would compare key facts between expected and actual
        return 0  # Placeholder - requires manual input
    
    @staticmethod
    def score_completeness(expected: str, actual: str) -> int:
        """Manual scoring of answer completeness (0-10)"""
        # This requires manual evaluation
        return 0  # Placeholder - requires manual input

    @staticmethod
    def calculate_module_average(scores: List[float]) -> Dict:
        """Calculate average metrics for a module"""
        if not scores:
            return {'average': 0, 'count': 0, 'min': 0, 'max': 0}
        
        return {
            'average': sum(scores) / len(scores),
            'count': len(scores),
            'min': min(scores),
            'max': max(scores)
        }