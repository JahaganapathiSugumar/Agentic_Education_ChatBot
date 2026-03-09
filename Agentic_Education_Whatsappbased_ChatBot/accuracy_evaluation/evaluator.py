import json
import datetime
from typing import Dict, List
import os

class AccuracyEvaluator:
    def __init__(self):
        self.results = []
        self.test_cases = []
        
    def evaluate_whisper_accuracy(self, audio_file: str, expected_text: str) -> Dict:
        """Evaluate Whisper transcription accuracy"""
        # This would integrate with your existing transcribe_audio function
        actual_text = self.transcribe_test_audio(audio_file)
        
        return {
            'module': 'whisper_stt',
            'test_case': audio_file,
            'expected': expected_text,
            'actual': actual_text,
            'word_error_rate': self.calculate_wer(expected_text, actual_text),
            'accuracy_score': self.calculate_accuracy_score(expected_text, actual_text),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def evaluate_rag_accuracy(self, document: str, question: str, expected_answer: str) -> Dict:
        """Evaluate RAG system accuracy"""
        actual_answer = self.query_rag_system(question, document)
        
        return {
            'module': 'rag_system',
            'test_case': f"{document} - {question}",
            'expected': expected_answer,
            'actual': actual_answer,
            'factual_accuracy': self.score_factual_accuracy(expected_answer, actual_answer),
            'completeness': self.score_completeness(expected_answer, actual_answer),
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def save_results(self, filename: str = None):
        """Save evaluation results to JSON file"""
        if not filename:
            filename = f"accuracy_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_dir = "accuracy_evaluation/results"
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def generate_report(self):
        """Generate comprehensive accuracy report"""
        report = {
            'summary': self.calculate_summary_stats(),
            'module_breakdown': self.breakdown_by_module(),
            'recommendations': self.generate_recommendations(),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        report_dir = "accuracy_evaluation/results/accuracy_reports"
        os.makedirs(report_dir, exist_ok=True)
        
        report_file = os.path.join(report_dir, f"accuracy_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report