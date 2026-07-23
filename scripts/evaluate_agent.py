"""
Agent Evaluation Script
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import CustomerSupportAgent
from src.utils import logger

class AgentEvaluator:
    """Evaluate agent performance"""
    
    def __init__(self):
        self.agent = CustomerSupportAgent()
        self.results = []
    
    def load_test_cases(self, file_path: str) -> List[Dict]:
        """Load test cases from CSV"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test cases file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    
    def evaluate_query(self, test_case: Dict) -> Dict:
        """Evaluate single query"""
        query = test_case.get('query', '')
        expected_escalation = test_case.get('expected_escalation', False)
        expected_contains = test_case.get('expected_contains', '')
        
        try:
            result = self.agent.process_query(query)
            
            # Check escalation
            escalation_match = result.get('escalated') == expected_escalation
            
            # Check response content
            content_match = True
            if expected_contains:
                content_match = expected_contains.lower() in result.get('response', '').lower()
            
            return {
                'query': query,
                'expected_escalation': expected_escalation,
                'actual_escalation': result.get('escalated'),
                'escalation_match': escalation_match,
                'expected_contains': expected_contains,
                'content_match': content_match,
                'response': result.get('response', ''),
                'success': result.get('success', False),
                'sentiment': result.get('sentiment', 0.5)
            }
            
        except Exception as e:
            return {
                'query': query,
                'error': str(e),
                'success': False
            }
    
    def run_evaluation(self, test_cases: List[Dict]) -> Dict:
        """Run full evaluation"""
        logger.info(f"🚀 Running evaluation on {len(test_cases)} test cases...")
        
        results = []
        for i, test_case in enumerate(test_cases):
            logger.info(f"  Testing case {i+1}/{len(test_cases)}")
            result = self.evaluate_query(test_case)
            results.append(result)
        
        self.results = results
        
        # Calculate metrics
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        escalation_matches = sum(1 for r in results if r.get('escalation_match', False))
        content_matches = sum(1 for r in results if r.get('content_match', False))
        
        metrics = {
            'total_tests': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0,
            'escalation_accuracy': escalation_matches / total if total > 0 else 0,
            'content_accuracy': content_matches / total if total > 0 else 0,
            'avg_sentiment': sum(r.get('sentiment', 0) for r in results) / total if total > 0 else 0
        }
        
        return metrics
    
    def save_results(self, output_path: str):
        """Save evaluation results"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save raw results
        with open(output_path, 'w') as f:
            json.dump({
                'results': self.results,
                'timestamp': str(pd.Timestamp.now())
            }, f, indent=2)
        
        logger.info(f"✅ Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent")
    parser.add_argument(
        "--test-cases",
        required=True,
        help="Path to test cases CSV file"
    )
    parser.add_argument(
        "--output",
        default="data/evaluation_results.json",
        help="Output file path"
    )
    args = parser.parse_args()
    
    evaluator = AgentEvaluator()
    
    # Load test cases
    test_cases = evaluator.load_test_cases(args.test_cases)
    
    # Run evaluation
    metrics = evaluator.run_evaluation(test_cases)
    
    # Print results
    print("\n📊 Evaluation Results:")
    print("=" * 50)
    print(f"Total Tests: {metrics['total_tests']}")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Escalation Accuracy: {metrics['escalation_accuracy']:.2%}")
    print(f"Content Accuracy: {metrics['content_accuracy']:.2%}")
    print(f"Average Sentiment: {metrics['avg_sentiment']:.2f}")
    print("=" * 50)
    
    # Save results
    evaluator.save_results(args.output)

if __name__ == "__main__":
    main()