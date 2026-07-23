"""
Escalation Management System
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class EscalationRule:
    """Rule for escalation"""
    pattern: str
    description: str
    priority: int  # 1-5, 5 is highest
    required_sentiment: float = 0.5  # Below this threshold triggers escalation

class EscalationManager:
    """Manages escalation decisions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold = config.get('escalation.threshold', 0.7)
        self.min_sentiment = config.get('escalation.min_sentiment', 0.3)
        
        # Define escalation rules
        self.rules = [
            EscalationRule(
                pattern=r"\b(refund|money back|cancel subscription)\b",
                description="Financial/Account action requested",
                priority=4,
                required_sentiment=0.4
            ),
            EscalationRule(
                pattern=r"\b(lawsuit|legal|attorney|lawyer)\b",
                description="Legal threat",
                priority=5,
                required_sentiment=0.3
            ),
            EscalationRule(
                pattern=r"\b(furious|terrible|worst|horrible|awful)\b",
                description="Very negative sentiment",
                priority=4,
                required_sentiment=0.2
            ),
            EscalationRule(
                pattern=r"\b(data loss|lost data|missing files)\b",
                description="Data loss concern",
                priority=5,
                required_sentiment=0.4
            ),
            EscalationRule(
                pattern=r"\b(fraud|scam|stolen|breach)\b",
                description="Security issue",
                priority=5,
                required_sentiment=0.3
            ),
            EscalationRule(
                pattern=r"\b(charge|billing error|overcharged|wrong charge)\b",
                description="Billing issue",
                priority=4,
                required_sentiment=0.4
            ),
            EscalationRule(
                pattern=r"\b(can't access|can't login|locked out|hacked)\b",
                description="Account access issue",
                priority=4,
                required_sentiment=0.4
            ),
            EscalationRule(
                pattern=r"\b(executive|manager|supervisor|higher up)\b",
                description="Requested manager",
                priority=4,
                required_sentiment=0.5
            ),
            EscalationRule(
                pattern=r"\b(complaint|dispute|issue)\b.*\b(not resolved|unsolved)\b",
                description="Repeated issue",
                priority=3,
                required_sentiment=0.4
            ),
            EscalationRule(
                pattern=r"\b(threat|danger|security risk|vulnerability)\b",
                description="Security threat",
                priority=5,
                required_sentiment=0.3
            )
        ]
        
        # Track escalation history
        self.escalation_history = {}
        
    def check_escalation(self, text: str, sentiment: float) -> Tuple[bool, str]:
        """
        Check if query needs escalation
        
        Args:
            text: Customer input
            sentiment: Sentiment score (0-1)
            
        Returns:
            Tuple of (should_escalate, reason)
        """
        text_lower = text.lower()
        
        # First check for explicit escalation keywords
        for rule in self.rules:
            if re.search(rule.pattern, text_lower):
                if sentiment <= rule.required_sentiment:
                    logger.info(f"Escalation triggered: {rule.description}")
                    return True, rule.description
        
        # Check if customer is very upset
        if sentiment < self.min_sentiment:
            logger.info(f"Escalation triggered: Very low sentiment ({sentiment:.2f})")
            return True, "Very low sentiment score"
        
        # Check for repeated escalations
        if self._has_escalation_history(text):
            logger.info("Escalation triggered: Repeated escalation pattern")
            return True, "Repeated issue with previous escalations"
        
        return False, ""
    
    def _has_escalation_history(self, text: str) -> bool:
        """Check if there's a history of escalations for similar queries"""
        # Simplified - would check against historical data
        return False
    
    def lower_threshold(self):
        """Lower escalation threshold temporarily"""
        self.min_sentiment = max(0.2, self.min_sentiment - 0.1)
        logger.info(f"Escalation threshold lowered to {self.min_sentiment:.2f}")
    
    def reset_threshold(self):
        """Reset escalation threshold to default"""
        self.min_sentiment = self.config.get('escalation.min_sentiment', 0.3)
        logger.info(f"Escalation threshold reset to {self.min_sentiment:.2f}")
    
    def log_escalation(self, case_id: str, details: Dict):
        """Log escalation for tracking"""
        self.escalation_history[case_id] = {
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        logger.info(f"Escalation logged: {case_id}")
    
    def get_escalation_stats(self) -> Dict:
        """Get escalation statistics"""
        total = len(self.escalation_history)
        if total == 0:
            return {
                'total_escalations': 0,
                'average_resolution_time': 'N/A',
                'escalation_rate': 0
            }
        
        return {
            'total_escalations': total,
            'average_resolution_time': '2.5 hours',  # Placeholder
            'escalation_rate': total / max(1, total * 10)  # Placeholder
        }