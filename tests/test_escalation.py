"""
Escalation System Tests
"""

import pytest
from src.escalation import EscalationManager, EscalationRule

class TestEscalation:
    """Test escalation management"""
    
    @pytest.fixture
    def escalation_manager(self, test_config):
        """Create escalation manager"""
        return EscalationManager(test_config)
    
    def test_escalation_initialization(self, escalation_manager):
        """Test escalation manager initialization"""
        assert escalation_manager is not None
        assert len(escalation_manager.rules) > 0
        assert escalation_manager.threshold == 0.7
        assert escalation_manager.min_sentiment == 0.3
    
    def test_escalation_detection_refund(self, escalation_manager):
        """Test refund-related escalation"""
        text = "I want a refund for my subscription"
        sentiment = 0.4
        should_escalate, reason = escalation_manager.check_escalation(text, sentiment)
        
        assert should_escalate is True
        assert "refund" in reason.lower() or "financial" in reason.lower()
    
    def test_escalation_detection_negative_sentiment(self, escalation_manager):
        """Test negative sentiment escalation"""
        text = "This is the worst product I've ever used"
        sentiment = 0.2
        should_escalate, reason = escalation_manager.check_escalation(text, sentiment)
        
        assert should_escalate is True
        assert "sentiment" in reason.lower()
    
    def test_no_escalation_positive_sentiment(self, escalation_manager):
        """Test no escalation with positive sentiment"""
        text = "Great product, just a quick question about features"
        sentiment = 0.8
        should_escalate, reason = escalation_manager.check_escalation(text, sentiment)
        
        assert should_escalate is False
        assert reason == ""
    
    def test_escalation_with_mixed_sentiment(self, escalation_manager):
        """Test escalation with mixed signals"""
        text = "The product is good but I have a billing issue"
        sentiment = 0.5
        should_escalate, reason = escalation_manager.check_escalation(text, sentiment)
        
        # Should escalate due to billing issue
        assert should_escalate is True
        assert "billing" in reason.lower() or "financial" in reason.lower()
    
    def test_escalation_threshold_lowering(self, escalation_manager):
        """Test lowering escalation threshold"""
        original_threshold = escalation_manager.min_sentiment
        escalation_manager.lower_threshold()
        
        assert escalation_manager.min_sentiment < original_threshold
        assert escalation_manager.min_sentiment >= 0.2
    
    def test_escalation_threshold_reset(self, escalation_manager):
        """Test resetting escalation threshold"""
        escalation_manager.lower_threshold()
        escalation_manager.reset_threshold()
        
        assert escalation_manager.min_sentiment == 0.3
    
    def test_escalation_history_logging(self, escalation_manager):
        """Test escalation history logging"""
        case_id = "CS-TEST-123"
        details = {
            'query': 'This product is broken',
            'sentiment': 0.2,
            'reason': 'Negative sentiment'
        }
        
        escalation_manager.log_escalation(case_id, details)
        
        assert case_id in escalation_manager.escalation_history
        assert escalation_manager.escalation_history[case_id]['details'] == details
    
    def test_escalation_stats(self, escalation_manager):
        """Test escalation statistics"""
        # Add some escalations
        for i in range(3):
            escalation_manager.log_escalation(f"CS-{i}", {'test': True})
        
        stats = escalation_manager.get_escalation_stats()
        assert stats['total_escalations'] == 3
        assert 'average_resolution_time' in stats
        assert 'escalation_rate' in stats
    
    def test_rule_pattern_matching(self, escalation_manager):
        """Test escalation rule pattern matching"""
        # Test each rule pattern
        test_cases = [
            ("I want a refund", True),
            ("This is terrible service", True),
            ("My data was lost", True),
            ("How do I reset my password?", False),
            ("What are your features?", False)
        ]
        
        for text, expected in test_cases:
            should_escalate, _ = escalation_manager.check_escalation(text, 0.3)
            assert should_escalate == expected, f"Failed for: {text}"
    
    def test_escalation_priority(self, escalation_manager):
        """Test escalation priority assignment"""
        # Find highest priority rule
        max_priority = max(rule.priority for rule in escalation_manager.rules)
        assert max_priority == 5  # Security/legal issues
        
        # Test highest priority trigger
        text = "I need to talk to your legal team about a lawsuit"
        sentiment = 0.3
        should_escalate, reason = escalation_manager.check_escalation(text, sentiment)
        
        assert should_escalate is True
        assert "legal" in reason.lower() or "threat" in reason.lower()