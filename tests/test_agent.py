"""
Test Agent Functionality
"""

import pytest
from src.agent import CustomerSupportAgent
from src.knowledge_base import KnowledgeBaseManager
from src.escalation import EscalationManager

class TestAgent:
    """Test customer support agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance"""
        return CustomerSupportAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent is not None
        assert agent.llm is not None
        assert agent.vectorstore is not None
        assert agent.graph is not None
    
    def test_query_processing(self, agent):
        """Test query processing"""
        result = agent.process_query("How do I reset my password?")
        assert result['success'] is True
        assert 'response' in result
        assert len(result['response']) > 0
    
    def test_escalation_detection(self, agent):
        """Test escalation detection"""
        # Test query that should escalate
        result = agent.process_query("I want a refund, this product is broken")
        assert result['escalated'] is True
        assert result['escalation_reason'] is not None
    
    def test_knowledge_retrieval(self, agent):
        """Test knowledge retrieval"""
        state = {
            "user_input": "What is CloudSync Pro?",
            "messages": []
        }
        result = agent.retrieve_context(state)
        assert 'retrieved_context' in result
        assert len(result['retrieved_context']) > 0
    
    def test_sentiment_analysis(self, agent):
        """Test sentiment analysis"""
        # Positive sentiment
        state = {"user_input": "I love this product! It's amazing!"}
        result = agent.analyze_sentiment(state)
        assert result['sentiment_score'] > 0.6
        
        # Negative sentiment
        state = {"user_input": "This is terrible, I'm very disappointed"}
        result = agent.analyze_sentiment(state)
        assert result['sentiment_score'] < 0.4
    
    def test_conversation_memory(self, agent):
        """Test conversation memory"""
        user_id = "test_user_123"
        
        # Save conversation
        agent.process_query("Hello", user_id)
        agent.process_query("How are you?", user_id)
        
        # Get history
        history = agent.get_conversation_history(user_id)
        assert len(history) == 2