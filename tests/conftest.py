"""
Pytest Configuration and Fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from src.agent import CustomerSupportAgent
from src.knowledge_base import KnowledgeBaseManager
from src.escalation import EscalationManager

@pytest.fixture(scope="session")
def test_config():
    """Test configuration"""
    return {
        'llm': {
            'model': 'gpt-4o-mini',
            'temperature': 0.2,
            'max_tokens': 100
        },
        'embeddings': {
            'model': 'text-embedding-ada-002'
        },
        'knowledge_base': {
            'path': 'knowledge_base',
            'chunk_size': 100,
            'chunk_overlap': 20
        },
        'vectorstore': {
            'path': 'data/vectorstore'
        },
        'escalation': {
            'threshold': 0.7,
            'min_sentiment': 0.3
        },
        'retrieval': {
            'similarity_threshold': 0.5
        }
    }

@pytest.fixture
def temp_kb_dir():
    """Create temporary knowledge base directory"""
    temp_dir = tempfile.mkdtemp()
    kb_path = Path(temp_dir) / "knowledge_base"
    kb_path.mkdir(parents=True)
    
    # Create sample KB files
    (kb_path / "sample.txt").write_text("This is a sample knowledge base document.")
    (kb_path / "support").mkdir()
    (kb_path / "support" / "help.txt").write_text("Help documentation for testing.")
    
    yield str(kb_path)
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = Mock()
    mock.invoke.return_value.content = "This is a test response from the LLM."
    return mock

@pytest.fixture
def agent_with_mock_llm(monkeypatch, test_config):
    """Agent with mocked LLM"""
    with patch('src.agent.ChatOpenAI') as mock_llm_class:
        mock_llm = Mock()
        mock_llm.invoke.return_value.content = "Mock response from LLM"
        mock_llm_class.return_value = mock_llm
        
        # Mock embeddings
        with patch('src.agent.OpenAIEmbeddings'):
            agent = CustomerSupportAgent()
            agent.llm = mock_llm
            yield agent

@pytest.fixture
def sample_conversations():
    """Sample conversation data"""
    return [
        {
            'user_id': 'test_user',
            'session_id': 'session1',
            'user_input': 'How do I reset my password?',
            'response': 'You can reset your password by clicking "Forgot Password" on the login page.',
            'escalated': False,
            'sentiment': 0.7
        },
        {
            'user_id': 'test_user',
            'session_id': 'session2',
            'user_input': 'I need a refund, this product is broken!',
            'response': 'I understand your frustration. Let me escalate this to a specialist.',
            'escalated': True,
            'sentiment': 0.2
        }
    ]

@pytest.fixture
def escalation_rules():
    """Sample escalation rules"""
    return [
        {
            'pattern': r'\b(refund|money back)\b',
            'description': 'Financial request',
            'priority': 4,
            'required_sentiment': 0.4
        },
        {
            'pattern': r'\b(furious|terrible|worst)\b',
            'description': 'Negative sentiment',
            'priority': 4,
            'required_sentiment': 0.2
        }
    ]