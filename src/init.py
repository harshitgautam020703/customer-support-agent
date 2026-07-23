"""
Customer Support Agent with RAG - Source Package
"""

from .agent import CustomerSupportAgent
from .knowledge_base import KnowledgeBaseManager
from .escalation import EscalationManager
from .memory import ConversationMemory
from .utils import load_config, logger, format_response

__all__ = [
    'CustomerSupportAgent',
    'KnowledgeBaseManager',
    'EscalationManager',
    'ConversationMemory',
    'load_config',
    'logger',
    'format_response'
]

__version__ = '1.0.0'