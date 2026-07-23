"""
RAG System Tests
"""

import pytest
from unittest.mock import Mock, patch
from src.knowledge_base import KnowledgeBaseManager
from src.agent import CustomerSupportAgent

class TestRAGSystem:
    """Test RAG functionality"""
    
    def test_knowledge_base_loading(self, temp_kb_dir, test_config):
        """Test knowledge base loading"""
        test_config['knowledge_base']['path'] = temp_kb_dir
        kb_manager = KnowledgeBaseManager(test_config)
        documents = kb_manager.load_documents()
        
        assert documents is not None
        assert len(documents) > 0
        
    def test_vectorstore_building(self, temp_kb_dir, test_config):
        """Test vector store building"""
        test_config['knowledge_base']['path'] = temp_kb_dir
        kb_manager = KnowledgeBaseManager(test_config)
        documents = kb_manager.load_documents()
        
        with patch('src.knowledge_base.FAISS.from_documents') as mock_faiss:
            mock_faiss.return_value = Mock()
            vectorstore = kb_manager.build_vectorstore(documents)
            assert vectorstore is not None
    
    def test_context_retrieval(self, agent_with_mock_llm):
        """Test context retrieval"""
        agent = agent_with_mock_llm
        
        # Create test state
        state = {
            'user_input': 'How do I reset my password?',
            'messages': []
        }
        
        # Mock vectorstore
        mock_doc = Mock()
        mock_doc.page_content = "Password reset instructions: Click Forgot Password on login page."
        agent.vectorstore.similarity_search_with_score = Mock(return_value=[(mock_doc, 0.8)])
        
        result = agent.retrieve_context(state)
        assert 'retrieved_context' in result
        assert len(result['retrieved_context']) > 0
        
    def test_similarity_threshold(self, agent_with_mock_llm):
        """Test similarity threshold filtering"""
        agent = agent_with_mock_llm
        
        state = {
            'user_input': 'random query',
            'messages': []
        }
        
        # Mock low similarity score
        mock_doc = Mock()
        mock_doc.page_content = "Some content"
        agent.vectorstore.similarity_search_with_score = Mock(return_value=[(mock_doc, 0.3)])
        
        result = agent.retrieve_context(state)
        # Should fall back to general context
        assert 'retrieved_context' in result
        
    def test_fallback_context(self, temp_kb_dir, test_config):
        """Test fallback context when no documents found"""
        test_config['knowledge_base']['path'] = temp_kb_dir
        kb_manager = KnowledgeBaseManager(test_config)
        
        fallback = kb_manager.get_fallback_context()
        assert fallback is not None
        assert 'CloudSync Pro' in fallback or 'support' in fallback.lower()
        
    def test_document_metadata(self, temp_kb_dir, test_config):
        """Test document metadata extraction"""
        test_config['knowledge_base']['path'] = temp_kb_dir
        kb_manager = KnowledgeBaseManager(test_config)
        documents = kb_manager.load_documents()
        
        for doc in documents[:3]:
            assert 'source' in doc.metadata
            assert 'category' in doc.metadata
    
    def test_text_splitting(self, test_config):
        """Test text splitting functionality"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=test_config['knowledge_base']['chunk_size'],
            chunk_overlap=test_config['knowledge_base']['chunk_overlap']
        )
        
        text = "This is a long text that needs to be split into smaller chunks for processing."
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= test_config['knowledge_base']['chunk_size'] + 50
    
    def test_knowledge_add_document(self, temp_kb_dir, test_config):
        """Test adding new documents"""
        test_config['knowledge_base']['path'] = temp_kb_dir
        kb_manager = KnowledgeBaseManager(test_config)
        
        new_content = "New document content for testing."
        metadata = {'category': 'test', 'source': 'test_doc.txt'}
        
        with patch('src.knowledge_base.FAISS') as mock_faiss:
            mock_faiss.return_value = Mock()
            kb_manager.vectorstore = Mock()
            kb_manager.add_document(new_content, metadata)
            
            # Verify save operation
            kb_manager.vectorstore.add_documents.assert_called_once()