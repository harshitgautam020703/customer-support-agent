"""
Knowledge Base Management System
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import json
import yaml
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """Manages knowledge base documents and retrieval"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_path = Path(config.get('knowledge_base.path', 'knowledge_base'))
        self.chunk_size = config.get('knowledge_base.chunk_size', 300)
        self.chunk_overlap = config.get('knowledge_base.chunk_overlap', 50)
        self.embeddings = OpenAIEmbeddings(
            model=config.get('embeddings.model', 'text-embedding-ada-002')
        )
        self.vectorstore = None
        
    def load_documents(self) -> List[Document]:
        """Load all documents from knowledge base directory"""
        documents = []
        
        # Load text files
        text_files = list(self.base_path.rglob("*.txt"))
        for file_path in text_files:
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata['source'] = str(file_path)
                    doc.metadata['category'] = self._get_category(file_path)
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        # Load markdown files
        md_files = list(self.base_path.rglob("*.md"))
        for file_path in md_files:
            try:
                loader = UnstructuredMarkdownLoader(str(file_path))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = str(file_path)
                    doc.metadata['category'] = self._get_category(file_path)
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        # Load YAML/JSON configs
        config_files = list(self.base_path.rglob("*.yaml")) + list(self.base_path.rglob("*.json"))
        for file_path in config_files:
            try:
                with open(file_path, 'r') as f:
                    if file_path.suffix == '.yaml':
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                    
                    # Convert to text
                    text = self._dict_to_text(data)
                    doc = Document(
                        page_content=text,
                        metadata={
                            'source': str(file_path),
                            'category': self._get_category(file_path),
                            'type': 'config'
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _get_category(self, file_path: Path) -> str:
        """Get category from file path"""
        # Try to get from parent directory
        parent = file_path.parent.name
        if parent in ['products', 'support', 'policies']:
            return parent
        
        # Try to get from filename
        if 'product' in file_path.name.lower():
            return 'products'
        elif 'support' in file_path.name.lower() or 'help' in file_path.name.lower():
            return 'support'
        elif 'policy' in file_path.name.lower():
            return 'policies'
        else:
            return 'general'
    
    def _dict_to_text(self, data: Dict, prefix: str = "") -> str:
        """Convert dictionary to text representation"""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.extend(self._dict_to_text(value, f"{prefix}{key}."))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.extend(self._dict_to_text(item, f"{prefix}{key}_{i+1}."))
                    else:
                        lines.append(f"{prefix}{key}_{i+1}: {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)
    
    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """Build vector store from documents"""
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save if configured
        save_path = self.config.get('vectorstore.path', 'data/vectorstore')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.vectorstore.save_local(save_path)
            logger.info(f"Saved vectorstore to {save_path}")
        
        return self.vectorstore
    
    def retrieve_context(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant context for query"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        return docs
    
    def get_fallback_context(self) -> str:
        """Get fallback context when no relevant documents found"""
        return """
        CloudSync Pro is a cloud storage and synchronization service.
        
        Basic Features:
        - File synchronization across devices
        - Cloud storage
        - File sharing
        - Version history
        
        Support Options:
        - Email: support@cloudsyncpro.com
        - Phone: 1-800-555-0123
        - Live Chat: Available on website
        
        More information can be found in our help center.
        """
    
    def add_document(self, content: str, metadata: Dict = None) -> None:
        """Add a new document to the knowledge base"""
        if metadata is None:
            metadata = {}
            
        doc = Document(page_content=content, metadata=metadata)
        self.vectorstore.add_documents([doc])
        
        # Also save as file
        if metadata.get('source'):
            file_path = Path(metadata['source'])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
        
        logger.info(f"Added new document: {metadata.get('source', 'unknown')}")