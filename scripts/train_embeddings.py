"""
Train and Optimize Embeddings
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_base import KnowledgeBaseManager
from src.utils import load_config, logger

def train_embeddings(model_name: str = "text-embedding-ada-002", force: bool = False):
    """Train embeddings with specified model"""
    logger.info(f"🚀 Training embeddings with {model_name}")
    
    config = load_config()
    config['embeddings']['model'] = model_name
    
    # Initialize KB manager
    kb_manager = KnowledgeBaseManager(config)
    
    # Load documents
    logger.info("📄 Loading documents...")
    documents = kb_manager.load_documents()
    
    if not documents:
        logger.error("❌ No documents found")
        return
    
    # Build vectorstore with new embeddings
    logger.info("🔨 Building vectorstore...")
    vectorstore = kb_manager.build_vectorstore(documents)
    
    logger.info("✅ Embeddings training complete!")
    
    # Test retrieval
    test_queries = [
        "How do I reset my password?",
        "What are the pricing plans?",
        "How do I cancel my subscription?"
    ]
    
    logger.info("\n🔍 Testing retrieval:")
    for query in test_queries:
        docs = vectorstore.similarity_search(query, k=2)
        logger.info(f"\n  Query: {query}")
        for i, doc in enumerate(docs, 1):
            logger.info(f"    {i}. {doc.page_content[:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Train Embeddings")
    parser.add_argument(
        "--model",
        default="text-embedding-ada-002",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining"
    )
    args = parser.parse_args()
    
    train_embeddings(args.model, args.force)

if __name__ == "__main__":
    main()