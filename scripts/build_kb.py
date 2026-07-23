"""
Build Knowledge Base Script
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge_base import KnowledgeBaseManager
from src.utils import load_config, logger

def build_knowledge_base(kb_dir: str = None, force_rebuild: bool = False):
    """Build knowledge base from files"""
    logger.info("🚀 Building Knowledge Base...")
    
    # Load config
    config = load_config()
    
    # Override KB path if provided
    if kb_dir:
        config['knowledge_base']['path'] = kb_dir
    
    # Check if vectorstore exists
    vectorstore_path = config.get('vectorstore', {}).get('path', 'data/vectorstore')
    if Path(vectorstore_path).exists() and not force_rebuild:
        logger.info(f"Vectorstore already exists at {vectorstore_path}")
        response = input("Rebuild? (y/n): ")
        if response.lower() != 'y':
            logger.info("Skipping rebuild")
            return
    
    # Initialize knowledge base manager
    kb_manager = KnowledgeBaseManager(config)
    
    # Load documents
    logger.info("📄 Loading documents...")
    documents = kb_manager.load_documents()
    
    if not documents:
        logger.error("❌ No documents found in knowledge base directory")
        return
    
    logger.info(f"✅ Loaded {len(documents)} documents")
    
    # Build vectorstore
    logger.info("🔨 Building vectorstore...")
    kb_manager.build_vectorstore(documents)
    
    logger.info("✅ Knowledge base built successfully!")
    logger.info(f"📁 Vectorstore saved to: {vectorstore_path}")
    logger.info(f"📊 Total documents: {len(documents)}")
    
    # Print statistics
    categories = {}
    for doc in documents:
        category = doc.metadata.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    logger.info("\n📊 Document Categories:")
    for category, count in categories.items():
        logger.info(f"  - {category}: {count} documents")

def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Base")
    parser.add_argument(
        "--kb-dir",
        help="Knowledge base directory path"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if vectorstore exists"
    )
    args = parser.parse_args()
    
    build_knowledge_base(args.kb_dir, args.force)

if __name__ == "__main__":
    main()