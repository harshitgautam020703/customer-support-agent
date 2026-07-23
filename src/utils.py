"""
Utility Functions
"""

import yaml
import json
import logging
from typing import Dict, Any
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return get_default_config()

def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'llm': {
            'model': 'gpt-4o-mini',
            'temperature': 0.2,
            'max_tokens': 1000
        },
        'embeddings': {
            'model': 'text-embedding-ada-002'
        },
        'knowledge_base': {
            'path': 'knowledge_base',
            'chunk_size': 300,
            'chunk_overlap': 50
        },
        'vectorstore': {
            'path': 'data/vectorstore'
        },
        'escalation': {
            'threshold': 0.7,
            'min_sentiment': 0.3
        },
        'retrieval': {
            'similarity_threshold': 0.7
        }
    }

def format_response(text: str) -> str:
    """Format response text with proper structure"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    formatted = []
    
    for para in paragraphs:
        if len(para) > 80:
            # Break long paragraphs
            words = para.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > 80:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    current_line.append(word)
                    current_length += len(word) + 1
            
            if current_line:
                lines.append(' '.join(current_line))
            formatted.append('\n'.join(lines))
        else:
            formatted.append(para)
    
    return '\n\n'.join(formatted)

def validate_query(query: str) -> bool:
    """Validate user query"""
    if not query or len(query.strip()) < 2:
        return False
    if len(query) > 5000:
        return False
    return True

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    # Remove common stopwords
    stopwords = {'the', 'and', 'for', 'with', 'you', 'your', 'can', 'please'}
    return list(set(words) - stopwords)

def generate_case_id(user_input: str) -> str:
    """Generate unique case ID"""
    import hashlib
    timestamp = datetime.now().strftime('%Y%m%d')
    hash_val = hashlib.md5(user_input.encode()).hexdigest()[:8]
    return f"CS-{timestamp}-{hash_val.upper()}"

def get_timestamp() -> str:
    """Get current timestamp"""
   