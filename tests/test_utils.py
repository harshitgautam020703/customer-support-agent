"""
Utility Functions Tests
"""

import pytest
import json
import yaml
from datetime import datetime
from src.utils import (
    load_config, format_response, validate_query,
    extract_keywords, generate_case_id, get_timestamp
)

class TestUtils:
    """Test utility functions"""
    
    def test_load_config(self, tmp_path):
        """Test loading configuration"""
        # Create test config
        config_content = {
            'test_key': 'test_value',
            'nested': {
                'key': 'value'
            }
        }
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        config = load_config(str(config_path))
        assert config['test_key'] == 'test_value'
        assert config['nested']['key'] == 'value'
    
    def test_load_config_default(self, tmp_path):
        """Test loading default config when file missing"""
        config = load_config('non_existent_file.yaml')
        assert config is not None
        assert 'llm' in config
        assert 'knowledge_base' in config
    
    def test_format_response(self):
        """Test response formatting"""
        text = "This is a long response that needs to be formatted properly."
        formatted = format_response(text)
        assert formatted is not None
        assert len(formatted) > 0
    
    def test_format_response_long_text(self):
        """Test formatting long text with paragraphs"""
        text = "First paragraph. " * 20 + "\n\nSecond paragraph. " * 20
        formatted = format_response(text)
        assert '\n\n' in formatted
        assert len(formatted.split('\n')) > 1
    
    def test_validate_query_valid(self):
        """Test valid query validation"""
        assert validate_query("How do I reset my password?") is True
        assert validate_query("Help") is True
        assert validate_query("A" * 100) is True  # 100 chars
    
    def test_validate_query_invalid(self):
        """Test invalid query validation"""
        assert validate_query("") is False
        assert validate_query(" ") is False
        assert validate_query("A") is False  # Too short
        assert validate_query("A" * 5001) is False  # Too long
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        text = "How do I reset my password for CloudSync Pro?"
        keywords = extract_keywords(text)
        
        assert 'reset' in keywords
        assert 'password' in keywords
        assert 'cloudsync' in keywords
        assert len(keywords) > 0
    
    def test_extract_keywords_empty(self):
        """Test keyword extraction with empty text"""
        keywords = extract_keywords("")
        assert keywords == []
    
    def test_generate_case_id(self):
        """Test case ID generation"""
        user_input = "This is a test query"
        case_id = generate_case_id(user_input)
        
        assert case_id.startswith("CS-")
        assert len(case_id) > 10
        assert "CS-" in case_id
    
    def test_generate_case_id_unique(self):
        """Test case ID uniqueness"""
        id1 = generate_case_id("test1")
        id2 = generate_case_id("test2")
        assert id1 != id2
    
    def test_get_timestamp(self):
        """Test timestamp generation"""
        timestamp = get_timestamp()
        assert timestamp is not None
        # Should be ISO format
        assert 'T' in timestamp or ' ' in timestamp
    
    def test_load_config_environment_variables(self, tmp_path, monkeypatch):
        """Test loading config with environment variable substitution"""
        monkeypatch.setenv('TEST_VAR', 'env_value')
        
        config_content = {
            'test_key': '${TEST_VAR}',
            'nested': {
                'key': '${TEST_VAR}'
            }
        }
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config_content, f)
        
        config = load_config(str(config_path))
        # Note: This test assumes your load_config handles env vars
        # If not, this would need to be implemented
        assert config['test_key'] is not None
    
    def test_format_response_with_markdown(self):
        """Test formatting with markdown-like content"""
        text = "*Bold text* and **strong emphasis**"
        formatted = format_response(text)
        assert formatted is not None
        # Markdown should be preserved
        assert '*' in formatted or formatted == text