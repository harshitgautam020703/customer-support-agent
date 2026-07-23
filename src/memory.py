"""
Conversation Memory Management
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import sqlite3
import threading
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    """Thread-safe conversation memory management"""
    
    def __init__(self, db_path: str = "data/conversations.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        
    def _init_db(self):
        """Initialize database"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user_input TEXT NOT NULL,
                    response TEXT NOT NULL,
                    escalated BOOLEAN DEFAULT FALSE,
                    sentiment REAL DEFAULT 0.5,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_context (
                    user_id TEXT PRIMARY KEY,
                    last_seen TIMESTAMP,
                    total_conversations INTEGER DEFAULT 0,
                    average_sentiment REAL DEFAULT 0.5,
                    escalation_count INTEGER DEFAULT 0,
                    preferences TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
    def save_conversation(self, user_id: str, data: Dict):
        """Save conversation to memory"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO conversations (
                    user_id, session_id, timestamp, user_input, 
                    response, escalated, sentiment, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                data.get('session_id', ''),
                data.get('timestamp', datetime.now().isoformat()),
                data.get('user_input', ''),
                data.get('response', ''),
                data.get('escalated', False),
                data.get('sentiment', 0.5),
                json.dumps(data.get('metadata', {}))
            ))
            
            # Update user context
            cursor.execute("""
                INSERT INTO user_context (user_id, last_seen, total_conversations)
                VALUES (?, ?, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_seen = ?,
                    total_conversations = total_conversations + 1
            """, (user_id, datetime.now().isoformat(), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            logger.info(f"Saved conversation for user {user_id}")
            
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history for a user"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM conversations
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
    def get_user_context(self, user_id: str) -> Dict:
        """Get user context"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM user_context
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            return dict(row) if row else {}
            
    def update_user_preferences(self, user_id: str, preferences: Dict):
        """Update user preferences"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE user_context
                SET preferences = ?
                WHERE user_id = ?
            """, (json.dumps(preferences), user_id))
            
            conn.commit()
            conn.close()
            
    def get_recent_escalations(self, hours: int = 24) -> List[Dict]:
        """Get recent escalations"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM conversations
                WHERE escalated = TRUE
                AND timestamp > ?
                ORDER BY timestamp DESC
            """, (cutoff.isoformat(),))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
    def clear_old_conversations(self, days: int = 30):
        """Clear conversations older than specified days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM conversations
                WHERE timestamp < ?
            """, (cutoff.isoformat(),))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {deleted} conversations older than {days} days")
            return deleted