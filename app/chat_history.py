from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine
from app.config import config
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatHistoryManager:
    def __init__(self):
        self.engine = create_engine(config.DATABASE_URL)
        
    def get_session_history(self, session_id: str) -> SQLChatMessageHistory:
        """Get chat history for a specific session"""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=config.DATABASE_URL
        )
    
    def create_session(self, user_id: Optional[str] = None, title: Optional[str] = None) -> str:
        """Create a new chat session and return session_id"""
        session_id = str(uuid.uuid4())
        
        with self.engine.connect() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (session_id, user_id, title) VALUES (%s, %s, %s)",
                (session_id, user_id, title)
            )
            conn.commit()
        
        return session_id
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        with self.engine.connect() as conn:
            result = conn.execute(
                "SELECT session_id, title, created_at FROM chat_sessions WHERE user_id = %s ORDER BY created_at DESC",
                (user_id,)
            )
            return [dict(row) for row in result]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            with self.engine.connect() as conn:
                conn.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
                conn.commit()
            return True
        except Exception:
            return False

# Instancia global
chat_history_manager = ChatHistoryManager()