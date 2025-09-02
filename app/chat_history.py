from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine, text
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
            # Usar text() para SQLAlchemy 2.x
            stmt = text("INSERT INTO chat_sessions (session_id, user_id, title) VALUES (:session_id, :user_id, :title)")
            conn.execute(stmt, {"session_id": session_id, "user_id": user_id, "title": title})
            conn.commit()

        return session_id

    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user"""
        with self.engine.connect() as conn:
            stmt = text("SELECT session_id, title, created_at FROM chat_sessions WHERE user_id = :user_id ORDER BY created_at DESC")
            result = conn.execute(stmt, {"user_id": user_id})
            return [dict(row._mapping) for row in result]

    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            with self.engine.connect() as conn:
                stmt = text("DELETE FROM chat_sessions WHERE session_id = :session_id")
                conn.execute(stmt, {"session_id": session_id})
                conn.commit()
            return True
        except Exception:
            return False

# Instancia global
chat_history_manager = ChatHistoryManager()