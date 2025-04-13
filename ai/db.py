from typing import List, Optional
import os
from datetime import datetime
import json
from uuid import UUID
import asyncio
from functools import wraps

import asyncpg
from dotenv import load_dotenv

load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv('POSTGRES_URL')

def with_connection(func):
    """Decorator to handle database connections."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            return await func(conn, *args, **kwargs)
        finally:
            await conn.close()
    return wrapper

class MemoryDB:
    @staticmethod
    @with_connection
    async def store_memory(conn: asyncpg.Connection, 
                          content: str,
                          timestamp: datetime,
                          memory_type: str,
                          importance: float,
                          embedding: List[float],
                          user_id: UUID) -> UUID:
        """Store a memory in the database."""
        query = """
        INSERT INTO "Memory" (content, timestamp, type, importance, embedding, "userId")
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """
        memory_id = await conn.fetchval(
            query,
            content,
            timestamp,
            memory_type,
            importance,
            embedding,
            user_id
        )
        return memory_id

    @staticmethod
    @with_connection
    async def get_memories(conn: asyncpg.Connection,
                          user_id: UUID,
                          memory_type: Optional[str] = None,
                          limit: int = 100) -> List[dict]:
        """Retrieve memories from the database."""
        query = """
        SELECT id, content, timestamp, type, importance, embedding
        FROM "Memory"
        WHERE "userId" = $1
        """
        if memory_type:
            query += ' AND type = $2'
            query += ' ORDER BY timestamp DESC LIMIT $3'
            rows = await conn.fetch(query, user_id, memory_type, limit)
        else:
            query += ' ORDER BY timestamp DESC LIMIT $2'
            rows = await conn.fetch(query, user_id, limit)

        return [dict(row) for row in rows]

    @staticmethod
    @with_connection
    async def find_similar_memories(conn: asyncpg.Connection,
                                  query_embedding: List[float],
                                  user_id: UUID,
                                  match_threshold: float = 0.7,
                                  match_count: int = 5) -> List[dict]:
        """Find similar memories using vector similarity search."""
        query = """
        SELECT * FROM match_memories($1, $2, $3, $4)
        """
        rows = await conn.fetch(
            query,
            query_embedding,
            match_threshold,
            match_count,
            user_id
        )
        return [dict(row) for row in rows]

    @staticmethod
    @with_connection
    async def delete_old_memories(conn: asyncpg.Connection,
                                user_id: UUID,
                                memory_type: str,
                                keep_count: int) -> int:
        """Delete old memories keeping only the most recent ones."""
        query = """
        WITH to_delete AS (
            SELECT id FROM "Memory"
            WHERE "userId" = $1 AND type = $2
            ORDER BY timestamp DESC
            OFFSET $3
        )
        DELETE FROM "Memory"
        WHERE id IN (SELECT id FROM to_delete)
        RETURNING id
        """
        deleted = await conn.fetch(query, user_id, memory_type, keep_count)
        return len(deleted)

def run_async(coro):
    """Helper function to run async code in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)
