import asyncio
import asyncpg
import json
import hashlib
import os
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

class PostgresCache:
    def __init__(self):
        self.pool = None
    
    async def initialize_and_clear_cache(self):
        """
        Wipes all data from the documents table.
        The CASCADE option ensures all related chunks are also deleted.
        """
        if not self.pool:
            print("⚠️ Cannot clear cache, database pool is not initialized.")
            return
        
        async with self.pool.acquire() as connection:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    url_hash VARCHAR(64) UNIQUE NOT NULL,
                    url TEXT NOT NULL,
                    content TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR(384)
                );
            """)
            print("✅ Database tables checked/created.")

            # Step 2: If the flag is set, run a safe truncate command.
            if os.getenv("CLEAR_CACHE_ON_RESTART", "false").lower() == "true":
                print("🔥 CLEAR_CACHE_ON_RESTART is true. Wiping database cache...")
                # This DO block checks if the table exists before truncating.
                # This is the SQL equivalent of "TRUNCATE IF EXISTS".
                await connection.execute("""
                    DO $$
                    BEGIN
                    IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename  = 'documents') THEN
                        TRUNCATE TABLE documents CASCADE;
                    END IF;
                    END $$;
                """)
                print("✅ Database cache cleared successfully.")
            
    async def init_pool(self, database_url: str):
        self.pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=5,
            server_settings={'jit': 'off'}
        )
        await self._create_tables()
    
    async def _create_tables(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS doc_cache (
                    url_hash TEXT PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    chunks JSONB NOT NULL,
                    embeddings BYTEA,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_doc_cache_created ON doc_cache(created_at);
                
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    url_hash TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_query_cache_lookup ON query_cache(url_hash, query_hash);
            """)
    
    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    async def get_doc_cache(self, url: str) -> Optional[Dict]:
        if not self.pool:
            return None
        url_hash = self._get_hash(url)
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT text_content, chunks, embeddings FROM doc_cache WHERE url_hash = $1",
                url_hash
            )
            if row:
                return {
                    'text': row['text_content'],
                    'chunks': json.loads(row['chunks']),
                    'embeddings': row['embeddings']
                }
        return None
    
    async def set_doc_cache(self, url: str, text: str, chunks: List[str], embeddings: bytes):
        if not self.pool:
            return
        url_hash = self._get_hash(url)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO doc_cache (url_hash, text_content, chunks, embeddings) 
                   VALUES ($1, $2, $3, $4) ON CONFLICT (url_hash) DO NOTHING""",
                url_hash, text, json.dumps(chunks), embeddings
            )
    
    async def get_query_cache(self, url: str, question: str) -> Optional[str]:
        if not self.pool:
            return None
        url_hash = self._get_hash(url)
        query_hash = self._get_hash(f"{url}:{question}")
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT answer FROM query_cache WHERE url_hash = $1 AND query_hash = $2",
                url_hash, query_hash
            )
            return row['answer'] if row else None
    
    async def set_query_cache(self, url: str, question: str, answer: str):
        if not self.pool:
            return
        url_hash = self._get_hash(url)
        query_hash = self._get_hash(f"{url}:{question}")
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO query_cache (query_hash, url_hash, question, answer) 
                   VALUES ($1, $2, $3, $4) ON CONFLICT (query_hash) DO NOTHING""",
                query_hash, url_hash, question, answer
            )
    
    async def cleanup_old_cache(self, days: int = 7):
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM doc_cache WHERE created_at < NOW() - INTERVAL '%s days'",
                days
            )
            await conn.execute(
                "DELETE FROM query_cache WHERE created_at < NOW() - INTERVAL '%s days'",
                days
            )

db_cache = PostgresCache()