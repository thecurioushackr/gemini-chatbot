-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create Memory table
CREATE TABLE IF NOT EXISTS "Memory" (
  "id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
  "content" text NOT NULL,
  "timestamp" timestamp NOT NULL,
  "type" varchar(16) NOT NULL,
  "importance" real NOT NULL,
  "embedding" vector(768),
  "userId" uuid NOT NULL REFERENCES "User"(id)
);

-- Create vector index using HNSW for faster similarity search
CREATE INDEX ON "Memory" USING hnsw (embedding vector_cosine_ops)
WITH (
  m = 16,
  ef_construction = 64
);

-- Create function for similarity search
CREATE OR REPLACE FUNCTION match_memories(
  query_embedding vector(768),
  match_threshold float,
  match_count int,
  user_id uuid
)
RETURNS TABLE (
  id uuid,
  content text,
  timestamp timestamp,
  type varchar(16),
  importance real,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    m.id,
    m.content,
    m.timestamp,
    m.type,
    m.importance,
    1 - (m.embedding <=> query_embedding) as similarity
  FROM "Memory" m
  WHERE 
    m."userId" = user_id 
    AND 1 - (m.embedding <=> query_embedding) > match_threshold
  ORDER BY m.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
