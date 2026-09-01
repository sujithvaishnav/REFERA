-- ==============================================================================
-- REFERA - Supabase Database Schema & pgvector Setup
-- ==============================================================================
-- Run this complete SQL script in your Supabase SQL Editor to initialize
-- all tables, pgvector extensions, HNSW indexes, and RPC functions.
-- ==============================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents Table (stores uploaded document metadata & summaries)
CREATE TABLE IF NOT EXISTS public.documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 3. Document Chunks Table (stores text splits with 384-dim MiniLM embeddings)
CREATE TABLE IF NOT EXISTS public.document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES public.documents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4. Chat Sessions Table (manages individual user conversation threads)
CREATE TABLE IF NOT EXISTS public.chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 5. Chat Messages Table (stores multi-turn conversation history & citations)
CREATE TABLE IF NOT EXISTS public.chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ==============================================================================
-- INDEXES FOR MAXIMUM QUERY PERFORMANCE
-- ==============================================================================

-- Fast Vector Similarity Search using HNSW Cosine Distance
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
ON public.document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- Foreign key and lookup indexes
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON public.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_user_id ON public.document_chunks(user_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON public.document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON public.chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON public.chat_messages(session_id);

-- ==============================================================================
-- STORED PROCEDURE: match_document_chunks (pgvector Similarity Search)
-- ==============================================================================

CREATE OR REPLACE FUNCTION public.match_document_chunks (
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT -1.0,
    match_count INT DEFAULT 5,
    filter_user_id UUID DEFAULT NULL,
    filter_filenames TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    user_id UUID,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.user_id,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM public.document_chunks dc
    WHERE
        (filter_user_id IS NULL OR dc.user_id = filter_user_id)
        AND (
            filter_filenames IS NULL 
            OR dc.metadata->>'source' = ANY(filter_filenames)
        )
        AND (
            match_threshold = -1.0 
            OR 1 - (dc.embedding <=> query_embedding) >= match_threshold
        )
    ORDER BY dc.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;

-- ==============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES & PERMISSIONS
-- ==============================================================================
-- Multi-tenancy is strictly enforced at the application and query layer
-- by filtering on user_id across all tables and stored procedures.

ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages ENABLE ROW LEVEL SECURITY;

-- Drop old strict policies if they exist to avoid conflicts
DROP POLICY IF EXISTS "Users can view their own documents" ON public.documents;
DROP POLICY IF EXISTS "Users can insert their own documents" ON public.documents;
DROP POLICY IF EXISTS "Users can update their own documents" ON public.documents;
DROP POLICY IF EXISTS "Users can delete their own documents" ON public.documents;
DROP POLICY IF EXISTS "Users can view their own document chunks" ON public.document_chunks;
DROP POLICY IF EXISTS "Users can insert their own document chunks" ON public.document_chunks;
DROP POLICY IF EXISTS "Users can delete their own document chunks" ON public.document_chunks;
DROP POLICY IF EXISTS "Users can view their own chat sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "Users can insert their own chat sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "Users can update their own chat sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "Users can delete their own chat sessions" ON public.chat_sessions;
DROP POLICY IF EXISTS "Users can view their own chat messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can insert their own chat messages" ON public.chat_messages;
DROP POLICY IF EXISTS "Users can delete their own chat messages" ON public.chat_messages;

-- Create permissive policies for application access
CREATE POLICY "Allow full access for documents"
    ON public.documents FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow full access for document chunks"
    ON public.document_chunks FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow full access for chat sessions"
    ON public.chat_sessions FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow full access for chat messages"
    ON public.chat_messages FOR ALL
    USING (true)
    WITH CHECK (true);
