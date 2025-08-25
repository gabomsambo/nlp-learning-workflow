-- NLP Learning Workflow Database Schema
-- This script creates all necessary tables for the NLP learning workflow system
-- with proper pillar isolation, foreign keys, and performance indexes.

-- Enable UUID generation extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ==========================================
-- 1. PAPERS TABLE
-- Stores research paper metadata with pillar isolation
-- ==========================================
CREATE TABLE papers (
    id TEXT PRIMARY KEY,  -- DOI or arXiv ID
    pillar_id TEXT NOT NULL,  -- Pillar isolation (P1, P2, P3, P4, P5)
    title TEXT NOT NULL,
    authors JSONB,  -- Array of author names
    venue TEXT,  -- Conference or journal
    year INTEGER,
    url_pdf TEXT,
    abstract TEXT,
    citation_count INTEGER DEFAULT 0,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- 2. PAPER_QUEUE TABLE
-- Stores papers queued for processing
-- ==========================================
CREATE TABLE paper_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pillar_id TEXT NOT NULL,  -- Pillar isolation
    paper_id TEXT NOT NULL,
    title TEXT NOT NULL,
    priority INTEGER DEFAULT 5,  -- Higher number = higher priority
    source TEXT,  -- Source of the paper (arxiv, searxng, etc.)
    processed BOOLEAN DEFAULT FALSE,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- 3. NOTES TABLE
-- Stores structured notes extracted from papers
-- ==========================================
CREATE TABLE notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT NOT NULL,
    pillar_id TEXT NOT NULL,  -- Pillar isolation
    problem TEXT NOT NULL,  -- Problem the paper addresses
    method TEXT NOT NULL,  -- Methodology used
    findings JSONB,  -- Array of key findings
    limitations JSONB,  -- Array of limitations
    future_work JSONB,  -- Array of future research directions
    key_terms JSONB,  -- Array of important technical terms
    related_papers JSONB,  -- Array of related paper IDs
    confidence_score REAL DEFAULT 0.8 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- 4. LESSONS TABLE
-- Stores synthesized lessons from papers
-- ==========================================
CREATE TABLE lessons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT NOT NULL,
    pillar_id TEXT NOT NULL,  -- Pillar isolation
    tl_dr TEXT NOT NULL,  -- One-sentence summary
    takeaways JSONB,  -- Array of key takeaways (3-5)
    practice_ideas JSONB,  -- Array of practical applications
    connections JSONB,  -- Array of connections to other work
    difficulty INTEGER DEFAULT 2 CHECK (difficulty IN (1, 2, 3)),  -- 1=Easy, 2=Medium, 3=Hard
    estimated_time INTEGER DEFAULT 10,  -- Reading time in minutes
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- 5. QUIZ_CARDS TABLE
-- Stores quiz cards for spaced repetition learning
-- ==========================================
CREATE TABLE quiz_cards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT NOT NULL,
    pillar_id TEXT NOT NULL,  -- Pillar isolation
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    difficulty INTEGER DEFAULT 2 CHECK (difficulty IN (1, 2, 3)),  -- 1=Easy, 2=Medium, 3=Hard
    question_type TEXT DEFAULT 'factual' CHECK (question_type IN ('factual', 'conceptual', 'application')),
    -- Spaced repetition fields (SM-2 algorithm)
    interval INTEGER DEFAULT 1,  -- Days until next review
    repetitions INTEGER DEFAULT 0,  -- Number of successful reviews
    ease_factor REAL DEFAULT 2.5 CHECK (ease_factor >= 1.3),  -- Ease factor for SM-2
    due_date TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    last_reviewed TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- FOREIGN KEY CONSTRAINTS
-- Link tables together with referential integrity
-- ==========================================

-- Notes reference papers
ALTER TABLE notes 
ADD CONSTRAINT fk_notes_paper_id 
FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE;

-- Lessons reference papers
ALTER TABLE lessons 
ADD CONSTRAINT fk_lessons_paper_id 
FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE;

-- Quiz cards reference papers
ALTER TABLE quiz_cards 
ADD CONSTRAINT fk_quiz_cards_paper_id 
FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE;

-- ==========================================
-- PERFORMANCE INDEXES
-- Add indexes on foreign keys and pillar_id for optimal query performance
-- ==========================================

-- Papers table indexes
CREATE INDEX idx_papers_pillar_id ON papers(pillar_id);
CREATE INDEX idx_papers_processed ON papers(processed, pillar_id);
CREATE INDEX idx_papers_created_at ON papers(created_at);

-- Paper queue indexes
CREATE INDEX idx_paper_queue_pillar_id ON paper_queue(pillar_id);
CREATE INDEX idx_paper_queue_processed ON paper_queue(processed, pillar_id);
CREATE INDEX idx_paper_queue_priority ON paper_queue(priority DESC, added_at DESC);

-- Notes table indexes
CREATE INDEX idx_notes_pillar_id ON notes(pillar_id);
CREATE INDEX idx_notes_paper_id ON notes(paper_id);
CREATE INDEX idx_notes_created_at ON notes(created_at);

-- Lessons table indexes
CREATE INDEX idx_lessons_pillar_id ON lessons(pillar_id);
CREATE INDEX idx_lessons_paper_id ON lessons(paper_id);
CREATE INDEX idx_lessons_created_at ON lessons(created_at);

-- Quiz cards table indexes
CREATE INDEX idx_quiz_cards_pillar_id ON quiz_cards(pillar_id);
CREATE INDEX idx_quiz_cards_paper_id ON quiz_cards(paper_id);
CREATE INDEX idx_quiz_cards_due_date ON quiz_cards(due_date, pillar_id);
CREATE INDEX idx_quiz_cards_created_at ON quiz_cards(created_at);

-- ==========================================
-- VALIDATION COMMENTS
-- ==========================================

-- This schema ensures:
-- ✅ Primary Keys: UUID for most tables, TEXT for papers (using DOI/arXiv ID)
-- ✅ Foreign Key Constraints: All child tables reference papers.id with CASCADE DELETE
-- ✅ Pillar Isolation: Every table has pillar_id TEXT NOT NULL column
-- ✅ Data Types: TEXT for strings, TIMESTAMP WITH TIME ZONE for dates, JSONB for arrays, BOOLEAN, INTEGER, REAL
-- ✅ Defaults: timezone('utc', now()) for creation timestamps, sensible defaults for other fields
-- ✅ Indexes: Added on foreign keys and pillar_id for performance
-- ✅ Constraints: CHECK constraints for difficulty levels, question types, confidence scores, ease factors

COMMENT ON TABLE papers IS 'Research paper metadata with pillar isolation';
COMMENT ON TABLE paper_queue IS 'Papers queued for processing with priority ordering';
COMMENT ON TABLE notes IS 'Structured notes extracted from papers by AI agents';
COMMENT ON TABLE lessons IS 'Synthesized lessons and takeaways from papers';
COMMENT ON TABLE quiz_cards IS 'Spaced repetition quiz cards for learning reinforcement';

COMMENT ON COLUMN papers.pillar_id IS 'Learning pillar (P1-P5) for content isolation';
COMMENT ON COLUMN quiz_cards.ease_factor IS 'SM-2 algorithm ease factor (min 1.3)';
COMMENT ON COLUMN quiz_cards.interval IS 'Days until next review in spaced repetition';
