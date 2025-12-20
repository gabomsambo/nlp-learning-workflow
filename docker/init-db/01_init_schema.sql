-- NLP Learning Workflow Database Schema
-- For use with Supabase (PostgreSQL)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================
-- Core Tables
-- =====================================

-- Pillars table
CREATE TABLE IF NOT EXISTS pillars (
    id VARCHAR(2) PRIMARY KEY CHECK (id IN ('P1', 'P2', 'P3', 'P4', 'P5')),
    name VARCHAR(100) NOT NULL,
    goal TEXT,
    papers_per_day INTEGER DEFAULT 2 CHECK (papers_per_day BETWEEN 1 AND 10),
    focus_areas JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_active TIMESTAMP WITH TIME ZONE,
    settings JSONB DEFAULT '{}'::jsonb
);

-- Papers table
CREATE TABLE IF NOT EXISTS papers (
    id VARCHAR(100) PRIMARY KEY,  -- DOI or arXiv ID
    pillar_id VARCHAR(2) REFERENCES pillars(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    authors JSONB NOT NULL DEFAULT '[]'::jsonb,
    venue VARCHAR(200),
    year INTEGER,
    url_pdf TEXT,
    abstract TEXT,
    citation_count INTEGER DEFAULT 0,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Paper Notes table
CREATE TABLE IF NOT EXISTS notes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id VARCHAR(100) REFERENCES papers(id) ON DELETE CASCADE,
    pillar_id VARCHAR(2) REFERENCES pillars(id) ON DELETE CASCADE,
    problem TEXT NOT NULL,
    method TEXT NOT NULL,
    findings JSONB NOT NULL DEFAULT '[]'::jsonb,
    limitations JSONB NOT NULL DEFAULT '[]'::jsonb,
    future_work JSONB NOT NULL DEFAULT '[]'::jsonb,
    key_terms JSONB NOT NULL DEFAULT '[]'::jsonb,
    related_papers JSONB DEFAULT '[]'::jsonb,
    confidence_score REAL DEFAULT 0.8 CHECK (confidence_score BETWEEN 0 AND 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Lessons table
CREATE TABLE IF NOT EXISTS lessons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id VARCHAR(100) REFERENCES papers(id) ON DELETE CASCADE,
    pillar_id VARCHAR(2) REFERENCES pillars(id) ON DELETE CASCADE,
    tl_dr TEXT NOT NULL,
    takeaways JSONB NOT NULL DEFAULT '[]'::jsonb,
    practice_ideas JSONB NOT NULL DEFAULT '[]'::jsonb,
    connections JSONB DEFAULT '[]'::jsonb,
    difficulty INTEGER DEFAULT 2 CHECK (difficulty BETWEEN 1 AND 3),
    estimated_time INTEGER DEFAULT 10,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Quiz Cards table (with spaced repetition)
CREATE TABLE IF NOT EXISTS quiz_cards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id VARCHAR(100) REFERENCES papers(id) ON DELETE CASCADE,
    pillar_id VARCHAR(2) REFERENCES pillars(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    difficulty INTEGER DEFAULT 2 CHECK (difficulty BETWEEN 1 AND 3),
    question_type VARCHAR(20) CHECK (question_type IN ('factual', 'conceptual', 'application')),
    -- Spaced Repetition (SM-2) fields
    interval INTEGER DEFAULT 1,
    repetitions INTEGER DEFAULT 0,
    ease_factor REAL DEFAULT 2.5,
    due_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_reviewed TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Podcast Scripts table
CREATE TABLE IF NOT EXISTS podcast_scripts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id VARCHAR(100) REFERENCES papers(id) ON DELETE CASCADE,
    pillar_id VARCHAR(2) REFERENCES pillars(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    duration_minutes INTEGER DEFAULT 12,
    host_cs TEXT NOT NULL,
    host_ling TEXT NOT NULL,
    key_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    conclusion TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Learning Progress table
CREATE TABLE IF NOT EXISTS progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pillar_id VARCHAR(2) REFERENCES pillars(id) ON DELETE CASCADE,
    user_id VARCHAR(100) DEFAULT 'default',  -- For future multi-user support
    papers_read INTEGER DEFAULT 0,
    papers_queued INTEGER DEFAULT 0,
    quizzes_completed INTEGER DEFAULT 0,
    current_streak INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    total_time_minutes INTEGER DEFAULT 0,
    last_activity TIMESTAMP WITH TIME ZONE,
    next_review TIMESTAMP WITH TIME ZONE,
    mastery_score REAL DEFAULT 0.0 CHECK (mastery_score BETWEEN 0 AND 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(pillar_id, user_id)
);

-- Daily Sessions table
CREATE TABLE IF NOT EXISTS daily_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pillar_id VARCHAR(2) REFERENCES pillars(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    papers_processed JSONB DEFAULT '[]'::jsonb,
    lessons_generated INTEGER DEFAULT 0,
    quizzes_created INTEGER DEFAULT 0,
    quizzes_reviewed INTEGER DEFAULT 0,
    time_spent_minutes INTEGER DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(pillar_id, session_date)
);

-- Paper Queue table
CREATE TABLE IF NOT EXISTS paper_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pillar_id VARCHAR(2) REFERENCES pillars(id) ON DELETE CASCADE,
    paper_id VARCHAR(100),
    title TEXT NOT NULL,
    priority INTEGER DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    source VARCHAR(50),  -- 'arxiv', 'semantic_scholar', etc.
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    scheduled_for DATE,
    processed BOOLEAN DEFAULT FALSE
);

-- =====================================
-- Indexes for Performance
-- =====================================

CREATE INDEX idx_papers_pillar ON papers(pillar_id) WHERE processed = FALSE;
CREATE INDEX idx_papers_processed ON papers(processed, added_at DESC);
CREATE INDEX idx_notes_paper ON notes(paper_id);
CREATE INDEX idx_notes_pillar ON notes(pillar_id);
CREATE INDEX idx_lessons_paper ON lessons(paper_id);
CREATE INDEX idx_lessons_pillar ON lessons(pillar_id);
CREATE INDEX idx_quiz_due ON quiz_cards(pillar_id, due_date);
CREATE INDEX idx_quiz_paper ON quiz_cards(paper_id);
CREATE INDEX idx_progress_user ON progress(user_id, pillar_id);
CREATE INDEX idx_sessions_date ON daily_sessions(pillar_id, session_date DESC);
CREATE INDEX idx_queue_priority ON paper_queue(pillar_id, priority DESC) WHERE processed = FALSE;

-- =====================================
-- Functions and Triggers
-- =====================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add update triggers
CREATE TRIGGER update_notes_updated_at BEFORE UPDATE ON notes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_progress_updated_at BEFORE UPDATE ON progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update progress stats
CREATE OR REPLACE FUNCTION update_progress_stats(
    p_pillar_id VARCHAR(2),
    p_papers_read INTEGER DEFAULT 0,
    p_quizzes_completed INTEGER DEFAULT 0,
    p_time_spent INTEGER DEFAULT 0
)
RETURNS void AS $$
BEGIN
    INSERT INTO progress (pillar_id, user_id, papers_read, quizzes_completed, total_time_minutes, last_activity)
    VALUES (p_pillar_id, 'default', p_papers_read, p_quizzes_completed, p_time_spent, NOW())
    ON CONFLICT (pillar_id, user_id) DO UPDATE SET
        papers_read = progress.papers_read + EXCLUDED.papers_read,
        quizzes_completed = progress.quizzes_completed + EXCLUDED.quizzes_completed,
        total_time_minutes = progress.total_time_minutes + EXCLUDED.total_time_minutes,
        last_activity = NOW(),
        current_streak = CASE 
            WHEN progress.last_activity::date = CURRENT_DATE - INTERVAL '1 day' 
            THEN progress.current_streak + 1
            WHEN progress.last_activity::date < CURRENT_DATE - INTERVAL '1 day'
            THEN 1
            ELSE progress.current_streak
        END,
        longest_streak = GREATEST(progress.longest_streak, progress.current_streak);
END;
$$ LANGUAGE plpgsql;

-- =====================================
-- Initial Data
-- =====================================

-- Insert default pillars
INSERT INTO pillars (id, name, goal) VALUES
    ('P1', 'Linguistic & Cognitive Foundations', 'Master core linguistic theory and cognitive alignment between humans and AI'),
    ('P2', 'Models & Architectures', 'Understand cutting-edge model architectures and emerging paradigms'),
    ('P3', 'Data, Training & Methodologies', 'Master data curation, training techniques, and multilingual challenges'),
    ('P4', 'Evaluation & Interpretability', 'Develop expertise in model evaluation, analysis, and interpretability'),
    ('P5', 'Ethics & Applications', 'Understand ethical implications and real-world applications')
ON CONFLICT (id) DO NOTHING;

-- Initialize progress for all pillars
INSERT INTO progress (pillar_id, user_id)
SELECT id, 'default' FROM pillars
ON CONFLICT (pillar_id, user_id) DO NOTHING;

-- =====================================
-- Views for Easy Querying
-- =====================================

-- View for papers with complete information
CREATE OR REPLACE VIEW v_papers_full AS
SELECT 
    p.*,
    pi.name as pillar_name,
    n.problem,
    n.method,
    l.tl_dr,
    COUNT(DISTINCT q.id) as quiz_count
FROM papers p
LEFT JOIN pillars pi ON p.pillar_id = pi.id
LEFT JOIN notes n ON p.id = n.paper_id
LEFT JOIN lessons l ON p.id = l.paper_id
LEFT JOIN quiz_cards q ON p.id = q.paper_id
GROUP BY p.id, pi.name, n.problem, n.method, l.tl_dr;

-- View for daily learning summary
CREATE OR REPLACE VIEW v_daily_summary AS
SELECT 
    ds.pillar_id,
    pi.name as pillar_name,
    ds.session_date,
    ds.papers_processed,
    ds.lessons_generated,
    ds.quizzes_created,
    ds.quizzes_reviewed,
    ds.time_spent_minutes
FROM daily_sessions ds
JOIN pillars pi ON ds.pillar_id = pi.id
ORDER BY ds.session_date DESC;

-- View for quiz cards due for review
CREATE OR REPLACE VIEW v_quiz_due AS
SELECT 
    q.*,
    p.title as paper_title,
    pi.name as pillar_name
FROM quiz_cards q
JOIN papers p ON q.paper_id = p.id
JOIN pillars pi ON q.pillar_id = pi.id
WHERE q.due_date <= NOW()
ORDER BY q.due_date, q.difficulty;

-- =====================================
-- Row Level Security (RLS) - Optional
-- =====================================

-- Enable RLS on tables (uncomment if using Supabase Auth)
-- ALTER TABLE papers ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE notes ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE lessons ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE quiz_cards ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE progress ENABLE ROW LEVEL SECURITY;

-- Create policies (example for multi-user support)
-- CREATE POLICY "Users can view their own progress" ON progress
--     FOR ALL USING (auth.uid()::text = user_id);
-- Migration: Add Pillars Table
-- This migration adds a pillars table to support dynamic pillar management
-- Allows users to create, read, update, and delete custom learning pillars

-- ==========================================
-- PILLARS TABLE
-- Stores dynamic pillar configurations
-- ==========================================
CREATE TABLE pillars (
    id TEXT PRIMARY KEY,  -- URL-friendly slug (auto-generated from name)
    name TEXT NOT NULL UNIQUE,  -- Human-readable pillar name
    goal TEXT NOT NULL,  -- Learning goal/objective for this pillar
    focus_areas JSONB DEFAULT '[]'::jsonb,  -- Array of focus area strings
    papers_per_day INTEGER DEFAULT 2 CHECK (papers_per_day >= 1 AND papers_per_day <= 10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    last_active TIMESTAMP WITH TIME ZONE
);

-- ==========================================
-- INDEXES FOR PERFORMANCE
-- ==========================================
CREATE INDEX idx_pillars_name ON pillars(name);
CREATE INDEX idx_pillars_created_at ON pillars(created_at);
CREATE INDEX idx_pillars_last_active ON pillars(last_active);

-- ==========================================
-- UPDATE TRIGGER FOR updated_at
-- ==========================================
CREATE OR REPLACE FUNCTION update_pillars_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc', now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_pillars_updated_at
    BEFORE UPDATE ON pillars
    FOR EACH ROW
    EXECUTE FUNCTION update_pillars_updated_at();

-- ==========================================
-- SEED DEFAULT PILLARS
-- Insert the existing 5 pillars as initial data
-- ==========================================
INSERT INTO pillars (id, name, goal, focus_areas, papers_per_day) VALUES
('linguistic-cognitive-foundations', 'Linguistic & Cognitive Foundations', 
 'Understanding the theoretical foundations of language and cognition in NLP', 
 '["syntax", "semantics", "pragmatics", "cognitive science", "psycholinguistics"]'::jsonb, 2),
 
('models-architectures', 'Models & Architectures', 
 'Mastering neural network architectures and model designs for NLP', 
 '["transformers", "attention mechanisms", "RNNs", "CNNs", "model architecture"]'::jsonb, 2),
 
('data-training-methodologies', 'Data, Training & Methodologies', 
 'Learning about data preprocessing, training techniques, and methodologies', 
 '["data preprocessing", "training methods", "optimization", "regularization", "fine-tuning"]'::jsonb, 2),
 
('evaluation-interpretability', 'Evaluation & Interpretability', 
 'Understanding how to evaluate and interpret NLP models', 
 '["evaluation metrics", "interpretability", "explainability", "bias detection", "robustness"]'::jsonb, 2),
 
('ethics-applications', 'Ethics & Applications', 
 'Exploring ethical considerations and real-world applications of NLP', 
 '["ethics", "fairness", "applications", "deployment", "societal impact"]'::jsonb, 2);

-- ==========================================
-- COMMENTS
-- ==========================================
COMMENT ON TABLE pillars IS 'Dynamic pillar configurations for personalized learning paths';
COMMENT ON COLUMN pillars.id IS 'URL-friendly slug generated from name (e.g., "machine-learning-basics")';
COMMENT ON COLUMN pillars.name IS 'Human-readable pillar name displayed in UI';
COMMENT ON COLUMN pillars.goal IS 'Learning objective or description of what this pillar aims to teach';
COMMENT ON COLUMN pillars.focus_areas IS 'Array of topic areas this pillar focuses on';
COMMENT ON COLUMN pillars.papers_per_day IS 'Target number of papers to process per day (1-10)';
-- Migration 003: Add FSRS tables for personalized spaced repetition
-- Adds review_logs and user_fsrs_parameters tables to support FSRS algorithm

-- ==========================================
-- 1. REVIEW_LOGS TABLE
-- Stores complete history of every quiz interaction
-- ==========================================
CREATE TABLE review_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    card_id UUID NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default_user',  -- For future multi-user support
    pillar_id TEXT NOT NULL,  -- For performance and filtering
    paper_id TEXT NOT NULL,   -- For analytics
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 4),  -- FSRS rating: 1=Again, 2=Hard, 3=Good, 4=Easy
    review_timestamp TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    
    -- Card state at time of review (for FSRS optimization)
    difficulty REAL NOT NULL,     -- FSRS difficulty parameter
    stability REAL NOT NULL,      -- FSRS stability parameter
    retrievability REAL,          -- Optional: calculated retrievability at review time
    
    -- Previous interval info
    previous_due_date TIMESTAMP WITH TIME ZONE,
    days_overdue INTEGER DEFAULT 0,  -- How many days late was this review
    
    -- Context information
    session_id TEXT,              -- Group reviews into sessions
    response_time_ms INTEGER,     -- Time taken to answer (optional)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- 2. USER_FSRS_PARAMETERS TABLE
-- Stores personalized FSRS parameters per user
-- ==========================================
CREATE TABLE user_fsrs_parameters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL DEFAULT 'default_user',
    pillar_id TEXT,  -- NULL means global parameters, specific pillar_id means pillar-specific
    
    -- FSRS algorithm parameters (optimized per user)
    w0 REAL NOT NULL DEFAULT 0.4,     -- Initial stability for new cards
    w1 REAL NOT NULL DEFAULT 0.6,     -- Initial stability for learning cards
    w2 REAL NOT NULL DEFAULT 2.4,     -- Initial stability multiplier
    w3 REAL NOT NULL DEFAULT 5.8,     -- Initial difficulty offset
    w4 REAL NOT NULL DEFAULT 4.93,    -- Difficulty weight for Again
    w5 REAL NOT NULL DEFAULT 0.94,    -- Difficulty weight for Hard
    w6 REAL NOT NULL DEFAULT 0.86,    -- Difficulty weight for Good
    w7 REAL NOT NULL DEFAULT 0.01,    -- Difficulty weight for Easy
    w8 REAL NOT NULL DEFAULT 1.49,    -- Stability multiplier for Again
    w9 REAL NOT NULL DEFAULT 0.14,    -- Stability multiplier for Hard
    w10 REAL NOT NULL DEFAULT 0.94,   -- Stability multiplier for Good
    w11 REAL NOT NULL DEFAULT 2.18,   -- Stability multiplier for Easy
    w12 REAL NOT NULL DEFAULT 0.05,   -- Difficulty decay for Again
    w13 REAL NOT NULL DEFAULT 0.34,   -- Difficulty decay for Hard
    w14 REAL NOT NULL DEFAULT 0.67,   -- Difficulty decay for Good
    w15 REAL NOT NULL DEFAULT 2.74,   -- Difficulty decay for Easy
    w16 REAL NOT NULL DEFAULT 0.0,    -- Forgetting curve parameter
    w17 REAL NOT NULL DEFAULT 2.0,    -- Stability increase parameter
    
    -- Metadata
    review_count INTEGER DEFAULT 0,   -- Number of reviews used for optimization
    last_optimized TIMESTAMP WITH TIME ZONE,
    optimization_score REAL,         -- Quality score of the optimization (R² or similar)
    is_default BOOLEAN DEFAULT FALSE, -- True if these are default parameters
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    
    -- Ensure unique parameters per user/pillar combination
    UNIQUE(user_id, pillar_id)
);

-- ==========================================
-- 3. UPDATE QUIZ_CARDS TABLE
-- Add FSRS-specific fields while maintaining backward compatibility
-- ==========================================

-- Add FSRS fields to existing quiz_cards table
ALTER TABLE quiz_cards ADD COLUMN IF NOT EXISTS difficulty_fsrs REAL DEFAULT 0.0;
ALTER TABLE quiz_cards ADD COLUMN IF NOT EXISTS stability REAL DEFAULT 0.0;
ALTER TABLE quiz_cards ADD COLUMN IF NOT EXISTS retrievability REAL;
ALTER TABLE quiz_cards ADD COLUMN IF NOT EXISTS last_review_date TIMESTAMP WITH TIME ZONE;
ALTER TABLE quiz_cards ADD COLUMN IF NOT EXISTS next_review_date TIMESTAMP WITH TIME ZONE;
ALTER TABLE quiz_cards ADD COLUMN IF NOT EXISTS state TEXT DEFAULT 'new' CHECK (state IN ('new', 'learning', 'review', 'relearning'));
ALTER TABLE quiz_cards ADD COLUMN IF NOT EXISTS lapses INTEGER DEFAULT 0;  -- Number of times card was forgotten
ALTER TABLE quiz_cards ADD COLUMN IF NOT EXISTS user_id TEXT DEFAULT 'default_user';

-- ==========================================
-- FOREIGN KEY CONSTRAINTS
-- ==========================================

-- Review logs reference quiz cards
ALTER TABLE review_logs 
ADD CONSTRAINT fk_review_logs_card_id 
FOREIGN KEY (card_id) REFERENCES quiz_cards(id) ON DELETE CASCADE;

-- ==========================================
-- PERFORMANCE INDEXES
-- ==========================================

-- Review logs indexes
CREATE INDEX idx_review_logs_user_id ON review_logs(user_id);
CREATE INDEX idx_review_logs_card_id ON review_logs(card_id);
CREATE INDEX idx_review_logs_pillar_id ON review_logs(pillar_id);
CREATE INDEX idx_review_logs_timestamp ON review_logs(review_timestamp);
CREATE INDEX idx_review_logs_user_pillar ON review_logs(user_id, pillar_id);
CREATE INDEX idx_review_logs_session ON review_logs(session_id) WHERE session_id IS NOT NULL;

-- User FSRS parameters indexes
CREATE INDEX idx_user_fsrs_parameters_user_id ON user_fsrs_parameters(user_id);
CREATE INDEX idx_user_fsrs_parameters_pillar ON user_fsrs_parameters(user_id, pillar_id);

-- Quiz cards FSRS indexes
CREATE INDEX idx_quiz_cards_user_id ON quiz_cards(user_id);
CREATE INDEX idx_quiz_cards_next_review ON quiz_cards(next_review_date, user_id) WHERE next_review_date IS NOT NULL;
CREATE INDEX idx_quiz_cards_state ON quiz_cards(state, user_id);

-- ==========================================
-- DEFAULT FSRS PARAMETERS
-- Insert default parameters for the default user
-- ==========================================

-- Global default parameters
INSERT INTO user_fsrs_parameters (
    user_id, pillar_id, 
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17,
    is_default, review_count
) VALUES (
    'default_user', NULL,
    0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 0.67, 2.74, 0.0, 2.0,
    TRUE, 0
) ON CONFLICT (user_id, pillar_id) DO NOTHING;

-- Pillar-specific default parameters (optional)
INSERT INTO user_fsrs_parameters (
    user_id, pillar_id, 
    w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17,
    is_default, review_count
) 
SELECT 
    'default_user', pillar_value,
    0.4, 0.6, 2.4, 5.8, 4.93, 0.94, 0.86, 0.01, 1.49, 0.14, 0.94, 2.18, 0.05, 0.34, 0.67, 2.74, 0.0, 2.0,
    TRUE, 0
FROM unnest(ARRAY['P1', 'P2', 'P3', 'P4', 'P5']) AS pillar_value
ON CONFLICT (user_id, pillar_id) DO NOTHING;

-- ==========================================
-- COMMENTS
-- ==========================================

COMMENT ON TABLE review_logs IS 'Complete history of every quiz review for FSRS optimization';
COMMENT ON TABLE user_fsrs_parameters IS 'Personalized FSRS algorithm parameters per user and optionally per pillar';

COMMENT ON COLUMN review_logs.rating IS 'FSRS rating: 1=Again, 2=Hard, 3=Good, 4=Easy';
COMMENT ON COLUMN review_logs.difficulty IS 'FSRS difficulty parameter at time of review';
COMMENT ON COLUMN review_logs.stability IS 'FSRS stability parameter at time of review';
COMMENT ON COLUMN quiz_cards.difficulty_fsrs IS 'FSRS difficulty (different from legacy difficulty column)';
COMMENT ON COLUMN quiz_cards.stability IS 'FSRS stability parameter';
COMMENT ON COLUMN quiz_cards.state IS 'FSRS card state: new, learning, review, relearning';

-- ==========================================
-- VALIDATION
-- ==========================================

-- This migration adds:
-- ✅ review_logs table for complete review history tracking
-- ✅ user_fsrs_parameters table for personalized algorithm parameters  
-- ✅ FSRS fields to quiz_cards table while maintaining backward compatibility
-- ✅ Foreign key constraints linking review logs to quiz cards
-- ✅ Performance indexes for efficient querying
-- ✅ Default FSRS parameters for immediate use
-- ✅ Multi-user support structure (ready for future expansion)
-- ✅ Proper constraints and data types for FSRS algorithm requirements
-- Migration: Allow papers to exist in multiple pillars
-- Changes papers primary key from (id) to (id, pillar_id)
-- Updates foreign key constraints on child tables

-- =====================================
-- Step 1: Drop existing foreign key constraints
-- =====================================

-- Drop foreign keys that reference papers(id)
ALTER TABLE notes DROP CONSTRAINT IF EXISTS notes_paper_id_fkey;
ALTER TABLE lessons DROP CONSTRAINT IF EXISTS lessons_paper_id_fkey;
ALTER TABLE quiz_cards DROP CONSTRAINT IF EXISTS quiz_cards_paper_id_fkey;
ALTER TABLE podcast_scripts DROP CONSTRAINT IF EXISTS podcast_scripts_paper_id_fkey;

-- =====================================
-- Step 2: Drop existing primary key on papers
-- =====================================

ALTER TABLE papers DROP CONSTRAINT IF EXISTS papers_pkey;

-- =====================================
-- Step 3: Create composite primary key
-- =====================================

-- Make pillar_id NOT NULL before adding to primary key
ALTER TABLE papers ALTER COLUMN pillar_id SET NOT NULL;

-- Create composite primary key (id, pillar_id)
ALTER TABLE papers ADD PRIMARY KEY (id, pillar_id);

-- =====================================
-- Step 4: Add composite foreign key constraints
-- =====================================

-- Notes table - reference papers(id, pillar_id)
ALTER TABLE notes ADD CONSTRAINT notes_paper_pillar_fkey
    FOREIGN KEY (paper_id, pillar_id) REFERENCES papers(id, pillar_id) ON DELETE CASCADE;

-- Lessons table - reference papers(id, pillar_id)
ALTER TABLE lessons ADD CONSTRAINT lessons_paper_pillar_fkey
    FOREIGN KEY (paper_id, pillar_id) REFERENCES papers(id, pillar_id) ON DELETE CASCADE;

-- Quiz cards table - reference papers(id, pillar_id)
ALTER TABLE quiz_cards ADD CONSTRAINT quiz_cards_paper_pillar_fkey
    FOREIGN KEY (paper_id, pillar_id) REFERENCES papers(id, pillar_id) ON DELETE CASCADE;

-- Podcast scripts table - reference papers(id, pillar_id)
ALTER TABLE podcast_scripts ADD CONSTRAINT podcast_scripts_paper_pillar_fkey
    FOREIGN KEY (paper_id, pillar_id) REFERENCES papers(id, pillar_id) ON DELETE CASCADE;

-- =====================================
-- Step 5: Update indexes for new primary key
-- =====================================

-- The existing idx_papers_pillar index is still valid
-- Add index for queries by paper_id alone (common lookup pattern)
CREATE INDEX IF NOT EXISTS idx_papers_id ON papers(id);

-- =====================================
-- Step 6: Update views that depend on papers table
-- =====================================

-- Recreate v_papers_full view to handle composite key
CREATE OR REPLACE VIEW v_papers_full AS
SELECT
    p.*,
    pi.name as pillar_name,
    n.problem,
    n.method,
    l.tl_dr,
    COUNT(DISTINCT q.id) as quiz_count
FROM papers p
LEFT JOIN pillars pi ON p.pillar_id = pi.id
LEFT JOIN notes n ON p.id = n.paper_id AND p.pillar_id = n.pillar_id
LEFT JOIN lessons l ON p.id = l.paper_id AND p.pillar_id = l.pillar_id
LEFT JOIN quiz_cards q ON p.id = q.paper_id AND p.pillar_id = q.pillar_id
GROUP BY p.id, p.pillar_id, pi.name, n.problem, n.method, l.tl_dr;

-- Recreate v_quiz_due view with composite join
CREATE OR REPLACE VIEW v_quiz_due AS
SELECT
    q.*,
    p.title as paper_title,
    pi.name as pillar_name
FROM quiz_cards q
JOIN papers p ON q.paper_id = p.id AND q.pillar_id = p.pillar_id
JOIN pillars pi ON q.pillar_id = pi.id
WHERE q.due_date <= NOW()
ORDER BY q.due_date, q.difficulty;
-- Migration: 005_add_paper_citations.sql
-- Description: Add paper_citations table for storing citation relationships
-- Created: 2024

-- =============================================================================
-- Paper Citations Table
-- =============================================================================
-- Stores citation relationships between papers for citation network discovery.
-- Each row represents a citation relationship between two papers.

CREATE TABLE IF NOT EXISTS paper_citations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id VARCHAR(100) NOT NULL,           -- The paper (our paper)
    cited_paper_id VARCHAR(100) NOT NULL,     -- The related paper
    citation_direction VARCHAR(10) NOT NULL   -- 'outgoing' (paper cites cited_paper) or 'incoming' (cited_paper cites paper)
        CHECK (citation_direction IN ('outgoing', 'incoming')),
    is_influential BOOLEAN DEFAULT FALSE,     -- Semantic Scholar influential citation flag
    citation_context TEXT,                    -- Optional: surrounding context text
    source VARCHAR(50) DEFAULT 'semantic_scholar',  -- Source API
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure unique citation relationships
    UNIQUE(paper_id, cited_paper_id, citation_direction)
);

-- =============================================================================
-- Indexes for efficient querying
-- =============================================================================

-- Index for finding all citations for a paper
CREATE INDEX idx_citations_paper ON paper_citations(paper_id);

-- Index for finding papers that cite a specific paper
CREATE INDEX idx_citations_cited ON paper_citations(cited_paper_id);

-- Index for filtering by direction
CREATE INDEX idx_citations_direction ON paper_citations(citation_direction);

-- Partial index for influential citations (faster queries for high-impact papers)
CREATE INDEX idx_citations_influential ON paper_citations(is_influential)
    WHERE is_influential = TRUE;

-- Composite index for common query patterns
CREATE INDEX idx_citations_paper_direction ON paper_citations(paper_id, citation_direction);

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE paper_citations IS 'Citation relationships between papers for network-based discovery';
COMMENT ON COLUMN paper_citations.paper_id IS 'Paper ID (DOI/arXiv format)';
COMMENT ON COLUMN paper_citations.cited_paper_id IS 'Related paper ID (DOI/arXiv format)';
COMMENT ON COLUMN paper_citations.citation_direction IS 'outgoing = paper cites cited_paper, incoming = cited_paper cites paper';
COMMENT ON COLUMN paper_citations.is_influential IS 'True if citation is marked as influential by Semantic Scholar';
COMMENT ON COLUMN paper_citations.citation_context IS 'Text context around the citation if available';
COMMENT ON COLUMN paper_citations.source IS 'API source that provided the citation data';
-- Migration: Dynamic Pillars System
-- Migrates from hardcoded P1-P5 VARCHAR(2) to dynamic VARCHAR(100) slugs
--
-- IMPORTANT: This migration assumes 001_initial_schema was applied but NOT 004_composite_paper_key
-- It will set up the composite key structure as part of this migration

BEGIN;

-- =====================================
-- PHASE 1: DROP ALL VIEWS
-- Must drop before altering any columns they depend on
-- =====================================
DROP VIEW IF EXISTS v_papers_full CASCADE;
DROP VIEW IF EXISTS v_daily_summary CASCADE;
DROP VIEW IF EXISTS v_quiz_due CASCADE;

-- =====================================
-- PHASE 2: DROP FK CONSTRAINTS ON paper_id
-- These are the ORIGINAL names from migration 001
-- =====================================
ALTER TABLE notes DROP CONSTRAINT IF EXISTS notes_paper_id_fkey;
ALTER TABLE lessons DROP CONSTRAINT IF EXISTS lessons_paper_id_fkey;
ALTER TABLE quiz_cards DROP CONSTRAINT IF EXISTS quiz_cards_paper_id_fkey;
ALTER TABLE podcast_scripts DROP CONSTRAINT IF EXISTS podcast_scripts_paper_id_fkey;

-- =====================================
-- PHASE 3: DROP FK CONSTRAINTS ON pillar_id
-- Drop all possible constraint names for safety
-- =====================================
ALTER TABLE papers DROP CONSTRAINT IF EXISTS papers_pillar_id_fkey;
ALTER TABLE notes DROP CONSTRAINT IF EXISTS notes_pillar_id_fkey;
ALTER TABLE lessons DROP CONSTRAINT IF EXISTS lessons_pillar_id_fkey;
ALTER TABLE quiz_cards DROP CONSTRAINT IF EXISTS quiz_cards_pillar_id_fkey;
ALTER TABLE podcast_scripts DROP CONSTRAINT IF EXISTS podcast_scripts_pillar_id_fkey;
ALTER TABLE progress DROP CONSTRAINT IF EXISTS progress_pillar_id_fkey;
ALTER TABLE daily_sessions DROP CONSTRAINT IF EXISTS daily_sessions_pillar_id_fkey;
ALTER TABLE paper_queue DROP CONSTRAINT IF EXISTS paper_queue_pillar_id_fkey;

-- =====================================
-- PHASE 4: DROP CHECK CONSTRAINT AND PRIMARY KEY
-- =====================================
ALTER TABLE pillars DROP CONSTRAINT IF EXISTS pillars_id_check;
ALTER TABLE papers DROP CONSTRAINT IF EXISTS papers_pkey;

-- =====================================
-- PHASE 5: ALTER COLUMN TYPES
-- Increase VARCHAR(2) to VARCHAR(100) for all pillar_id columns
-- =====================================

-- Parent table first
ALTER TABLE pillars ALTER COLUMN id TYPE VARCHAR(100);

-- All child table pillar_id columns
ALTER TABLE papers ALTER COLUMN pillar_id TYPE VARCHAR(100);
ALTER TABLE notes ALTER COLUMN pillar_id TYPE VARCHAR(100);
ALTER TABLE lessons ALTER COLUMN pillar_id TYPE VARCHAR(100);
ALTER TABLE quiz_cards ALTER COLUMN pillar_id TYPE VARCHAR(100);
ALTER TABLE podcast_scripts ALTER COLUMN pillar_id TYPE VARCHAR(100);
ALTER TABLE progress ALTER COLUMN pillar_id TYPE VARCHAR(100);
ALTER TABLE daily_sessions ALTER COLUMN pillar_id TYPE VARCHAR(100);
ALTER TABLE paper_queue ALTER COLUMN pillar_id TYPE VARCHAR(100);

-- =====================================
-- PHASE 6: ADD NEW COLUMNS TO PILLARS
-- =====================================
ALTER TABLE pillars ADD COLUMN IF NOT EXISTS abbreviation VARCHAR(10);
ALTER TABLE pillars ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- =====================================
-- PHASE 7: UPDATE ALL DATA - CHILD TABLES FIRST
-- Must update children before parent to maintain referential integrity
-- =====================================

-- Update papers table
UPDATE papers SET pillar_id = 'linguistic-cognitive-foundations' WHERE pillar_id = 'P1';
UPDATE papers SET pillar_id = 'models-architectures' WHERE pillar_id = 'P2';
UPDATE papers SET pillar_id = 'data-training-methodologies' WHERE pillar_id = 'P3';
UPDATE papers SET pillar_id = 'evaluation-interpretability' WHERE pillar_id = 'P4';
UPDATE papers SET pillar_id = 'ethics-applications' WHERE pillar_id = 'P5';

-- Update notes table
UPDATE notes SET pillar_id = 'linguistic-cognitive-foundations' WHERE pillar_id = 'P1';
UPDATE notes SET pillar_id = 'models-architectures' WHERE pillar_id = 'P2';
UPDATE notes SET pillar_id = 'data-training-methodologies' WHERE pillar_id = 'P3';
UPDATE notes SET pillar_id = 'evaluation-interpretability' WHERE pillar_id = 'P4';
UPDATE notes SET pillar_id = 'ethics-applications' WHERE pillar_id = 'P5';

-- Update lessons table
UPDATE lessons SET pillar_id = 'linguistic-cognitive-foundations' WHERE pillar_id = 'P1';
UPDATE lessons SET pillar_id = 'models-architectures' WHERE pillar_id = 'P2';
UPDATE lessons SET pillar_id = 'data-training-methodologies' WHERE pillar_id = 'P3';
UPDATE lessons SET pillar_id = 'evaluation-interpretability' WHERE pillar_id = 'P4';
UPDATE lessons SET pillar_id = 'ethics-applications' WHERE pillar_id = 'P5';

-- Update quiz_cards table
UPDATE quiz_cards SET pillar_id = 'linguistic-cognitive-foundations' WHERE pillar_id = 'P1';
UPDATE quiz_cards SET pillar_id = 'models-architectures' WHERE pillar_id = 'P2';
UPDATE quiz_cards SET pillar_id = 'data-training-methodologies' WHERE pillar_id = 'P3';
UPDATE quiz_cards SET pillar_id = 'evaluation-interpretability' WHERE pillar_id = 'P4';
UPDATE quiz_cards SET pillar_id = 'ethics-applications' WHERE pillar_id = 'P5';

-- Update podcast_scripts table
UPDATE podcast_scripts SET pillar_id = 'linguistic-cognitive-foundations' WHERE pillar_id = 'P1';
UPDATE podcast_scripts SET pillar_id = 'models-architectures' WHERE pillar_id = 'P2';
UPDATE podcast_scripts SET pillar_id = 'data-training-methodologies' WHERE pillar_id = 'P3';
UPDATE podcast_scripts SET pillar_id = 'evaluation-interpretability' WHERE pillar_id = 'P4';
UPDATE podcast_scripts SET pillar_id = 'ethics-applications' WHERE pillar_id = 'P5';

-- Update progress table
UPDATE progress SET pillar_id = 'linguistic-cognitive-foundations' WHERE pillar_id = 'P1';
UPDATE progress SET pillar_id = 'models-architectures' WHERE pillar_id = 'P2';
UPDATE progress SET pillar_id = 'data-training-methodologies' WHERE pillar_id = 'P3';
UPDATE progress SET pillar_id = 'evaluation-interpretability' WHERE pillar_id = 'P4';
UPDATE progress SET pillar_id = 'ethics-applications' WHERE pillar_id = 'P5';

-- Update daily_sessions table
UPDATE daily_sessions SET pillar_id = 'linguistic-cognitive-foundations' WHERE pillar_id = 'P1';
UPDATE daily_sessions SET pillar_id = 'models-architectures' WHERE pillar_id = 'P2';
UPDATE daily_sessions SET pillar_id = 'data-training-methodologies' WHERE pillar_id = 'P3';
UPDATE daily_sessions SET pillar_id = 'evaluation-interpretability' WHERE pillar_id = 'P4';
UPDATE daily_sessions SET pillar_id = 'ethics-applications' WHERE pillar_id = 'P5';

-- Update paper_queue table
UPDATE paper_queue SET pillar_id = 'linguistic-cognitive-foundations' WHERE pillar_id = 'P1';
UPDATE paper_queue SET pillar_id = 'models-architectures' WHERE pillar_id = 'P2';
UPDATE paper_queue SET pillar_id = 'data-training-methodologies' WHERE pillar_id = 'P3';
UPDATE paper_queue SET pillar_id = 'evaluation-interpretability' WHERE pillar_id = 'P4';
UPDATE paper_queue SET pillar_id = 'ethics-applications' WHERE pillar_id = 'P5';

-- =====================================
-- PHASE 8: UPDATE PARENT TABLE (PILLARS)
-- Now safe because all child references are updated
-- =====================================
UPDATE pillars SET
    id = 'linguistic-cognitive-foundations',
    abbreviation = 'LingCog',
    updated_at = NOW()
WHERE id = 'P1';

UPDATE pillars SET
    id = 'models-architectures',
    abbreviation = 'ModArch',
    updated_at = NOW()
WHERE id = 'P2';

UPDATE pillars SET
    id = 'data-training-methodologies',
    abbreviation = 'DataTrn',
    updated_at = NOW()
WHERE id = 'P3';

UPDATE pillars SET
    id = 'evaluation-interpretability',
    abbreviation = 'EvalInt',
    updated_at = NOW()
WHERE id = 'P4';

UPDATE pillars SET
    id = 'ethics-applications',
    abbreviation = 'EthApp',
    updated_at = NOW()
WHERE id = 'P5';

-- =====================================
-- PHASE 9: SET NOT NULL AND CREATE COMPOSITE PRIMARY KEY
-- =====================================

-- Set NOT NULL on pillar_id columns for composite keys
ALTER TABLE papers ALTER COLUMN pillar_id SET NOT NULL;
ALTER TABLE notes ALTER COLUMN pillar_id SET NOT NULL;
ALTER TABLE lessons ALTER COLUMN pillar_id SET NOT NULL;
ALTER TABLE quiz_cards ALTER COLUMN pillar_id SET NOT NULL;
ALTER TABLE podcast_scripts ALTER COLUMN pillar_id SET NOT NULL;

-- Create composite primary key on papers
ALTER TABLE papers ADD PRIMARY KEY (id, pillar_id);

-- Create index for lookups by paper_id alone
CREATE INDEX IF NOT EXISTS idx_papers_id ON papers(id);

-- =====================================
-- PHASE 10: UPDATE FUNCTION SIGNATURE
-- =====================================
CREATE OR REPLACE FUNCTION update_progress_stats(
    p_pillar_id VARCHAR(100),
    p_papers_read INTEGER DEFAULT 0,
    p_quizzes_completed INTEGER DEFAULT 0,
    p_time_spent INTEGER DEFAULT 0
)
RETURNS void AS $$
BEGIN
    INSERT INTO progress (pillar_id, user_id, papers_read, quizzes_completed, total_time_minutes, last_activity)
    VALUES (p_pillar_id, 'default', p_papers_read, p_quizzes_completed, p_time_spent, NOW())
    ON CONFLICT (pillar_id, user_id) DO UPDATE SET
        papers_read = progress.papers_read + EXCLUDED.papers_read,
        quizzes_completed = progress.quizzes_completed + EXCLUDED.quizzes_completed,
        total_time_minutes = progress.total_time_minutes + EXCLUDED.total_time_minutes,
        last_activity = NOW(),
        current_streak = CASE
            WHEN progress.last_activity::date = CURRENT_DATE - INTERVAL '1 day'
            THEN progress.current_streak + 1
            WHEN progress.last_activity::date < CURRENT_DATE - INTERVAL '1 day'
            THEN 1
            ELSE progress.current_streak
        END,
        longest_streak = GREATEST(progress.longest_streak, progress.current_streak);
END;
$$ LANGUAGE plpgsql;

-- =====================================
-- PHASE 11: CREATE TRIGGERS
-- =====================================
DROP TRIGGER IF EXISTS update_pillars_updated_at ON pillars;
CREATE TRIGGER update_pillars_updated_at BEFORE UPDATE ON pillars
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================
-- PHASE 12: RECREATE VIEWS
-- Using composite key joins where applicable
-- =====================================

-- View 1: v_papers_full
CREATE OR REPLACE VIEW v_papers_full AS
SELECT
    p.*,
    pi.name as pillar_name,
    n.problem,
    n.method,
    l.tl_dr,
    COUNT(DISTINCT q.id) as quiz_count
FROM papers p
LEFT JOIN pillars pi ON p.pillar_id = pi.id
LEFT JOIN notes n ON p.id = n.paper_id AND p.pillar_id = n.pillar_id
LEFT JOIN lessons l ON p.id = l.paper_id AND p.pillar_id = l.pillar_id
LEFT JOIN quiz_cards q ON p.id = q.paper_id AND p.pillar_id = q.pillar_id
GROUP BY p.id, p.pillar_id, pi.name, n.problem, n.method, l.tl_dr;

-- View 2: v_daily_summary
CREATE OR REPLACE VIEW v_daily_summary AS
SELECT
    ds.pillar_id,
    pi.name as pillar_name,
    ds.session_date,
    ds.papers_processed,
    ds.lessons_generated,
    ds.quizzes_created,
    ds.quizzes_reviewed,
    ds.time_spent_minutes
FROM daily_sessions ds
JOIN pillars pi ON ds.pillar_id = pi.id
ORDER BY ds.session_date DESC;

-- View 3: v_quiz_due
CREATE OR REPLACE VIEW v_quiz_due AS
SELECT
    q.*,
    p.title as paper_title,
    pi.name as pillar_name
FROM quiz_cards q
JOIN papers p ON q.paper_id = p.id AND q.pillar_id = p.pillar_id
JOIN pillars pi ON q.pillar_id = pi.id
WHERE q.due_date <= NOW()
ORDER BY q.due_date, q.difficulty;

-- =====================================
-- PHASE 13: CLEAN UP ORPHANED RECORDS
-- Remove records from child tables that don't have matching papers
-- This can happen if papers were deleted but child records weren't
-- =====================================

-- Delete orphaned notes (where paper_id + pillar_id doesn't exist in papers)
DELETE FROM notes n
WHERE NOT EXISTS (
    SELECT 1 FROM papers p
    WHERE p.id = n.paper_id AND p.pillar_id = n.pillar_id
);

-- Delete orphaned lessons
DELETE FROM lessons l
WHERE NOT EXISTS (
    SELECT 1 FROM papers p
    WHERE p.id = l.paper_id AND p.pillar_id = l.pillar_id
);

-- Delete orphaned quiz_cards
DELETE FROM quiz_cards q
WHERE NOT EXISTS (
    SELECT 1 FROM papers p
    WHERE p.id = q.paper_id AND p.pillar_id = q.pillar_id
);

-- Delete orphaned podcast_scripts
DELETE FROM podcast_scripts ps
WHERE NOT EXISTS (
    SELECT 1 FROM papers p
    WHERE p.id = ps.paper_id AND p.pillar_id = ps.pillar_id
);

-- =====================================
-- PHASE 14: RECREATE FOREIGN KEY CONSTRAINTS
-- Using composite keys for child tables referencing papers
-- =====================================

-- Child tables -> papers (composite FK)
ALTER TABLE notes ADD CONSTRAINT notes_paper_pillar_fkey
    FOREIGN KEY (paper_id, pillar_id) REFERENCES papers(id, pillar_id) ON DELETE CASCADE;

ALTER TABLE lessons ADD CONSTRAINT lessons_paper_pillar_fkey
    FOREIGN KEY (paper_id, pillar_id) REFERENCES papers(id, pillar_id) ON DELETE CASCADE;

ALTER TABLE quiz_cards ADD CONSTRAINT quiz_cards_paper_pillar_fkey
    FOREIGN KEY (paper_id, pillar_id) REFERENCES papers(id, pillar_id) ON DELETE CASCADE;

ALTER TABLE podcast_scripts ADD CONSTRAINT podcast_scripts_paper_pillar_fkey
    FOREIGN KEY (paper_id, pillar_id) REFERENCES papers(id, pillar_id) ON DELETE CASCADE;

-- Tables -> pillars (simple FK)
ALTER TABLE progress ADD CONSTRAINT progress_pillar_id_fkey
    FOREIGN KEY (pillar_id) REFERENCES pillars(id) ON DELETE CASCADE;

ALTER TABLE daily_sessions ADD CONSTRAINT daily_sessions_pillar_id_fkey
    FOREIGN KEY (pillar_id) REFERENCES pillars(id) ON DELETE CASCADE;

ALTER TABLE paper_queue ADD CONSTRAINT paper_queue_pillar_id_fkey
    FOREIGN KEY (pillar_id) REFERENCES pillars(id) ON DELETE CASCADE;

COMMIT;

-- =====================================
-- VERIFICATION QUERIES (run after migration)
-- =====================================
-- SELECT id, name, abbreviation FROM pillars;
-- SELECT COUNT(*) as total, pillar_id FROM papers GROUP BY pillar_id;
-- SELECT DISTINCT pillar_id FROM notes;
-- \d papers  -- Check primary key and constraints
-- Migration: Update podcast_scripts table for single host format
-- Run this in Supabase SQL editor

-- Add new columns for single host format
ALTER TABLE podcast_scripts
  ADD COLUMN IF NOT EXISTS script TEXT,
  ADD COLUMN IF NOT EXISTS word_count INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS ground_pack JSONB DEFAULT '{}'::jsonb;

-- Make old columns nullable (keep for backward compatibility)
ALTER TABLE podcast_scripts
  ALTER COLUMN host_cs DROP NOT NULL,
  ALTER COLUMN host_ling DROP NOT NULL;

-- Add index for paper lookup
CREATE INDEX IF NOT EXISTS idx_podcast_scripts_paper ON podcast_scripts(paper_id);

-- Add index for pillar lookup
CREATE INDEX IF NOT EXISTS idx_podcast_scripts_pillar ON podcast_scripts(pillar_id);
