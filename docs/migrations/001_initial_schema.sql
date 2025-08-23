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
CREATE INDEX idx_quiz_due ON quiz_cards(pillar_id, due_date) WHERE due_date <= NOW();
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
