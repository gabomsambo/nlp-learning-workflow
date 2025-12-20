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
