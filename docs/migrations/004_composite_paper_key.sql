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
