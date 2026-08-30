-- NLP Learning Workflow Database Schema
-- This script creates all necessary tables for the NLP learning workflow system
-- with proper pillar isolation, foreign keys, and performance indexes.
--
-- ==========================================================================
-- PROVENANCE — READ THIS BEFORE TRUSTING ANY COLUMN
-- ==========================================================================
-- The original hosted Supabase project was reaped and its host no longer
-- resolves in DNS. NOTHING here was recovered from that database.
--
-- Tables 1-5 (papers, paper_queue, notes, lessons, quiz_cards) are the
-- original hand-written definitions from this file, EXTENDED where the code
-- demonstrably reads or writes columns the old definition lacked.
--
-- Tables 6-11 (pillars, podcast_scripts, review_logs, user_fsrs_parameters,
-- paper_citations, scripts) are RECONSTRUCTED FROM CODE — derived from the
-- `_*_to_dict` / `_dict_to_*` converters in nlp_pillars/db.py, the query
-- parameters in webui/services/postgrest_client.py, and the Pydantic models
-- in nlp_pillars/schemas.py.
--
-- Every place a type, nullability, default or constraint had to be INFERRED
-- rather than read off a call site is marked with an `INFERRED:` comment.
-- Nobody can confirm what the old tables actually looked like.
-- ==========================================================================

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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    -- ADDED during reconstruction. webui/services/postgrest_client.py both
    -- selects and orders by `added_at` on this table (lines 106 and 161:
    -- "select": "id,title,...,added_at", "order": "added_at.desc"), while
    -- nlp_pillars/db.py only ever writes `created_at`. The old table must
    -- have carried both or the papers list in the web UI was broken.
    -- INFERRED: type, default. Kept separate from created_at rather than
    -- renamed, so both access paths work.
    added_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
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
    source TEXT,  -- What the identifier is: 'arxiv' or 'url'
    -- PDF URL the candidate was discovered with. NULL only for arXiv-id rows,
    -- where pop_from_paper_queue reconstructs it from paper_id. Added by
    -- docs/migrations/008_paper_queue_url_pdf.sql.
    url_pdf TEXT,
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
-- NOTE: the Lesson Pydantic model carries title/content/examples/podcast_script,
-- but _lesson_to_dict() deliberately does not persist them and _dict_to_lesson()
-- synthesises them on read ("Generate title as it's not stored in DB").
-- No columns added for them — that asymmetry is intentional in the code.
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
-- The original definition here only covered the legacy SM-2 fields. The FSRS
-- work added a whole second set of columns that _quiz_card_to_dict() writes on
-- every insert and update; without them every quiz write would fail with
-- PGRST204 (column not found). Those are marked ADDED below.
CREATE TABLE quiz_cards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT NOT NULL,
    pillar_id TEXT NOT NULL,  -- Pillar isolation
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    difficulty INTEGER DEFAULT 2 CHECK (difficulty IN (1, 2, 3)),  -- 1=Easy, 2=Medium, 3=Hard
    question_type TEXT DEFAULT 'factual' CHECK (question_type IN ('factual', 'conceptual', 'application')),

    -- ADDED: written by _quiz_card_to_dict(), read by _dict_to_quiz_card().
    -- INFERRED: TEXT + default matching QuizCard.user_id's default.
    user_id TEXT NOT NULL DEFAULT 'default_user',

    -- ADDED: FSRS algorithm fields (nlp_pillars/db.py _quiz_card_to_dict).
    -- INFERRED: DOUBLE PRECISION for the float params; the Pydantic defaults
    -- (0.0/0.0/None/'new'/0) are mirrored as column defaults.
    difficulty_fsrs DOUBLE PRECISION DEFAULT 0.0,  -- FSRS difficulty parameter
    stability DOUBLE PRECISION DEFAULT 0.0,        -- FSRS stability parameter
    retrievability DOUBLE PRECISION,               -- nullable: Optional[float] in schema
    state TEXT DEFAULT 'new' CHECK (state IN ('new', 'learning', 'review', 'relearning')),
    lapses INTEGER DEFAULT 0,
    last_review_date TIMESTAMP WITH TIME ZONE,
    next_review_date TIMESTAMP WITH TIME ZONE,

    -- Spaced repetition fields (SM-2 algorithm, legacy but still written)
    interval INTEGER DEFAULT 1,  -- Days until next review
    repetitions INTEGER DEFAULT 0,  -- Number of successful reviews
    ease_factor REAL DEFAULT 2.5 CHECK (ease_factor >= 1.3),  -- Ease factor for SM-2
    due_date TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    last_reviewed TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- 6. PILLARS TABLE  (RECONSTRUCTED FROM CODE)
-- Dynamic learning pillars. Replaces the static PILLAR_CONFIGS dict, which
-- config.get_pillar_config() now only falls back to when this lookup fails.
-- ==========================================
-- Source: nlp_pillars/db.py:1362 _pillar_to_dict() writes id/name/goal/
-- focus_areas/papers_per_day; _dict_to_pillar() additionally reads created_at,
-- updated_at and last_active. create_pillars.py posts the same five fields.
CREATE TABLE pillars (
    -- id is a slug generated from the name by _generate_slug(), not a UUID.
    id TEXT PRIMARY KEY,
    -- INFERRED UNIQUE: create_pillar() at db.py:1414 branches on a 409 /
    -- "duplicate" error to raise PillarExistsError("Pillar with name ...
    -- already exists"). That error path can only ever fire if the database
    -- enforces uniqueness on the NAME specifically.
    name TEXT NOT NULL UNIQUE,
    goal TEXT NOT NULL,
    -- List[str] in PillarBase; stored as a JSONB array like every other list
    -- column in this schema. INFERRED: default '[]' so the column is never
    -- NULL (_dict_to_pillar uses row.get('focus_areas', []) and would cope,
    -- but an empty array is the honest representation).
    focus_areas JSONB DEFAULT '[]'::jsonb,
    -- Bounds mirror PillarBase.papers_per_day (ge=1, le=10).
    papers_per_day INTEGER DEFAULT 2 CHECK (papers_per_day >= 1 AND papers_per_day <= 10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    -- updated_at is READ by _dict_to_pillar() but NEVER WRITTEN by
    -- _pillar_to_dict() or update_pillar(). It therefore has to be maintained
    -- by the database. INFERRED: a trigger (defined below) does that.
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    -- Optional[datetime] in the Pillar model; nothing in the codebase writes
    -- it. Left nullable with no default.
    last_active TIMESTAMP WITH TIME ZONE
);

-- ==========================================
-- 7. PODCAST_SCRIPTS TABLE  (RECONSTRUCTED FROM CODE)
-- Generated single-host podcast scripts, one per paper.
-- ==========================================
-- Source: nlp_pillars/db.py _podcast_script_to_dict() / _dict_to_podcast_script();
-- webui/services/postgrest_client.py:178 selects
-- "id,paper_id,pillar_id,title,word_count,created_at" ordered by created_at.desc.
CREATE TABLE podcast_scripts (
    -- INFERRED UUID: add_podcast_script() reads response['data'][0].get('id')
    -- back after insert without ever supplying one, so the database generates it.
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT NOT NULL,
    pillar_id TEXT NOT NULL,  -- Pillar isolation
    title TEXT NOT NULL,      -- Episode title; read unguarded as row['title']
    -- Full script in [HOST]: format. Read as row.get('script', ''), so the
    -- code tolerates NULL, but the model requires it on write.
    script TEXT NOT NULL,
    word_count INTEGER DEFAULT 0,
    key_points JSONB DEFAULT '[]'::jsonb,   -- List[str] of discussion points
    ground_pack JSONB DEFAULT '{}'::jsonb,  -- dict: 4 prompt outputs for reference
    -- What the script was written from; see schemas.py::SourceMaterial.
    -- Nullable: pre-011 rows have no value and are read back as level="full".
    -- Added by docs/migrations/011_podcast_source_material.sql.
    source_material JSONB,
    -- What the script was aimed at: field/audience/length/tone; see
    -- schemas.py::PodcastOptions and nlp_pillars/podcast_options.py.
    -- Nullable: pre-012 rows were generated when nothing was configurable and
    -- are read back as an empty PodcastOptions.
    -- Added by docs/migrations/012_podcast_options.sql.
    options JSONB,
    -- Per-section model provenance for the four Ground Pack calls; see
    -- schemas.py::GroundPackCallRecord and nlp_pillars/podcast_models.py.
    -- Nullable: pre-013 rows have no value and are read back as {}.
    -- Added by docs/migrations/013_podcast_ground_pack_calls.sql.
    ground_pack_calls JSONB,
    -- Generated episode audio metadata; see schemas.py::AudioMetadata.
    -- Added by docs/migrations/014_podcast_audio.sql.
    audio_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- 8. REVIEW_LOGS TABLE  (RECONSTRUCTED FROM CODE)
-- One row per quiz card review. Feeds FSRS parameter optimisation.
-- ==========================================
-- Source: nlp_pillars/db.py _review_log_to_dict() / _dict_to_review_log(),
-- inserted at db.py:1006 after every FSRS review and read back by
-- get_review_logs() for optimisation.
CREATE TABLE review_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- No FK to quiz_cards: see the FOREIGN KEY section below for why the
    -- reconstructed tables deliberately carry no new referential constraints.
    card_id UUID NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default_user',
    pillar_id TEXT NOT NULL,
    paper_id TEXT NOT NULL,
    -- FSRSRating enum: 1=Again, 2=Hard, 3=Good, 4=Easy.
    rating INTEGER NOT NULL CHECK (rating IN (1, 2, 3, 4)),
    review_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    -- Card state captured at review time, for FSRS optimisation.
    -- Required (not Optional) in the ReviewLog model, hence NOT NULL.
    difficulty DOUBLE PRECISION NOT NULL,
    stability DOUBLE PRECISION NOT NULL,
    retrievability DOUBLE PRECISION,  -- Optional[float] in the model
    -- Context information
    previous_due_date TIMESTAMP WITH TIME ZONE,
    days_overdue INTEGER DEFAULT 0,
    session_id TEXT,
    response_time_ms INTEGER
);

-- ==========================================
-- 9. USER_FSRS_PARAMETERS TABLE  (RECONSTRUCTED FROM CODE)
-- Personalised FSRS weights, globally or per pillar.
-- ==========================================
-- Source: nlp_pillars/db.py _user_fsrs_parameters_to_dict() /
-- _dict_to_user_fsrs_parameters(), used by get_user_fsrs_parameters() and
-- upsert_user_fsrs_parameters().
--
-- NOTE the off-by-one in the model's own docstring: UserFSRSParameters says
-- "FSRS algorithm weights (17 parameters)" but declares w0 THROUGH w17, and
-- from_weights_list() raises unless it is handed exactly 18. There are 18
-- weight columns here, w0..w17.
CREATE TABLE user_fsrs_parameters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL DEFAULT 'default_user',
    -- NULL means "global parameters"; get_user_fsrs_parameters() tries the
    -- pillar-specific row first and falls back to the row where this IS NULL.
    pillar_id TEXT,

    -- Defaults mirror the Pydantic field defaults exactly.
    w0  DOUBLE PRECISION DEFAULT 0.4,   -- Initial stability for new cards
    w1  DOUBLE PRECISION DEFAULT 0.6,   -- Initial stability for learning cards
    w2  DOUBLE PRECISION DEFAULT 2.4,   -- Initial stability multiplier
    w3  DOUBLE PRECISION DEFAULT 5.8,   -- Initial difficulty offset
    w4  DOUBLE PRECISION DEFAULT 4.93,  -- Difficulty weight for Again
    w5  DOUBLE PRECISION DEFAULT 0.94,  -- Difficulty weight for Hard
    w6  DOUBLE PRECISION DEFAULT 0.86,  -- Difficulty weight for Good
    w7  DOUBLE PRECISION DEFAULT 0.01,  -- Difficulty weight for Easy
    w8  DOUBLE PRECISION DEFAULT 1.49,  -- Stability multiplier for Again
    w9  DOUBLE PRECISION DEFAULT 0.14,  -- Stability multiplier for Hard
    w10 DOUBLE PRECISION DEFAULT 0.94,  -- Stability multiplier for Good
    w11 DOUBLE PRECISION DEFAULT 2.18,  -- Stability multiplier for Easy
    w12 DOUBLE PRECISION DEFAULT 0.05,  -- Difficulty decay for Again
    w13 DOUBLE PRECISION DEFAULT 0.34,  -- Difficulty decay for Hard
    w14 DOUBLE PRECISION DEFAULT 0.67,  -- Difficulty decay for Good
    w15 DOUBLE PRECISION DEFAULT 2.74,  -- Difficulty decay for Easy
    w16 DOUBLE PRECISION DEFAULT 0.0,   -- Forgetting curve parameter
    w17 DOUBLE PRECISION DEFAULT 2.0,   -- Stability increase parameter

    -- Metadata
    review_count INTEGER DEFAULT 0,        -- Reviews used for optimization
    last_optimized TIMESTAMP WITH TIME ZONE,
    optimization_score DOUBLE PRECISION,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- INFERRED: upsert_user_fsrs_parameters() treats (user_id, pillar_id) as the
-- identity of a row — it updates on that pair and only inserts if the update
-- affected nothing. A plain UNIQUE(user_id, pillar_id) would NOT constrain the
-- global rows, because in SQL every NULL is distinct, so two global rows for
-- the same user could coexist. Two partial indexes cover both cases.
CREATE UNIQUE INDEX uq_user_fsrs_parameters_pillar
    ON user_fsrs_parameters(user_id, pillar_id) WHERE pillar_id IS NOT NULL;
CREATE UNIQUE INDEX uq_user_fsrs_parameters_global
    ON user_fsrs_parameters(user_id) WHERE pillar_id IS NULL;

-- ==========================================
-- 10. PAPER_CITATIONS TABLE  (RECONSTRUCTED FROM CODE)
-- Citation edges between papers, used for citation-network discovery.
-- ==========================================
-- Source: nlp_pillars/db.py _paper_citation_to_dict() / _dict_to_paper_citation(),
-- with call sites add_paper_citations(), get_citations_for_paper(),
-- citation_exists() and get_citation_network_papers().
CREATE TABLE paper_citations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT NOT NULL,
    -- Deliberately NOT a foreign key to papers(id): cited papers are returned
    -- by external APIs (Semantic Scholar) and are usually NOT rows in `papers`.
    -- An FK here would reject most real citation inserts.
    cited_paper_id TEXT NOT NULL,
    citation_direction TEXT NOT NULL CHECK (citation_direction IN ('outgoing', 'incoming')),
    is_influential BOOLEAN DEFAULT FALSE,
    citation_context TEXT,  -- Text context around the citation
    source TEXT DEFAULT 'semantic_scholar',  -- Source API
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- INFERRED UNIQUE: add_paper_citations() explicitly swallows insert errors
-- whose text contains 'duplicate', 'conflict' or 'unique' ("Skip duplicates
-- silently"), and citation_exists() probes exactly this triple. Both only make
-- sense against a uniqueness constraint on these three columns.
CREATE UNIQUE INDEX uq_paper_citations_edge
    ON paper_citations(paper_id, cited_paper_id, citation_direction);

-- ==========================================
-- 11. SCRIPTS TABLE  (PLACEHOLDER — SEE CAVEAT)
-- ==========================================
-- CAVEAT: unlike every other table here, `scripts` has NO live call site.
-- Its single mention in the codebase is a commented-out example inside the
-- docstring of get_script_from_db() at webui/routers/api/script_download.py:234
-- ("# - script = await supabase.table('scripts').select('*')..."), a function
-- whose body is `return None  # Replace with actual implementation`.
--
-- The table is created anyway so the REST surface is complete and nothing is
-- removed, but its shape is taken from the "Expected structure" block in that
-- same docstring, not from any executed code. This one is the least trustworthy
-- definition in the file — treat it as a stub, not a recovered table.
CREATE TABLE scripts (
    id TEXT PRIMARY KEY,  -- docstring shows a string id ("script_123"), not a UUID
    title TEXT,
    content TEXT,            -- plain text content, OR sections below
    sections JSONB,          -- array of {type, speaker?, text} objects
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- ==========================================
-- 12. PIPELINE_RUNS TABLE
-- ==========================================
-- Not reconstructed — this table is new (migration 009). It records one row per
-- pipeline execution so the web UI can return immediately and poll for progress
-- instead of holding the HTTP request open for the whole run.
--
-- Postgres and not an in-memory registry because the scheduler runs the same
-- pipeline in a SEPARATE container, so webui-process state could never see a
-- nightly run.
CREATE TABLE pipeline_runs (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pillar_id        TEXT NOT NULL,
    trigger_source   TEXT NOT NULL
        CONSTRAINT pipeline_runs_trigger_source_check
        CHECK (trigger_source IN ('ui_pipeline', 'ui_select', 'ui_discover',
                                  'scheduler', 'ui_podcast_audio', 'ui_upload',
                                  'ui_podcast_script')),
    kind             TEXT NOT NULL
        CONSTRAINT pipeline_runs_kind_check
        CHECK (kind IN ('run_daily', 'process_selected', 'discover',
                        'podcast_audio', 'upload', 'podcast_script')),
    -- NOT NULL with a default on purpose: TableQuery.eq() renders Python None as the
    -- string "None", so nothing here may be filtered while nullable. Lookups use
    -- status, never "finished_at IS NULL".
    status           TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'succeeded', 'failed',
                          'cancelled', 'interrupted')),
    current_stage    TEXT,
    papers_processed INTEGER NOT NULL DEFAULT 0,
    papers_failed    INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    created_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    started_at       TIMESTAMP WITH TIME ZONE,
    finished_at      TIMESTAMP WITH TIME ZONE,
    heartbeat_at     TIMESTAMP WITH TIME ZONE,
    -- Terminal payload for the run kinds that produce one. 'discover' stores the
    -- candidate list the user chooses from, 'podcast_audio' the generated episode,
    -- 'upload' the paper it added and what the post-upload steps did,
    -- 'podcast_script' the generated script metadata (and the full text when
    -- saved=false); the rest leave it NULL. Written once, at the end, in a single
    -- PATCH — so unlike the stage list it has no read-modify-write problem and does
    -- not need a child table.
    result           JSONB
);

-- ==========================================
-- 13. PIPELINE_RUN_STAGES TABLE
-- ==========================================
-- One row per stage of a run, seeded 'pending' at run creation so the UI has a full
-- skeleton before work starts. A child table and NOT a JSONB column on pipeline_runs:
-- PostgREST cannot partially update JSONB, so an array column would mean
-- read-modify-write eleven times per run from a worker thread while the browser polls
-- the same row, which loses updates.
CREATE TABLE pipeline_run_stages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES pipeline_runs(id) ON DELETE CASCADE,
    seq         SMALLINT NOT NULL,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    detail      TEXT,
    started_at  TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    UNIQUE (run_id, seq)
);

-- ==========================================
-- TRIGGERS
-- ==========================================

-- pillars.updated_at is read by the application but never written by it.
-- Maintain it in the database so the column means what the code assumes.
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = timezone('utc', now());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_pillars_updated_at
    BEFORE UPDATE ON pillars
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- user_fsrs_parameters.updated_at IS written explicitly by
-- _user_fsrs_parameters_to_dict(), so it gets no trigger — the application
-- value would only be overwritten.

-- ==========================================
-- FOREIGN KEY CONSTRAINTS
-- Link tables together with referential integrity
-- ==========================================
--
-- SCOPE NOTE: only the three original foreign keys are defined. No FK was
-- added for any reconstructed table, and in particular none from *.pillar_id
-- to pillars(id), for two reasons:
--   1. Nothing proves the old database had them. delete_pillar() at
--      db.py:1545 checks for associated papers with an explicit SELECT and
--      refuses the delete in application code — which is what you write when
--      the database is NOT enforcing the relationship for you.
--   2. Papers, queue entries and citations are written for pillar ids and
--      external paper ids that may not have rows yet. A wrong FK here fails
--      inserts at runtime with an opaque PostgREST error, which is a worse
--      outcome for a personal tool than a permitted orphan row.
-- Indexes are still provided on every one of those columns below.

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
-- webui orders the papers list by added_at.desc
CREATE INDEX idx_papers_added_at ON papers(added_at);

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
-- FSRS scheduling reads: due cards are found by next_review_date per user.
CREATE INDEX idx_quiz_cards_next_review ON quiz_cards(next_review_date, pillar_id);
CREATE INDEX idx_quiz_cards_user_id ON quiz_cards(user_id);

-- Pillars table indexes
-- get_pillars() orders by created_at ascending.
CREATE INDEX idx_pillars_created_at ON pillars(created_at);

-- Podcast scripts indexes
CREATE INDEX idx_podcast_scripts_pillar_id ON podcast_scripts(pillar_id);
CREATE INDEX idx_podcast_scripts_paper_id ON podcast_scripts(paper_id);
CREATE INDEX idx_podcast_scripts_created_at ON podcast_scripts(created_at);

-- Review logs indexes
CREATE INDEX idx_review_logs_card_id ON review_logs(card_id);
-- get_review_logs() filters by user_id and optionally pillar_id.
CREATE INDEX idx_review_logs_user_pillar ON review_logs(user_id, pillar_id);
CREATE INDEX idx_review_logs_timestamp ON review_logs(review_timestamp);

-- Paper citations indexes
-- get_citations_for_paper() filters on paper_id, optionally + citation_direction.
CREATE INDEX idx_paper_citations_paper_id ON paper_citations(paper_id, citation_direction);
CREATE INDEX idx_paper_citations_cited_paper_id ON paper_citations(cited_paper_id);

-- Pipeline run indexes
-- The partial unique index is the one-active-run-per-pillar guard. It is a
-- correctness constraint, not a performance index: a check-then-insert in Python
-- races, this does not. A duplicate surfaces through PostgREST as 409 / 23505.
--
-- Uploads and podcast_script runs are exempt (migrations 015 / 016). The guard is
-- about two writers driving one pillar's queue at once; both exempted kinds are
-- scoped to one paper the user handed over, so refusing them because a discovery
-- run is in flight would refuse work for a conflict that does not exist.
CREATE UNIQUE INDEX pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running')
      AND kind NOT IN ('upload', 'podcast_script');
CREATE INDEX idx_pipeline_runs_pillar ON pipeline_runs(pillar_id, created_at DESC);
CREATE INDEX idx_pipeline_runs_active ON pipeline_runs(status)
    WHERE status IN ('pending', 'running');
CREATE INDEX idx_pipeline_run_stages_run ON pipeline_run_stages(run_id, seq);

COMMENT ON TABLE pipeline_runs IS
    'One row per pipeline execution, from any trigger. status=interrupted means the '
    'process died mid-run and the startup sweep found the row, NOT that the pipeline '
    'failed - a killed process cannot write its own epitaph.';
COMMENT ON TABLE pipeline_run_stages IS
    'Per-stage progress for a run. Seeded pending at run creation; seq matches the '
    'Step N ordering in nlp_pillars/orchestrator.py.';

-- ==========================================
-- VALIDATION COMMENTS
-- ==========================================

-- This schema ensures:
-- ✅ Primary Keys: UUID for most tables, TEXT for papers (DOI/arXiv ID), TEXT slug for pillars
-- ✅ Foreign Key Constraints: original child tables reference papers.id with CASCADE DELETE
-- ✅ Pillar Isolation: every content table has a pillar_id TEXT NOT NULL column
-- ✅ Data Types: TEXT for strings, TIMESTAMP WITH TIME ZONE for dates, JSONB for arrays/objects,
--    BOOLEAN, INTEGER, REAL/DOUBLE PRECISION for floats
-- ✅ Defaults: timezone('utc', now()) for creation timestamps, model defaults mirrored elsewhere
-- ✅ Indexes: on foreign keys, pillar_id, and every column the code sorts or filters by
-- ✅ Constraints: CHECK constraints for difficulty, question types, FSRS card state, ratings,
--    citation direction, confidence scores, ease factors, papers_per_day bounds
-- ⚠️  Reconstructed from application code, NOT recovered from the old database.

COMMENT ON TABLE papers IS 'Research paper metadata with pillar isolation';
COMMENT ON TABLE paper_queue IS 'Papers queued for processing with priority ordering';
COMMENT ON TABLE notes IS 'Structured notes extracted from papers by AI agents';
COMMENT ON TABLE lessons IS 'Synthesized lessons and takeaways from papers';
COMMENT ON TABLE quiz_cards IS 'Spaced repetition quiz cards for learning reinforcement';
COMMENT ON TABLE pillars IS 'Dynamic learning pillars; reconstructed from code, id is a slug';
COMMENT ON TABLE podcast_scripts IS 'Generated single-host podcast scripts; reconstructed from code';
COMMENT ON TABLE review_logs IS 'Per-review FSRS history; reconstructed from code';
COMMENT ON TABLE user_fsrs_parameters IS 'Personalised FSRS weights w0-w17; reconstructed from code';
COMMENT ON TABLE paper_citations IS 'Citation edges for discovery; reconstructed from code';
COMMENT ON TABLE scripts IS 'STUB: no live call site, shape taken from a docstring example only';

COMMENT ON COLUMN papers.pillar_id IS 'Learning pillar (slug) for content isolation';
COMMENT ON COLUMN papers.added_at IS 'Required by the web UI papers list; db.py writes created_at instead';
COMMENT ON COLUMN quiz_cards.ease_factor IS 'SM-2 algorithm ease factor (min 1.3)';
COMMENT ON COLUMN quiz_cards.interval IS 'Days until next review in spaced repetition';
COMMENT ON COLUMN pillars.id IS 'URL-friendly slug generated from name by _generate_slug()';
COMMENT ON COLUMN pillars.updated_at IS 'Maintained by trigger; the application reads but never writes it';
COMMENT ON COLUMN user_fsrs_parameters.pillar_id IS 'NULL means global parameters for the user';
