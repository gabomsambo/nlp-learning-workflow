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
-- inserted by services/quiz_reviews.py before the card row is touched, read
-- back to fold a card's schedule (fsrs_fold.py) and by get_review_logs() for
-- optimisation. This table, not quiz_cards, is the source of truth.
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
    response_time_ms INTEGER,
    -- Where the review was made. Added by docs/migrations/025_review_log_origin.sql.
    -- id is also the review's idempotency key: the API inserts its event_id here.
    device_id TEXT,  -- NULL for the desktop app
    origin TEXT NOT NULL DEFAULT 'desktop' CHECK (origin IN ('desktop', 'cloud'))
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
                                  'ui_podcast_script', 'ui_podcast_factcheck',
                                  'auto_podcast_factcheck', 'ui_playground',
                                  'ui_map')),
    kind             TEXT NOT NULL
        CONSTRAINT pipeline_runs_kind_check
        CHECK (kind IN ('run_daily', 'process_selected', 'discover',
                        'podcast_audio', 'upload', 'podcast_script',
                        'podcast_factcheck', 'playground_explain',
                        'playground_align', 'map_build')),
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
    -- saved=false). 'podcast_factcheck' stores script_id at insert so the
    -- one-active-per-script index can see it, then replaces that object with
    -- counts when the run finishes. 'playground_explain' stores playground_id
    -- at insert for the same reason, and 'playground_align' script_id plus
    -- playground_id. The rest leave it NULL. Written in a
    -- single PATCH — so unlike the stage list it has no read-modify-write
    -- problem and does not need a child table.
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
-- PAPER TEXT, CHUNKS, AND EMBEDDINGS
-- Source of truth for paper text and for the vectors computed from it.
-- Same objects as docs/migrations/017_paper_text.sql,
-- docs/migrations/018_chunk_embeddings.sql,
-- docs/migrations/019_vector_index_state.sql, and
-- docs/migrations/021_paper_vectors.sql, which existing volumes apply by
-- hand. Vector columns are real[] (float4), not bytea and not pgvector:
-- PostgREST returns a JSON array. Qdrant is a rebuildable projection of
-- the chunk rows (019). paper_chunks.id is the surrogate chunk_embeddings
-- references. paper_vectors has no foreign key: an external paper is not a
-- row in papers.
-- ==========================================
CREATE TABLE paper_texts (
    paper_id TEXT PRIMARY KEY,
    full_text TEXT NOT NULL,
    extractor TEXT NOT NULL,
    extracted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now())
);

CREATE TABLE paper_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK (token_count >= 0),
    chunker TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    chunk_size INTEGER NOT NULL CHECK (chunk_size > 0),
    chunk_overlap INTEGER NOT NULL CHECK (chunk_overlap >= 0),
    encoding TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now())
);

CREATE UNIQUE INDEX paper_chunks_paper_idx_version
    ON paper_chunks (paper_id, chunk_index, chunker_version);

-- One vector per chunk per model. real[] is float4. The CHECK keeps dims
-- honest. Unique (chunk_id, model_id) is what makes the backfill idempotent.
CREATE TABLE chunk_embeddings (
    chunk_id UUID NOT NULL,
    model_id TEXT NOT NULL CHECK (btrim(model_id) <> ''),
    dims INTEGER NOT NULL CHECK (dims > 0),
    embedding REAL[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT chunk_embeddings_dims_match CHECK (cardinality(embedding) = dims)
);

CREATE UNIQUE INDEX chunk_embeddings_chunk_model
    ON chunk_embeddings (chunk_id, model_id);

-- Which chunks are already in which Qdrant. Same object as
-- docs/migrations/019_vector_index_state.sql (hand-applied on an existing
-- volume). No foreign key: re-ingest deletes the chunk, and the state row
-- has to survive so the projector can delete the Qdrant point.
CREATE TABLE vector_index_state (
    target TEXT NOT NULL CHECK (target IN ('local', 'cloud')),
    chunk_id UUID NOT NULL,
    model_id TEXT NOT NULL CHECK (btrim(model_id) <> ''),
    indexed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    PRIMARY KEY (target, chunk_id, model_id)
);

CREATE INDEX vector_index_state_target_model
    ON vector_index_state (target, model_id);

-- What the cloud reader already holds. Same object as
-- docs/migrations/026_cloud_publish_state.sql (hand-applied on an existing
-- volume). No foreign key: a deleted paper's row outlives the paper until the
-- publisher has told the cloud to drop the document.
CREATE TABLE cloud_publish_state (
    doc_key TEXT PRIMARY KEY CHECK (btrim(doc_key) <> ''),
    content_hash TEXT NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    byte_size BIGINT NOT NULL DEFAULT 0 CHECK (byte_size >= 0),
    published_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now())
);

-- docs/migrations/027_cloud_sync_state.sql (hand-applied on an existing volume).
-- Cursors for the cloud reader's review events: reviews_seq (folded through) and
-- reviews_pending (a pull in progress). Not a source of truth.
CREATE TABLE cloud_sync_state (
    key TEXT PRIMARY KEY CHECK (btrim(key) <> ''),
    value BIGINT NOT NULL DEFAULT 0 CHECK (value >= 0),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now())
);

-- Paper Playground. Same object as docs/migrations/028_playgrounds.sql
-- (hand-applied on an existing volume). One current playground per paper; the
-- stored shape of notebook, steps and datasets is docs/PLAYGROUND.md.
CREATE TABLE playgrounds (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id      TEXT NOT NULL UNIQUE REFERENCES papers(id) ON DELETE CASCADE,
    title         TEXT NOT NULL,
    source_kind   TEXT NOT NULL DEFAULT 'notebook'
        CHECK (source_kind IN ('notebook', 'generated')),
    notebook      JSONB NOT NULL,
    notebook_name TEXT,
    steps         JSONB NOT NULL DEFAULT '[]'::jsonb,
    datasets      JSONB NOT NULL DEFAULT '[]'::jsonb,
    status        TEXT NOT NULL DEFAULT 'imported'
        CHECK (status IN ('imported', 'ready', 'partial', 'failed')),
    model         TEXT,
    error         TEXT,
    -- docs/migrations/030_playground_setup_code.sql. Run before any cell on
    -- every runner start and restart. NULL: use the suggestion.
    setup_code    TEXT,
    created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    updated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now())
);

-- Playground follow-along. Same object as docs/migrations/029_playground_links.sql
-- (hand-applied on an existing volume). Which script lines of an episode are
-- about which code cell of a playground; see docs/PLAYGROUND.md.
CREATE TABLE playground_links (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_id     UUID NOT NULL REFERENCES podcast_scripts(id) ON DELETE CASCADE,
    playground_id UUID NOT NULL REFERENCES playgrounds(id) ON DELETE CASCADE,
    mapping       JSONB NOT NULL DEFAULT '[]'::jsonb,
    model         TEXT,
    status        TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'ready', 'failed')),
    error         TEXT,
    script_sha256 TEXT,
    created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    updated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    UNIQUE (script_id, playground_id)
);
CREATE INDEX idx_playground_links_playground ON playground_links (playground_id);


-- One transaction: a failed insert rolls back the delete.
CREATE OR REPLACE FUNCTION replace_paper_text_and_chunks(
    p_paper_id TEXT,
    p_full_text TEXT,
    p_extractor TEXT,
    p_chunks JSONB
) RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    inserted INTEGER;
BEGIN
    IF p_paper_id IS NULL OR btrim(p_paper_id) = '' THEN
        RAISE EXCEPTION 'paper text write missing paper_id';
    END IF;
    IF p_full_text IS NULL OR btrim(p_full_text) = '' THEN
        RAISE EXCEPTION 'refusing to store empty paper text for %', p_paper_id;
    END IF;
    IF p_extractor IS NULL OR btrim(p_extractor) = '' THEN
        RAISE EXCEPTION 'paper text write for % missing extractor', p_paper_id;
    END IF;
    IF p_chunks IS NULL
       OR jsonb_typeof(p_chunks) <> 'array'
       OR jsonb_array_length(p_chunks) = 0 THEN
        RAISE EXCEPTION 'refusing to replace chunks for % with an empty set', p_paper_id;
    END IF;

    INSERT INTO paper_texts (paper_id, full_text, extractor, extracted_at)
    VALUES (p_paper_id, p_full_text, p_extractor, timezone('utc', now()))
    ON CONFLICT (paper_id) DO UPDATE
        SET full_text = EXCLUDED.full_text,
            extractor = EXCLUDED.extractor,
            extracted_at = timezone('utc', now());

    DELETE FROM paper_chunks WHERE paper_id = p_paper_id;

    INSERT INTO paper_chunks (
        paper_id, chunk_index, text, token_count,
        chunker, chunker_version, chunk_size, chunk_overlap, encoding,
        content_hash
    )
    SELECT
        p_paper_id,
        (elem->>'chunk_index')::INTEGER,
        elem->>'text',
        (elem->>'token_count')::INTEGER,
        elem->>'chunker',
        elem->>'chunker_version',
        (elem->>'chunk_size')::INTEGER,
        (elem->>'chunk_overlap')::INTEGER,
        elem->>'encoding',
        elem->>'content_hash'
    FROM jsonb_array_elements(p_chunks) AS elem;

    GET DIAGNOSTICS inserted = ROW_COUNT;
    IF inserted <> jsonb_array_length(p_chunks) THEN
        RAISE EXCEPTION 'stored % chunks for %, expected %',
            inserted, p_paper_id, jsonb_array_length(p_chunks);
    END IF;
    RETURN inserted;
END;
$$;

-- Chunks that have no row yet for this model. p_paper_id NULL is every paper.
CREATE OR REPLACE FUNCTION chunks_missing_embeddings(
    p_model_id TEXT,
    p_paper_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 200,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    chunk_id UUID,
    paper_id TEXT,
    chunk_index INTEGER,
    chunk_text TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_model_id IS NULL OR btrim(p_model_id) = '' THEN
        RAISE EXCEPTION 'chunks_missing_embeddings missing model_id';
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        RAISE EXCEPTION 'chunks_missing_embeddings limit must be positive';
    END IF;
    IF p_offset IS NULL OR p_offset < 0 THEN
        RAISE EXCEPTION 'chunks_missing_embeddings offset must be >= 0';
    END IF;

    RETURN QUERY
    SELECT c.id, c.paper_id, c.chunk_index, c.text
    FROM paper_chunks c
    WHERE (p_paper_id IS NULL OR c.paper_id = p_paper_id)
      AND NOT EXISTS (
          SELECT 1
          FROM chunk_embeddings e
          WHERE e.chunk_id = c.id
            AND e.model_id = p_model_id
      )
    ORDER BY c.paper_id, c.chunk_index, c.id
    LIMIT p_limit OFFSET p_offset;
END;
$$;

-- Chunks that have an embedding for this model, joined to the paper's pillar.
CREATE OR REPLACE FUNCTION chunks_with_embeddings(
    p_model_id TEXT,
    p_paper_id TEXT DEFAULT NULL,
    p_limit INTEGER DEFAULT 200,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    chunk_id UUID,
    paper_id TEXT,
    pillar_id TEXT,
    chunk_index INTEGER,
    chunk_text TEXT,
    dims INTEGER,
    embedding REAL[]
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_model_id IS NULL OR btrim(p_model_id) = '' THEN
        RAISE EXCEPTION 'chunks_with_embeddings missing model_id';
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        RAISE EXCEPTION 'chunks_with_embeddings limit must be positive';
    END IF;
    IF p_offset IS NULL OR p_offset < 0 THEN
        RAISE EXCEPTION 'chunks_with_embeddings offset must be >= 0';
    END IF;

    RETURN QUERY
    SELECT c.id, c.paper_id, p.pillar_id, c.chunk_index, c.text, e.dims, e.embedding
    FROM paper_chunks c
    JOIN chunk_embeddings e
      ON e.chunk_id = c.id
     AND e.model_id = p_model_id
    JOIN papers p
      ON p.id = c.paper_id
    WHERE p_paper_id IS NULL OR c.paper_id = p_paper_id
    ORDER BY c.paper_id, c.chunk_index, c.id
    LIMIT p_limit OFFSET p_offset;
END;
$$;

-- State rows whose chunk has been deleted.
CREATE OR REPLACE FUNCTION projection_stale_chunks(
    p_target TEXT,
    p_model_id TEXT,
    p_limit INTEGER DEFAULT 200,
    p_offset INTEGER DEFAULT 0
) RETURNS TABLE (
    chunk_id UUID
)
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_target IS NULL OR p_target NOT IN ('local', 'cloud') THEN
        RAISE EXCEPTION 'projection_stale_chunks target must be local or cloud';
    END IF;
    IF p_model_id IS NULL OR btrim(p_model_id) = '' THEN
        RAISE EXCEPTION 'projection_stale_chunks missing model_id';
    END IF;
    IF p_limit IS NULL OR p_limit < 1 THEN
        RAISE EXCEPTION 'projection_stale_chunks limit must be positive';
    END IF;
    IF p_offset IS NULL OR p_offset < 0 THEN
        RAISE EXCEPTION 'projection_stale_chunks offset must be >= 0';
    END IF;

    RETURN QUERY
    SELECT s.chunk_id
    FROM vector_index_state s
    WHERE s.target = p_target
      AND s.model_id = p_model_id
      AND NOT EXISTS (
          SELECT 1 FROM paper_chunks c WHERE c.id = s.chunk_id
      )
    ORDER BY s.chunk_id
    LIMIT p_limit OFFSET p_offset;
END;
$$;

-- 03-grants.sql covers tables created above. It does not grant functions.
GRANT EXECUTE ON FUNCTION replace_paper_text_and_chunks(TEXT, TEXT, TEXT, JSONB) TO web_anon;
GRANT EXECUTE ON FUNCTION chunks_missing_embeddings(TEXT, TEXT, INTEGER, INTEGER) TO web_anon;
GRANT EXECUTE ON FUNCTION chunks_with_embeddings(TEXT, TEXT, INTEGER, INTEGER) TO web_anon;
GRANT EXECUTE ON FUNCTION projection_stale_chunks(TEXT, TEXT, INTEGER, INTEGER) TO web_anon;

-- One title+abstract vector per paper per model. Graph related_to reads
-- these rows. Chunk embeddings stay the source the search index is rebuilt
-- from. Unique (ref_kind, ref_id, model_id, source) is what makes
-- embeddings papers idempotent. dims is checked against the array length.
CREATE TABLE paper_vectors (
    ref_kind TEXT NOT NULL CHECK (ref_kind IN ('paper', 'external_paper')),
    ref_id TEXT NOT NULL CHECK (btrim(ref_id) <> ''),
    model_id TEXT NOT NULL CHECK (btrim(model_id) <> ''),
    source TEXT NOT NULL DEFAULT 'title_abstract' CHECK (btrim(source) <> ''),
    dims INTEGER NOT NULL CHECK (dims > 0),
    vec REAL[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT paper_vectors_dims_match CHECK (cardinality(vec) = dims)
);

CREATE UNIQUE INDEX paper_vectors_ref_model_source
    ON paper_vectors (ref_kind, ref_id, model_id, source);

-- ==========================================
-- KNOWLEDGE GRAPH
-- Same objects as docs/migrations/020_knowledge_graph.sql. Existing volumes
-- apply that file by hand. These rows are a projection: graph build deletes
-- the derived ones and writes them again.
-- Concept endpoints in graph_edges are the canonical name, not concepts.id.
-- ==========================================
CREATE TABLE concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL CHECK (btrim(name) <> ''),
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT concepts_aliases_array CHECK (jsonb_typeof(aliases) = 'array')
);

CREATE UNIQUE INDEX concepts_name_key ON concepts (name);

CREATE TABLE paper_concepts (
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    concept_id UUID NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    source TEXT NOT NULL CHECK (btrim(source) <> ''),
    weight REAL,
    PRIMARY KEY (paper_id, concept_id)
);

CREATE INDEX paper_concepts_concept_id ON paper_concepts (concept_id);

CREATE TABLE graph_edges (
    src_kind TEXT NOT NULL CHECK (btrim(src_kind) <> ''),
    src_id TEXT NOT NULL CHECK (btrim(src_id) <> ''),
    dst_kind TEXT NOT NULL CHECK (btrim(dst_kind) <> ''),
    dst_id TEXT NOT NULL CHECK (btrim(dst_id) <> ''),
    kind TEXT NOT NULL CHECK (btrim(kind) <> ''),
    weight REAL,
    provenance TEXT NOT NULL CHECK (btrim(provenance) <> ''),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now())
);

CREATE UNIQUE INDEX graph_edges_edge_key
    ON graph_edges (src_kind, src_id, dst_kind, dst_id, kind);

CREATE INDEX graph_edges_src ON graph_edges (src_kind, src_id);
CREATE INDEX graph_edges_dst ON graph_edges (dst_kind, dst_id);
CREATE INDEX graph_edges_provenance ON graph_edges (provenance);

COMMENT ON TABLE concepts IS
    'Canonical concept names built from notes.key_terms. Rebuildable.';
COMMENT ON TABLE paper_concepts IS
    'Paper to concept. source key_terms is replaced by graph build.';
COMMENT ON TABLE graph_edges IS
    'Directed learner-graph edges. Unique per endpoints and kind. Derived rows are rebuilt.';
COMMENT ON COLUMN graph_edges.provenance IS
    'derived:structure, derived:key_terms, derived:embedding:<model>, semantic_scholar, user, or llm:<model>.';

CREATE OR REPLACE FUNCTION replace_derived_graph(
    p_concepts JSONB,
    p_links JSONB,
    p_edges JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    concept_count INTEGER;
    link_count INTEGER;
    edge_count INTEGER;
    got INTEGER;
BEGIN
    IF p_concepts IS NULL OR jsonb_typeof(p_concepts) <> 'array' THEN
        RAISE EXCEPTION 'replace_derived_graph concepts must be a JSON array';
    END IF;
    IF p_links IS NULL OR jsonb_typeof(p_links) <> 'array' THEN
        RAISE EXCEPTION 'replace_derived_graph links must be a JSON array';
    END IF;
    IF p_edges IS NULL OR jsonb_typeof(p_edges) <> 'array' THEN
        RAISE EXCEPTION 'replace_derived_graph edges must be a JSON array';
    END IF;

    INSERT INTO concepts (name, aliases)
    SELECT btrim(elem->>'name'),
           CASE
               WHEN jsonb_typeof(elem->'aliases') = 'array' THEN elem->'aliases'
               ELSE '[]'::jsonb
           END
    FROM jsonb_array_elements(p_concepts) AS elem
    WHERE btrim(elem->>'name') <> ''
    ON CONFLICT (name) DO UPDATE
        SET aliases = EXCLUDED.aliases,
            updated_at = timezone('utc', now());

    DELETE FROM paper_concepts WHERE source = 'key_terms';

    INSERT INTO paper_concepts (paper_id, concept_id, source, weight)
    SELECT btrim(elem->>'paper_id'),
           c.id,
           COALESCE(NULLIF(btrim(elem->>'source'), ''), 'key_terms'),
           (elem->>'weight')::real
    FROM jsonb_array_elements(p_links) AS elem
    JOIN concepts c ON c.name = btrim(elem->>'name')
    WHERE btrim(elem->>'paper_id') <> ''
      AND btrim(elem->>'name') <> ''
    ON CONFLICT (paper_id, concept_id) DO UPDATE
        SET source = EXCLUDED.source,
            weight = EXCLUDED.weight
        WHERE paper_concepts.source NOT LIKE 'llm:%';

    GET DIAGNOSTICS link_count = ROW_COUNT;
    DELETE FROM concepts c
    WHERE NOT EXISTS (
        SELECT 1 FROM paper_concepts pc WHERE pc.concept_id = c.id
    );

    DELETE FROM graph_edges
    WHERE provenance LIKE 'derived:%'
       OR provenance = 'semantic_scholar';

    INSERT INTO graph_edges (
        src_kind, src_id, dst_kind, dst_id, kind, weight, provenance
    )
    SELECT btrim(elem->>'src_kind'),
           btrim(elem->>'src_id'),
           btrim(elem->>'dst_kind'),
           btrim(elem->>'dst_id'),
           btrim(elem->>'kind'),
           (elem->>'weight')::real,
           btrim(elem->>'provenance')
    FROM jsonb_array_elements(p_edges) AS elem
    ON CONFLICT (src_kind, src_id, dst_kind, dst_id, kind) DO UPDATE
        SET weight = EXCLUDED.weight,
            provenance = EXCLUDED.provenance
        WHERE graph_edges.provenance NOT LIKE 'llm:%';

    GET DIAGNOSTICS edge_count = ROW_COUNT;
    SELECT count(*) INTO concept_count FROM concepts;
    SELECT count(*) INTO got FROM graph_edges;

    RETURN jsonb_build_object(
        'concepts', concept_count,
        'paper_concepts', link_count,
        'edges_written', edge_count,
        'edges', got
    );
END;
$$;

GRANT EXECUTE ON FUNCTION replace_derived_graph(JSONB, JSONB, JSONB) TO web_anon;

-- Vocabulary-aware summarization replaces only one paper's paid concept
-- choices. The graph rebuild above preserves these llm:% rows when its free
-- key_terms projection names the same concept.
CREATE OR REPLACE FUNCTION replace_paper_concepts(
    p_paper_id TEXT,
    p_concepts JSONB,
    p_source TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    link_count INTEGER;
    edge_count INTEGER;
BEGIN
    IF btrim(COALESCE(p_paper_id, '')) = '' THEN
        RAISE EXCEPTION 'replace_paper_concepts paper_id must not be empty';
    END IF;
    IF p_concepts IS NULL OR jsonb_typeof(p_concepts) <> 'array' THEN
        RAISE EXCEPTION 'replace_paper_concepts concepts must be a JSON array';
    END IF;
    IF jsonb_array_length(p_concepts) > 10 THEN
        RAISE EXCEPTION 'replace_paper_concepts accepts at most 10 concepts';
    END IF;
    IF btrim(COALESCE(p_source, '')) NOT LIKE 'llm:%' THEN
        RAISE EXCEPTION 'replace_paper_concepts source must start with llm:';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM papers WHERE id = btrim(p_paper_id)) THEN
        RAISE EXCEPTION 'replace_paper_concepts paper % does not exist', p_paper_id;
    END IF;
    IF EXISTS (
        SELECT 1
        FROM jsonb_array_elements(p_concepts) AS elem
        WHERE btrim(COALESCE(elem->>'name', '')) = ''
    ) THEN
        RAISE EXCEPTION 'replace_paper_concepts concept names must not be empty';
    END IF;

    INSERT INTO concepts (name, aliases)
    SELECT DISTINCT btrim(elem->>'name'), '[]'::jsonb
    FROM jsonb_array_elements(p_concepts) AS elem
    ON CONFLICT (name) DO UPDATE
        SET updated_at = timezone('utc', now());

    DELETE FROM graph_edges
    WHERE src_kind = 'paper'
      AND src_id = btrim(p_paper_id)
      AND dst_kind = 'concept'
      AND kind = 'covers'
      AND provenance LIKE 'llm:%';
    DELETE FROM paper_concepts
    WHERE paper_id = btrim(p_paper_id)
      AND source LIKE 'llm:%';

    INSERT INTO paper_concepts (paper_id, concept_id, source, weight)
    SELECT btrim(p_paper_id), c.id, btrim(p_source),
           COALESCE((elem->>'weight')::real, 1.0)
    FROM jsonb_array_elements(p_concepts) AS elem
    JOIN concepts c ON c.name = btrim(elem->>'name')
    ON CONFLICT (paper_id, concept_id) DO UPDATE
        SET source = EXCLUDED.source,
            weight = EXCLUDED.weight;

    INSERT INTO graph_edges (
        src_kind, src_id, dst_kind, dst_id, kind, weight, provenance
    )
    SELECT 'paper', btrim(p_paper_id), 'concept', c.name, 'covers',
           COALESCE((elem->>'weight')::real, 1.0), btrim(p_source)
    FROM jsonb_array_elements(p_concepts) AS elem
    JOIN concepts c ON c.name = btrim(elem->>'name')
    ON CONFLICT (src_kind, src_id, dst_kind, dst_id, kind) DO UPDATE
        SET weight = EXCLUDED.weight,
            provenance = EXCLUDED.provenance;

    DELETE FROM concepts c
    WHERE NOT EXISTS (
        SELECT 1 FROM paper_concepts pc WHERE pc.concept_id = c.id
    );

    SELECT count(*) INTO link_count
    FROM paper_concepts
    WHERE paper_id = btrim(p_paper_id) AND source = btrim(p_source);
    SELECT count(*) INTO edge_count
    FROM graph_edges
    WHERE src_kind = 'paper' AND src_id = btrim(p_paper_id)
      AND dst_kind = 'concept' AND kind = 'covers'
      AND provenance = btrim(p_source);
    RETURN jsonb_build_object(
        'paper_id', btrim(p_paper_id), 'concepts', link_count,
        'edges', edge_count, 'source', btrim(p_source)
    );
END;
$$;

-- Current FSRS-6 retrievability, averaged over every card inherited through
-- a concept's papers. A new/uninitialised card contributes 0; no cards is NULL.
CREATE OR REPLACE FUNCTION concept_mastery(
    p_user_id TEXT DEFAULT 'default_user',
    p_pillar_id TEXT DEFAULT NULL
) RETURNS TABLE (
    concept_id UUID,
    name TEXT,
    retrievability DOUBLE PRECISION,
    card_count BIGINT,
    reviewed_card_count BIGINT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        c.id AS concept_id,
        c.name,
        CASE
            WHEN count(q.id) = 0 THEN NULL
            ELSE avg(
                CASE
                    WHEN q.last_review_date IS NULL
                      OR q.stability IS NULL
                      OR q.stability <= 0 THEN 0.0
                    ELSE power(
                        1.0 + (19.0 / 81.0) *
                        greatest(
                            0.0,
                            floor(
                                extract(epoch FROM (
                                    timezone('utc', now()) - q.last_review_date
                                )) / 86400.0
                            )
                        ) / q.stability,
                        -0.5
                    )
                END
            )::double precision
        END AS retrievability,
        count(q.id) AS card_count,
        count(q.id) FILTER (
            WHERE q.last_review_date IS NOT NULL
              AND q.stability IS NOT NULL
              AND q.stability > 0
        ) AS reviewed_card_count
    FROM concepts c
    JOIN paper_concepts pc ON pc.concept_id = c.id
    JOIN papers p ON p.id = pc.paper_id
    LEFT JOIN quiz_cards q
      ON q.paper_id = pc.paper_id
     AND q.user_id = COALESCE(NULLIF(btrim(p_user_id), ''), 'default_user')
    WHERE p_pillar_id IS NULL OR p.pillar_id = p_pillar_id
    GROUP BY c.id, c.name
    ORDER BY c.name;
$$;

GRANT EXECUTE ON FUNCTION replace_paper_concepts(TEXT, JSONB, TEXT) TO web_anon;
GRANT EXECUTE ON FUNCTION concept_mastery(TEXT, TEXT) TO web_anon;

-- ==========================================
-- PODCAST FACT CHECKS
-- Same objects as docs/migrations/022_podcast_fact_checks.sql. Existing
-- volumes apply that file by hand. A check reads paper_texts and writes
-- these two tables. It does not update podcast_scripts.
-- ==========================================
CREATE TABLE podcast_fact_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    script_id UUID NOT NULL REFERENCES podcast_scripts(id) ON DELETE CASCADE,
    script_sha256 TEXT NOT NULL CHECK (btrim(script_sha256) <> ''),
    paper_text_sha256 TEXT NOT NULL CHECK (btrim(paper_text_sha256) <> ''),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    models JSONB NOT NULL DEFAULT '{}'::jsonb,
    prompt_version TEXT NOT NULL CHECK (btrim(prompt_version) <> ''),
    counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT podcast_fact_checks_models_object
        CHECK (jsonb_typeof(models) = 'object'),
    CONSTRAINT podcast_fact_checks_counts_object
        CHECK (jsonb_typeof(counts) = 'object')
);

CREATE INDEX podcast_fact_checks_script_id
    ON podcast_fact_checks (script_id, created_at DESC);

CREATE TABLE podcast_claims (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fact_check_id UUID NOT NULL REFERENCES podcast_fact_checks(id) ON DELETE CASCADE,
    line_index INTEGER NOT NULL CHECK (line_index >= 0),
    span_start INTEGER NOT NULL CHECK (span_start >= 0),
    span_end INTEGER NOT NULL CHECK (span_end >= span_start),
    span_text TEXT NOT NULL,
    claim TEXT NOT NULL CHECK (btrim(claim) <> ''),
    scope TEXT NOT NULL CHECK (scope IN ('paper', 'background')),
    category TEXT NOT NULL CHECK (category IN (
        'number', 'method', 'setup', 'result', 'comparison',
        'attribution', 'limitation', 'implication', 'background'
    )),
    verify_marker BOOLEAN NOT NULL DEFAULT false,
    verdict TEXT NOT NULL CHECK (verdict IN (
        'supported', 'partially_supported', 'contradicted',
        'unsupported', 'not_checkable'
    )),
    error_type TEXT NOT NULL CHECK (error_type IN (
        'number_mismatch', 'invented_detail', 'overstatement',
        'wrong_attribution', 'method_misdescribed',
        'outside_knowledge', 'other', 'none'
    )),
    severity TEXT NOT NULL CHECK (severity IN ('high', 'medium', 'low', 'none')),
    rationale TEXT NOT NULL DEFAULT '',
    source_says TEXT NOT NULL DEFAULT '',
    evidence JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_status TEXT NOT NULL CHECK (evidence_status IN (
        'located', 'not_located', 'not_required'
    )),
    judge_trail JSONB NOT NULL DEFAULT '[]'::jsonb,
    dismissed_at TIMESTAMP WITH TIME ZONE,
    dismissed_note TEXT,
    CONSTRAINT podcast_claims_evidence_array
        CHECK (jsonb_typeof(evidence) = 'array'),
    CONSTRAINT podcast_claims_judge_trail_array
        CHECK (jsonb_typeof(judge_trail) = 'array')
);

CREATE INDEX podcast_claims_fact_check_id ON podcast_claims (fact_check_id);

CREATE OR REPLACE FUNCTION insert_podcast_fact_check(
    p_check JSONB,
    p_claims JSONB
) RETURNS UUID
LANGUAGE plpgsql
AS $$
DECLARE
    new_id UUID;
    inserted INTEGER;
BEGIN
    IF p_check IS NULL OR jsonb_typeof(p_check) <> 'object' THEN
        RAISE EXCEPTION 'fact check payload must be an object';
    END IF;
    IF p_claims IS NULL OR jsonb_typeof(p_claims) <> 'array' THEN
        RAISE EXCEPTION 'fact check claims must be an array';
    END IF;
    IF p_check->>'status' IS DISTINCT FROM 'succeeded' THEN
        RAISE EXCEPTION 'fact check insert only accepts status succeeded';
    END IF;

    INSERT INTO podcast_fact_checks (
        script_id, script_sha256, paper_text_sha256, status,
        models, prompt_version, counts, cost_usd
    ) VALUES (
        (p_check->>'script_id')::uuid,
        p_check->>'script_sha256',
        p_check->>'paper_text_sha256',
        p_check->>'status',
        COALESCE(p_check->'models', '{}'::jsonb),
        p_check->>'prompt_version',
        COALESCE(p_check->'counts', '{}'::jsonb),
        COALESCE((p_check->>'cost_usd')::numeric, 0)
    )
    RETURNING id INTO new_id;

    INSERT INTO podcast_claims (
        fact_check_id, line_index, span_start, span_end, span_text,
        claim, scope, category, verify_marker,
        verdict, error_type, severity, rationale, source_says,
        evidence, evidence_status, judge_trail
    )
    SELECT
        new_id,
        (elem->>'line_index')::integer,
        (elem->>'span_start')::integer,
        (elem->>'span_end')::integer,
        elem->>'span_text',
        elem->>'claim',
        elem->>'scope',
        elem->>'category',
        COALESCE((elem->>'verify_marker')::boolean, false),
        elem->>'verdict',
        elem->>'error_type',
        elem->>'severity',
        COALESCE(elem->>'rationale', ''),
        COALESCE(elem->>'source_says', ''),
        COALESCE(elem->'evidence', '[]'::jsonb),
        elem->>'evidence_status',
        COALESCE(elem->'judge_trail', '[]'::jsonb)
    FROM jsonb_array_elements(p_claims) AS elem;

    GET DIAGNOSTICS inserted = ROW_COUNT;
    IF inserted <> jsonb_array_length(p_claims) THEN
        RAISE EXCEPTION 'stored % claims, expected %',
            inserted, jsonb_array_length(p_claims);
    END IF;
    RETURN new_id;
END;
$$;

GRANT EXECUTE ON FUNCTION insert_podcast_fact_check(JSONB, JSONB) TO web_anon;

-- Dismiss-with-note and its append-only log. Same objects as
-- docs/migrations/033_claim_dismissals.sql.
CREATE TABLE podcast_claim_reviews (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    claim_id   UUID NOT NULL REFERENCES podcast_claims(id) ON DELETE CASCADE,
    action     TEXT NOT NULL CHECK (action IN ('dismissed', 'restored')),
    note       TEXT NOT NULL DEFAULT '' CHECK (char_length(note) <= 1000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT podcast_claim_reviews_dismiss_has_note
        CHECK (action <> 'dismissed' OR btrim(note) <> '')
);

CREATE INDEX podcast_claim_reviews_claim_id
    ON podcast_claim_reviews (claim_id, created_at);

COMMENT ON TABLE podcast_claim_reviews IS
    'Append-only log of dismissals and restores of fact-check claims. Rows are never updated.';

CREATE OR REPLACE FUNCTION set_podcast_claim_dismissal(
    p_claim_id UUID,
    p_dismissed BOOLEAN,
    p_note TEXT
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    clean TEXT := btrim(COALESCE(p_note, ''));
    updated podcast_claims%ROWTYPE;
BEGIN
    IF p_dismissed AND clean = '' THEN
        RAISE EXCEPTION 'a dismissal needs a note';
    END IF;
    IF char_length(clean) > 1000 THEN
        RAISE EXCEPTION 'a dismissal note is at most 1000 characters';
    END IF;

    UPDATE podcast_claims
       SET dismissed_at   = CASE WHEN p_dismissed THEN timezone('utc', now()) END,
           dismissed_note = CASE WHEN p_dismissed THEN clean END
     WHERE id = p_claim_id
    RETURNING * INTO updated;

    IF NOT FOUND THEN
        RETURN NULL;
    END IF;

    INSERT INTO podcast_claim_reviews (claim_id, action, note)
    VALUES (
        p_claim_id,
        CASE WHEN p_dismissed THEN 'dismissed' ELSE 'restored' END,
        clean
    );
    RETURN to_jsonb(updated);
END;
$$;

-- Explicit, not relying on ALTER DEFAULT PRIVILEGES (per-grantor; see 009).
-- The log is read and appended to by the app, never edited.
GRANT SELECT, INSERT ON podcast_claim_reviews TO web_anon;
GRANT EXECUTE ON FUNCTION set_podcast_claim_dismissal(UUID, BOOLEAN, TEXT) TO web_anon;

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
-- SCOPE NOTE: the original foreign keys are the three below, plus paper_texts
-- and paper_chunks (migration 017), which reference papers(id) because those
-- rows are written only after the paper row exists. No FK was added for any
-- other reconstructed table, and in particular none from *.pillar_id to
-- pillars(id), for two reasons:
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

-- Stored text references papers. Written only after the paper row exists.
ALTER TABLE paper_texts
ADD CONSTRAINT fk_paper_texts_paper_id
FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE;

ALTER TABLE paper_chunks
ADD CONSTRAINT fk_paper_chunks_paper_id
FOREIGN KEY (paper_id) REFERENCES papers(id) ON DELETE CASCADE;

-- Embeddings follow their chunk. Re-ingest deletes the chunk rows, and the
-- vectors go with them.
ALTER TABLE chunk_embeddings
ADD CONSTRAINT fk_chunk_embeddings_chunk_id
FOREIGN KEY (chunk_id) REFERENCES paper_chunks(id) ON DELETE CASCADE;

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
-- The fold reads one card's events in (review_timestamp, id) order.
CREATE INDEX idx_review_logs_card_fold ON review_logs(card_id, review_timestamp, id);

-- Paper citations indexes
-- get_citations_for_paper() filters on paper_id, optionally + citation_direction.
CREATE INDEX idx_paper_citations_paper_id ON paper_citations(paper_id, citation_direction);
CREATE INDEX idx_paper_citations_cited_paper_id ON paper_citations(cited_paper_id);

-- Pipeline run indexes
-- The partial unique index is the one-active-run-per-pillar guard. It is a
-- correctness constraint, not a performance index: a check-then-insert in Python
-- races, this does not. A duplicate surfaces through PostgREST as 409 / 23505.
--
-- Uploads, podcast_script, podcast_factcheck, playground_explain,
-- playground_align and map_build runs are exempt (migrations 015 / 016 / 023 /
-- 028 / 029 / 034). The guard is about two writers
-- driving one pillar's queue at once. Those kinds are scoped to one paper or
-- one script, so refusing them because a discovery run is in flight would
-- refuse work for a conflict that does not exist. One script still has at most
-- one active fact-check, one playground one active explain pass, and one
-- pillar one active map build.
CREATE UNIQUE INDEX pipeline_runs_one_active_per_pillar
    ON pipeline_runs (pillar_id)
    WHERE status IN ('pending', 'running')
      AND kind NOT IN ('upload', 'podcast_script', 'podcast_factcheck',
                       'playground_explain', 'playground_align', 'map_build');
CREATE UNIQUE INDEX pipeline_runs_one_active_map_build
    ON pipeline_runs (pillar_id)
    WHERE kind = 'map_build'
      AND status IN ('pending', 'running');
CREATE UNIQUE INDEX pipeline_runs_one_active_factcheck_per_script
    ON pipeline_runs ((result->>'script_id'))
    WHERE kind = 'podcast_factcheck'
      AND status IN ('pending', 'running');
CREATE UNIQUE INDEX pipeline_runs_one_active_explain_per_playground
    ON pipeline_runs ((result->>'playground_id'))
    WHERE kind = 'playground_explain'
      AND status IN ('pending', 'running');
CREATE UNIQUE INDEX pipeline_runs_one_active_align_per_link
    ON pipeline_runs ((result->>'script_id'), (result->>'playground_id'))
    WHERE kind = 'playground_align'
      AND status IN ('pending', 'running');
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
COMMENT ON TABLE podcast_fact_checks IS 'One finished fact-check of a podcast script against paper_texts. Does not change the script.';
COMMENT ON TABLE podcast_claims IS 'Claims from one fact-check. Evidence quotes are only those located in the paper text.';
COMMENT ON TABLE review_logs IS 'Per-review FSRS history; reconstructed from code';
COMMENT ON TABLE user_fsrs_parameters IS 'Personalised FSRS weights w0-w17; reconstructed from code';
COMMENT ON TABLE paper_citations IS 'Citation edges for discovery; reconstructed from code';
COMMENT ON TABLE scripts IS 'STUB: no live call site, shape taken from a docstring example only';
COMMENT ON TABLE paper_texts IS 'Full text of a paper. Source of truth; vector indexes are rebuilt from here.';
COMMENT ON TABLE paper_chunks IS 'Chunk text for one chunker config. Vectors live in chunk_embeddings, keyed by id.';
COMMENT ON TABLE chunk_embeddings IS 'One float32 vector per chunk per model. real[] so PostgREST returns JSON. Qdrant is rebuilt from these rows.';
COMMENT ON TABLE vector_index_state IS 'Chunks projected into one Qdrant target. Not a foreign key: the row outlives the chunk so the point can be deleted.';
COMMENT ON TABLE cloud_publish_state IS 'One row per document the cloud reader holds: its content hash and when it was uploaded. The cloud is a rebuildable projection.';
COMMENT ON COLUMN cloud_publish_state.doc_key IS 'Bundle path, e.g. papers/arxiv_3a1706.03762.json or audio/<sha256>.mp3.';
COMMENT ON COLUMN cloud_publish_state.content_hash IS 'sha256 (hex) of the bytes uploaded, as listed in the bundle manifest.';
COMMENT ON COLUMN cloud_publish_state.byte_size IS 'Size of the uploaded object, summed to keep the bucket under its cap.';
COMMENT ON TABLE cloud_sync_state IS 'Cursors for syncing with the cloud reader: reviews_seq (folded through) and reviews_pending (a pull in progress).';
COMMENT ON TABLE playgrounds IS 'Paper Playground: one notebook per paper, its study guide (steps) and cached dataset records. See docs/PLAYGROUND.md.';
COMMENT ON TABLE playground_links IS 'Playground follow-along: which script lines of an episode are about which code cell of a playground. See docs/PLAYGROUND.md.';
COMMENT ON COLUMN chunk_embeddings.embedding IS 'float4[] (PostgreSQL real[]). Chosen over bytea so the array is readable without a codec, and over pgvector so the Postgres image stays unchanged.';
COMMENT ON COLUMN chunk_embeddings.model_id IS 'Provider model id including the pinned revision, e.g. BAAI/bge-base-en-v1.5@<onnx sha>.';
COMMENT ON TABLE paper_vectors IS 'One float32 title+abstract vector per paper per model. Rebuildable. Not the chunk index.';
COMMENT ON COLUMN paper_vectors.vec IS 'float4[] (PostgreSQL real[]). Same storage choice as chunk_embeddings.embedding.';
COMMENT ON COLUMN paper_vectors.source IS 'What was embedded. title_abstract is the graph related_to input.';
COMMENT ON COLUMN paper_vectors.ref_kind IS 'paper for a library row, external_paper for a paper that is not in the library.';
COMMENT ON COLUMN paper_chunks.chunker_version IS 'Chunker identity including its config, e.g. semchunk==4.1.1|250/25|cl100k_base';

COMMENT ON COLUMN papers.pillar_id IS 'Learning pillar (slug) for content isolation';
COMMENT ON COLUMN papers.added_at IS 'Required by the web UI papers list; db.py writes created_at instead';
COMMENT ON COLUMN quiz_cards.ease_factor IS 'SM-2 algorithm ease factor (min 1.3)';
COMMENT ON COLUMN quiz_cards.interval IS 'Days until next review in spaced repetition';
COMMENT ON COLUMN pillars.id IS 'URL-friendly slug generated from name by _generate_slug()';
COMMENT ON COLUMN pillars.updated_at IS 'Maintained by trigger; the application reads but never writes it';
COMMENT ON COLUMN user_fsrs_parameters.pillar_id IS 'NULL means global parameters for the user';

-- ==========================================
-- PAPER TABLES
-- Same objects as docs/migrations/032_paper_tables.sql. Existing volumes
-- apply that file by hand.
-- ==========================================
CREATE TABLE paper_tables (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    paper_id       TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    table_index    INTEGER NOT NULL CHECK (table_index >= 0),
    label          TEXT NOT NULL,
    page           INTEGER NOT NULL CHECK (page >= 1),
    caption        TEXT NOT NULL DEFAULT '',
    columns        JSONB NOT NULL DEFAULT '[]'::jsonb,
    rows           JSONB NOT NULL DEFAULT '[]'::jsonb,
    model          TEXT NOT NULL,
    agreement      NUMERIC(5, 4) CHECK (agreement IS NULL OR agreement BETWEEN 0 AND 1),
    numerals       INTEGER NOT NULL DEFAULT 0 CHECK (numerals >= 0),
    matched        INTEGER NOT NULL DEFAULT 0 CHECK (matched >= 0),
    missing_before INTEGER NOT NULL DEFAULT 0 CHECK (missing_before >= 0),
    status         TEXT NOT NULL
        CHECK (status IN ('spliced', 'flagged', 'unverified', 'unplaced', 'too_large')),
    reason         TEXT NOT NULL DEFAULT '',
    cost_usd       NUMERIC(12, 6) NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT paper_tables_columns_array CHECK (jsonb_typeof(columns) = 'array'),
    CONSTRAINT paper_tables_rows_array CHECK (jsonb_typeof(rows) = 'array')
);

CREATE UNIQUE INDEX paper_tables_paper_index
    ON paper_tables (paper_id, table_index);

COMMENT ON TABLE paper_tables IS
    'Tables transcribed from page images at ingestion, with page, caption, rows, model and numeric agreement against the PDF text layer.';

-- One transaction: delete the paper's rows, insert the new set. An empty
-- array clears the paper's rows.
CREATE OR REPLACE FUNCTION replace_paper_tables(
    p_paper_id TEXT,
    p_tables JSONB
) RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    inserted INTEGER;
BEGIN
    IF p_paper_id IS NULL OR btrim(p_paper_id) = '' THEN
        RAISE EXCEPTION 'paper tables write missing paper_id';
    END IF;
    IF p_tables IS NULL OR jsonb_typeof(p_tables) <> 'array' THEN
        RAISE EXCEPTION 'paper tables for % must be a JSON array', p_paper_id;
    END IF;

    DELETE FROM paper_tables WHERE paper_id = p_paper_id;

    INSERT INTO paper_tables (
        paper_id, table_index, label, page, caption, columns, rows, model,
        agreement, numerals, matched, missing_before, status, reason, cost_usd
    )
    SELECT
        p_paper_id,
        (elem->>'table_index')::INTEGER,
        elem->>'label',
        (elem->>'page')::INTEGER,
        COALESCE(elem->>'caption', ''),
        COALESCE(elem->'columns', '[]'::jsonb),
        COALESCE(elem->'rows', '[]'::jsonb),
        elem->>'model',
        (elem->>'agreement')::NUMERIC,
        COALESCE((elem->>'numerals')::INTEGER, 0),
        COALESCE((elem->>'matched')::INTEGER, 0),
        COALESCE((elem->>'missing_before')::INTEGER, 0),
        elem->>'status',
        COALESCE(elem->>'reason', ''),
        COALESCE((elem->>'cost_usd')::NUMERIC, 0)
    FROM jsonb_array_elements(p_tables) AS elem;

    GET DIAGNOSTICS inserted = ROW_COUNT;
    RETURN inserted;
END;
$$;

-- Explicit grants: default privileges do not cover a hand-applied migration.
GRANT SELECT, INSERT, UPDATE, DELETE ON paper_tables TO web_anon;
GRANT EXECUTE ON FUNCTION replace_paper_tables(TEXT, JSONB) TO web_anon;

-- WORK IDENTITY AND TYPED CITATIONS  (docs/migrations/031_works_identity.sql)
-- Last in the file on purpose: it renames the paper_citations table above to
-- paper_citations_legacy and puts a read-only view of work_citations in its
-- place, so every index and comment on the table must already have run. A
-- brand-new volume ends up identical to one migrated by hand.
-- ==========================================

CREATE TABLE works (
    id                         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    s2_paper_id                TEXT,
    corpus_id                  BIGINT,
    title                      TEXT,
    year                       INTEGER,
    publication_date           DATE,
    venue                      TEXT,
    authors                    JSONB NOT NULL DEFAULT '[]'::jsonb,
    author_count               INTEGER,
    abstract                   TEXT,
    tldr                       TEXT,
    citation_count             INTEGER,
    influential_citation_count INTEGER,
    reference_count            INTEGER,
    open_pdf_url               TEXT,
    fetched_at                 TIMESTAMPTZ,
    refreshed_at               TIMESTAMPTZ,
    references_synced_at       TIMESTAMPTZ,
    citers_synced_at           TIMESTAMPTZ,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT works_authors_array CHECK (jsonb_typeof(authors) = 'array')
);

CREATE UNIQUE INDEX works_s2_paper_id_key
    ON works (s2_paper_id) WHERE s2_paper_id IS NOT NULL;

CREATE TABLE work_ids (
    kind    TEXT NOT NULL
        CHECK (kind IN ('s2', 'arxiv', 'doi', 'acl', 'corpus', 'title_hash')),
    value   TEXT NOT NULL CHECK (btrim(value) <> ''),
    work_id UUID NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    PRIMARY KEY (kind, value)
);

CREATE INDEX work_ids_work_id ON work_ids (work_id);

ALTER TABLE papers
    ADD COLUMN work_id UUID REFERENCES works(id) ON DELETE SET NULL;

CREATE INDEX papers_work_id
    ON papers (work_id) WHERE work_id IS NOT NULL;

CREATE TABLE work_citations (
    citing         UUID NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    cited          UUID NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    is_influential BOOLEAN NOT NULL DEFAULT FALSE,
    intents        TEXT[] NOT NULL DEFAULT '{}',
    contexts       TEXT[] NOT NULL DEFAULT '{}',
    source         TEXT NOT NULL DEFAULT 'semantic_scholar',
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    PRIMARY KEY (citing, cited),
    CONSTRAINT work_citations_not_self CHECK (citing <> cited),
    CONSTRAINT work_citations_contexts_cap CHECK (cardinality(contexts) <= 3)
);

CREATE INDEX work_citations_cited ON work_citations (cited);

-- The bare id the rest of the app has always shown for an external paper:
-- arXiv, else DOI, else S2 paperId (SemanticScholarTool._convert_to_paper_ref).
CREATE OR REPLACE FUNCTION work_display_id(p_work UUID) RETURNS TEXT
LANGUAGE sql STABLE
AS $$
    SELECT value
    FROM work_ids
    WHERE work_id = p_work AND kind IN ('arxiv', 'doi', 's2')
    ORDER BY CASE kind WHEN 'arxiv' THEN 1 WHEN 'doi' THEN 2 ELSE 3 END
    LIMIT 1
$$;

-- Retire the old table, once. A re-run finds the view and does nothing here.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = current_schema() AND c.relname = 'paper_citations'
          AND c.relkind IN ('r', 'p')
    ) THEN
        ALTER TABLE paper_citations RENAME TO paper_citations_legacy;
    END IF;
END
$$;

CREATE OR REPLACE VIEW paper_citations AS
    SELECT md5(p.id || '>' || wc.cited::text)            AS id,
           p.id                                           AS paper_id,
           COALESCE(op.id, work_display_id(wc.cited))     AS cited_paper_id,
           'outgoing'::text                               AS citation_direction,
           wc.is_influential                              AS is_influential,
           wc.contexts[1]                                 AS citation_context,
           wc.source                                      AS source,
           wc.fetched_at                                  AS fetched_at,
           ow.citation_count                              AS other_citation_count
    FROM work_citations wc
    JOIN papers p ON p.work_id = wc.citing
    JOIN works ow ON ow.id = wc.cited
    LEFT JOIN papers op ON op.work_id = wc.cited
    UNION ALL
    SELECT md5(p.id || '<' || wc.citing::text),
           p.id,
           COALESCE(op.id, work_display_id(wc.citing)),
           'incoming'::text,
           wc.is_influential,
           wc.contexts[1],
           wc.source,
           wc.fetched_at,
           ow.citation_count
    FROM work_citations wc
    JOIN papers p ON p.work_id = wc.cited
    JOIN works ow ON ow.id = wc.citing
    LEFT JOIN papers op ON op.work_id = wc.citing;

-- One transaction. Python has already resolved identity, so every uuid here is
-- final; the function only has to keep a write from making a stored row worse.
--   p_works   [{id, s2_paper_id, corpus_id, title, year, publication_date,
--               venue, authors[], author_count, abstract, tldr, citation_count,
--               influential_citation_count, reference_count, open_pdf_url,
--               described, references_synced, citers_synced}]
--             `described` = S2 described this work (sets fetched_at once and
--             refreshed_at every time). A stub row leaves every field NULL.
--   p_ids     [{kind, value, work_id}]
--   p_edges   [{citing, cited, is_influential, intents[], contexts[], source,
--               detail}]
--             detail=true is a typed edge from S2's references/citations
--             endpoint and replaces what is stored; detail=false is an id-only
--             edge and never overwrites one.
--   p_papers  [{id, work_id, citation_count}]  library rows to link
CREATE OR REPLACE FUNCTION apply_work_batch(
    p_works  JSONB,
    p_ids    JSONB,
    p_edges  JSONB,
    p_papers JSONB
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    works_n    INTEGER;
    ids_n      INTEGER;
    edges_n    INTEGER := 0;
    thin_n     INTEGER := 0;
    papers_n   INTEGER;
BEGIN
    IF p_works IS NULL OR jsonb_typeof(p_works) <> 'array'
       OR p_ids IS NULL OR jsonb_typeof(p_ids) <> 'array'
       OR p_edges IS NULL OR jsonb_typeof(p_edges) <> 'array'
       OR p_papers IS NULL OR jsonb_typeof(p_papers) <> 'array' THEN
        RAISE EXCEPTION 'apply_work_batch arguments must be JSON arrays';
    END IF;

    INSERT INTO works (
        id, s2_paper_id, corpus_id, title, year, publication_date, venue,
        authors, author_count, abstract, tldr, citation_count,
        influential_citation_count, reference_count, open_pdf_url,
        fetched_at, refreshed_at, references_synced_at, citers_synced_at
    )
    SELECT r.id, r.s2_paper_id, r.corpus_id, r.title, r.year,
           r.publication_date, r.venue,
           COALESCE(r.authors, '[]'::jsonb), r.author_count, r.abstract,
           r.tldr, r.citation_count, r.influential_citation_count,
           r.reference_count, r.open_pdf_url,
           CASE WHEN r.described THEN timezone('utc', now()) END,
           CASE WHEN r.described THEN timezone('utc', now()) END,
           CASE WHEN r.references_synced THEN timezone('utc', now()) END,
           CASE WHEN r.citers_synced THEN timezone('utc', now()) END
    FROM jsonb_to_recordset(p_works) AS r(
        id UUID, s2_paper_id TEXT, corpus_id BIGINT, title TEXT, year INTEGER,
        publication_date DATE, venue TEXT, authors JSONB, author_count INTEGER,
        abstract TEXT, tldr TEXT, citation_count INTEGER,
        influential_citation_count INTEGER, reference_count INTEGER,
        open_pdf_url TEXT, described BOOLEAN, references_synced BOOLEAN,
        citers_synced BOOLEAN
    )
    ON CONFLICT (id) DO UPDATE SET
        s2_paper_id = COALESCE(works.s2_paper_id, EXCLUDED.s2_paper_id),
        corpus_id = COALESCE(EXCLUDED.corpus_id, works.corpus_id),
        title = COALESCE(EXCLUDED.title, works.title),
        year = COALESCE(EXCLUDED.year, works.year),
        publication_date = COALESCE(EXCLUDED.publication_date, works.publication_date),
        venue = COALESCE(EXCLUDED.venue, works.venue),
        authors = CASE WHEN jsonb_array_length(EXCLUDED.authors) > 0
                       THEN EXCLUDED.authors ELSE works.authors END,
        author_count = COALESCE(EXCLUDED.author_count, works.author_count),
        abstract = COALESCE(EXCLUDED.abstract, works.abstract),
        tldr = COALESCE(EXCLUDED.tldr, works.tldr),
        citation_count = COALESCE(EXCLUDED.citation_count, works.citation_count),
        influential_citation_count = COALESCE(
            EXCLUDED.influential_citation_count, works.influential_citation_count),
        reference_count = COALESCE(EXCLUDED.reference_count, works.reference_count),
        open_pdf_url = COALESCE(EXCLUDED.open_pdf_url, works.open_pdf_url),
        fetched_at = COALESCE(works.fetched_at, EXCLUDED.fetched_at),
        refreshed_at = COALESCE(EXCLUDED.refreshed_at, works.refreshed_at),
        references_synced_at = COALESCE(
            EXCLUDED.references_synced_at, works.references_synced_at),
        citers_synced_at = COALESCE(EXCLUDED.citers_synced_at, works.citers_synced_at);
    GET DIAGNOSTICS works_n = ROW_COUNT;

    INSERT INTO work_ids (kind, value, work_id)
    SELECT r.kind, r.value, r.work_id
    FROM jsonb_to_recordset(p_ids) AS r(kind TEXT, value TEXT, work_id UUID)
    ON CONFLICT (kind, value) DO NOTHING;
    GET DIAGNOSTICS ids_n = ROW_COUNT;

    INSERT INTO work_citations (
        citing, cited, is_influential, intents, contexts, source
    )
    SELECT e.citing, e.cited, COALESCE(e.is_influential, FALSE),
           ARRAY(SELECT jsonb_array_elements_text(COALESCE(e.intents, '[]'::jsonb))),
           ARRAY(SELECT jsonb_array_elements_text(COALESCE(e.contexts, '[]'::jsonb))),
           COALESCE(e.source, 'semantic_scholar')
    FROM jsonb_to_recordset(p_edges) AS e(
        citing UUID, cited UUID, is_influential BOOLEAN, intents JSONB,
        contexts JSONB, source TEXT, detail BOOLEAN
    )
    WHERE COALESCE(e.detail, FALSE)
    ON CONFLICT (citing, cited) DO UPDATE SET
        is_influential = EXCLUDED.is_influential,
        intents = EXCLUDED.intents,
        contexts = EXCLUDED.contexts,
        source = EXCLUDED.source,
        fetched_at = timezone('utc', now());
    GET DIAGNOSTICS edges_n = ROW_COUNT;

    INSERT INTO work_citations (
        citing, cited, is_influential, source
    )
    SELECT e.citing, e.cited, COALESCE(e.is_influential, FALSE),
           COALESCE(e.source, 'semantic_scholar')
    FROM jsonb_to_recordset(p_edges) AS e(
        citing UUID, cited UUID, is_influential BOOLEAN, intents JSONB,
        contexts JSONB, source TEXT, detail BOOLEAN
    )
    WHERE NOT COALESCE(e.detail, FALSE)
    ON CONFLICT (citing, cited) DO NOTHING;
    GET DIAGNOSTICS thin_n = ROW_COUNT;

    UPDATE papers p
    SET work_id = COALESCE(r.work_id, p.work_id),
        citation_count = COALESCE(r.citation_count, p.citation_count)
    FROM jsonb_to_recordset(p_papers) AS r(
        id TEXT, work_id UUID, citation_count INTEGER
    )
    WHERE p.id = r.id;
    GET DIAGNOSTICS papers_n = ROW_COUNT;

    RETURN jsonb_build_object(
        'works', works_n,
        'ids_added', ids_n,
        'edges', edges_n + thin_n,
        'papers', papers_n
    );
END;
$$;

COMMENT ON TABLE works IS
    'One scholarly work, in the library or not. title NULL = a stub known only by an id; fetched_at NULL = S2 has not described it.';
COMMENT ON TABLE work_ids IS
    'Every normalised id a work goes by. (kind, value) belongs to exactly one work. Normal forms: nlp_pillars/work_ids.py.';
COMMENT ON TABLE work_citations IS
    'citing -> cited, with S2 intents, up to three citing sentences and the influential flag. The citation truth; paper_citations is a view of it.';
COMMENT ON COLUMN papers.work_id IS
    'The library paper''s work. NULL for documents Semantic Scholar cannot know (file: uploads).';
COMMENT ON VIEW paper_citations IS
    'Read-only compatibility view over work_citations in the pre-031 column shape. Write through nlp_pillars.works_store.';

GRANT EXECUTE ON FUNCTION work_display_id(UUID) TO web_anon;
GRANT EXECUTE ON FUNCTION apply_work_batch(JSONB, JSONB, JSONB, JSONB) TO web_anon;

-- ==========================================
-- PAPER MAP SNAPSHOTS  (docs/migrations/034_map_snapshots.sql)
-- One rebuildable snapshot per pillar, written by `map build`
-- (nlp_pillars/map_build.py, nlp_pillars/map_store.py).
-- ==========================================

CREATE TABLE map_snapshots (
    scope        TEXT PRIMARY KEY CHECK (btrim(scope) <> ''),
    built_at     TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    params_hash  TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    model_id     TEXT,
    run_id       UUID,
    stats        JSONB NOT NULL DEFAULT '{}'::jsonb,
    positions    JSONB NOT NULL DEFAULT '{}'::jsonb,
    layout_key   TEXT,
    payload      JSONB NOT NULL,
    CONSTRAINT map_snapshots_payload_object CHECK (jsonb_typeof(payload) = 'object'),
    CONSTRAINT map_snapshots_positions_object CHECK (jsonb_typeof(positions) = 'object')
);

COMMENT ON TABLE map_snapshots IS
    'The paper map per scope (pillar id): a rebuildable projection written by `map build`. See nlp_pillars/map_build.py.';
COMMENT ON COLUMN map_snapshots.positions IS
    'node id -> [x, y] from payload.nodes, so the next build can warm-start its layout cheaply.';
COMMENT ON COLUMN map_snapshots.layout_key IS
    'Hash of the graph the positions were laid out for. An identical graph reuses them unchanged, so an unchanged map keeps its payload_hash.';

-- Returns {scope, built_at, changed}. changed is false when the stored payload
-- already had this hash: the map did not move, only built_at did.
CREATE OR REPLACE FUNCTION put_map_snapshot(
    p_scope        TEXT,
    p_params_hash  TEXT,
    p_payload_hash TEXT,
    p_model_id     TEXT,
    p_stats        JSONB,
    p_positions    JSONB,
    p_layout_key   TEXT,
    p_payload      JSONB,
    p_run_id       UUID
) RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    previous TEXT;
    stamped  TIMESTAMPTZ := timezone('utc', now());
BEGIN
    SELECT payload_hash INTO previous FROM map_snapshots WHERE scope = p_scope;
    INSERT INTO map_snapshots (
        scope, built_at, params_hash, payload_hash, model_id, run_id,
        stats, positions, layout_key, payload
    )
    VALUES (
        p_scope, stamped, p_params_hash, p_payload_hash, p_model_id, p_run_id,
        COALESCE(p_stats, '{}'::jsonb), COALESCE(p_positions, '{}'::jsonb),
        p_layout_key, p_payload
    )
    ON CONFLICT (scope) DO UPDATE SET
        built_at = EXCLUDED.built_at,
        params_hash = EXCLUDED.params_hash,
        payload_hash = EXCLUDED.payload_hash,
        model_id = EXCLUDED.model_id,
        run_id = EXCLUDED.run_id,
        stats = EXCLUDED.stats,
        positions = EXCLUDED.positions,
        layout_key = EXCLUDED.layout_key,
        payload = EXCLUDED.payload;
    RETURN jsonb_build_object(
        'scope', p_scope,
        'built_at', stamped,
        'changed', previous IS DISTINCT FROM p_payload_hash
    );
END;
$$;

GRANT SELECT, INSERT, UPDATE, DELETE ON map_snapshots TO web_anon;
GRANT EXECUTE ON FUNCTION put_map_snapshot(TEXT, TEXT, TEXT, TEXT, JSONB, JSONB, TEXT, JSONB, UUID) TO web_anon;
