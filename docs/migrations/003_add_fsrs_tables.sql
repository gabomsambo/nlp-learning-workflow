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
