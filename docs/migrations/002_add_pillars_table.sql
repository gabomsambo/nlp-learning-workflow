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
