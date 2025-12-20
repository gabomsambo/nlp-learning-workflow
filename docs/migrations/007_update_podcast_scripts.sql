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
