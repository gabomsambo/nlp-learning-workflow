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
