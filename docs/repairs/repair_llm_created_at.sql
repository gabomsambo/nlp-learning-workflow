-- Restamp lessons.created_at and notes.created_at that an LLM invented.
--
-- The structured-output schema used to include created_at. The model filled it
-- with a training-era date (2023–2024) and that value was stored. Quiz cards
-- were never affected: their created_at is stamped in the server when the row
-- is written. New notes and lessons are stamped the same way. This file only
-- corrects rows already stored.
--
-- It was not applied by the change that added it. Apply it by hand, once,
-- against the database that holds the invented rows:
--
--   docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
--     < docs/repairs/repair_llm_created_at.sql
--
-- Preview the SQL without connecting:
--
--   python -m nlp_pillars.cli dates repair
--
-- Timestamp chosen for each row, first one that exists. This is the same
-- order as nlp_pillars.created_at_repair.choose_real_timestamp:
--
--   1. pipeline_run_stages.finished_at of the completed summarize (notes) or
--      synthesize (lessons) stage for that paper. A daily or select run stores
--      the paper id in the stage detail. An upload stores it on
--      pipeline_runs.result ->> 'paper_id'. The earliest finished_at is when
--      that work was first done.
--   2. earliest quiz_cards.created_at for the same paper. Those rows are
--      server-stamped in the same processing pass.
--   3. papers.added_at, then papers.created_at.
--
-- A row with none of these is left unchanged. Safe to re-run: a row already
-- equal to the chosen timestamp is not updated.

BEGIN;

UPDATE notes AS n
SET created_at = chosen.real_at
FROM (
    SELECT
        n2.id,
        COALESCE(
            stage.finished_at,
            quiz.created_at,
            p.added_at,
            p.created_at
        ) AS real_at
    FROM notes AS n2
    LEFT JOIN papers AS p ON p.id = n2.paper_id
    LEFT JOIN LATERAL (
        SELECT MIN(s.finished_at) AS finished_at
        FROM pipeline_run_stages AS s
        JOIN pipeline_runs AS r ON r.id = s.run_id
        WHERE s.name = 'summarize'
          AND s.status = 'completed'
          AND s.finished_at IS NOT NULL
          AND (
              s.detail = n2.paper_id
              OR (r.result ->> 'paper_id') = n2.paper_id
          )
    ) AS stage ON TRUE
    LEFT JOIN LATERAL (
        SELECT MIN(q.created_at) AS created_at
        FROM quiz_cards AS q
        WHERE q.paper_id = n2.paper_id
    ) AS quiz ON TRUE
) AS chosen
WHERE n.id = chosen.id
  AND chosen.real_at IS NOT NULL
  AND n.created_at IS DISTINCT FROM chosen.real_at;

UPDATE lessons AS n
SET created_at = chosen.real_at
FROM (
    SELECT
        n2.id,
        COALESCE(
            stage.finished_at,
            quiz.created_at,
            p.added_at,
            p.created_at
        ) AS real_at
    FROM lessons AS n2
    LEFT JOIN papers AS p ON p.id = n2.paper_id
    LEFT JOIN LATERAL (
        SELECT MIN(s.finished_at) AS finished_at
        FROM pipeline_run_stages AS s
        JOIN pipeline_runs AS r ON r.id = s.run_id
        WHERE s.name = 'synthesize'
          AND s.status = 'completed'
          AND s.finished_at IS NOT NULL
          AND (
              s.detail = n2.paper_id
              OR (r.result ->> 'paper_id') = n2.paper_id
          )
    ) AS stage ON TRUE
    LEFT JOIN LATERAL (
        SELECT MIN(q.created_at) AS created_at
        FROM quiz_cards AS q
        WHERE q.paper_id = n2.paper_id
    ) AS quiz ON TRUE
) AS chosen
WHERE n.id = chosen.id
  AND chosen.real_at IS NOT NULL
  AND n.created_at IS DISTINCT FROM chosen.real_at;

COMMIT;
