# The paper map's knowledge layer

What sits on top of the free map (`map_build.py`) to explain it in words. Design:
the paper-map scout report, sections 3.3, 5 and 7 (row M8). Postgres is the only
store; nothing here is a second source of truth.

| Piece | Where | Paid? |
|---|---|---|
| Typed lesson connections | `connections.py`, `LessonDraft.connections` | one extra field on the existing lesson call |
| Learner confirm / reject / edit | `connections.py`, `/api/connections`, `learner_connections` | no |
| Community names and summaries | `map_enrich.py`, route `map_names` | Haiku, about $0.0003 a community, cached |
| Bridge sentences | `map_enrich.py`, route `map_bridge` | Haiku, about $0.0002 a pair, cached |
| Concept anchoring, merges, parents | `concept_anchoring.py`, `vocabulary.py` | exact matches free; the rest one Haiku call each (about $0.0001) |

## The switch

`MAP_ENRICHMENT_ENABLED` (default **true**) turns every paid call in this layer
off. Off, the map is named from its title terms, every connection is explained
by a template sentence over its stored evidence, and anchoring makes exact
label/synonym matches only. Names and bridges already paid for are still shown:
reuse is free.

A paid pass is also refused, with the reason on the run panel, when
`DAILY_SPEND_CAP_USD` is reached or today's spend cannot be read
(`enrichment.require_budget`). One broken client (no key, an unpriced model)
ends the paid pass after one attempt and the map keeps its templates.

Models: `LLM_MAP_NAMES_MODEL`, `LLM_MAP_BRIDGE_MODEL`, `LLM_CONCEPT_ANCHOR_MODEL`
(all `claude-haiku-5-5`, fallback `claude-sonnet-5-5`, thinking off; see
`llm_routes.py`).

## Cost, and where it is shown

Every call goes through `llm_client.structured`, so it is priced from the
provider's token counts and logged at WARNING. The `map_enrich` stage's detail
carries the pass total in the line `spend.py` sums
(`6 named, 0 cached, 2 bridge(s), 0 rejected · 8 calls · claude-haiku-5-5 · 7,300 in · 810 out · $0.002400`),
and `map_snapshots.stats.enrichment` keeps it per build. The **payload never
holds a cost**, so a rebuild that changes nothing keeps its `payload_hash` and
costs $0: names are cached by the community's `member_hash`, bridges by a hash
of the evidence they were written from (a rejected bridge is cached too, so the
same evidence is not paid for twice). At most 60 bridges are written per build;
the rest are `deferred` to the next one.

`python -m nlp_pillars.cli map build --dry-run` prices the pass and calls nothing.

## Bridge sentences are checked, and marked

One sentence per connected pair of **library** papers, written only from what is
stored for that pair: the citing sentence (S2 context), both abstracts, and the
lesson's own relation. The model returns the sentence **and** 1-3 verbatim quotes
with the number of the source each came from. A bridge is kept only if

- every quote is found in its source, character for character up to case,
  whitespace and quote marks (`factcheck.anchor.find_in_text`; no fuzzy match);
- every number in the sentence occurs in a source or in the build's own counts;
- it is one line of at most 320 characters.

Anything else is `rejected` (the pair keeps its template sentence); a model that
sees no concrete connection may return an empty sentence (`abstained`, no
penalty). A kept bridge carries `generated: true`, `reviewed: false`, its model,
its quotes with offsets, and what each quote is from. A pair missing either
abstract is `no_abstract` and costs nothing.

**The check proves the quotes are real and no number is invented, not that the
sentence is a fair reading of them.** Hand-labelled on 43 pairs
(`evals/bridges/`): 35 of the 36 bridges kept were faithful and one overreached;
with the first prompt 2 of 40 stated something no source said. So a surface
should show a bridge with its quotes and as written by a model, not reviewed.

## Payload additions (additive; `MAP_SNAPSHOT_VERSION` is unchanged)

- `communities.{regions,topics}[]`: `name`, `summary`, `name_source`
  (`model` / `terms` / `pillar`), `name_model`.
- `evidence[]`: `status` (`suggested` / `confirmed` / `rejected`).
- `connections[]`: one entry per pair. `provenance` is `lesson` or `user`;
  `relation`, `status`, `sentence` (the template, or the learner's phrase), and
  for lesson entries `why`, `mention`, `paper_id`, `legacy`.
- `bridges[]`: `a`, `b`, `status`, `evidence_hash`, and when `generated` the
  text and quotes (each with what it is from).
- `enrichment`: `{version, paid}`.

A rejected connection is flagged, never removed, and never explained or billed.

## Typed lesson connections

The lesson call now writes each connection as `{work, title, relation, why}`
(`relation` is one of builds_on, uses, critiques, compares, alternative,
background). `lessons.connections` (prose) is derived from them, so the lesson
page, the modal, the cloud reader and the CLI are untouched;
`lessons.connection_links` keeps the typed copy. On every map build the typed
items are resolved against the paper's own reference list: first author's
surname, year within one, and a title match when one is given. A mention that fits
two references or none stays unresolved and is counted; nothing is guessed.
Lessons written before this exist only as prose: author-year mentions are read
from the text and the relation guessed from its words (marked `legacy`).

`python -m nlp_pillars.cli connections resolve` shows what resolves, read-only.

## The learner's decisions

`GET/PUT/DELETE /api/connections`. Confirm (optionally with another relation and
a phrase of the learner's own, at most 400 characters), reject, or undo. Stored
in `learner_connections` with `provenance = 'user'`, one row per unordered
pair. A map build reads them and never writes them, so they survive every
rebuild; a confirmed pair shows the learner's phrase.

## Concept anchoring

`python -m nlp_pillars.cli concepts anchor [--dry-run] [--reanchor] [--max-calls N]`.
Per unanchored concept, cheapest first: exact label/synonym match in the
vocabulary; else the five nearest fields by bge cosine and one Haiku call that
answers with a candidate's **position** (never an id) and whether the concept
**is** that field (`same`) or a kind of it (`narrower`); else nothing, if no
field is near enough (`CANDIDATE_FLOOR`). Concepts that are `same` as one field
are merged (the other names become aliases, paper links and graph edges move);
a `narrower` concept gets `parent_id` pointing at the concept that is its field,
else at the nearest ancestor field the library has a concept for. All of it is
written by one transaction, `apply_concept_anchoring`.

Vocabularies: the NLP-KG fields of study (421 fields, 583 synonyms, MIT) are
bundled in `nlp_pillars/vocab/nlp_kg_fields.json`, regenerated by
`scripts/build_nlp_kg_vocabulary.py`. The Papers with Code archive is CC BY-SA
4.0 and is not bundled; `VOCABULARY_FILES` loads a local conversion in the same
shape under its own key (`pwc:<id>`).

## Migration 035 (hand-applied)

`docs/migrations/035_knowledge_layer.sql`: `lessons.connection_links`,
`concepts.anchor / anchor_relation / anchored_at / parent_id`,
`learner_connections`, and the functions `apply_concept_anchoring`,
`put_learner_connection`, `delete_learner_connection`. Until it is applied: new
lessons still save (without their typed copy, logged), `map build` runs without
the learner's decisions and says so in its warnings, and `concepts anchor` and
`/api/connections` name the file.
