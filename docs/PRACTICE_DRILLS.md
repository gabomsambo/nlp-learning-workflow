# Practice drills

Practice drills are custom answer tracks for speaking out loud, stored as normal
episodes. A drill is a `podcast_scripts` row with
`source_material.kind = "drill"` and `paper_id = NULL`; no model is called and
nothing is added to podcast spend. The Listen page labels it **Practice drill**,
`POST /api/podcast/tts/generate` voices it like any other script, read-along
alignment works on the stored text, and `python -m nlp_pillars.cli cloud publish`
puts it on the phone reader.

Apply migration 038 first on an existing database: it drops the old
`NOT NULL` on `podcast_scripts.paper_id`.

```bash
docker exec -i nlp_postgres psql -U nlp -d nlp -v ON_ERROR_STOP=1 -f - \
  < docs/migrations/038_podcast_drill_scripts.sql
```

## From a drill sheet (CLI)

The sheet format is the one used by the ThetaRay answer drills:

```markdown
# Day 5 drill: Graph methods for money laundering

Intro: Day five, part two. Graph methods. Two questions.

## Q1. How would you detect round-tripping?

**Answer.** Round-tripping is a cycle in the directed payments graph. ...

**Key points.** Cycles. Strongly connected components. Amount similarity.
```

```bash
python -m nlp_pillars.cli podcast import data/sheets/d5-graph-aml.md \
  --pillar ml-systems-production \
  --title "Day 5 drill: graph methods"
```

`--title` is optional and defaults to the sheet's `#` heading. The command keeps
the spoken-form substitutions the drill build tool uses (`PR-AUC` becomes
"P R A U C", `0.905` becomes "zero point nine zero five"), numbers the questions
in words, and writes `[PAUSE 5s]` after each question and `[PAUSE 8s]` after
each answer. A 1.5 s gap separates the intro, each key-points recap and the next
question. The script ends with "End of drill."

The command prints the new script id. To make audio, queue it with
`POST /api/podcast/tts/generate` (`{"script_id": "...", "voice_path": "..."}`)
or the **Make audio** control on the Listen page.

## From supplied text (API)

`POST /api/podcast/scripts/import` stores any text already in the app's script
shape. It never calls a model.

```json
{
  "title": "Day 5 drill: graph methods",
  "pillar_id": "ml-systems-production",
  "paper_id": null,
  "script": "[HOST]: Question one. ...\n[PAUSE 5s]\n[HOST]: ...\n"
}
```

`title`, `pillar_id` and `script` are required and must not be blank;
`paper_id` is optional. An unknown pillar is **404**, blank input is **422**,
and a row that cannot be stored is **503** (naming migration 038 when the cause
is the old `paper_id NOT NULL`). The response is **201** with `script_id`.

## Timed pauses

`nlp_pillars/tts/cue_parser.py` accepts `[PAUSE 5s]` alongside the existing
fixed `[PAUSE]` (400 ms, unchanged). Whole seconds are not required:
`[PAUSE 2.5s]` works. Values are clamped to **0.5-15 s**, in or out of range.
Timed pauses are silence in the assembled audio and are never sent to IndexTTS
or counted as spoken words.

## Automatic fact-check

The automatic fact-check after a saved script skips `kind = "drill"` before the
spend gate: a drill has no paper text to check against. The manual
`POST /api/podcast/{id}/factcheck` also answers **422** for a drill rather than
starting a check that cannot succeed. Nothing about a failed check changes the
script, and drills are unaffected either way.
