#!/usr/bin/env python3
"""Print the five podcast prompts for a given set of options. No model calls.

The prompts are the artifact that matters when podcast aiming changes, and they
are otherwise only visible by paying $0.27 and reading a log that the container
suppresses. This renders them exactly as ``PodcastAgent`` assembles them, for
free, so a prompt change can be reviewed as a diff.

    python scripts/render_podcast_prompts.py                     # the defaults
    python scripts/render_podcast_prompts.py field=biology length=45
    python scripts/render_podcast_prompts.py field=__custom__ field_custom="marine ecology"
    python scripts/render_podcast_prompts.py writer_mode=evidence    # ledger builder + evidence writer
    python scripts/render_podcast_prompts.py writer_mode=both        # free, then evidence

The paper body, Ground Pack and ledger are placeholders; everything else is real.
Options are the same flat key=value shape the browser posts — see
nlp_pillars/podcast_options.py. ``writer_mode`` (free | evidence | both, default
free) is not an option: it selects which writer's prompts are printed, the way
PODCAST_WRITER_MODE selects the writer. Unlike the environment variable it never
defaults to the environment, so a render is reproducible.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Rendering touches no client, but PodcastAgent's module imports the Anthropic
# SDK, which does not need a key until it is constructed.
from nlp_pillars.agents import podcast_agent as pa  # noqa: E402
from nlp_pillars.evidence import WRITER_MODES  # noqa: E402
from nlp_pillars.evidence import ledger as ev_ledger  # noqa: E402
from nlp_pillars.evidence import prompts as ev_prompts  # noqa: E402
from nlp_pillars.podcast_options import (  # noqa: E402
    PRECEDENCE_NOTE, PodcastOptionError, build_variables, resolve, settings_block,
)

PAPER = "<<PAPER CONTENT: title, authors, year, abstract and full text go here>>"
GROUND_PACK = "<<GROUND PACK: the four analysis outputs go here>>"
LEDGER = (
    "F1 [identity] <<the paper's title, authors and year, from the database>>\n"
    "F2 [method; Section 3] <<one self-contained fact, quote-verified against the paper>>\n"
    "... (the ledger builder's verified F entries)\n"
    "T1 [table_row] <<Table 9 (caption), row \"name\": column=value; column=value>>\n"
    "... (one T entry per repaired table row)"
)


def render_evidence(raw):
    """The ledger builder's call and the evidence writer's call. No model is called."""
    options = resolve(raw)
    variables = build_variables(options)
    block = settings_block(options)
    lo, hi = ev_ledger.entry_target(60_000)
    ledger_user = ev_prompts.LEDGER_USER.format(
        paper=ev_ledger.paper_view(
            "<<title>>", ["<<authors>>"], "<<year>>", "<<abstract>>", PAPER)
    )
    writer_user = pa.EVIDENCE_PROMPT.format(
        settings_block=block, precedence_note=PRECEDENCE_NOTE,
        ledger=LEDGER, **variables,
    )
    calls = [
        ("A — Evidence ledger (Haiku, structured output)",
         f"prompt {ev_prompts.LEDGER_PROMPT_VERSION}; entry target for a 60K-char paper",
         ev_prompts.LEDGER_SYSTEM.format(lo=lo, hi=hi), ledger_user),
        ("B — Evidence writer (call 5)",
         f"prompt {ev_prompts.EVIDENCE_PROMPT_VERSION}; temperature not sent (Sonnet 5)",
         pa.EVIDENCE_SYSTEM, writer_user),
    ]
    out = ["OPTIONS", "-------"]
    for key, choice in options.choices.items():
        source = "preset" if choice.preset else "custom"
        out.append(f"  {key:9s} {choice.label}   [{source}]")
    out += ["", "WRITER MODE: evidence (2 calls; the Ground Pack is not used)"]
    for name, note, system, user in calls:
        out += [
            "", "=" * 78, f"CALL {name}   ({note})", "=" * 78, "",
            "--- SYSTEM ---", system, "", "--- USER ---", user,
        ]
    return "\n".join(out)


def render(raw):
    options = resolve(raw)
    variables = build_variables(options)
    block = settings_block(options)

    def ground_pack_user(lead_in):
        return pa.GROUND_PACK_USER_TEMPLATE.format(
            settings_block=block,
            precedence_note=PRECEDENCE_NOTE,
            rules=pa.GROUND_PACK_RULES,
            lead_in=lead_in,
            paper_content=PAPER,
        )

    calls = [
        ("1 — Facts-Only Outline", pa.TEMPERATURE_EXTRACTION,
         pa.PROMPT_1_FACTS_OUTLINE.format(**variables),
         ground_pack_user("Analyze this paper:")),
        ("2 — Core Concepts", pa.TEMPERATURE_ANALYSIS,
         pa.PROMPT_2_CORE_CONCEPTS.format(**variables),
         ground_pack_user("Identify the 3 most complex or novel concepts from this paper:")),
        ("3 — Metrics & Datasets", pa.TEMPERATURE_EXTRACTION,
         pa.PROMPT_3_METRICS_DATASETS.format(**variables),
         ground_pack_user("Extract all quantitative data, datasets, and benchmarks from this paper:")),
        ("4 — Limitations", pa.TEMPERATURE_ANALYSIS,
         pa.PROMPT_4_LIMITATIONS.format(**variables),
         ground_pack_user("Critically analyze the limitations and weaknesses of this paper:")),
        ("5 — Final synthesis", pa.TEMPERATURE_SCRIPT,
         pa.FINAL_SYNTHESIS_SYSTEM,
         pa.FINAL_SYNTHESIS_PROMPT.format(
             settings_block=block, precedence_note=PRECEDENCE_NOTE,
             ground_pack=GROUND_PACK, paper_content=PAPER, **variables,
         )),
    ]

    out = ["OPTIONS", "-------"]
    for key, choice in options.choices.items():
        source = "preset" if choice.preset else "custom"
        out.append(f"  {key:9s} {choice.label}   [{source}]")

    for name, temperature, system, user in calls:
        out += [
            "",
            "=" * 78,
            f"CALL {name}   (temperature {temperature})",
            "=" * 78,
            "",
            "--- SYSTEM ---",
            system,
            "",
            "--- USER ---",
            user,
        ]
    return "\n".join(out)


def main(argv):
    raw = {}
    for arg in argv:
        if "=" not in arg:
            raise SystemExit(f"expected key=value, got {arg!r}")
        key, _, value = arg.partition("=")
        raw[key] = value
    mode = raw.pop("writer_mode", "free")
    if mode not in (*WRITER_MODES, "both"):
        raise SystemExit(f"error: writer_mode must be free, evidence or both, got {mode!r}")
    try:
        if mode in ("free", "both"):
            print(render(raw))
        if mode == "both":
            print("\n\n" + "#" * 78 + "\n")
        if mode in ("evidence", "both"):
            print(render_evidence(raw))
    except PodcastOptionError as e:
        raise SystemExit(f"error: {e}") from None
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
