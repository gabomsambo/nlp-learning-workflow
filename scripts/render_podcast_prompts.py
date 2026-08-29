#!/usr/bin/env python3
"""Print the five podcast prompts for a given set of options. No model calls.

The prompts are the artifact that matters when podcast aiming changes, and they
are otherwise only visible by paying $0.27 and reading a log that the container
suppresses. This renders them exactly as ``PodcastAgent`` assembles them, for
free, so a prompt change can be reviewed as a diff.

    python scripts/render_podcast_prompts.py                     # the defaults
    python scripts/render_podcast_prompts.py field=biology length=45
    python scripts/render_podcast_prompts.py field=__custom__ field_custom="marine ecology"

The paper body and Ground Pack are placeholders; everything else is real.
Options are the same flat key=value shape the browser posts — see
nlp_pillars/podcast_options.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Rendering touches no client, but PodcastAgent's module imports the Anthropic
# SDK, which does not need a key until it is constructed.
from nlp_pillars.agents import podcast_agent as pa  # noqa: E402
from nlp_pillars.podcast_options import (  # noqa: E402
    PRECEDENCE_NOTE, PodcastOptionError, build_variables, resolve, settings_block,
)

PAPER = "<<PAPER CONTENT: title, authors, year, abstract and full text go here>>"
GROUND_PACK = "<<GROUND PACK: the four analysis outputs go here>>"


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
    try:
        print(render(raw))
    except PodcastOptionError as e:
        raise SystemExit(f"error: {e}") from None
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
