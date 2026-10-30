"""The evidence writer's prompts, versioned.

``EVIDENCE_PROMPT_VERSION`` changes whenever any text here changes, and is
stored with every ledger, so a quality shift is attributable to a prompt and
not guessed from logs.

The writer's system prompt is the free writer's system prompt with its
GROUNDING RULES block replaced. It is derived rather than copied so the OUTPUT
FORMAT, TTS and EPISODE SETTINGS text cannot drift between the two modes; if
the free prompt is edited so the markers below vanish, importing this module
fails loudly instead of quietly writing a different prompt.
"""

from __future__ import annotations

LEDGER_PROMPT_VERSION = "ledger-v1"
EVIDENCE_PROMPT_VERSION = "evidence-v1"

LEDGER_SYSTEM = """You build an EVIDENCE LEDGER from one research paper, for a podcast writer who will be allowed to state only what the ledger contains.
Produce {lo} to {hi} entries covering, in this order of importance: the paper's identity (title, authors, year); the problem and motivation as the AUTHORS state them; the method step by step; datasets, benchmarks and baselines; every headline number together with the metric, dataset, model/setting it belongs to; the main comparisons exactly as the authors state them; ablations and what the authors conclude from them; limitations and future work the authors themselves state; definitions of the paper's own terms.
For each entry give, IN THIS ORDER: quote (copied VERBATIM from the paper text or from a TABLE block, 20-250 characters, the shortest passage that makes the statement true; for a table cell use that table row's text), where (section or table), statement (one self-contained fact in plain words; every number stays attached to the thing it measures; keep the paper's hedges such as 'may' and 'in our experiments'), kind (one of identity, problem, method, data, result, comparison, ablation, limitation, definition).
Rules: copy numbers exactly. If a number's row or column cannot be determined, do not write an entry for it. Do not compute anything (no sums, differences, ratios, counts of wins). Do not add any fact the paper does not state, including who the authors are beyond what is printed, what came later, or what is true in the world. Do not paraphrase a quote."""

LEDGER_USER = """Build the evidence ledger for this paper.

{paper}"""

# Replaces the free writer's "GROUNDING RULES (STRICT)" block.
EVIDENCE_RULES = """GROUNDING RULES (STRICT, EVIDENCE MODE)
- The ONLY facts you may state about the paper (its method, data, numbers, results, comparisons, the authors' claims, limitations) are the numbered ledger entries (F# = narrative facts, T# = table rows, each already checked against the paper) in the user message. Do not use anything else: not your own knowledge, not the web, not what you remember about this paper or its authors.
- Every [HOST] line that states any such fact MUST end with the ids it relies on in double square brackets, for example: "... gives 61.7. [[F12,T3]]". Cite every entry you used. A line that states no paper facts (greeting, transition, question to the listener, analogy, the host's own opinion framed as opinion) ends with [[-]]. Cue lines such as [PAUSE] take no tag. The tags are bookkeeping that is removed before the script is read aloud: never mention them, the ledger or entry ids in what the host says.
- Numbers: use digits (or their spoken form) only from the cited entries, kept attached to what they measure. Do NOT compute new numbers: no sums, differences, ratios, percent changes, rankings, or counts such as "six of eight datasets", unless a cited entry states the result. Compare two results only if a cited entry states the comparison. For a table row, the column=value pairs of THAT row are the only meaning of its numbers.
- Do not attribute arguments, motives, or advice to the authors ("the authors argue/suggest/make the case") unless a cited entry says so. Do not say the paper "never" or "does not" do something unless a cited entry says so.
- Background about the world or the field (history, who invented what, what is "still widely used today", what came later) is an outside fact: leave it out. Analogies are fine if they add no fact about the paper.
- Keep the authors' hedges ("may", "in our experiments", "on these datasets"). Do not strengthen them with words like "consistently", "dramatically", "always", "proves".
- Everything else (tone, structure, length, audience) follows the instructions below.
"""

LEDGER_SECTION = """EVIDENCE LEDGER (the only facts you may state about the paper)
{ledger}

Cover the ledger broadly: reach the length target by using more of the F entries and the table rows that matter, not by elaborating beyond what an entry says."""

EVIDENCE_REMINDER = """REMINDER — these outrank anything in the EPISODE SETTINGS block above:
- State only facts found in the EVIDENCE LEDGER above, and end every [HOST] line with the ids it relies on in double square brackets, or [[-]] when it states no fact about the paper.
- Do not compute new numbers and do not add background from outside the ledger.
- Every line must begin with [HOST]: apart from standalone [MUSIC], [SFX], [PAUSE] and [TRANSITION] cues.

OUTPUT ONLY THE SCRIPT IN THE REQUIRED SPEAKER FORMAT."""

_SYSTEM_START = "GROUNDING RULES (STRICT)\n"
_SYSTEM_END = "OUTPUT FORMAT (STRICT)"
_USER_END = "RESOURCES (GROUND PACK)"


def build_evidence_system(free_system: str) -> str:
    """The free writer's system prompt with its grounding block swapped out."""
    start = free_system.find(_SYSTEM_START)
    end = free_system.find(_SYSTEM_END)
    if start < 0 or end < start:
        raise RuntimeError(
            "the free writer's system prompt no longer has a GROUNDING RULES "
            "block followed by OUTPUT FORMAT; update nlp_pillars/evidence/prompts.py"
        )
    return free_system[:start] + EVIDENCE_RULES + "\n" + free_system[end:]


def build_evidence_user_template(free_user: str) -> str:
    """The free writer's user prompt up to its Ground Pack, then the ledger.

    The result still has the free prompt's ``{placeholders}`` plus ``{ledger}``;
    callers fill them in one ``format`` pass.
    """
    end = free_user.find(_USER_END)
    if end < 0:
        raise RuntimeError(
            "the free writer's user prompt no longer has a RESOURCES (GROUND PACK) "
            "section; update nlp_pillars/evidence/prompts.py"
        )
    head = free_user[:end]
    return head + LEDGER_SECTION + "\n\n" + EVIDENCE_REMINDER
