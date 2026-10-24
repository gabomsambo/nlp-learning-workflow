"""Measured v2 prompts.

Text is the fact-check research report, section 9 (2026-10-24). The schema
field order that goes with the verifier prompt lives on ``ClaimVerdict``.
"""

from __future__ import annotations

EXTRACT_INSTRUCTIONS = """\
You extract checkable factual claims from a podcast script about ONE research paper, so that each claim can later be checked against the paper's text.
The script is given as numbered lines "L<n>: <text>". You will be told which line range to extract from; read the whole script for context but emit claims ONLY for lines in that range.
SELECT. Emit a claim for every statement that asserts a fact:
- about the paper: its title, authors, year, venue, problem, method, architecture, data, training setup, hyperparameters, experiments, numbers, results, comparisons, stated limitations, stated future work;
- about the world, presented as fact: history of the field, other systems or papers, how things generally work ("background").
Do NOT emit: greetings, sign-offs, signposting ("let's dive in"), rhetorical questions, analogies or metaphors framed as such, the host's opinions or enthusiasm ("this is clever"), and speculation clearly hedged by the host ("maybe", "I'd guess"). When a sentence mixes an opinion with a fact, emit the fact.
DECONTEXTUALIZE. Every claim must be understandable on its own: replace pronouns and "they", "the model", "this approach" with the names the script uses. Keep exactly the qualifiers and hedges the script uses: do not make a claim stronger, weaker, more specific or more general than the script says it.
GRANULARITY. One checkable fact per claim. Keep a number together with what it measures (metric, dataset, model size, setting). A sentence listing four datasets is one claim; a sentence giving two different results is two claims.
For each claim give: line; span (the shortest EXACT verbatim substring of that line carrying the claim); claim (standalone); scope "paper"|"background"; category number|method|setup|result|comparison|attribution|limitation|implication|background; verify_marker (line has [VERIFY] or says the paper does not specify something).
"""

VERIFY_INSTRUCTIONS = """\
You are a strict fact checker. For each CLAIM, decide whether the SOURCE supports it. The source is text from one research paper (extracted from its PDF, so tables and equations may be garbled).
Only the source counts. Your own knowledge of this paper, its authors or the field is NOT evidence, even when you are sure it is true. A claim about the wider world is supported only if the paper itself states it.
Verdicts:
- supported: the source states it, or it follows directly. Paraphrase, rounding, and spoken number forms ("about three thousand nine hundred" for 3,912; "seven billion" for 7B) count as the same.
- partially_supported: the core is supported but a detail is not: a number, qualifier, scope, or condition is missing from the source, or the claim states something more strongly or more generally than the source does.
- contradicted: the source says something incompatible: a different number, a different method or dataset, the wrong author, the opposite direction of a result, or a result attributed to the wrong system.
- unsupported: the source does not address it.
- not_checkable: an opinion, analogy, or prediction that no text could confirm.
Claims of ABSENCE ("the paper does not test X", "the authors never discuss Y") are checkable: supported if the source indeed does not do or discuss it, contradicted if it does. Quote the closest passage that shows the scope.
For every verdict except not_checkable, give evidence: up to 3 quotes copied VERBATIM from the source (each under 300 characters) … "rationale" is one short sentence. Work in this order: find the quotes, state what the source says, reason, THEN choose the verdict, so the verdict follows from the evidence.
error_type: number_mismatch | invented_detail | overstatement | wrong_attribution | method_misdescribed | outside_knowledge | other | none.
"""

REJUDGE_ADDENDUM = """\
Be careful in both directions: do not excuse a wrong number or a wrong attribution, and do not flag a claim just because the wording differs. Papers use abbreviations … resolve them before deciding. Tables extracted from PDFs often lose their row labels or checkmarks; when a table is ambiguous, use the paper's prose about that table to decide which row is which.
"""


def verification_instructions(*, rejudge: bool) -> str:
    """Screen prompt, or the same prompt plus the re-judge addendum."""
    if not rejudge:
        return VERIFY_INSTRUCTIONS
    return VERIFY_INSTRUCTIONS.rstrip() + "\n\n" + REJUDGE_ADDENDUM
