"""
M_15 — NumericGuardrail (shared)
================================
Deterministic guardrail used by M_07 (PolicySummary), M_08 (PolicyQA) and
M_09 (AgenticRAG).

Rule: every numeric claim that appears in an outbound response MUST also appear
in the grounding context the agent was given (retrieved product metadata, plan
data, policy wording, regulations, or treatment-cost tables). If a number in the
response is NOT found in the allowed context, the response is flagged — the
calling agent then falls back to a safe, number-free phrasing.

This catches hallucinated premiums / waiting periods / sums insured before they
ever reach the user. It is intentionally simple and deterministic for Phase 0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Import the contract. Support both "python subagents/guardrails.py" (script) and
# "from subagents.guardrails import ..." (package) execution.
try:
    from .base import AgentID
except ImportError:  # pragma: no cover - script execution
    from base import AgentID  # type: ignore


# A "number token" — integers/decimals, optionally with separators, %, or an
# Indian-currency / magnitude suffix. We normalise before comparing.
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])                 # not preceded by a word char or dot
    (?:₹|rs\.?\s*|inr\s*)?     # optional currency prefix
    (\d{1,3}(?:[,\d]*\d)?(?:\.\d+)?)   # the digits (allow thousands separators)
    \s*
    (%|percent|lakhs?|lacs?|crores?|cr|k|months?|days?|years?|yrs?)?  # optional suffix
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Words that are numbers but carry no factual risk — safe to ignore if they leak.
_IGNORE_BARE = {"0", "1", "2", "3"}  # small ordinals/counts ("3 options", "1 plan")


@dataclass
class GuardrailReport:
    ok: bool
    offending: list[str]        # numeric tokens found in response but not in context
    checked: list[str]          # all numeric tokens found in the response


def _normalise(num: str, suffix: str | None) -> str:
    """Canonical form of a numeric token for set comparison."""
    digits = num.replace(",", "").rstrip(".")
    suf = (suffix or "").lower()
    # Collapse suffix synonyms.
    if suf in ("percent",):
        suf = "%"
    elif suf in ("lac", "lacs", "lakh", "lakhs"):
        suf = "lakh"
    elif suf in ("cr", "crore", "crores"):
        suf = "crore"
    elif suf in ("yr", "yrs", "year", "years"):
        suf = "year"
    elif suf in ("month", "months"):
        suf = "month"
    elif suf in ("day", "days"):
        suf = "day"
    return f"{digits}{suf}"


def extract_numbers(text: str) -> list[str]:
    """Return the normalised numeric tokens present in `text`."""
    out: list[str] = []
    for m in _NUMBER_RE.finditer(text or ""):
        num, suffix = m.group(1), m.group(2)
        if not num:
            continue
        token = _normalise(num, suffix)
        out.append(token)
    return out


def validate(response_text: str, context_text: str) -> GuardrailReport:
    """
    Check that every numeric token in `response_text` also occurs in
    `context_text`. Bare small counts (0-3 with no unit) are ignored.
    """
    ctx_numbers = set(extract_numbers(context_text))
    checked = extract_numbers(response_text)
    offending = []
    for tok in checked:
        if tok in _IGNORE_BARE:
            continue
        if tok in ctx_numbers:
            continue
        # Also accept a digits-only match against a suffixed context token, and
        # vice-versa (e.g. response "12 months" vs context "12").
        digits_only = re.sub(r"[^\d]", "", tok)
        if digits_only and any(re.sub(r"[^\d]", "", c) == digits_only for c in ctx_numbers):
            continue
        offending.append(tok)
    return GuardrailReport(ok=not offending, offending=offending, checked=checked)


class NumericGuardrail:
    """Thin agent-shaped wrapper so the guardrail is addressable by AgentID."""

    agent_id = AgentID.NUMERIC_GUARDRAIL

    def check(self, response_text: str, context_text: str) -> GuardrailReport:
        return validate(response_text, context_text)


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ctx = (
        "annual_premium_inr: 6200; si_inr: 500000; "
        "waiting.ped_months: 12; opd_limit_inr: 5000; ncb 50%"
    )

    # Grounded response — every number present in context.
    r1 = validate("Your premium is ₹6,200 with a 12 month PED wait and 50% NCB.", ctx)
    assert r1.ok, r1.offending

    # Hallucinated number — 3500 not in context.
    r2 = validate("The premium is just ₹3,500 a year.", ctx)
    assert not r2.ok and "3500" in r2.offending, r2

    # Bare small counts are ignored.
    r3 = validate("I have 3 options for you.", ctx)
    assert r3.ok, r3.offending

    # Suffix/synonym normalisation: "5 lakh" vs context "500000" → digits differ,
    # so this SHOULD flag (different representation, we don't convert magnitudes).
    r4 = validate("Cover of 5 lakh.", ctx)
    assert not r4.ok, "5 lakh should not match 500000 without magnitude conversion"

    print("guardrails.py (M_15) self-test passed.")
