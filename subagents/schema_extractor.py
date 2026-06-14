"""
M_04 — SchemaExtractor (deterministic NER, fallback layer)
==========================================================
Role     : extract structured UserSchema values from the latest user message
           using deterministic parsers only.
Trigger  : every turn, right after the M_01 understanding agent. M_01's merged
           LLM call is the primary extractor; M_04 is a deterministic top-up that
           fills any fields the LLM missed (and the sole extractor when the LLM is
           unavailable, e.g. offline tests).
Output   : AgentResult.schema_updates = {field_key: value} plus a per-field
           confidence map in meta["confidence"]. The orchestrator applies the
           updates via schema.set() (which validates against the glossary).
Fallback : if nothing can be parsed, returns empty schema_updates (no-op).

Deterministic passes (precision over recall)
--------------------------------------------
  1. EXPECTED-FIELD pass — the field the agent most likely just asked
     (schema.next_missing_field()) is parsed from the answer with high confidence.
     Age here accepts bare ("42") and spelled-out ("forty two") numbers.
  2. GLOBAL keyword pass — any other field with an unambiguous keyword signal is
     also captured at medium confidence. Age uses an ANCHORED parser here
     ("<n> years"/"age <n>"/"i am <n>") so it never grabs a stray number.
A field already set on the schema is never overwritten.
"""

from __future__ import annotations

import os
import re
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from .base import AgentContext, AgentResult, AgentID
except ImportError:  # pragma: no cover - script execution
    from base import AgentContext, AgentResult, AgentID  # type: ignore

from attribute_glossary import get_entry


# ---------------------------------------------------------------------------
# Keyword → enum-value maps (sourced from glossary valid_values semantics)
# ---------------------------------------------------------------------------

_AFFIRM = {"yes", "yeah", "yep", "yup", "sure", "haan", "ha", "correct", "true",
           "i do", "i have", "definitely", "of course", "ji haan"}
_NEGATE = {"no", "nope", "nah", "nahi", "none", "don't", "do not", "false",
           "not really", "no thanks", "i don't", "i do not"}

_BUYER_TYPE = [
    (("employer_large",), ("50 employees", "more than 50", "large company",
                            "big company", "500 employees", "1000 employees")),
    (("employer_sme",),   ("employees", "my company", "my staff", "my team",
                           "for my business", "small business", "sme", "startup")),
    (("gig_worker",),     ("gig", "delivery", "cab driver", "uber", "ola",
                           "freelance", "self-employed", "daily wage", "domestic worker")),
    (("individual",),     ("myself", "for me", "my family", "personal",
                           "individual", "just me", "me and my")),
]

_GENDER = [
    ("female", ("female", "woman", "lady", "girl", "she", "her", "f ")),
    ("male",   ("male", "man", "boy", "he ", "him", "m ")),
    ("other",  ("other", "non-binary", "nonbinary", "prefer not")),
]

_PRIMARY_NEED = [
    ("maternity",        ("maternity", "pregnan", "delivery", "newborn", "baby")),
    ("cancer",           ("cancer", "tumour", "tumor", "oncolog")),
    ("critical_illness", ("critical illness", "heart attack", "stroke", "lump sum",
                          "lumpsum", "serious illness")),
    ("accident",         ("accident", "disability", "accidental")),
    ("top_up",           ("top up", "top-up", "topup", "additional cover",
                          "extra cover on top", "deductible")),
    ("international",     ("international", "abroad", "overseas", "outside india",
                           "foreign")),
    ("daily_cash",       ("daily cash", "cash per day", "hospital cash", "per day")),
    ("hospitalisation",  ("hospitalisation", "hospitalization", "hospital stay",
                           "in-patient", "inpatient", "general health", "standard")),
]

_PED_TYPE = [
    ("diabetes_cardiac", ("diabetes", "diabetic", "sugar", "cardiac", "heart",
                          "hypertension", "blood pressure", "bp", "cholesterol")),
    ("other_ped",        ("thyroid", "arthritis", "kidney", "asthma", "liver",
                          "other condition")),
    ("none",             ("none", "no condition", "nothing", "healthy")),
]

_BUDGET_BAND = [
    ("micro",   ("under 2000", "less than 2000", "1000", "1500", "micro",
                 "very cheap", "3 per day", "5 per day")),
    ("budget",  ("2000", "5000", "10000", "budget", "affordable", "cheap")),
    ("mid",     ("15000", "20000", "25000", "30000", "mid", "moderate")),
    ("premium", ("above 30000", "50000", "100000", "premium", "high end",
                 "no limit", "money no problem")),
]

_FAMILY_COVER = [
    ("floater_joint",   ("parents", "joint family", "3 generation", "three generation",
                          "in-laws", "grandparent", "extended family")),
    ("floater_nuclear", ("spouse", "wife", "husband", "children", "kids", "family",
                          "nuclear")),
    ("individual",      ("just me", "only me", "myself", "individual", "single", "just myself")),
]

_SI_PREFERENCE = [
    ("50L_plus", ("50 lakh", "50 lakhs", "1 crore", "crore", "50l", "50 lac")),
    ("10_25L",   ("10 lakh", "15 lakh", "20 lakh", "25 lakh", "10l", "10-25",
                  "comprehensive")),
    ("3_5L",     ("3 lakh", "4 lakh", "5 lakh", "3l", "5l", "3-5")),
    ("1_2L",     ("1 lakh", "2 lakh", "1l", "2l", "1-2", "small cover")),
]


# ---------------------------------------------------------------------------
# Primitive parsers
# ---------------------------------------------------------------------------

def _first_match(text: str, table: list) -> str | None:
    """Return the value whose any keyword appears in text (table order = priority)."""
    for value, keywords in table:
        if any(k in text for k in keywords):
            return value[0] if isinstance(value, tuple) else value
    return None


def _parse_bool(text: str) -> bool | None:
    # Normalise punctuation so "no," / "yes!" match on word boundaries.
    norm = " " + re.sub(r"[^a-z0-9\s]", " ", text) + " "
    norm = re.sub(r"\s+", " ", norm)
    if any(f" {w} " in norm for w in _NEGATE):
        return False
    if any(f" {w} " in norm for w in _AFFIRM):
        return True
    return None


_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_int(text: str) -> int | None:
    """Parse a small spelled-out number (0–120) like 'twenty five' or 'thirty-one'."""
    total = None
    for tok in re.findall(r"[a-z]+", text.lower()):
        if tok in _TENS:
            total = (total or 0) + _TENS[tok]
        elif tok in _ONES:
            total = (total or 0) + _ONES[tok]
        elif total is not None:
            break   # stop once a non-number word follows a started number
    return total


def _parse_age_anchored(text: str) -> int | None:
    """Age only when explicitly anchored ("<n> years", "age <n>", "i am <n>").
    Safe to run on any message in the global scan — won't grab stray numbers."""
    m = re.search(r"\b(\d{1,3})\s*(?:years?|yrs?|yo|y/o|years old)\b", text)
    if not m:
        m = re.search(r"\b(?:age|aged|i am|i'm|im)\s*(?:is\s*)?(\d{1,3})\b", text)
    if m:
        val = int(m.group(1))
        if 0 <= val <= 120:
            return val
    return None


def _parse_age(text: str) -> int | None:
    # Anchored match first; else a bare number; finally a spelled-out number
    # ("twenty five" → 25). The bare/spelled paths are only safe when age is the
    # expected field (Pass 1), where the whole reply is the answer.
    val = _parse_age_anchored(text)
    if val is not None:
        return val
    m = re.fullmatch(r"\s*(\d{1,3})\s*", text)  # bare number answer
    if m:
        val = int(m.group(1))
        if 0 <= val <= 120:
            return val
    word = _words_to_int(text)
    if word is not None and 0 <= word <= 120:
        return word
    return None


def _parse_count(text: str) -> int | None:
    m = re.search(r"\b(\d{1,2})\s*(?:people|members|persons|of us|family members)\b", text)
    if not m:
        m = re.fullmatch(r"\s*(\d{1,2})\s*", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 8:
            return val
    return None


# field -> (parser, high-confidence value)
def _parse_field(key: str, text: str):
    if key == "buyer_type":   return _first_match(text, _BUYER_TYPE)
    if key == "age":          return _parse_age(text)
    if key == "gender":       return _first_match(text, _GENDER)
    if key == "primary_need": return _first_match(text, _PRIMARY_NEED)
    if key == "has_ped":
        # A named condition implies has_ped True even without an explicit "yes".
        if _first_match(text, _PED_TYPE) in ("diabetes_cardiac", "other_ped"):
            return True
        return _parse_bool(text)
    if key == "ped_type":     return _first_match(text, _PED_TYPE)
    if key == "needs_opd":    return _parse_bool(text)
    if key == "budget_band":  return _first_match(text, _BUDGET_BAND)
    if key == "family_cover": return _first_match(text, _FAMILY_COVER)
    if key == "family_size":  return _parse_count(text)
    if key == "si_preference":return _first_match(text, _SI_PREFERENCE)
    return None


# Fields safe to scan globally every turn (unambiguous keyword signals). `age`
# uses an anchored parser (see global scan loop) so it can be picked up from a
# multi-field utterance without grabbing stray numbers.
_GLOBAL_SCAN = ["buyer_type", "age", "gender", "primary_need", "ped_type",
                "budget_band", "family_cover", "si_preference"]

# Fields the LLM pass is allowed to extract (matches deterministic coverage).
_EXTRACTABLE = ["buyer_type", "age", "gender", "primary_need", "has_ped",
                "ped_type", "needs_opd", "budget_band", "family_cover",
                "family_size", "si_preference"]

# Integer field ranges (mirror user_schema.set() validation) — out-of-range rejected.
_INT_RANGE = {"age": (0, 120), "family_size": (1, 8)}

_BOOL_TRUE = {"true", "yes", "y", "1"}
_BOOL_FALSE = {"false", "no", "n", "0"}


def _coerce_valid(key: str, value):
    """Coerce an LLM-supplied value to a glossary-valid value, or None if invalid.

    Guarantees the orchestrator's schema.set(key, value) cannot raise.
    """
    if value is None:
        return None
    entry = get_entry(key)
    if not entry:
        return None
    typ = entry.get("type")

    if typ == "int":
        try:
            ival = int(value)
        except (TypeError, ValueError):
            return None
        lo, hi = _INT_RANGE.get(key, (None, None))
        if lo is not None and not (lo <= ival <= hi):
            return None
        return ival

    if typ == "bool":
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        if token in _BOOL_TRUE:
            return True
        if token in _BOOL_FALSE:
            return False
        return None

    # str_enum: value must be exactly one of the glossary's valid_values keys.
    valid = entry.get("valid_values")
    if isinstance(valid, dict):
        token = str(value).strip()
        if token in {str(k) for k in valid.keys()}:
            return token
    return None


class SchemaExtractor:
    agent_id = AgentID.SCHEMA_EXTRACTOR

    def run(self, ctx: AgentContext) -> AgentResult:
        text = (ctx.message or "").lower()
        schema = ctx.record.schema
        updates: dict = {}
        confidence: dict = {}

        # ── Pass 1: the field we most likely just asked (strong prior) ──────
        expected_entry = schema.next_missing_field()
        expected = expected_entry["key"] if expected_entry else None
        if expected:
            val = _parse_field(expected, text)
            if val is not None:
                updates[expected] = val
                confidence[expected] = 0.95
            # If we just asked has_ped and got a condition, also fill ped_type.
            if expected == "has_ped" and updates.get("has_ped") is True:
                pt = _first_match(text, _PED_TYPE)
                if pt and pt != "none":
                    updates.setdefault("ped_type", pt)
                    confidence["ped_type"] = 0.8

        # ── Pass 2: global keyword scan for other clearly-signalled fields ──
        for key in _GLOBAL_SCAN:
            if key in updates:
                continue
            if getattr(schema, key, None) is not None:
                continue   # never overwrite an already-collected field
            # age is scanned with the anchored parser (no stray-number grabs).
            val = _parse_age_anchored(text) if key == "age" else _parse_field(key, text)
            if val is not None:
                updates[key] = val
                confidence[key] = 0.6

        return AgentResult(
            agent_id=self.agent_id,
            output_text=None,
            schema_updates=updates,
            meta={"confidence": confidence},
        )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = SchemaExtractor()

    def extract(msg, preset=None):
        rec = ConversationRecord.new(session_id="m04_test")
        for k, v in (preset or {}).items():
            rec.schema.set(k, v)
        res = agent.run(AgentContext(record=rec, message=msg))
        return res.schema_updates, res.meta["confidence"]

    # Expected-field prior: empty schema → first missing is buyer_type.
    u, c = extract("just for myself and my family")
    assert u.get("buyer_type") == "individual", u

    # Age answer to the age question.
    u, _ = extract("I'm 42", preset={"buyer_type": "individual"})
    assert u.get("age") == 42, u

    # Gender.
    u, _ = extract("female", preset={"buyer_type": "individual", "age": 30})
    assert u.get("gender") == "female", u

    # Primary need keyword.
    u, _ = extract("mainly maternity cover please",
                   preset={"buyer_type": "individual", "age": 30, "gender": "female"})
    assert u.get("primary_need") == "maternity", u

    # has_ped via named condition also fills ped_type.
    u, _ = extract("yes I have diabetes",
                   preset={"buyer_type": "individual", "age": 50, "gender": "male",
                           "primary_need": "hospitalisation"})
    assert u.get("has_ped") is True and u.get("ped_type") == "diabetes_cardiac", u

    # Negation for needs_opd.
    u, _ = extract("no, hospital only",
                   preset={"buyer_type": "individual", "age": 50, "gender": "male",
                           "primary_need": "hospitalisation", "has_ped": False})
    assert u.get("needs_opd") is False, u

    # Never overwrite a set field.
    u, _ = extract("actually employees of my company",
                   preset={"buyer_type": "individual", "age": 30})
    assert "buyer_type" not in u, u

    print("schema_extractor.py (M_04) self-test passed.")
