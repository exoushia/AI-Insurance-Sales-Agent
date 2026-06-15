"""
M_11 — ResponseQueue
====================
Role     : assemble the final spoken turn from ordered response segments and
           normalise it for text-to-speech.
Trigger  : outbound stage, final step before the turn is returned (runs AFTER
           M_12 validation and M_10 translation — it is the last text stage
           before the string is handed to the TTS engine).
Tools    : none (deterministic FIFO join + TTS normalisation).
Output   : output_text — the single, ordered, speech-ready response string.

TTS normalisation (Bulbul reads the raw string literally):
  - acronyms → spaced letters so they are spelt out, not phoneticised
    (e.g. "IRDAI" → "I R D A I");
  - bracketed asides / clause citations are dropped whole (never spoken);
  - markdown / section symbols stripped;
  - long bare integers get thousands separators (Bulbul reads "10,000" as a
    whole number, "10000" digit-by-digit);
  - English-only: a stray "hai" token is corrected to "hi" (skipped for
    Hindi/Hinglish where "hai"/है is a legitimate, common word).

Payload contract:
  ctx.payload["segments"]      — ordered list of response fragments (optional).
  ctx.payload["response_text"] — fallback single response if no segments given.
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

_WS_RE = re.compile(r"\s+")

# Acronyms spoken letter-by-letter (spaced) so Bulbul spells them out instead of
# reading them as a word. \b boundaries keep these from firing inside longer
# tokens. "SI" expands to plain words since the letters alone are ambiguous.
_ACRONYM_MAP = {
    "IRDAI": "I R D A I",
    "IRDA": "I R D A",
    "PED": "P E D",
    "TPA": "T P A",
    "OPD": "O P D",
    "ICU": "I C U",
    "NCB": "N C B",
    "SI": "sum insured",
}
_ACRONYM_RES = [(re.compile(rf"\b{re.escape(k)}\b"), v) for k, v in _ACRONYM_MAP.items()]

# Bracketed asides and citations the model sometimes emits — dropped whole so
# they are never voiced. (Char class excludes all bracket types → no nesting.)
_PARENS_RE = re.compile(r"\s*[\(\[\{][^()\[\]{}]*[\)\]\}]")
# "Clause 8.1 –" / "Per Clause 4.6," style citations and "§17" section refs.
_CLAUSE_RE = re.compile(r"(?i)\b(?:as\s+per\s+|per\s+)?clause\s+\d+(?:\.\d+)*\s*[–—:-]?\s*")
_SECTION_RE = re.compile(r"§\s*\d+(?:\.\d+)*")
_MARKDOWN_RE = re.compile(r"[*_#`]")
# Rupee amounts: "₹13,800" / "₹ 5000" / "Rs. 1,20,000" → spoken "13,800 rupees".
# The TTS engine does not voice the ₹ glyph, so convert it to the spoken word
# and move it after the number where it reads naturally.
_RUPEE_RE = re.compile(r"(?:₹|\brs\.?|\binr\b)\s*([\d][\d,]*(?:\.\d+)?)", re.IGNORECASE)
# Bare integers of 5+ digits not already grouped with commas.
_BIGNUM_RE = re.compile(r"(?<![\d,])\d{5,}(?![\d,])")
# Leftover space before punctuation after removals.
_SPACE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
# Stray romanised "hai" (English mode only).
_HAI_RE = re.compile(r"\b([Hh])ai\b")

_ENGLISH_LANGS = {"english", "en", "en-in", "en_in", ""}


class ResponseQueue:
    agent_id = AgentID.RESPONSE_QUEUE

    def run(self, ctx: AgentContext) -> AgentResult:
        segments = ctx.payload.get("segments")
        if not segments:
            single = ctx.payload.get("response_text", "") or ""
            segments = [single] if single else []

        ordered = [self._normalise(s) for s in segments if s and str(s).strip()]
        combined = " ".join(ordered).strip()
        combined = _WS_RE.sub(" ", combined)

        language = getattr(ctx.record.schema, "language", None)
        combined = self._normalize_for_tts(combined, language)

        return AgentResult(
            agent_id=self.agent_id,
            output_text=combined,
            meta={"segment_count": len(ordered)},
        )

    @staticmethod
    def _normalise(segment: str) -> str:
        return _WS_RE.sub(" ", str(segment).strip())

    @staticmethod
    def _normalize_for_tts(text: str, language: str | None = None) -> str:
        """Make the assembled string speech-ready for Bulbul (see module doc)."""
        if not text:
            return text

        out = _PARENS_RE.sub(" ", text)
        out = _SECTION_RE.sub(" ", out)
        out = _CLAUSE_RE.sub("", out)
        out = _MARKDOWN_RE.sub("", out)

        # Convert rupee amounts to spoken form BEFORE big-number grouping so the
        # currency word is attached and the digits keep their comma grouping.
        out = _RUPEE_RE.sub(lambda m: f"{m.group(1)} rupees", out)

        for rx, repl in _ACRONYM_RES:
            out = rx.sub(repl, out)

        out = _BIGNUM_RE.sub(lambda m: format(int(m.group(0)), ","), out)

        lang = (language or "english").strip().lower()
        if lang in _ENGLISH_LANGS:
            out = _HAI_RE.sub(lambda m: "Hi" if m.group(1).isupper() else "hi", out)

        out = _WS_RE.sub(" ", out).strip()
        out = _SPACE_PUNCT_RE.sub(r"\1", out)
        return out


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = ResponseQueue()

    def assemble(payload):
        rec = ConversationRecord.new(session_id="m11_test")
        return agent.run(AgentContext(record=rec, payload=payload))

    # Multiple ordered segments → joined in FIFO order.
    r = assemble({"segments": ["Great choice.", "Here's how the plan works.", "Shall I proceed?"]})
    assert r.output_text == "Great choice. Here's how the plan works. Shall I proceed?", r.output_text
    assert r.meta["segment_count"] == 3

    # Single response_text fallback.
    r = assemble({"response_text": "  Your   plan is   ready.  "})
    assert r.output_text == "Your plan is ready.", repr(r.output_text)

    # Empty payload → empty string, zero segments.
    r = assemble({})
    assert r.output_text == "" and r.meta["segment_count"] == 0

    # --- TTS normalisation -------------------------------------------------
    norm = ResponseQueue._normalize_for_tts

    # Acronyms spelt out as spaced letters.
    assert norm("As per IRDAI rules") == "As per I R D A I rules", norm("As per IRDAI rules")
    assert "P E D" in norm("the PED waiting period"), norm("the PED waiting period")
    assert norm("a high SI plan") == "a high sum insured plan", norm("a high SI plan")

    # Bracketed asides and clause/section citations dropped whole.
    assert norm("Cover is wide (as per IRDAI §17) for you.") == "Cover is wide for you.", \
        norm("Cover is wide (as per IRDAI §17) for you.")
    assert norm("Clause 8.1 – Cashless is supported.") == "Cashless is supported.", \
        norm("Clause 8.1 – Cashless is supported.")

    # Markdown stripped; long numbers grouped.
    assert norm("**Premium** is 300000 rupees") == "Premium is 300,000 rupees", \
        norm("**Premium** is 300000 rupees")
    # Already-grouped numbers and 4-digit years untouched.
    assert norm("In 2024 the cap is 1,50,000") == "In 2024 the cap is 1,50,000", \
        norm("In 2024 the cap is 1,50,000")

    # "hai" fixed in English, preserved in Hinglish.
    assert norm("Hai there, welcome") == "Hi there, welcome", norm("Hai there, welcome")
    assert norm("aapka plan ready hai", language="hinglish") == "aapka plan ready hai", \
        norm("aapka plan ready hai", language="hinglish")

    print("response_queue.py (M_11) self-test passed.")
