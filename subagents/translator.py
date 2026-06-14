"""
M_10 — TranslatorAgent
======================
Role     : render the validated response in the user's language.
Trigger  : outbound stage, when schema.language is not English.
Tools    : Sarvam translation API (ctx.sarvam), M_15 NumericGuardrail.
Output   : output_text in the target language + meta {target_language, translated}.

Phase 1 behaviour
-----------------
English  → identity passthrough (translated=False).
Hindi    → Sarvam translate to hi-IN (native script).
Hinglish → Sarvam translate to hi-IN in code-mixed roman script.
Numbers are kept in international (digit) form and re-checked by M_15 against the
original English text; if a figure was mangled or the API fails, the agent falls
back to the original English text (identity) so amounts are never corrupted.

Payload contract:
  ctx.payload["response_text"] — the text to render (required).
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from .base import AgentContext, AgentResult, AgentID
    from .guardrails import NumericGuardrail
except ImportError:  # pragma: no cover - script execution
    from base import AgentContext, AgentResult, AgentID  # type: ignore
    from guardrails import NumericGuardrail  # type: ignore

from llm_gateway import log_llm_call

_ENGLISH = {"english", "en", "en-in", "en_in", "", None}

# language label → Sarvam translate parameters
_HINDI = {"hindi", "hi", "hi-in", "hi_in"}
_HINGLISH = {"hinglish", "hi-en", "hinglish-in", "code-mixed"}


class TranslatorAgent:
    agent_id = AgentID.TRANSLATOR

    def __init__(self):
        self._guard = NumericGuardrail()

    def run(self, ctx: AgentContext) -> AgentResult:
        text = ctx.payload.get("response_text", "") or ""
        language = (getattr(ctx.record.schema, "language", None) or "english")
        lang_norm = str(language).strip().lower()

        if lang_norm in _ENGLISH:
            return AgentResult(
                agent_id=self.agent_id,
                output_text=text,
                meta={"target_language": "english", "translated": False},
            )

        params = self._sarvam_params(lang_norm)
        sarvam = ctx.sarvam
        if params is None or sarvam is None or not getattr(sarvam, "is_available", False):
            # Unsupported language or no Sarvam access → identity passthrough.
            log_llm_call(self.agent_id, "fallback", "sarvam", False,
                         f"no-translate {lang_norm}")
            return AgentResult(
                agent_id=self.agent_id,
                output_text=text,
                meta={"target_language": lang_norm, "translated": False, "stub": True},
            )

        translated = sarvam.translate(text, **params)
        if translated:
            # Protect numbers: every figure in the translation must exist in the
            # original English text; otherwise fall back to English.
            report = self._guard.check(translated, text)
            if report.ok:
                log_llm_call(self.agent_id, "llm", "sarvam", True, f"translated {lang_norm}")
                return AgentResult(
                    agent_id=self.agent_id,
                    output_text=translated,
                    meta={"target_language": lang_norm, "translated": True},
                )
            log_llm_call(self.agent_id, "fallback", "sarvam", False,
                         f"guardrail {report.offending}")
        else:
            log_llm_call(self.agent_id, "fallback", "sarvam", False, "translate-empty")

        # Fallback: original English text (numbers intact).
        return AgentResult(
            agent_id=self.agent_id,
            output_text=text,
            meta={"target_language": lang_norm, "translated": False, "fallback": True},
        )

    @staticmethod
    def _sarvam_params(lang_norm: str) -> dict | None:
        if lang_norm in _HINDI:
            return {"target_language_code": "hi-IN", "mode": "modern-colloquial",
                    "output_script": "fully-native"}
        if lang_norm in _HINGLISH:
            return {"target_language_code": "hi-IN", "mode": "code-mixed",
                    "output_script": "roman"}
        return None


# ---------------------------------------------------------------------------
# SELF-TEST  (no network — ctx.sarvam is None → identity passthrough)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = TranslatorAgent()

    def translate(text, language=None):
        rec = ConversationRecord.new(session_id="m10_test")
        if language is not None:
            rec.schema.language = language
        return agent.run(AgentContext(record=rec, payload={"response_text": text}))

    # English → passthrough, translated False.
    r = translate("Your plan covers maternity after 24 months.")
    assert r.output_text.startswith("Your plan") and r.meta["translated"] is False

    # Non-English with no Sarvam handle → identity passthrough, language recorded.
    r = translate("Aaapka plan bees din mein khatam hojaega.", language="hindi")
    assert r.meta["target_language"] == "hindi" and r.meta["translated"] is False
    assert r.output_text  # text preserved
    print(r.output_text)
    print("translator.py (M_10) self-test passed.")

