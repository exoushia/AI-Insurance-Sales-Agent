"""
M_01 — UnderstandingAgent (intent + NER, merged)
================================================
Role     : in ONE frontier-model call, read the latest user message and return
           BOTH the intent AND every structured profile field the user stated.
Trigger  : every turn (first agent the orchestrator runs).
Output   : AgentResult with
             meta = {"intent", "confidence", "source"[, "llm_label"]}
             schema_updates = {field_key: coerced_value}   (the extracted fields)
           The orchestrator reads ctx.intent from meta, applies schema_updates,
           then lets M_04 deterministically top up any still-missing fields.

Why merged: intent classification and field extraction both "understand" the same
           sentence. One reasoning call replaces two brittle LLM hops (old M_01 +
           M_04's LLM pass), halving per-turn structured calls and removing the
           regex-only blind spots (e.g. multi-field utterances, spelled-out ages).

Safety / fallback:
  - Safety-critical signals (unsafe/want_human/frustrated) stay 100% deterministic
    (classify_intent) and ALWAYS override the LLM — they are never delegated.
  - A low-confidence LLM label (< _CONFIDENCE_FLOOR) is coerced to UNRECOGNISED so
    the flow asks the user to clarify (orchestrator treats UNRECOGNISED as a failed
    turn, escalating after MAX_FAILURES). Extracted fields are still kept.
  - LLM unavailable/failed → deterministic classify_intent for the intent; fields
    are left to M_04's deterministic extractor.

Confidence (deterministic fallback path):
             1.0  → a definitive keyword/phrase matched (any non-UNRECOGNISED)
             0.3  → no rule matched (UNRECOGNISED)
"""

from __future__ import annotations

import os
import sys

# Ensure the package dir (parent of subagents/) is importable for sibling modules
# like `fsm` — needed when this file is run directly as a script.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from .base import AgentContext, AgentResult, AgentID
except ImportError:  # pragma: no cover - script execution
    from base import AgentContext, AgentResult, AgentID  # type: ignore

from fsm import classify_intent, IntentSignal, INTENT_DEFINITIONS
from prompts_template import UNDERSTANDING_SYSTEM, UNDERSTANDING_USER_TEMPLATE
from llm_gateway import log_llm_call
from attribute_glossary import llm_context_block
from .schema_extractor import _EXTRACTABLE, _coerce_valid

# Signals that must never be overridden by the LLM.
_SAFETY_CRITICAL = {IntentSignal.UNSAFE, IntentSignal.WANT_HUMAN, IntentSignal.FRUSTRATED}
# LLM labels below this confidence are coerced to UNRECOGNISED (ask to clarify).
_CONFIDENCE_FLOOR = 0.5


class IntentClassifier:
    """M_01 — merged understanding agent (intent + NER in one LLM call)."""

    agent_id = AgentID.INTENT_CLASSIFIER

    def run(self, ctx: AgentContext) -> AgentResult:
        deterministic = classify_intent(ctx.message, ctx.record.schema)

        # Safety-critical signals always win — never sent to the LLM, no fields.
        if deterministic in _SAFETY_CRITICAL:
            log_llm_call(self.agent_id, "fallback", "-", True, f"safety={deterministic.value}")
            return self._result(deterministic, 1.0, source="deterministic-safety")

        llm = ctx.llm
        if llm is None or not getattr(llm, "is_available", False):
            log_llm_call(self.agent_id, "fallback", "-", True, "no-llm")
            return self._result(deterministic, self._det_conf(deterministic),
                                source="deterministic")

        model = ctx.config.models.understanding if ctx.config else "?"
        parsed = self._understand_llm(ctx, llm)
        if parsed is None:
            # LLM failed: deterministic intent; fields left to M_04's deterministic pass.
            log_llm_call(self.agent_id, "fallback", model, False, "understand-empty")
            return self._result(deterministic, self._det_conf(deterministic),
                                source="deterministic-fallback")

        intent, confidence, updates = parsed

        if confidence < _CONFIDENCE_FLOOR:
            # Low confidence → clarify: coerce intent to UNRECOGNISED so the flow
            # re-asks. Extracted fields are independent of intent, so keep them.
            log_llm_call(self.agent_id, "llm", model, True,
                         f"low-conf {intent.value}@{confidence:.2f} -> clarify +{','.join(updates) or 'none'}")
            return self._result(IntentSignal.UNRECOGNISED, confidence,
                                source="llm-low-confidence", llm_label=intent.value,
                                schema_updates=updates)

        log_llm_call(self.agent_id, "llm", model, True,
                     f"{intent.value}@{confidence:.2f} +{','.join(updates) or 'none'}")
        return self._result(intent, confidence, source="llm", llm_label=intent.value,
                            schema_updates=updates)

    # -- helpers ------------------------------------------------------------

    def _understand_llm(
        self, ctx: AgentContext, llm
    ) -> tuple[IntentSignal, float, dict] | None:
        """One call → (intent, confidence, coerced schema_updates) or None on failure."""
        intent_menu = "\n".join(f'- "{l}": {m}' for l, m in INTENT_DEFINITIONS.items())
        schema = ctx.record.schema
        known = {k: getattr(schema, k) for k in _EXTRACTABLE
                 if getattr(schema, k, None) is not None}
        expected_entry = schema.next_missing_field()
        expected = expected_entry["key"] if expected_entry else "none"
        recent = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in ctx.record.messages[-4:]
            if isinstance(m.get("content"), str)
        ) or "(no prior turns)"

        payload = llm.complete_json(
            UNDERSTANDING_SYSTEM.format(
                intent_menu=intent_menu,
                field_catalog=llm_context_block(_EXTRACTABLE),
            ),
            UNDERSTANDING_USER_TEMPLATE.format(
                state=ctx.record.state.value,
                resolved_product=schema.resolved_product_id or "none",
                known_fields=known or "(none yet)",
                expected_field=expected,
                recent_context=recent,
                user_message=ctx.message,
            ),
            model=ctx.config.models.understanding if ctx.config else None,
        )
        if not payload:
            return None
        try:
            intent = IntentSignal(str(payload.get("intent", "")).strip().lower())
        except ValueError:
            return None
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        # Coerce each extracted field to a glossary-valid value; skip already-set
        # fields and anything that fails validation (so schema.set() can't raise).
        updates: dict = {}
        raw_fields = payload.get("fields")
        if isinstance(raw_fields, dict):
            for key, raw in raw_fields.items():
                if key not in _EXTRACTABLE:
                    continue
                if getattr(schema, key, None) is not None:
                    continue
                val = _coerce_valid(key, raw)
                if val is not None:
                    updates[key] = val
        return intent, confidence, updates

    @staticmethod
    def _det_conf(intent: IntentSignal) -> float:
        return 0.3 if intent == IntentSignal.UNRECOGNISED else 1.0

    def _result(self, intent: IntentSignal, confidence: float, *,
                source: str, llm_label: str | None = None,
                schema_updates: dict | None = None) -> AgentResult:
        meta = {"intent": intent, "confidence": confidence, "source": source}
        if llm_label is not None:
            meta["llm_label"] = llm_label
        return AgentResult(agent_id=self.agent_id, output_text=None,
                           schema_updates=schema_updates or {}, meta=meta)


# ---------------------------------------------------------------------------
# SELF-TEST  (deterministic path only — ctx.llm is None, no network)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = IntentClassifier()
    rec = ConversationRecord.new(session_id="m01_test")

    cases = [
        ("I want to buy this",        IntentSignal.PROSPECTIVE, 1.0),
        ("can I speak to a human",    IntentSignal.WANT_HUMAN,  1.0),
        ("this is a waste of time",   IntentSignal.FRUSTRATED,  1.0),
        ("thanks, bye",               IntentSignal.DONE,        1.0),
        ("just looking for now",      IntentSignal.EXPLORATORY, 1.0),
        ("what is the waiting period?", IntentSignal.INQUIRY,   1.0),
        ("qwerty zxcv",               IntentSignal.UNRECOGNISED, 0.3),
    ]
    for text, expected_intent, expected_conf in cases:
        ctx = AgentContext(record=rec, message=text)   # ctx.llm is None
        res = agent.run(ctx)
        got = res.meta["intent"]
        assert got == expected_intent, f"{text!r}: expected {expected_intent}, got {got}"
        assert res.meta["confidence"] == expected_conf, f"{text!r}: conf {res.meta['confidence']}"
        assert res.output_text is None
        print(f"  OK: {text!r:35} -> {got.value} (conf={res.meta['confidence']})")

    print("intent_classifier.py (M_01) self-test passed.")
