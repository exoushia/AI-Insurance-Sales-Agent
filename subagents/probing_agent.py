"""
M_05 — ProbingAgent 
=====================================================
Role     : during S1_DISCOVERY, decide the next field to ask and emit a
           voice-phrased question. Fully deterministic: the field choice and the
           spoken phrasing both come from the conversational templates below.
Trigger  : state == S1_DISCOVERY.
Tools    : schema queue + filter_products probe selection (read-only).
Output   :
  - output_text  : the question to speak (None when action == "proceed").
  - meta["action"]: "ask" | "probe" | "proceed".
  - meta["field"] : the field key asked (when action == "ask").
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from .base import AgentContext, AgentResult, AgentID
except ImportError:  # pragma: no cover - script execution
    from base import AgentContext, AgentResult, AgentID  # type: ignore

from attribute_glossary import get_question
from retrieval_tools import _select_probe_question, MAX_CANDIDATES_WITHOUT_PROBE


MAX_DISCOVERY_TURNS = 8

# Voice phrasing variants per field (the agent is voice-first). These are
# deliberately OPEN and conversational — we do NOT read out every option like an
# IVR menu; the schema extractor (M_04) maps whatever the caller says back to the
# right value, so the questions stay short and human.
QUESTION_TEMPLATES: dict[str, str] = {
    "buyer_type":   "To start — are we sorting out cover for you and your family, "
                    "or for a business and its team?",
    "age":          "And how old are you?",
    "gender":       "Got it — may I ask, are you male or female?",
    "primary_need": "What's the main thing you're hoping this policy will take care of?",
    "has_ped":      "Any ongoing health conditions I should keep in mind — "
                    "things like diabetes or blood pressure?",
    "ped_type":     "Got it — what kind of condition is it?",
    "needs_opd":    "Would you like it to cover regular doctor visits and medicines too, "
                    "or mainly hospital stays?",
    "budget_band":  "Roughly what yearly budget did you have in mind?",
    "family_cover": "Who would you like on the policy with you — "
                    "a spouse, kids, parents?",
    "family_size":  "And how many people in total, including yourself?",
    "si_preference":"And how much cover are you after — a few lakhs, ten lakhs, or more?",
}

TRANSITION_TEMPLATES = {
    "ped_to_opd":   "Got it.",
    "budget_pivot": "Makes sense.",
    "family_pivot": "Perfect.",
    "generic":      "Thanks.",
    "first_q":      "",
    "pre_recommend":"That gives me a clear picture.",
}

PROBE_TEMPLATE = (
    "I have a few good options for you. "
    "To narrow it down to the best fit — {probe_question}"
)


def _pick_transition(last_collected: str | None, next_field: str) -> str:
    if last_collected is None:
        return TRANSITION_TEMPLATES["first_q"]
    if last_collected == "has_ped" and next_field == "needs_opd":
        return TRANSITION_TEMPLATES["ped_to_opd"]
    if next_field == "budget_band":
        return TRANSITION_TEMPLATES["budget_pivot"]
    if next_field in ("family_cover", "family_size"):
        return TRANSITION_TEMPLATES["family_pivot"]
    if next_field == "si_preference":
        return TRANSITION_TEMPLATES["pre_recommend"]
    return TRANSITION_TEMPLATES["generic"]


class ProbingAgent:
    agent_id = AgentID.PROBING

    def run(self, ctx: AgentContext) -> AgentResult:
        schema = ctx.record.schema
        retrieval_result = ctx.retrieval_result or ctx.record.retrieval_result
        turn_count = getattr(schema, "turn_count", 0)

        # ── Gate 1: too many discovery turns → force a recommendation attempt
        if turn_count >= MAX_DISCOVERY_TURNS and schema.sufficient_for_retrieval():
            return self._proceed()

        # ── Gate 2: everything needed is already collected
        if schema.sufficient_for_recommendation() and schema.sufficient_for_plan_selection():
            return self._proceed()

        # ── Gate 3: probe to narrow a wide candidate set
        probe_q = self._should_probe(schema, retrieval_result)
        if probe_q:
            base = PROBE_TEMPLATE.format(probe_question=probe_q)
            return AgentResult(
                agent_id=self.agent_id,
                output_text=base,
                meta={"action": "probe"},
            )

        # ── Gate 4: next deterministic field from the queue
        nxt = schema.next_missing_field()
        if nxt is None:
            return self._proceed()

        field_key = nxt["key"]
        question = self._phrase(field_key, schema)
        return AgentResult(
            agent_id=self.agent_id,
            output_text=question,
            meta={"action": "ask", "field": field_key},
        )

    # -- helpers ------------------------------------------------------------

    def _proceed(self) -> AgentResult:
        return AgentResult(agent_id=self.agent_id, output_text=None,
                           meta={"action": "proceed"})

    def _should_probe(self, schema, retrieval_result) -> str | None:
        if not retrieval_result or schema.probe_asked:
            return None
        top = [c["product_id"] for c in retrieval_result.get("candidates", [])
               if c["score"] > 0.10]
        if len(top) <= MAX_CANDIDATES_WITHOUT_PROBE:
            return None
        return _select_probe_question(top, schema.to_tool_input())

    def _phrase(self, field_key: str, schema) -> str:
        base = QUESTION_TEMPLATES.get(field_key) or get_question(field_key) or ""
        collected = list(schema.collected_collection_fields().keys())
        last_collected = collected[-1] if collected else None
        transition = _pick_transition(last_collected, field_key)
        prefix = f"{transition} " if transition else ""
        return f"{prefix}{base}"


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = ProbingAgent()

    def ask(preset):
        rec = ConversationRecord.new(session_id="m05_test")
        for k, v in preset.items():
            rec.schema.set(k, v)
        return agent.run(AgentContext(record=rec))

    # Empty schema → first question is buyer_type.
    r = ask({})
    assert r.meta["action"] == "ask" and r.meta["field"] == "buyer_type", r.meta
    assert r.output_text and "team" in r.output_text

    # After buyer_type → age.
    r = ask({"buyer_type": "individual"})
    assert r.meta["field"] == "age", r.meta

    # has_ped just collected → next is needs_opd with the ped_to_opd transition.
    r = ask({"buyer_type": "individual", "age": 40, "gender": "male",
             "primary_need": "hospitalisation", "has_ped": False})
    assert r.meta["field"] == "needs_opd", r.meta
    assert r.output_text.startswith("Got it."), r.output_text

    # Fully collected → proceed.
    r = ask({"buyer_type": "individual", "age": 40, "gender": "male",
             "primary_need": "hospitalisation", "has_ped": False, "needs_opd": False,
             "budget_band": "mid", "family_cover": "individual", "si_preference": "3_5L",
             "resolved_product_id": "SP001", "resolved_plan_id": "SP001_SILVER"})
    assert r.meta["action"] == "proceed", r.meta
    assert r.output_text is None

    print("probing_agent.py (M_05) self-test passed.")
