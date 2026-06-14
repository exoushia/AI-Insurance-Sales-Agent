"""
M_08 — PolicyQAAgent
====================
Role     : answer a user's question about the resolved product during
           S3_POLICY_QA, grounded strictly in that product's wording document.
Trigger  : state == S3_POLICY_QA.
Tools    : search_policy_wording (isolated to the resolved product), M_15.
Behaviour (deterministic, Phase 0):
  1. If no product is resolved yet, hand off to M_09 (general agentic RAG).
  2. Keyword-search the resolved product's wording. If a clause clears the
     relevance threshold, quote it as the grounded answer and validate numbers
     with M_15.
  3. Otherwise hand off to M_09 for a broader search.
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

from retrieval_tools import search_policy_wording
import json
from prompts_template import POLICY_QA_ANSWER_ENGINE_PROMPT
from llm_gateway import log_llm_call

_MIN_CLAUSE_SCORE = 2   # at least this many distinct query terms must hit a clause


class PolicyQAAgent:
    agent_id = AgentID.POLICY_QA

    def __init__(self):
        self._guard = NumericGuardrail()

    def run(self, ctx: AgentContext) -> AgentResult:
        schema = ctx.record.schema
        pid = schema.resolved_product_id
        question = ctx.message or ""

        if not pid:
            return AgentResult(
                agent_id=self.agent_id,
                output_text=None,
                handoff_to=AgentID.AGENTIC_RAG,
                meta={"reason": "no_resolved_product"},
            )

        result = search_policy_wording(pid, question)
        chunks = result.get("chunks", [])
        best = chunks[0] if chunks else None

        if not best or best.get("score", 0) < _MIN_CLAUSE_SCORE:
            return AgentResult(
                agent_id=self.agent_id,
                output_text=None,
                handoff_to=AgentID.AGENTIC_RAG,
                tool_calls=["search_policy_wording"],
                meta={"reason": "no_confident_clause", "pid": pid},
            )

        clause = best["text"].strip()
        # Use the top relevant clauses as the grounding context for both the LLM
        # and the numeric guardrail.
        top_chunks = [c for c in chunks if c.get("score", 0) >= 1][:4] or [best]
        clauses_text = "\n\n".join(c["text"].strip() for c in top_chunks)

        det_answer = f"Here's what the {pid} policy wording says: {clause}"
        answer, source = det_answer, "deterministic"

        llm = ctx.llm
        if llm is not None and getattr(llm, "is_available", False):
            llm_answer = self._answer_llm(ctx, llm, pid, clauses_text, question)
            if not llm_answer:
                log_llm_call(self.agent_id, "fallback", "response_generator", False, "llm-empty")
            else:
                rep = self._guard.check(llm_answer, clauses_text)
                if rep.ok:
                    answer, source = llm_answer, "llm"
                    log_llm_call(self.agent_id, "llm", "response_generator", True, "grounded")
                else:
                    log_llm_call(self.agent_id, "fallback", "response_generator", False,
                                 f"guardrail {rep.offending}")

        report = self._guard.check(answer, clauses_text)

        return AgentResult(
            agent_id=self.agent_id,
            output_text=answer,
            tool_calls=["search_policy_wording", self._guard.agent_id],
            meta={
                "pid": pid,
                "clause_score": best["score"],
                "guardrail_ok": report.ok,
                "guardrail_offending": report.offending,
                "source": source,
                "_retrieved_context": clauses_text,
            },
        )

    # -- helpers ------------------------------------------------------------

    def _answer_llm(self, ctx: AgentContext, llm, pid: str, clauses_text: str,
                    question: str) -> str | None:
        """Compose a concise, clause-cited answer grounded only in `clauses_text`."""
        try:
            profile = json.dumps(ctx.record.schema.to_tool_input(), default=str)
        except Exception:
            profile = "{}"
        prompt = POLICY_QA_ANSWER_ENGINE_PROMPT.format(
            product_id=pid,
            user_profile_json=profile,
            retrieved_policy_clauses=clauses_text,
            user_query=question,
        )
        return llm.generate_response(prompt, question)


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord, FSMState

    agent = PolicyQAAgent()

    def ask(question, pid=None):
        rec = ConversationRecord.new(session_id="m08_test")
        if pid:
            rec.schema.set("resolved_product_id", pid)
        rec.state = FSMState.S3_POLICY_QA
        return agent.run(AgentContext(record=rec, message=question))

    # No resolved product → hand off to M_09.
    r = ask("what is the grace period?")
    assert r.handoff_to == AgentID.AGENTIC_RAG, r
    assert r.output_text is None

    # Resolved product + answerable question → grounded clause answer.
    r = ask("what is the waiting period for diabetes and insulin cover?", pid="SP015")
    assert r.output_text and r.output_text.startswith("Here's what the SP015"), r.output_text
    assert r.meta["guardrail_ok"], f"guardrail flagged: {r.meta.get('guardrail_offending')}"

    # Resolved product + irrelevant gibberish → hand off to M_09.
    r = ask("zxqw plfft brumble", pid="SP015")
    assert r.handoff_to == AgentID.AGENTIC_RAG, r

    print("policy_qa.py (M_08) self-test passed.")
