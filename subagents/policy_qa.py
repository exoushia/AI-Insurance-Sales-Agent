"""
M_08 — PolicyQAAgent
====================
Role     : entry agent for S3_POLICY_QA.
Trigger  : state == S3_POLICY_QA.
Behaviour:
  TEMP — the keyword search on product wording is disabled. M_08 now passes
  every policy question straight to M_09 (AgenticRAG), which searches the full
  semantic vector index (policy wording + IRDAI regulations + treatment costs)
  with a product-filtered merge. This avoids answering regulation questions from
  narrow keyword-matched product clauses.
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
        # TEMP: the keyword search on product wording is disabled for now.
        # It was intercepting questions (including IRDAI regulation queries) and
        # answering from the narrow, keyword-matched product clauses instead of
        # the semantic vector index. Route every policy question straight to
        # M_09 (AgenticRAG), which searches the full corpus — policy wording +
        # IRDAI regulations + treatment costs — with a product-filtered merge.
        pid = ctx.record.schema.resolved_product_id
        return AgentResult(
            agent_id=self.agent_id,
            output_text=None,
            handoff_to=AgentID.AGENTIC_RAG,
            meta={"reason": "keyword_search_disabled", "pid": pid},
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

    # Keyword search disabled → every question hands off to M_09, no answer here.
    r = ask("what is the grace period?")
    assert r.handoff_to == AgentID.AGENTIC_RAG, r
    assert r.output_text is None

    # Even with a resolved product, M_08 passes through to M_09.
    r = ask("what is the waiting period for diabetes and insulin cover?", pid="SP015")
    assert r.handoff_to == AgentID.AGENTIC_RAG, r
    assert r.output_text is None

    r = ask("zxqw plfft brumble", pid="SP015")
    assert r.handoff_to == AgentID.AGENTIC_RAG, r

    print("policy_qa.py (M_08) self-test passed.")
