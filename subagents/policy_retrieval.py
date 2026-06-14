"""
M_06 — PolicyRetrievalAgent
===========================
Role     : run the deterministic product filter and decide whether a single
           product is a clear winner (auto-resolve) or several should be
           presented for comparison.
Trigger  : on entry to S2_RECOMMENDATION (schema is sufficient_for_retrieval).
Tools    : filter_products(user_schema).
Output   :
  - schema_updates: {resolved_product_id, retrieval_score} when a clear winner
    exists (top score alone, or beating #2 by >= _CLEAR_WINNER_GAP).
  - meta["retrieval_result"]   : raw filter_products dict (orchestrator caches it).
  - meta["top_candidates"]     : up to 3 candidate dicts for M_07 to summarise.
  - handoff_to = M_07 (PolicySummaryAgent) in every case.
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

from retrieval_tools import filter_products

_SCORE_FLOOR = 0.10          # candidates at or below this are not viable
_CLEAR_WINNER_GAP = 0.25     # top must beat #2 by this to auto-resolve
_MAX_TO_PRESENT = 3


class PolicyRetrievalAgent:
    agent_id = AgentID.POLICY_RETRIEVAL

    def run(self, ctx: AgentContext) -> AgentResult:
        schema = ctx.record.schema
        result = filter_products(schema.to_tool_input())

        viable = [c for c in result.get("candidates", []) if c["score"] > _SCORE_FLOOR]
        top = viable[:_MAX_TO_PRESENT]

        updates: dict = {}
        resolved = None
        if len(viable) == 1:
            resolved = viable[0]
        elif len(viable) >= 2 and (viable[0]["score"] - viable[1]["score"]) >= _CLEAR_WINNER_GAP:
            resolved = viable[0]

        if resolved is not None:
            updates["resolved_product_id"] = resolved["product_id"]
            updates["retrieval_score"] = resolved["score"]

        return AgentResult(
            agent_id=self.agent_id,
            output_text=None,
            schema_updates=updates,
            handoff_to=AgentID.POLICY_SUMMARY,
            tool_calls=["filter_products"],
            meta={
                "retrieval_result": result,
                "top_candidates": top,
                "resolved": resolved["product_id"] if resolved else None,
                "probe_question": result.get("probe_question"),
            },
        )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord

    agent = PolicyRetrievalAgent()

    def retrieve(preset):
        rec = ConversationRecord.new(session_id="m06_test")
        for k, v in preset.items():
            rec.schema.set(k, v)
        return agent.run(AgentContext(record=rec))

    # Strongly-pointed need (maternity -> SP007) should auto-resolve.
    res = retrieve({"buyer_type": "individual", "age": 30, "gender": "female",
                    "primary_need": "maternity"})
    assert res.handoff_to == AgentID.POLICY_SUMMARY, res
    assert res.meta["retrieval_result"]["candidates"], "expected candidates"
    assert res.meta["resolved"] == "SP007", res.meta
    assert res.schema_updates.get("resolved_product_id") == "SP007", res.schema_updates
    assert "retrieval_score" in res.schema_updates

    # Broad need (hospitalisation) -> multiple candidates, present up to 3.
    res2 = retrieve({"buyer_type": "individual", "age": 30, "gender": "male",
                     "primary_need": "hospitalisation"})
    assert res2.handoff_to == AgentID.POLICY_SUMMARY
    assert len(res2.meta["top_candidates"]) <= 3
    assert res2.tool_calls == ["filter_products"]

    print("policy_retrieval.py (M_06) self-test passed.")
    print("  maternity  -> resolved:", res.meta["resolved"])
    print("  hospitalisation top:", [c["product_id"] for c in res2.meta["top_candidates"]])
