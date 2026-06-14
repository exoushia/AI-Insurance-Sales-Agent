"""
M_07 — PolicySummaryAgent  (formerly fsm.PolicySummaryAgent)
============================================================
Role     : turn retrieved product + plan data into a grounded recommendation
           speech. Deterministic template in Phase 0 (no LLM); every number in
           the speech is validated against the retrieved context by M_15.
Trigger  : handoff from M_06 (PolicyRetrievalAgent) in S2_RECOMMENDATION.
Tools    : get_product_features, get_plan_options, M_15 NumericGuardrail.
Output   :
  - output_text : the recommendation speech.
  - meta["products_presented"], meta["plan_presented"], meta["guardrail_ok"].
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

from retrieval_tools import get_product_features, get_plan_options
from prompts_template import POLICY_SUMMARY_SYSTEM, POLICY_SUMMARY_USER_TEMPLATE
from llm_gateway import log_llm_call

MAX_POLICIES_TO_PRESENT = 3
_CLEAR_WINNER_GAP = 0.25

_SUMMARY_EXCLUDED_PRODUCT_KEYS = {
    "_score_boosts", "uin", "product_id", "segment", "buyer_types",
    "primary_needs", "budget_bands", "medical_exam_required",
    "entry_age_min", "entry_age_max", "lifelong_renewal", "plans",
}


class PolicySummaryAgent:
    agent_id = AgentID.POLICY_SUMMARY

    def __init__(self):
        self._guard = NumericGuardrail()

    def run(self, ctx: AgentContext) -> AgentResult:
        schema = ctx.record.schema
        retrieval = ctx.retrieval_result or ctx.record.retrieval_result or {}
        candidates = retrieval.get("candidates", [])
        top = [c for c in candidates if c["score"] > 0.10][:MAX_POLICIES_TO_PRESENT]

        if not top:
            return AgentResult(
                agent_id=self.agent_id,
                output_text="I wasn't able to find a matching plan based on your details. "
                            "Let me connect you with our team to help further.",
                handoff_to=AgentID.ESCALATION,
                meta={"products_presented": [], "plan_presented": None},
            )

        # Single clear winner vs two-product comparison.
        present_single = (
            len(top) == 1
            or (len(top) >= 2 and (top[0]["score"] - top[1]["score"]) >= _CLEAR_WINNER_GAP)
        )
        to_present = [top[0]["product_id"]] if present_single else [c["product_id"] for c in top[:2]]

        tool_input = schema.to_tool_input()
        for pid in to_present:
            if pid not in ctx.feature_cache:
                ctx.feature_cache[pid] = get_product_features(pid, tool_input)
            if pid not in ctx.plan_cache:
                ctx.plan_cache[pid] = get_plan_options(pid, tool_input)

        context = self._build_context(to_present, ctx.feature_cache, ctx.plan_cache)
        det_speech = self._speech(to_present, ctx.feature_cache, ctx.plan_cache)

        # LLM writes the speech grounded ONLY in `context`; M_15 re-checks every
        # number. If the LLM hallucinates a figure (guardrail fail) or is
        # unavailable, fall back to the deterministic speech (grounded by build).
        speech, source = det_speech, "deterministic"
        llm = ctx.llm
        if llm is not None and getattr(llm, "is_available", False):
            llm_speech = self._compose_llm(ctx, llm, context, to_present)
            if not llm_speech:
                log_llm_call(self.agent_id, "fallback", "response_generator", False, "llm-empty")
            else:
                rep = self._guard.check(llm_speech, context)
                if rep.ok:
                    speech, source = llm_speech, "llm"
                    log_llm_call(self.agent_id, "llm", "response_generator", True, "grounded")
                else:
                    log_llm_call(self.agent_id, "fallback", "response_generator", False,
                                 f"guardrail {rep.offending}")

        report = self._guard.check(speech, context)

        plan_presented = ctx.plan_cache.get(to_present[0], {}).get("recommended_plan")

        return AgentResult(
            agent_id=self.agent_id,
            output_text=speech,
            tool_calls=["get_product_features", "get_plan_options", self._guard.agent_id],
            meta={
                "products_presented": to_present,
                "plan_presented": plan_presented,
                "guardrail_ok": report.ok,
                "guardrail_offending": report.offending,
                "source": source,
                "_retrieved_context": context,
            },
        )

    # -- helpers ------------------------------------------------------------

    def _compose_llm(self, ctx: AgentContext, llm, context: str, product_ids) -> str | None:
        """Ask the LLM to write the recommendation from `context` only."""
        recommended = ", ".join(
            f"{pid}: {ctx.plan_cache.get(pid, {}).get('recommended_plan') or 'n/a'}"
            for pid in product_ids
        )
        return llm.generate_response(
            POLICY_SUMMARY_SYSTEM,
            POLICY_SUMMARY_USER_TEMPLATE.format(recommended=recommended, context=context),
        )

    def _speech(self, product_ids, feats, plans) -> str:
        parts = []
        for pid in product_ids:
            feat = feats.get(pid, {})
            plan = plans.get(pid, {})
            product = feat.get("product", {})
            highlights = feat.get("highlight_fields", [])
            name = product.get("name", pid)

            parts.append(f"I'd like to tell you about {name}.")
            for h in highlights:
                parts.append(f"{h['label']} is {h['value']} — {h['why']}")

            rec_id = plan.get("recommended_plan")
            rec = next((p for p in plan.get("plans", []) if p["plan_id"] == rec_id), None)
            if rec:
                tail = f" with a sum insured of ₹{rec['si_inr']:,}." if rec.get("si_inr") else "."
                parts.append(
                    f"The recommended plan is {rec['label']} "
                    f"at ₹{rec['annual_premium_inr']:,} per year{tail}"
                )
        parts.append("Would you like to know more, or shall we go ahead?")
        return " ".join(parts)

    def _build_context(self, product_ids, feats, plans) -> str:
        """Single source of truth for the guardrail — every number the speech may use."""
        blocks = []
        for pid in product_ids:
            feat = feats.get(pid, {})
            plan = plans.get(pid, {})
            product = feat.get("product", {})
            highlights = feat.get("highlight_fields", [])

            blocks.append(f"=== {product.get('name', pid)} ({pid}) ===")
            blocks.append(f"WHY: {feat.get('why_this_product', '')}")
            for h in highlights:
                blocks.append(f"  {h['label']}: {h['value']} — {h['why']}")

            facts = {
                k: v for k, v in product.items()
                if k not in _SUMMARY_EXCLUDED_PRODUCT_KEYS
                and v not in (None, False, [], 0)
            }
            for k, v in list(facts.items())[:20]:
                blocks.append(f"  {k}: {v}")

            for p in plan.get("plans", [])[:4]:
                si = f"SI ₹{p['si_inr']:,}" if p.get("si_inr") else ""
                opd = f" OPD ₹{p['opd_limit_inr']:,}" if p.get("opd_limit_inr") else ""
                mat = f" delivery ₹{p['maternity_normal_inr']:,}" if p.get("maternity_normal_inr") else ""
                blocks.append(
                    f"  {p['plan_id']}: {p['label']} — {si} — "
                    f"₹{p['annual_premium_inr']:,}/yr{opd}{mat}"
                )
            blocks.append("")
        return "\n".join(blocks)


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord
    from retrieval_tools import filter_products

    agent = PolicySummaryAgent()

    def summarise(preset):
        rec = ConversationRecord.new(session_id="m07_test")
        for k, v in preset.items():
            rec.schema.set(k, v)
        retrieval = filter_products(rec.schema.to_tool_input())
        ctx = AgentContext(record=rec, retrieval_result=retrieval)
        return agent.run(ctx)

    # Maternity profile → SP007 single recommendation, numbers grounded.
    r = summarise({"buyer_type": "individual", "age": 30, "gender": "female",
                   "primary_need": "maternity"})
    assert r.output_text and "recommended plan" in r.output_text.lower(), r.output_text
    assert r.meta["products_presented"], r.meta
    assert r.meta["guardrail_ok"], f"guardrail flagged: {r.meta['guardrail_offending']}"

    # Broad hospitalisation → up to two products, still grounded.
    r2 = summarise({"buyer_type": "individual", "age": 30, "gender": "male",
                    "primary_need": "hospitalisation"})
    assert 1 <= len(r2.meta["products_presented"]) <= 2, r2.meta
    assert r2.meta["guardrail_ok"], f"guardrail flagged: {r2.meta['guardrail_offending']}"

    print("policy_summary.py (M_07) self-test passed.")
    print("  presented:", r.meta["products_presented"], "| plan:", r.meta["plan_presented"])
