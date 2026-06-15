"""
M_16 — SalesAgent  (agentic, native OpenAI tool-calling)
========================================================
The agentic orchestrator's brain. Where the deterministic FSM (S0-S6) walks the
customer through fixed states, M_16 lets the LLM drive the whole conversation by
calling tools. It reuses the SAME retrieval implementations the FSM path uses
(filter_products / get_product_features / get_plan_options / search_regulations
in retrieval_tools.py) plus two agentic-only tools (save_profile and
estimate_value_vs_cost) — so there is ONE source of truth for product facts.

Why we moved here (see fsm.py for the deterministic version we evolved from):
  - The FSM produced correct but rigid, "robotic" turns and mis-routed open
    follow-ups ("explain that more") into the wrong state.
  - Native tool-calling gives the model freedom to sequence discovery →
    recommendation → plan → close naturally, while every fact still comes from a
    tool (never the model's memory) and every number is re-checked by M_15.
  - Using the OpenAI SDK's tools API means the whole decision trace is visible in
    standard request/response logs for observability.

Guardrails retained from the FSM path:
  - M_15 NumericGuardrail re-checks every number in the reply against the tool
    outputs the model was actually given.
  - finalize_purchase only records a sale when a product AND a plan are confirmed.
  - The orchestrator (agentic_orchestrator.py) falls back to the FSM path if the
    LLM/tool loop is unavailable or errors.

Contract: implements the standard `run(ctx: AgentContext) -> AgentResult`.
"""

from __future__ import annotations

import ast
import json
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

from retrieval_tools import (
    filter_products,
    get_product_features,
    get_plan_options,
    search_regulations,
    search_policy_wording,
)
from prompts_template import AGENTIC_SALES_SYSTEM, AGENTIC_VALUE_FRAME_NOTE, DATA_DIR
from llm_gateway import log_llm_call


# ---------------------------------------------------------------------------
# OPENAI TOOL SCHEMAS  (native function-calling format)
# ---------------------------------------------------------------------------
# These mirror the retrieval implementations but DELIBERATELY do not ask the
# model to pass the user_schema back in — the dispatch injects the live schema
# from the conversation record. That keeps the model focused on the dialogue and
# removes a whole class of "stale schema" hallucinations.

_PROFILE_FIELDS = {
    "buyer_type": {
        "type": "string",
        "enum": ["individual", "employer_large", "employer_sme", "gig_worker"],
        "description": "Who is buying. Use 'individual' for a person buying for "
                       "THEMSELVES OR THEIR FAMILY (including spouse/kids/parents — "
                       "a family floater is still 'individual'). Only use an "
                       "employer_* type when a company is buying for its employees, "
                       "or 'gig_worker' for a self-employed daily-wage/cab/delivery worker.",
    },
    "age": {"type": "integer", "description": "Primary insured's age in years."},
    "primary_need": {
        "type": "string",
        "enum": ["hospitalisation", "critical_illness", "cancer", "accident",
                 "maternity", "top_up", "international", "daily_cash"],
        "description": "The single main thing the customer wants cover for. "
                       "Map their words to the closest value (e.g. 'planning a baby' "
                       "→ maternity, 'heart/stroke cover' → critical_illness, "
                       "'normal hospital cover' → hospitalisation).",
    },
    "has_ped": {"type": "boolean", "description": "Has a pre-existing condition."},
    "ped_type": {
        "type": "string",
        "enum": ["diabetes_cardiac", "other_ped", "none"],
        "description": "Pre-existing condition category. 'diabetes_cardiac' covers "
                       "diabetes, heart disease, hypertension; points strongly to SP015.",
    },
    "needs_opd": {"type": "boolean", "description": "Wants outpatient (OPD) cover."},
    "budget_band": {
        "type": "string",
        "enum": ["micro", "budget", "mid", "premium"],
        "description": "Annual premium comfort: micro (<2k), budget (2k-10k), "
                       "mid (10k-30k), premium (>30k).",
    },
    "family_cover": {
        "type": "string",
        "enum": ["individual", "floater_nuclear", "floater_joint"],
        "description": "Who is on the policy: individual, a nuclear-family floater, "
                       "or a joint-family floater.",
    },
    "family_size": {"type": "integer", "description": "Number of people to cover."},
    "si_preference": {
        "type": "string",
        "enum": ["1_2L", "3_5L", "10_25L", "50L_plus"],
        "description": "Preferred sum insured band.",
    },
}


def tool_schema_openai() -> list[dict]:
    """The agentic tool set in OpenAI chat-completions function format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "save_profile",
                "description": (
                    "Persist what you've learned about the customer. Call this whenever "
                    "you pick up a new detail (age, who's covered, main need, budget, "
                    "health condition) BEFORE recommending products. Send only the "
                    "fields you are confident about; omit unknown ones."
                ),
                "parameters": {
                    "type": "object",
                    "properties": _PROFILE_FIELDS,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_products",
                "description": (
                    "Rank the 20 Swasthya products for the current customer profile. "
                    "Call once you know who's covered, their age, and their main need. "
                    "If the result includes a probe_question, ask it before describing "
                    "any product. If no_exact_match is true, no product fits every "
                    "preference — recommend the top candidate as the closest fit and "
                    "briefly acknowledge the trade-off (relaxed_constraints says what "
                    "couldn't be met), then keep moving toward plans; do not loop. "
                    "Returns ranked candidates with product_id and score."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain_product",
                "description": (
                    "Get the full, authoritative feature set for ONE product before you "
                    "describe it or answer a detailed question about it. Optionally pass "
                    "an 'aspect' (e.g. 'maternity waiting period', 'pre-existing "
                    "conditions') to also pull the exact policy wording for that topic."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "Product code from recommend_products, e.g. 'SP015'.",
                        },
                        "aspect": {
                            "type": "string",
                            "description": "Optional specific topic to pull policy wording for.",
                        },
                    },
                    "required": ["product_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "show_plan_options",
                "description": (
                    "Get the plan tiers, sums insured and annual premiums for a product, "
                    "ranked by fit to the customer. Call when the customer asks about "
                    "price or coverage amounts, or is choosing a tier."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "Product code, e.g. 'SP002'.",
                        },
                    },
                    "required": ["product_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_value_vs_cost",
                "description": (
                    "Get REAL treatment-cost figures to put a premium in perspective when "
                    "the customer hesitates on price. Pass a 'concern' describing the "
                    "relevant event (e.g. 'delivery', 'heart', 'knee', 'cancer', "
                    "'diabetes', 'hospital stay'). Returns out-of-pocket cost ranges you "
                    "can contrast with the annual premium."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concern": {
                            "type": "string",
                            "description": "The treatment/event the customer is worried about.",
                        },
                    },
                    "required": ["concern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "answer_general_question",
                "description": (
                    "Search IRDAI regulations and general health-insurance knowledge for "
                    "questions about rights, portability, free-look, tax, grievance/"
                    "ombudsman, standard exclusions, or 'how does health insurance work'. "
                    "Do NOT use for a specific product's features — use explain_product."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The customer's question in natural language.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finalize_purchase",
                "description": (
                    "Record the purchase. ONLY call after the customer has clearly agreed "
                    "to buy AND a specific product and plan tier are confirmed. Confirm "
                    "the product and plan back to the customer first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "plan_id": {
                            "type": "string",
                            "description": "The chosen plan tier id from show_plan_options.",
                        },
                    },
                    "required": ["product_id", "plan_id"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# TREATMENT-COST REFERENCE  (for estimate_value_vs_cost)
# ---------------------------------------------------------------------------

def _load_treatment_costs() -> dict:
    """The file is `TREATMENT_COSTS = {...}` — a Python literal, not JSON."""
    try:
        with open(DATA_DIR / "treatment_costs.json", "r", encoding="utf-8") as f:
            text = f.read()
        brace = text.index("{")
        return ast.literal_eval(text[brace:])
    except (OSError, ValueError, SyntaxError):
        return {}


# Map loose customer concern words to the cost-table leaves most worth quoting.
_CONCERN_KEYWORDS = {
    "delivery":     [("common_procedures", "normal_delivery"),
                     ("common_procedures", "c_section_delivery")],
    "maternity":    [("common_procedures", "c_section_delivery"),
                     ("common_procedures", "normal_delivery")],
    "pregnan":      [("common_procedures", "c_section_delivery"),
                     ("common_procedures", "normal_delivery")],
    "heart":        [("common_procedures", "angioplasty_single_stent"),
                     ("common_procedures", "bypass_surgery_cabg")],
    "cardiac":      [("common_procedures", "angioplasty_single_stent"),
                     ("common_procedures", "bypass_surgery_cabg")],
    "bypass":       [("common_procedures", "bypass_surgery_cabg")],
    "knee":         [("common_procedures", "knee_replacement_single")],
    "joint":        [("common_procedures", "knee_replacement_single")],
    "cancer":       [("common_procedures", "cancer_chemotherapy_per_cycle")],
    "chemo":        [("common_procedures", "cancer_chemotherapy_per_cycle")],
    "dialysis":     [("common_procedures", "dialysis_per_session")],
    "kidney":       [("common_procedures", "dialysis_per_session")],
    "cataract":     [("common_procedures", "cataract_surgery_per_eye")],
    "eye":          [("common_procedures", "cataract_surgery_per_eye")],
    "appendi":      [("common_procedures", "appendectomy")],
    "covid":        [("common_procedures", "covid_moderate_10days")],
    "diabet":       [("annual_chronic_management", "diabetes_type2")],
    "hypertens":    [("annual_chronic_management", "hypertension")],
    "bp":           [("annual_chronic_management", "hypertension")],
    "asthma":       [("annual_chronic_management", "asthma")],
    "icu":          [("hospitalization_per_day", "private_icu")],
}


def _estimate_value_vs_cost(concern: str) -> dict:
    """Return real treatment-cost figures relevant to the stated concern."""
    costs = _load_treatment_costs()
    if not costs:
        return {"error": "treatment cost reference unavailable"}

    concern_l = (concern or "").lower()
    selected: list[dict] = []
    for keyword, leaves in _CONCERN_KEYWORDS.items():
        if keyword in concern_l:
            for category, leaf in leaves:
                value = costs.get(category, {}).get(leaf)
                if value is not None:
                    selected.append({
                        "item": leaf.replace("_", " "),
                        "category": category.replace("_", " "),
                        "cost_inr": value,
                    })
            break

    # Default: a private hospital stay + ICU/day, the universal value anchor.
    if not selected:
        hosp = costs.get("hospitalization_per_day", {})
        for leaf in ("private_single_ac", "private_icu"):
            if leaf in hosp:
                selected.append({
                    "item": f"{leaf.replace('_', ' ')} (per day)",
                    "category": "hospitalization per day",
                    "cost_inr": hosp[leaf],
                })

    return {
        "concern": concern,
        "out_of_pocket_costs": selected,
        "note": AGENTIC_VALUE_FRAME_NOTE,
    }


# ---------------------------------------------------------------------------
# SUB-AGENT
# ---------------------------------------------------------------------------

class SalesAgent:
    """M_16 — the agentic LLM tool-calling sales agent."""

    agent_id = AgentID.SALES_AGENT
    MAX_HISTORY_TURNS = 16   # plain user/assistant turns kept for context

    def __init__(self) -> None:
        self._guard = NumericGuardrail()

    def run(self, ctx: AgentContext) -> AgentResult:
        llm = ctx.llm
        if llm is None or not getattr(llm, "is_available", False):
            # No LLM → signal the orchestrator to use the deterministic path.
            return AgentResult(
                agent_id=self.agent_id,
                output_text="",
                meta={"agentic_ok": False, "reason": "llm_unavailable"},
            )

        schema = ctx.record.schema
        system = AGENTIC_SALES_SYSTEM.format(
            profile_json=json.dumps(schema.to_llm_context(), indent=2, default=str)
        )

        messages: list[dict] = [{"role": "system", "content": system}]
        messages.extend(self._history(ctx.record))
        messages.append({"role": "user", "content": ctx.message})

        result = llm.generate_with_tools(
            messages,
            tool_schema_openai(),
            lambda name, args: self._dispatch(name, args, ctx),
            agent_id=self.agent_id,
            obs_context={
                "session_id": schema.session_id,
                "turn": schema.turn_count,
                "state": ctx.record.state.value,
                "orchestration_mode": getattr(ctx.config, "orchestration_mode", "unknown"),
            },
        )

        if not result or not result.get("text"):
            return AgentResult(
                agent_id=self.agent_id,
                output_text="",
                meta={"agentic_ok": False, "reason": "no_text",
                      "tool_trace": (result or {}).get("tool_trace", [])},
            )

        text = result["text"]
        tool_trace = result.get("tool_trace", [])

        # M_15 — re-check every number in the reply against the tool outputs the
        # model actually received. If a number is unsupported, flag it (the
        # orchestrator decides whether to fall back).
        context_text = json.dumps(tool_trace, default=str)
        report = self._guard.check(text, context_text)
        log_llm_call(
            self.agent_id, "guardrail", "m15",
            report.ok, f"offending={report.offending}" if not report.ok else "",
        )

        return AgentResult(
            agent_id=self.agent_id,
            output_text=text,
            meta={
                "agentic_ok": True,
                "tool_trace": tool_trace,
                "hops": result.get("hops"),
                "guardrail_ok": report.ok,
                "guardrail_offending": report.offending,
            },
        )

    # ── history -------------------------------------------------------------
    def _history(self, record) -> list[dict]:
        """Reconstruct plain user/assistant turns (no raw tool exchanges) so the
        model has conversational memory without prompt bloat."""
        turns: list[dict] = []
        for m in record.messages[-(self.MAX_HISTORY_TURNS * 2):]:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                turns.append({"role": role, "content": content})
        return turns

    # ── tool dispatch -------------------------------------------------------
    def _dispatch(self, name: str, args: dict, ctx: AgentContext) -> dict:
        """Execute one tool call. Reuses retrieval_tools implementations and
        injects the LIVE schema so the model never has to round-trip it."""
        schema = ctx.record.schema
        user_schema = schema.to_tool_input()

        if name == "save_profile":
            return self._save_profile(args, ctx)

        if name == "recommend_products":
            result = filter_products(user_schema)
            result.pop("_eliminated", None)
            ctx.record.retrieval_result = result
            return result

        if name == "explain_product":
            pid = args.get("product_id", "")
            out = get_product_features(pid, user_schema)
            if "error" not in out:
                schema.set("resolved_product_id", pid)
                ctx.record.fetched_features[pid] = out.get("product", {})
            aspect = args.get("aspect")
            if aspect:
                wording = search_policy_wording(pid, aspect)
                out["policy_wording"] = wording.get("chunks", [])
            return out

        if name == "show_plan_options":
            pid = args.get("product_id", "")
            out = get_plan_options(pid, user_schema)
            if "error" not in out:
                schema.set("resolved_product_id", pid)
                ctx.record.fetched_plans[pid] = out
            return out

        if name == "estimate_value_vs_cost":
            return _estimate_value_vs_cost(args.get("concern", ""))

        if name == "answer_general_question":
            return search_regulations(args.get("query", ""))

        if name == "finalize_purchase":
            return self._finalize(args, ctx)

        return {"error": f"unknown tool {name!r}"}

    def _save_profile(self, args: dict, ctx: AgentContext) -> dict:
        schema = ctx.record.schema
        applied, rejected = {}, {}
        for key, value in (args or {}).items():
            if key not in _PROFILE_FIELDS or value is None:
                continue
            try:
                schema.set(key, value)
                applied[key] = value
            except Exception as exc:  # validation rejected the value
                rejected[key] = str(exc)
        return {
            "saved": applied,
            "rejected": rejected,
            "profile_now": schema.to_llm_context(),
        }

    def _finalize(self, args: dict, ctx: AgentContext) -> dict:
        schema = ctx.record.schema
        pid = args.get("product_id")
        plan = args.get("plan_id")
        if not pid or not plan:
            return {"ok": False, "reason": "need both product_id and plan_id to finalise"}
        plan = self._resolve_plan_id(pid, plan, ctx)
        schema.set("resolved_product_id", pid)
        schema.set("resolved_plan_id", plan)
        schema.set("purchased", True)
        return {"ok": True, "product_id": pid, "plan_id": plan,
                "message": "Purchase recorded."}

    @staticmethod
    def _resolve_plan_id(pid: str, plan: str, ctx: AgentContext) -> str:
        """Map a loose plan reference back to its real tier id using the plans
        cached from show_plan_options. The model may pass the exact id
        ('SP015-Silver'), a display label ('HDC Silver 3L' / 'HDC_Silver_3L'),
        or just the tier word ('silver', 'Silver tier'). Matching ignores case,
        spaces and punctuation. If nothing resolves, the input is returned
        unchanged so the sale is still recorded."""
        cached = ctx.record.fetched_plans.get(pid)
        if not cached:
            return plan
        plans = cached.get("plans", [])
        valid_ids = {p.get("plan_id") for p in plans}
        if plan in valid_ids:
            return plan

        def _norm(s: str) -> str:
            return "".join(ch for ch in (s or "").lower() if ch.isalnum())

        want = _norm(plan)
        if not want:
            return plan
        for p in plans:
            pid_norm = _norm(p.get("plan_id"))
            label_norm = _norm(p.get("label"))
            # tier suffix, e.g. 'SP015-Silver' -> 'silver'
            suffix = (p.get("plan_id") or "").rsplit("-", 1)[-1]
            suffix_norm = _norm(suffix)
            if want in (pid_norm, label_norm, suffix_norm):
                return p.get("plan_id", plan)
            # loose: 'silver tier' contains 'silver'; 'hdcsilver3l' contains want
            if suffix_norm and (suffix_norm in want or want in label_norm):
                return p.get("plan_id", plan)
        return plan


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Offline schema-free checks (no network): tool schema shape + value tool.
    tools = tool_schema_openai()
    names = [t["function"]["name"] for t in tools]
    assert names == [
        "save_profile", "recommend_products", "explain_product",
        "show_plan_options", "estimate_value_vs_cost",
        "answer_general_question", "finalize_purchase",
    ], names
    print("tool schema OK:", names)

    v = _estimate_value_vs_cost("worried about delivery costs")
    assert v["out_of_pocket_costs"], v
    print("value tool (delivery):", v["out_of_pocket_costs"])

    v2 = _estimate_value_vs_cost("my knee hurts")
    print("value tool (knee):", v2["out_of_pocket_costs"])

    v3 = _estimate_value_vs_cost("something unmapped")
    print("value tool (default hospital):", v3["out_of_pocket_costs"])
    print("ALL SALES_AGENT SELF-TESTS PASSED")
