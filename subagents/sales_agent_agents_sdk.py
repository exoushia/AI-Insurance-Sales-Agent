"""
M_16 — SalesAgentAgentsSDK  (openai-agents Runner implementation)
=================================================================
Drop-in alternative to SalesAgent that replaces the hand-rolled
chat.completions tool loop with the openai-agents Runner, gaining:

  • Real OpenAI Traces visible at platform.openai.com/traces (opt-in via
    OPENAI_AGENTS_TRACE=1 / AppConfig.agents_sdk_tracing=True).
  • Native span nesting: each tool call is a child span of the agent run,
    with input/output captured automatically.
  • Trace correlation: the SDK's trace id is returned in AgentResult.meta
    so it can be cross-referenced with local openai_events.jsonl logs.

RUNTIME CONTRACT
  Same as SalesAgent: implements run(ctx: AgentContext) -> AgentResult.
  Same 7 tools, same schema mutations, same guardrail placement.
  Same fallback signal: AgentResult.meta["agentic_ok"] = False when the
  SDK is unavailable or errors — the orchestrator falls back to the FSM.

BACKEND SELECTION
  Enabled by AGENTIC_BACKEND=agents_sdk (AppConfig.agentic_backend).
  Default is still "legacy" (the hand-rolled loop) so rollout is gated.

IMPORTANT THREADING NOTE
  openai-agents is async-first. The SDK provides agents.Runner.run_sync()
  for synchronous callers (no running event loop). The Pipecat voice
  pipeline runs inside asyncio; for that path use Runner.run() (awaitable)
  via asyncio.create_task or an executor. The current implementation uses
  run_sync() because the text-mode orchestrator is synchronous. The voice
  path refactor is tracked as a forward task.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time
from typing import Any

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    from .base import AgentContext, AgentResult, AgentID
    from .guardrails import NumericGuardrail
except ImportError:
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
from llm_gateway import log_llm_call, log_openai_event

# Agents SDK — imported lazily so the rest of the codebase never hard-fails when
# openai-agents is not installed (legacy path still works).
try:
    import agents as _agents_sdk  # openai-agents package
    _SDK_AVAILABLE = True
except ImportError:
    _agents_sdk = None  # type: ignore
    _SDK_AVAILABLE = False

# ---------------------------------------------------------------------------
# TREATMENT-COST REFERENCE  (ported from the former legacy implementation)
# ---------------------------------------------------------------------------

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


def _load_treatment_costs() -> dict:
    try:
        with open(DATA_DIR / "treatment_costs.json", "r", encoding="utf-8") as f:
            text = f.read()
        brace = text.index("{")
        return ast.literal_eval(text[brace:])
    except (OSError, ValueError, SyntaxError):
        return {}


def _estimate_value_vs_cost(concern: str) -> dict:
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
# PROFILE FIELD ENUM (kept identical to the former legacy implementation)
# ---------------------------------------------------------------------------

_PROFILE_FIELDS = {
    "buyer_type": {
        "type": "string",
        "enum": ["individual", "employer_large", "employer_sme", "gig_worker"],
    },
    "age":          {"type": "integer"},
    "primary_need": {
        "type": "string",
        "enum": ["hospitalisation", "critical_illness", "cancer", "accident",
                 "maternity", "top_up", "international", "daily_cash"],
    },
    "has_ped":      {"type": "boolean"},
    "ped_type":     {"type": "string", "enum": ["diabetes_cardiac", "other_ped", "none"]},
    "needs_opd":    {"type": "boolean"},
    "budget_band":  {"type": "string", "enum": ["micro", "budget", "mid", "premium"]},
    "family_cover": {"type": "string", "enum": ["individual", "floater_nuclear", "floater_joint"]},
    "family_size":  {"type": "integer"},
    "si_preference":{"type": "string", "enum": ["1_2L", "3_5L", "10_25L", "50L_plus"]},
}


# ---------------------------------------------------------------------------
# CONTEXT CARRIER
# ---------------------------------------------------------------------------
# The Agents SDK tools are plain functions; they need access to the per-turn
# AgentContext (schema, record, etc.) without globals. We use a thin context
# carrier that is populated before each SDK run and read by the tool functions.

class _RunContext:
    """Mutable per-turn context injected into tool closures."""
    ctx: AgentContext | None = None
    tool_trace: list[dict]   = []


_RUN_CTX = _RunContext()


# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# The functions below match the dispatch logic in SalesAgent._dispatch exactly.
# Each is registered as a @function_tool on the SDK agent.
# ---------------------------------------------------------------------------

def _save_profile(**kwargs: Any) -> str:
    ctx = _RUN_CTX.ctx
    schema = ctx.record.schema
    applied, rejected = {}, {}
    for key, value in kwargs.items():
        if key not in _PROFILE_FIELDS or value is None:
            continue
        try:
            schema.set(key, value)
            applied[key] = value
        except Exception as exc:
            rejected[key] = str(exc)
    result = {"saved": applied, "rejected": rejected, "profile_now": schema.to_llm_context()}
    _RUN_CTX.tool_trace.append({"name": "save_profile", "args": kwargs, "result": result})
    return json.dumps(result, default=str)


def _recommend_products() -> str:
    ctx = _RUN_CTX.ctx
    user_schema = ctx.record.schema.to_tool_input()
    result = filter_products(user_schema)
    result.pop("_eliminated", None)
    ctx.record.retrieval_result = result
    _RUN_CTX.tool_trace.append({"name": "recommend_products", "args": {}, "result": result})
    return json.dumps(result, default=str)


def _explain_product(product_id: str, aspect: str = "") -> str:
    ctx = _RUN_CTX.ctx
    user_schema = ctx.record.schema.to_tool_input()
    out = get_product_features(product_id, user_schema)
    if "error" not in out:
        ctx.record.schema.set("resolved_product_id", product_id)
        ctx.record.fetched_features[product_id] = out.get("product", {})
    if aspect:
        wording = search_policy_wording(product_id, aspect)
        out["policy_wording"] = wording.get("chunks", [])
    _RUN_CTX.tool_trace.append({"name": "explain_product", "args": {"product_id": product_id, "aspect": aspect}, "result": out})
    return json.dumps(out, default=str)


def _show_plan_options(product_id: str) -> str:
    ctx = _RUN_CTX.ctx
    user_schema = ctx.record.schema.to_tool_input()
    out = get_plan_options(product_id, user_schema)
    if "error" not in out:
        ctx.record.schema.set("resolved_product_id", product_id)
        ctx.record.fetched_plans[product_id] = out
    _RUN_CTX.tool_trace.append({"name": "show_plan_options", "args": {"product_id": product_id}, "result": out})
    return json.dumps(out, default=str)


def _estimate_value(concern: str) -> str:
    result = _estimate_value_vs_cost(concern)
    _RUN_CTX.tool_trace.append({"name": "estimate_value_vs_cost", "args": {"concern": concern}, "result": result})
    return json.dumps(result, default=str)


def _answer_general_question(query: str) -> str:
    result = search_regulations(query)
    _RUN_CTX.tool_trace.append({"name": "answer_general_question", "args": {"query": query}, "result": result})
    return json.dumps(result, default=str)


def _finalize_purchase(product_id: str, plan_id: str) -> str:
    ctx = _RUN_CTX.ctx
    schema = ctx.record.schema
    if not product_id or not plan_id:
        result = {"ok": False, "reason": "need both product_id and plan_id"}
        _RUN_CTX.tool_trace.append({"name": "finalize_purchase", "args": {"product_id": product_id, "plan_id": plan_id}, "result": result})
        return json.dumps(result)
    # Reuse the same plan-id normalization logic as the former legacy path
    plan_id = _resolve_plan_id(product_id, plan_id, ctx)
    schema.set("resolved_product_id", product_id)
    schema.set("resolved_plan_id", plan_id)
    schema.set("purchased", True)
    result = {"ok": True, "product_id": product_id, "plan_id": plan_id, "message": "Purchase recorded."}
    _RUN_CTX.tool_trace.append({"name": "finalize_purchase", "args": {"product_id": product_id, "plan_id": plan_id}, "result": result})
    return json.dumps(result)


def _resolve_plan_id(pid: str, plan: str, ctx: AgentContext) -> str:
    """Identical to SalesAgent._resolve_plan_id — kept in sync manually."""
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
        suffix = (p.get("plan_id") or "").rsplit("-", 1)[-1]
        suffix_norm = _norm(suffix)
        if want in (pid_norm, label_norm, suffix_norm):
            return p.get("plan_id", plan)
        if suffix_norm and (suffix_norm in want or want in label_norm):
            return p.get("plan_id", plan)
    return plan


# ---------------------------------------------------------------------------
# SDK AGENT FACTORY
# Built once per process; recreated if tracing config changes.
# ---------------------------------------------------------------------------

def _build_sdk_agent(system_prompt: str):
    """Build and return an openai-agents Agent with all 7 function tools."""
    if not _SDK_AVAILABLE:
        return None

    FunctionTool = _agents_sdk.FunctionTool
    Agent        = _agents_sdk.Agent

    # Wrap each plain function as a FunctionTool with the matching description.
    # Docstrings are used by the SDK as tool descriptions.
    _save_profile.__doc__ = (
        "Persist what you've learned about the customer. Call whenever you pick up "
        "a new detail (age, who's covered, main need, budget, health condition) BEFORE "
        "recommending products. Send only the fields you are confident about."
    )
    _recommend_products.__doc__ = (
        "Rank the 20 Swasthya products for the current profile. Call once you know who "
        "is covered, their age, and their main need. If no_exact_match is true, recommend "
        "the closest fit and briefly acknowledge the trade-off; do not loop."
    )
    _explain_product.__doc__ = (
        "Get the full feature set for one product before describing it. Optionally pass "
        "an 'aspect' to also pull the exact policy wording for that topic."
    )
    _show_plan_options.__doc__ = (
        "Get the plan tiers, sums insured and annual premiums for a product. Call when "
        "the customer asks about price or coverage amounts, or is choosing a tier."
    )
    _estimate_value.__doc__ = (
        "Get REAL treatment-cost figures to put a premium in perspective when the customer "
        "hesitates on price. Pass a 'concern' such as 'delivery', 'heart', 'knee', 'cancer'."
    )
    _answer_general_question.__doc__ = (
        "Search IRDAI regulations for questions about rights, portability, free-look, tax, "
        "exclusions. Do NOT use for a specific product's features — use explain_product."
    )
    _finalize_purchase.__doc__ = (
        "Record the purchase. ONLY call after the customer has clearly agreed to buy AND "
        "a specific product and plan tier are confirmed."
    )

    tools = [
        FunctionTool(_save_profile),
        FunctionTool(_recommend_products),
        FunctionTool(_explain_product),
        FunctionTool(_show_plan_options),
        FunctionTool(_estimate_value),
        FunctionTool(_answer_general_question),
        FunctionTool(_finalize_purchase),
    ]

    return Agent(
        name="SwasthyaSalesAgent",
        instructions=system_prompt,
        tools=tools,
    )


# ---------------------------------------------------------------------------
# SUB-AGENT
# ---------------------------------------------------------------------------

class SalesAgentAgentsSDK:
    """M_16 — Agents SDK backend for the agentic LLM tool-calling sales agent.

    Produces real OpenAI Traces (when OPENAI_AGENTS_TRACE=1) while preserving
    the exact same AgentResult contract as the legacy SalesAgent.
    """

    agent_id = AgentID.SALES_AGENT
    MAX_HISTORY_TURNS = 16

    def __init__(self) -> None:
        self._guard = NumericGuardrail()
        self._sdk_agent = None   # built lazily on first run per system prompt

    def run(self, ctx: AgentContext) -> AgentResult:
        if not _SDK_AVAILABLE:
            return AgentResult(
                agent_id=self.agent_id,
                output_text="",
                meta={"agentic_ok": False, "reason": "openai-agents not installed"},
            )

        schema = ctx.record.schema
        tracing_enabled = getattr(ctx.config, "agents_sdk_tracing", False)

        # Configure SDK tracing per AppConfig.
        if tracing_enabled:
            _agents_sdk.enable_tracing()
        else:
            _agents_sdk.disable_tracing()

        system = AGENTIC_SALES_SYSTEM.format(
            profile_json=json.dumps(schema.to_llm_context(), indent=2, default=str)
        )

        # Rebuild the SDK agent only when the system prompt changes (it embeds the
        # live profile JSON). Since this is per-turn we always rebuild — cost is
        # negligible (no network call; just Python object construction).
        sdk_agent = _build_sdk_agent(system)
        if sdk_agent is None:
            return AgentResult(
                agent_id=self.agent_id,
                output_text="",
                meta={"agentic_ok": False, "reason": "sdk_agent_build_failed"},
            )

        # Populate the per-turn context carrier (thread-local-safe for sync use).
        _RUN_CTX.ctx = ctx
        _RUN_CTX.tool_trace = []

        # Build conversation input: history + new user message.
        history = self._history(ctx.record)
        # The SDK accepts a list of messages or a plain string.  We pass the full
        # thread so the agent has memory, mirroring the legacy path.
        input_messages = history + [{"role": "user", "content": ctx.message}]

        started = time.perf_counter()
        trace_id: str | None = None
        text: str | None = None
        sdk_error: str | None = None

        try:
            Runner = _agents_sdk.Runner
            # run_sync drives the async Runner synchronously — safe outside asyncio.
            run_result = Runner.run_sync(sdk_agent, input_messages)
            latency_ms = (time.perf_counter() - started) * 1000.0

            # Extract final text output.
            text = run_result.final_output
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            # Extract trace id from the SDK run result metadata if available.
            trace_id = getattr(run_result, "trace_id", None) or getattr(
                getattr(run_result, "metadata", None), "trace_id", None
            )

            log_llm_call(
                self.agent_id, "agents_sdk", "gpt-4.1-mini",
                bool(text), f"latency={latency_ms:.0f}ms trace_id={trace_id}",
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            sdk_error = str(exc)
            log_llm_call(
                self.agent_id, "agents_sdk", "gpt-4.1-mini",
                False, f"error={sdk_error}",
            )
            return AgentResult(
                agent_id=self.agent_id,
                output_text="",
                meta={
                    "agentic_ok": False,
                    "reason": "sdk_exception",
                    "error": sdk_error,
                    "tool_trace": _RUN_CTX.tool_trace,
                },
            )

        if not text:
            return AgentResult(
                agent_id=self.agent_id,
                output_text="",
                meta={
                    "agentic_ok": False,
                    "reason": "no_text",
                    "tool_trace": _RUN_CTX.tool_trace,
                },
            )

        tool_trace = list(_RUN_CTX.tool_trace)

        # M_15 — same guardrail placement as the legacy SalesAgent.
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
                "hops": len({t["name"] for t in tool_trace}),
                "guardrail_ok": report.ok,
                "guardrail_offending": report.offending,
                # Trace correlation: SDK trace id for cross-referencing dashboard.
                "sdk_trace_id": trace_id,
                "backend": "agents_sdk",
            },
        )

    def _history(self, record) -> list[dict]:
        """Same history reconstruction as the legacy SalesAgent."""
        turns: list[dict] = []
        for m in record.messages[-(self.MAX_HISTORY_TURNS * 2):]:
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                turns.append({"role": role, "content": content})
        return turns
