"""
Sub-agent harness — shared contract
===================================
Every sub-agent in the Swasthya orchestration pipeline implements ONE uniform
interface so the orchestrator can route to, log, and validate them identically.

    result = agent.run(ctx)          # ctx: AgentContext  → result: AgentResult

Phase 0 is fully deterministic: no LLM/network calls. Each agent ships a
deterministic stub (regex/keyword extraction, template phrasing, rule-based
validation, identity translation) so the whole pipeline runs offline and the
wiring can be proven end-to-end.

Design rules
------------
- An agent NEVER mutates the ConversationRecord/UserSchema directly. It returns
  `schema_updates` and `next_state_hint`; the orchestrator applies them. This
  keeps state ownership with the orchestrator (deterministic, auditable).
- An agent may call retrieval tools (read-only) and record them in `tool_calls`.
- An agent may request a handoff to another agent via `handoff_to` (an M_ id);
  the orchestrator executes the handoff chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Imported lazily / only for typing to avoid heavy import cycles at runtime.
    from fsm import ConversationRecord, FSMState, IntentSignal
    from settings import AppConfig
    from llm_gateway import LLMGateway


# ---------------------------------------------------------------------------
# AGENT IDENTIFIERS  (single source of truth — mirrors states/intents scheme)
# ---------------------------------------------------------------------------

class AgentID:
    """Stable string ids for every sub-agent (M_<nn>)."""
    INTENT_CLASSIFIER  = "M_01"
    ESCALATION         = "M_02"
    CLOSURE            = "M_03"
    SCHEMA_EXTRACTOR   = "M_04"
    PROBING            = "M_05"
    POLICY_RETRIEVAL   = "M_06"
    POLICY_SUMMARY     = "M_07"
    POLICY_QA          = "M_08"
    AGENTIC_RAG        = "M_09"
    TRANSLATOR         = "M_10"
    RESPONSE_QUEUE     = "M_11"
    RESPONSE_VALIDATOR = "M_12"
    ANALYTICS_LOGGER   = "M_13"
    WHATSAPP           = "M_14"
    NUMERIC_GUARDRAIL  = "M_15"


# Human-readable names — used in logs and the analytics record.
AGENT_NAMES: dict[str, str] = {
    AgentID.INTENT_CLASSIFIER:  "IntentClassifier",
    AgentID.ESCALATION:         "EscalationAgent",
    AgentID.CLOSURE:            "ClosureAgent",
    AgentID.SCHEMA_EXTRACTOR:   "SchemaExtractor",
    AgentID.PROBING:            "ProbingAgent",
    AgentID.POLICY_RETRIEVAL:   "PolicyRetrievalAgent",
    AgentID.POLICY_SUMMARY:     "PolicySummaryAgent",
    AgentID.POLICY_QA:          "PolicyQAAgent",
    AgentID.AGENTIC_RAG:        "AgenticRAGAgent",
    AgentID.TRANSLATOR:         "TranslatorAgent",
    AgentID.RESPONSE_QUEUE:     "ResponseQueue",
    AgentID.RESPONSE_VALIDATOR: "ResponseValidator",
    AgentID.ANALYTICS_LOGGER:   "AnalyticsLogger",
    AgentID.WHATSAPP:           "WhatsAppAgent",
    AgentID.NUMERIC_GUARDRAIL:  "NumericGuardrail",
}


# ---------------------------------------------------------------------------
# AGENT CONTEXT  (input to every agent)
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    """
    Everything an agent may need for one turn. The orchestrator builds this once
    per turn and passes it (possibly enriched) to each agent it routes to.

    Agents READ from this; they do not mutate `record` directly (except the
    AnalyticsLogger, which is explicitly an I/O sink). All state changes flow
    back to the orchestrator via AgentResult.
    """
    record:           "ConversationRecord"
    message:          str = ""                      # latest user message (raw)
    intent:           "IntentSignal | None" = None  # set after M_01 runs
    confidence:       float | None = None           # intent confidence from M_01

    # Retrieval working set (shared across S2/S3 agents within a turn).
    retrieval_result: dict | None = None            # last filter_products() output
    feature_cache:    dict[str, dict] = field(default_factory=dict)  # pid -> features
    plan_cache:       dict[str, dict] = field(default_factory=dict)  # pid -> plans

    # Free-form payload for agent-to-agent handoffs (e.g. product_ids M_06→M_07).
    payload:          dict[str, Any] = field(default_factory=dict)

    # Infra handles (unused in Phase 0 — present so signatures are stable).
    config:           "AppConfig | None" = None
    llm:              "LLMGateway | None" = None
    sarvam:           "Any | None" = None           # SarvamGateway (M_10 translation)


# ---------------------------------------------------------------------------
# AGENT RESULT  (output of every agent)
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """
    Uniform return type. Only `agent_id` is required; everything else is optional
    so simple agents stay terse.

    Fields
    ------
    agent_id        : the M_ id of the agent that produced this result.
    output_text     : user-facing text this agent contributes (None if it produces
                      no speech, e.g. M_04/M_06/M_13).
    schema_updates  : {field_key: value} the orchestrator should apply via
                      schema.set(). Validation happens there.
    next_state_hint : an FSMState the agent suggests transitioning to (advisory;
                      the orchestrator's deterministic gates have final say).
    handoff_to      : an AgentID this agent wants the orchestrator to run next
                      (handoff chain), or None.
    tool_calls      : names of retrieval/IO tools this agent invoked (for logging).
    meta            : arbitrary structured extras (confidence maps, product_ids,
                      validation flags, etc.).
    """
    agent_id:        str
    output_text:     str | None = None
    schema_updates:  dict[str, Any] = field(default_factory=dict)
    next_state_hint: "FSMState | None" = None
    handoff_to:      str | None = None
    tool_calls:      list[str] = field(default_factory=list)
    meta:            dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AGENT PROTOCOL
# ---------------------------------------------------------------------------

@runtime_checkable
class BaseSubAgent(Protocol):
    """Structural type every sub-agent satisfies."""

    agent_id: str

    def run(self, ctx: AgentContext) -> AgentResult:
        ...


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal stand-alone smoke test (no orchestrator/fsm import needed).
    class _Echo:
        agent_id = AgentID.PROBING

        def run(self, ctx: AgentContext) -> AgentResult:
            return AgentResult(agent_id=self.agent_id, output_text=ctx.message)

    agent = _Echo()
    assert isinstance(agent, BaseSubAgent), "Echo must satisfy BaseSubAgent"

    ctx = AgentContext(record=None, message="hello")  # type: ignore[arg-type]
    res = agent.run(ctx)
    assert res.agent_id == "M_05"
    assert res.output_text == "hello"
    assert res.schema_updates == {} and res.tool_calls == []
    assert AGENT_NAMES[AgentID.NUMERIC_GUARDRAIL] == "NumericGuardrail"
    print("base.py self-test passed.")
