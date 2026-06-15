"""
Sub-agent package
=================
Exposes the uniform agent contract plus a single AGENT_REGISTRY mapping every
M_<id> to a ready-to-use agent instance. The orchestrator imports this registry
and routes turns by agent id — it never constructs agents itself.

All agents implement `run(ctx: AgentContext) -> AgentResult`, EXCEPT M_15
(NumericGuardrail), which is a utility used internally by M_07/M_08/M_09 via its
`check()` method. It is registered here for discoverability but is not routed.
"""

from __future__ import annotations

from .base import (
    AgentContext,
    AgentResult,
    AgentID,
    AGENT_NAMES,
    BaseSubAgent,
)

from .intent_classifier import IntentClassifier
from .escalation import EscalationAgent
from .closure import ClosureAgent
from .schema_extractor import SchemaExtractor
from .probing_agent import ProbingAgent
from .policy_retrieval import PolicyRetrievalAgent
from .policy_summary import PolicySummaryAgent
from .policy_qa import PolicyQAAgent
from .agentic_rag import AgenticRAGAgent
from .translator import TranslatorAgent
from .response_queue import ResponseQueue
from .response_validator import ResponseValidator
from .analytics_logger import AnalyticsLogger
from .whatsapp import WhatsAppAgent
from .guardrails import NumericGuardrail
from .sales_agent import SalesAgent

# Single source of truth: agent id → singleton instance.
AGENT_REGISTRY: dict[str, object] = {
    AgentID.INTENT_CLASSIFIER:  IntentClassifier(),
    AgentID.ESCALATION:         EscalationAgent(),
    AgentID.CLOSURE:            ClosureAgent(),
    AgentID.SCHEMA_EXTRACTOR:   SchemaExtractor(),
    AgentID.PROBING:            ProbingAgent(),
    AgentID.POLICY_RETRIEVAL:   PolicyRetrievalAgent(),
    AgentID.POLICY_SUMMARY:     PolicySummaryAgent(),
    AgentID.POLICY_QA:          PolicyQAAgent(),
    AgentID.AGENTIC_RAG:        AgenticRAGAgent(),
    AgentID.TRANSLATOR:         TranslatorAgent(),
    AgentID.RESPONSE_QUEUE:     ResponseQueue(),
    AgentID.RESPONSE_VALIDATOR: ResponseValidator(),
    AgentID.ANALYTICS_LOGGER:   AnalyticsLogger(),
    AgentID.WHATSAPP:           WhatsAppAgent(),
    AgentID.NUMERIC_GUARDRAIL:  NumericGuardrail(),
    AgentID.SALES_AGENT:        SalesAgent(),
}


def get_agent(agent_id: str) -> object:
    """Return the singleton agent instance for an M_<id>."""
    try:
        return AGENT_REGISTRY[agent_id]
    except KeyError as exc:
        raise KeyError(f"No agent registered for id {agent_id!r}") from exc


__all__ = [
    "AgentContext", "AgentResult", "AgentID", "AGENT_NAMES", "BaseSubAgent",
    "AGENT_REGISTRY", "get_agent",
    "IntentClassifier", "EscalationAgent", "ClosureAgent", "SchemaExtractor",
    "ProbingAgent", "PolicyRetrievalAgent", "PolicySummaryAgent", "PolicyQAAgent",
    "AgenticRAGAgent", "TranslatorAgent", "ResponseQueue", "ResponseValidator",
    "AnalyticsLogger", "WhatsAppAgent", "NumericGuardrail", "SalesAgent",
]
