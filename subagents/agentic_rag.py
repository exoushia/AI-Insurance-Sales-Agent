"""
M_09 — AgenticRAGAgent
======================
Role     : the fallback knowledge agent for questions M_08 could not answer from
           the resolved product's wording. Performs a semantic search across the
           unified vector index (policy wording + IRDAI regulations + treatment
           costs) and synthesises one grounded answer.
Trigger  : handoff from M_08 (PolicyQAAgent).
Retrieval: LlamaIndex + ChromaDB vector store (rag.index_store). When a product
           is resolved, a product-filtered policy query is merged with a general
           query so product-specific and cross-corpus context are both surfaced.
Validation : every number in the synthesised answer is checked by M_15 against
             the concatenated retrieved context.
Note       : if the index is unavailable (not built / no OpenAI) or returns no
             hits, M_09 escalates to a human (no keyword fallback by design).
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

from rag.index_store import get_rag_index
from rag import config as C
from prompts_template import RAG_SYNTHESIS_SYSTEM, RAG_SYNTHESIS_USER_TEMPLATE
from llm_gateway import log_llm_call

_TOP_K = 6   # final number of merged chunks fed to synthesis + guardrail


class AgenticRAGAgent:
    agent_id = AgentID.AGENTIC_RAG

    def __init__(self):
        self._guard = NumericGuardrail()

    def run(self, ctx: AgentContext) -> AgentResult:
        question = ctx.message or ""
        pid = ctx.record.schema.resolved_product_id

        hits = self._retrieve(question, pid)
        tool_calls = ["rag_vector_search"]

        if not hits:
            answer = ("I don't have a grounded answer for that in our policy or "
                      "regulatory references. Let me connect you with a specialist.")
            log_llm_call(self.agent_id, "fallback", C.EMBED_MODEL, False, "no-hits")
            return AgentResult(
                agent_id=self.agent_id,
                output_text=answer,
                tool_calls=tool_calls,
                handoff_to=AgentID.ESCALATION,
                meta={"reason": "no_grounded_context"},
            )

        context_parts = [h["text"].strip() for h in hits]
        context = "\n\n".join(context_parts)
        sources = [self._source_label(h["metadata"]) for h in hits]

        det_answer = "Here's what I found. " + " ".join(context_parts[:2])
        answer, source = det_answer, "deterministic"

        # LLM synthesises a coherent answer from the retrieved context; M_15
        # re-checks every number. Fall back to the concatenation on failure.
        llm = ctx.llm
        if llm is not None and getattr(llm, "is_available", False):
            llm_answer = self._synthesize_llm(ctx, llm, question, context)
            if not llm_answer:
                log_llm_call(self.agent_id, "fallback", "response_generator", False, "llm-empty")
            else:
                rep = self._guard.check(llm_answer, context)
                if rep.ok:
                    answer, source = llm_answer, "llm"
                    log_llm_call(self.agent_id, "llm", "response_generator", True, "grounded")
                else:
                    log_llm_call(self.agent_id, "fallback", "response_generator", False,
                                 f"guardrail {rep.offending}")

        report = self._guard.check(answer, context)

        tool_calls.append(self._guard.agent_id)
        return AgentResult(
            agent_id=self.agent_id,
            output_text=answer,
            tool_calls=tool_calls,
            meta={
                "sources": sources,
                "guardrail_ok": report.ok,
                "guardrail_offending": report.offending,
                "source": source,
                "returns_to": AgentID.POLICY_QA,
                "_retrieved_context": context,
            },
        )

    # -- helpers ------------------------------------------------------------

    def _retrieve(self, question: str, pid: str | None) -> list[dict]:
        """Light-agentic retrieval over the vector index.

        With a resolved product we run two queries — one filtered to that
        product's wording, one across the whole corpus — and merge them,
        deduping by text and keeping the highest-scoring _TOP_K chunks.
        """
        index = get_rag_index()
        if index is None:
            return []

        merged: dict[str, dict] = {}

        def _absorb(hits: list[dict]) -> None:
            for h in hits:
                key = h["text"]
                if key not in merged or h["score"] > merged[key]["score"]:
                    merged[key] = h

        if pid:
            _absorb(index.query(question, top_k=_TOP_K, product_id=pid))
        _absorb(index.query(question, top_k=_TOP_K))

        ranked = sorted(merged.values(), key=lambda h: h["score"], reverse=True)
        return ranked[:_TOP_K]

    @staticmethod
    def _source_label(meta: dict) -> str:
        """Human-readable citation for a retrieved chunk."""
        stype = meta.get(C.META_SOURCE_TYPE, "")
        if stype == C.SOURCE_POLICY:
            return f"{meta.get(C.META_PRODUCT_ID, '')} policy wording".strip()
        if stype == C.SOURCE_REGULATION:
            num = meta.get(C.META_SECTION_NUMBER, "")
            return f"IRDAI §{num}".strip() if num else "IRDAI regulation"
        if stype == C.SOURCE_COST:
            return "treatment cost reference"
        return stype or "reference"

    def _synthesize_llm(self, ctx: AgentContext, llm, question: str,
                        context: str) -> str | None:
        """Synthesise one grounded answer from the retrieved `context`."""
        return llm.generate_response(
            RAG_SYNTHESIS_SYSTEM,
            RAG_SYNTHESIS_USER_TEMPLATE.format(question=question, context=context),
        )


# ---------------------------------------------------------------------------
# SELF-TEST  (requires a built vector store + OPENAI_API_KEY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fsm import ConversationRecord, FSMState

    agent = AgenticRAGAgent()

    def ask(question, pid=None):
        rec = ConversationRecord.new(session_id="m09_test")
        if pid:
            rec.schema.set("resolved_product_id", pid)
        rec.state = FSMState.S3_POLICY_QA
        return agent.run(AgentContext(record=rec, message=question))

    if get_rag_index() is None:
        raise SystemExit(
            "Vector store not available. Build it first: python -m rag.ingest "
            "(and ensure OPENAI_API_KEY is set)."
        )

    # Regulation / policy question → grounded answer.
    r = ask("what is the free look period?")
    assert r.output_text, "expected a grounded answer"
    assert r.meta["guardrail_ok"], f"guardrail flagged: {r.meta.get('guardrail_offending')}"
    assert r.meta.get("sources"), r.meta
    print("  free-look sources:", r.meta["sources"][:3])

    # Cost question → treatment cost reference, numbers grounded.
    r = ask("how much does a normal delivery cost in a private hospital?")
    assert r.meta["guardrail_ok"], f"guardrail flagged: {r.meta.get('guardrail_offending')}"
    assert any("cost" in s for s in r.meta.get("sources", [])), r.meta["sources"]
    print("  cost sources:", r.meta["sources"][:3])

    # Pure gibberish → still retrieves some low-relevance chunks; the key
    # contract is that M_09 never crashes and produces a guardrail-checked
    # answer (vector RAG always returns nearest neighbours).
    r = ask("zxqw plffft brumble")
    assert r.output_text, r

    print("agentic_rag.py (M_09) self-test passed.")
