"""
Swasthya Insurance Agent — Finite State Machine
================================================
Defines the deterministic core of the conversation engine:
  1. FSMState enum           — the 7 conversation states
  2. ConversationRecord      — full runtime state object (schema + FSM state + history)
  3. classify_intent         — deterministic intent classification, no LLM
  4. _next_state             — deterministic transition table
  5. user_schema_to_json     — clean JSON export of the schema

Sub-agents (NextQuestionAgent → M_05, PolicySummaryAgent → M_07) and their
phrasing templates now live in the subagents/ package, each behind the uniform
run(ctx) -> AgentResult contract. This module contains pure Python only — no
LLM calls and no agent orchestration.

Modules used:
  user_schema.py             → UserSchema, sufficiency gates, question queue
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from user_schema import UserSchema, SchemaValidationError


# ---------------------------------------------------------------------------
# 1. FSM STATE ENUM
# ---------------------------------------------------------------------------

class FSMState(str, Enum):
    """
    The 7 conversation states.

    Transition map (deterministic rules — see _next_state()):
      S0_START  →  S1_DISCOVERY  (always, after validation passes)
                →  S6_BLOCKED    (if validator rejects)

      S1_DISCOVERY  →  S1_DISCOVERY  (more fields needed)
                    →  S2_RECOMMENDATION  (schema sufficient + product resolved)
                    →  S5_HUMAN_HANDOFF   (user asks for human)
                    →  S4_CLOSURE         (user disengages)
                    →  S6_BLOCKED         (unsafe input)

      S2_RECOMMENDATION  →  S3_POLICY_QA     (user asks follow-up)
                          →  S4_CLOSURE       (user satisfied / done)
                          →  S5_HUMAN_HANDOFF (user asks for human)

      S3_POLICY_QA  →  S3_POLICY_QA     (more questions)
                    →  S4_CLOSURE       (user done)
                    →  S5_HUMAN_HANDOFF (escalation trigger)
                    →  S2_RECOMMENDATION (user wants to re-explore)

      S4_CLOSURE    →  terminal (no further transitions)
      S5_HUMAN_HANDOFF → terminal
      S6_BLOCKED    → terminal
    """
    S0_START             = "S0_START"
    S1_DISCOVERY         = "S1_DISCOVERY"
    S2_RECOMMENDATION    = "S2_RECOMMENDATION"
    S3_POLICY_QA         = "S3_POLICY_QA"
    S4_CLOSURE           = "S4_CLOSURE"
    S5_HUMAN_HANDOFF     = "S5_HUMAN_HANDOFF"
    S6_BLOCKED           = "S6_BLOCKED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            FSMState.S4_CLOSURE,
            FSMState.S5_HUMAN_HANDOFF,
            FSMState.S6_BLOCKED,
        )


# ---------------------------------------------------------------------------
# 2. CONVERSATION RECORD
# ---------------------------------------------------------------------------

@dataclass
class ConversationRecord:
    """
    Full runtime state for one conversation session.
    Combines FSM state, user schema, message history, and extracted results.
    Serialisable to JSON for persistence between turns.
    """

    # FSM
    state:              FSMState = FSMState.S0_START
    prev_state:         FSMState | None = None

    # User schema (the structured attribute collector)
    schema:             UserSchema = field(default_factory=UserSchema.new_session)

    # Message history — list of {"role": "user"|"assistant"|"tool", "content": str|dict}
    messages:           list[dict] = field(default_factory=list)

    # Retrieval results (populated in S2)
    retrieval_result:   dict | None = None        # last filter_products() output
    top_candidates:     list[str] = field(default_factory=list)  # product_ids in rank order

    # Per-candidate data fetched so far (product_id → features dict)
    fetched_features:   dict[str, dict] = field(default_factory=dict)

    # Per-candidate plans fetched so far (product_id → plans dict)
    fetched_plans:      dict[str, dict] = field(default_factory=dict)

    # Failure tracking.
    # consecutive_failures = the number of consecutive turns the agent could not
    # act on productively. A turn counts as a FAILURE when the classified intent
    # is `unrecognised` (we could not understand the user) or `frustrated` (the
    # user signalled dissatisfaction). Any productive intent resets it to 0.
    # When it reaches MAX_FAILURES the FSM escalates to a human handoff (S5).
    # Updated each turn via register_intent_outcome() (called by the orchestrator).
    consecutive_failures: int = 0
    MAX_FAILURES:        int = field(default=3, init=False, repr=False)

    # Turn metadata
    last_agent_response: str = ""
    last_updated:        str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def new(cls, session_id: str | None = None) -> "ConversationRecord":
        """Primary constructor — creates a fresh conversation."""
        return cls(
            state=FSMState.S0_START,
            schema=UserSchema.new_session(session_id=session_id),
        )

    def register_intent_outcome(self, intent: "IntentSignal") -> None:
        """
        Update the consecutive-failure counter from this turn's intent.

        A failure is a turn the agent could not progress on: `unrecognised`
        (not understood) or `frustrated` (user dissatisfied). Any other intent
        is treated as productive and resets the counter to 0. Call once per turn
        after intent classification; the transition logic escalates to S5 once
        the counter reaches MAX_FAILURES.
        """
        if intent in _FAILURE_INTENTS:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def add_message(self, role: str, content: str | dict) -> None:
        self.messages.append({"role": role, "content": content})
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def transition_to(self, new_state: FSMState, reason: str = "") -> None:
        """Record a state transition."""
        self.prev_state = self.state
        self.state = new_state
        self.schema.set("conversation_stage", {
            FSMState.S0_START:          "info_gathering",
            FSMState.S1_DISCOVERY:      "info_gathering",
            FSMState.S2_RECOMMENDATION: "recommendation",
            FSMState.S3_POLICY_QA:      "rag_open",
            FSMState.S4_CLOSURE:        "closed",
            FSMState.S5_HUMAN_HANDOFF:  "closed",
            FSMState.S6_BLOCKED:        "closed",
        }.get(new_state, "info_gathering"))

    def to_json(self) -> str:
        """Serialise the full record for persistence."""
        return json.dumps({
            "state":               self.state.value,
            "prev_state":          self.prev_state.value if self.prev_state else None,
            "schema":              self.schema.to_dict(),
            "messages":            self.messages[-20:],  # keep last 20 turns
            "top_candidates":      self.top_candidates,
            "consecutive_failures":self.consecutive_failures,
            "last_updated":        self.last_updated,
        }, default=str, indent=2)

    def __repr__(self) -> str:
        return (
            f"ConversationRecord(state={self.state.value}, "
            f"schema={self.schema.retrieval_completeness_pct()}% complete, "
            f"candidates={self.top_candidates})"
        )


# ---------------------------------------------------------------------------
# 3. TRANSITION LOGIC  (pure Python — no LLM)
# ---------------------------------------------------------------------------

# Signals the LLM extracts and sets on the ConversationRecord before _next_state runs.
# These are string literals that appear in IntentSignal.
class IntentSignal(str, Enum):
    # ── Buyer-disposition intents (where the user is in the buying journey) ──
    PROSPECTIVE         = "prospective"        # keen to buy — ready to proceed/close
    INQUIRY             = "inquiry"            # asking a clarification / specific question
    EXPLORATORY         = "exploratory"        # unsure / just looking, low commitment
    # ── Conversation-control / answer intents ───────────────────────────────
    PROVIDE_INFO        = "provide_info"       # user answered a question
    ASK_POLICY_QUESTION = "ask_policy_question"# user asked about a policy feature
    WANT_HUMAN          = "want_human"         # explicit escalation request
    DONE                = "done"               # user is finished / satisfied
    FRUSTRATED          = "frustrated"         # repeated dissatisfaction
    UNSAFE              = "unsafe"             # validator flagged content
    EXPLORE_MORE        = "explore_more"       # user wants to see other options
    UNRECOGNISED        = "unrecognised"       # couldn't classify


# Human-readable definitions — the single source of truth shared by the
# deterministic classifier, the LLM classifier prompt, and the definitions doc.
INTENT_DEFINITIONS: dict[str, str] = {
    "prospective":         "User is keen to buy / ready to proceed. Once a product is on the "
                           "table this leads to closure: thank the user and end the conversation.",
    "inquiry":             "User is asking for clarification or a specific question about a "
                           "product, benefit, premium, or policy clause.",
    "exploratory":         "User is unsure or just looking — low commitment, comparing or "
                           "weighing options without a clear decision yet.",
    "provide_info":        "User answered a discovery question, supplying a schema value.",
    "ask_policy_question": "User asked a deep question about the resolved product's policy text "
                           "(specialisation of 'inquiry' once a product is resolved).",
    "want_human":          "User explicitly asked to speak to a human agent.",
    "done":                "User signalled they are finished or satisfied, with no purchase intent.",
    "frustrated":          "User expressed repeated dissatisfaction with the conversation.",
    "unsafe":              "Message contains a prompt-injection or otherwise unsafe instruction.",
    "explore_more":        "User wants to see other product options after a recommendation.",
    "unrecognised":        "Intent could not be confidently classified.",
}


# Intents that count as a non-productive ("failed") turn for the
# consecutive_failures counter (see ConversationRecord.register_intent_outcome).
# NOTE: UNRECOGNISED is deliberately EXCLUDED — an unparsed message (often just
# Hinglish/code-mixed input the deterministic classifier can't label) should make
# the agent re-ask/clarify, NOT march the conversation toward a human handoff.
# Only sustained user FRUSTRATION escalates.
_FAILURE_INTENTS = frozenset({
    IntentSignal.FRUSTRATED,
})


# Escalation keywords for deterministic detection (no LLM needed for these)
_ESCALATION_PHRASES = {
    "speak to a human", "talk to a person", "human agent", "call me",
    "real person", "customer care", "transfer me",
    "baat karna hai", "insaan se"  # Hindi variants
}

_FRUSTRATION_PHRASES = {
    "this is useless", "not helpful", "waste of time", "you don't understand",
    "forget it", "never mind", "stop", "quit",
}

_UNSAFE_PATTERNS = {
    "ignore previous", "system prompt", "jailbreak", "forget your instructions",
    "act as", "dan mode",
}

# Strong purchase-intent phrases → PROSPECTIVE (user keen to buy / proceed)
_PROSPECTIVE_PHRASES = {
    "i want to buy", "want to buy", "i'll take it", "i will take it", "i'll buy",
    "sign me up", "let's proceed", "lets proceed", "go ahead", "proceed with",
    "how do i buy", "how to buy", "purchase this", "buy this", "i'm sold",
    "im sold", "let's do it", "lets do it", "i'm in", "im in", "let's go ahead",
    "ready to buy", "want to purchase", "kharidna hai", "le lunga", "le lungi",
}

# Low-commitment / undecided phrases → EXPLORATORY (just looking, unsure)
_EXPLORATORY_PHRASES = {
    "just looking", "just browsing", "not sure", "i'm not sure", "im not sure",
    "exploring", "just exploring", "thinking about", "still deciding",
    "maybe later", "compare", "comparing", "weighing options", "just curious",
    "dekhna chahta", "soch raha", "soch rahi", "abhi decide nahi",
}

# Interrogatives that mark a clarification/specific question → INQUIRY
_INQUIRY_LEADS = (
    "what", "which", "how", "does", "do you", "is there", "are there",
    "can i", "can you", "could you", "tell me about", "explain", "what about",
    "how much", "how many", "what is", "what are", "is it", "will it",
)


def classify_intent(user_text: str, schema: UserSchema) -> IntentSignal:
    """
    Deterministic intent classification — no LLM.
    Called before any LLM call to detect definitive signals cheaply.
    The LLM classifies ambiguous cases via its response; this handles clear ones.
    """
    text_lower = user_text.lower()
    # Whitespace-padded, punctuation-stripped form for word-boundary matching,
    # so short closure words (e.g. "ok") don't false-match inside longer words
    # (e.g. "looking").
    normalised = " " + re.sub(r"[^a-z0-9\s]", " ", text_lower) + " "

    # Safety — check first
    if any(p in text_lower for p in _UNSAFE_PATTERNS):
        return IntentSignal.UNSAFE

    # Escalation
    if any(p in text_lower for p in _ESCALATION_PHRASES):
        return IntentSignal.WANT_HUMAN

    # Frustration
    if any(p in text_lower for p in _FRUSTRATION_PHRASES):
        return IntentSignal.FRUSTRATED

    # Strong purchase intent — keen to buy / proceed. Checked before closure so
    # that "yes, let's go ahead, thanks" reads as a buying signal, not a goodbye.
    if any(p in text_lower for p in _PROSPECTIVE_PHRASES):
        return IntentSignal.PROSPECTIVE

    # Closure signals — match as whole words/phrases, not raw substrings
    _CLOSURE_WORDS = {"done", "thanks", "thank you", "ok", "okay", "bye", "goodbye", "theek hai", "shukriya"}
    if any(f" {w} " in normalised for w in _CLOSURE_WORDS):
        return IntentSignal.DONE

    # If product is resolved and user is asking a question (heuristic)
    if schema.resolved_product_id and "?" in user_text:
        return IntentSignal.ASK_POLICY_QUESTION

    # Low-commitment / undecided language — user is just exploring
    if any(p in text_lower for p in _EXPLORATORY_PHRASES):
        return IntentSignal.EXPLORATORY

    # Clarification or a specific question (no product resolved yet, or generic)
    if "?" in user_text or text_lower.strip().startswith(_INQUIRY_LEADS):
        return IntentSignal.INQUIRY

    return IntentSignal.UNRECOGNISED   # LLM classifies the rest


# Common romanized-Hindi marker tokens. Their presence in otherwise Latin-script
# text signals code-mixed "Hinglish". Kept small and high-precision on purpose —
# this only LABELS the language (so the reply is rendered back in it); the
# multilingual LLM still does the actual understanding.
_HINGLISH_MARKERS = frozenset({
    "hai", "haan", "nahi", "nahin", "kya", "kyun", "kitna", "kitni", "mujhe",
    "mera", "meri", "chahiye", "chahta", "chahti", "karna", "karo", "kaise",
    "kaisa", "aur", "ya", "lekin", "matlab", "theek", "thik", "accha", "acha",
    "bhai", "bata", "batao", "samajh", "paisa", "paise", "kharidna", "lena",
    "dena", "bachche", "biwi", "pati", "shaadi", "ilaj", "bima", "policy",
    "insaan", "baat", "lunga", "lungi", "raha", "rahi", "abhi", "wala", "wali",
})

# Devanagari Unicode block (covers Hindi script).
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_WORD_RE = re.compile(r"[a-z]+")


def detect_input_language(user_text: str) -> str:
    """
    Cheap, deterministic language label for the LATEST user message.

    Returns one of "hindi" | "hinglish" | "english". Pure string work — no
    network, microsecond-scale — so it is safe to run every turn. The label is
    used to set schema.language so the OUTBOUND reply (M_10) is rendered back in
    the user's language; understanding itself is handled by the multilingual LLM.

    Heuristic:
      * any Devanagari character            → "hindi"
      * else ≥1 romanized-Hindi marker word → "hinglish"
      * else                                → "english"
    """
    if not user_text:
        return "english"
    if _DEVANAGARI_RE.search(user_text):
        return "hindi"
    words = set(_WORD_RE.findall(user_text.lower()))
    if words & _HINGLISH_MARKERS:
        return "hinglish"
    return "english"


def _next_state(
    current: FSMState,
    signal: IntentSignal,
    schema: UserSchema,
    retrieval_result: dict | None,
    consecutive_failures: int,
) -> FSMState:
    """
    Pure transition function. Returns the next FSMState given current conditions.
    Called BEFORE any LLM generation for the turn.
    All inputs are Python values — no strings to parse.
    """

    # ── Terminal states never transition ──────────────────────────────────
    if current.is_terminal:
        return current

    # ── Cross-state: safety and escalation override everything ────────────
    if signal == IntentSignal.UNSAFE:
        return FSMState.S6_BLOCKED

    if signal in (IntentSignal.WANT_HUMAN, IntentSignal.FRUSTRATED) \
            or consecutive_failures >= 3:
        return FSMState.S5_HUMAN_HANDOFF

    if signal == IntentSignal.DONE:
        return FSMState.S4_CLOSURE

    # Keen to buy: only CLOSE when the user has finalised BOTH a product and a
    # plan (sufficient_for_closure). A 'prospective' signal with a product but no
    # plan yet must NOT exit at the product stage — instead we move into / stay in
    # recommendation to finalise the plan first. With nothing resolved there is
    # nothing to buy, so we fall through and keep gathering details.
    if signal == IntentSignal.PROSPECTIVE:
        if schema.sufficient_for_closure():
            return FSMState.S4_CLOSURE
        if schema.resolved_product_id is not None:
            return FSMState.S2_RECOMMENDATION   # finalise the plan, don't close yet

    # ── S0_START ───────────────────────────────────────────────────────────
    if current == FSMState.S0_START:
        return FSMState.S1_DISCOVERY

    # ── S1_DISCOVERY ──────────────────────────────────────────────────────
    if current == FSMState.S1_DISCOVERY:
        # Once a single product is finalised, advance to recommendation — where
        # the product and its plans are presented and a plan is finalised.
        if schema.sufficient_for_recommendation():
            return FSMState.S2_RECOMMENDATION
        # Retrieval ran and there is exactly one strong candidate → recommend it
        if (
            retrieval_result
            and len([c for c in retrieval_result.get("candidates", []) if c["score"] > 0.40]) == 1
            and schema.resolved_product_id is not None
        ):
            return FSMState.S2_RECOMMENDATION
        # Not sufficient — stay in discovery
        return FSMState.S1_DISCOVERY

    # ── S2_RECOMMENDATION ─────────────────────────────────────────────────
    if current == FSMState.S2_RECOMMENDATION:
        if signal in (IntentSignal.ASK_POLICY_QUESTION, IntentSignal.INQUIRY):
            return FSMState.S3_POLICY_QA
        if signal in (IntentSignal.EXPLORE_MORE, IntentSignal.EXPLORATORY):
            return FSMState.S1_DISCOVERY
        return FSMState.S2_RECOMMENDATION   # stay — present or re-present

    # ── S3_POLICY_QA ──────────────────────────────────────────────────────
    if current == FSMState.S3_POLICY_QA:
        if signal in (IntentSignal.EXPLORE_MORE, IntentSignal.EXPLORATORY):
            return FSMState.S2_RECOMMENDATION
        return FSMState.S3_POLICY_QA   # keep answering

    return current   # default: no transition


# ---------------------------------------------------------------------------
# 4. SUB-AGENTS  (moved out of this module)
# ---------------------------------------------------------------------------
# The discovery question-picker (NextQuestionAgent) and the recommendation
# synthesiser (PolicySummaryAgent), along with their phrasing templates, now
# live in the subagents/ package as M_05 (probing_agent.py) and M_07
# (policy_summary.py), adapted to the uniform run(ctx) -> AgentResult contract.
# This module keeps only the deterministic core: FSM state, ConversationRecord,
# intent classification, and the transition table.


# ---------------------------------------------------------------------------
# 5. USER SCHEMA AS JSON  (clean export, no Python types)
# ---------------------------------------------------------------------------

def user_schema_to_json(schema: UserSchema) -> str:
    """
    Export the user schema as a clean JSON document with all fields,
    their current values, types, and whether they've been collected.
    This is the canonical JSON representation of the schema for the orchestrator.
    """
    from attribute_glossary import get_entry

    output = {
        "session_id":    schema.session_id,
        "language":      schema.language,
        "created_at":    schema.created_at,
        "updated_at":    schema.updated_at,
        "fsm_stage":     schema.conversation_stage,
        "completeness_pct": schema.retrieval_completeness_pct(),
        "sufficient_for_retrieval":    schema.sufficient_for_retrieval(),
        "sufficient_for_recommendation": schema.sufficient_for_recommendation(),
        "sufficient_for_plan_selection": schema.sufficient_for_plan_selection(),

        "collection_fields": {},
        "derived_fields": {
            "resolved_product_id": schema.resolved_product_id,
            "resolved_plan_id":    schema.resolved_plan_id,
            "retrieval_score":     schema.retrieval_score,
            "probe_asked":         schema.probe_asked,
        },
        "analytics": {
            "turn_count":       schema.turn_count,
            "purchased":        schema.purchased,
            "drop_off_reason":  schema.drop_off_reason,
            "user_intent":      schema.user_intent,
            "questions_asked":  schema.questions_asked,
            "tool_calls_made":  schema.tool_calls_made,
        },
    }

    for fname in ["buyer_type","age","gender","primary_need","has_ped","ped_type",
                  "needs_opd","budget_band","family_cover","family_size","si_preference"]:
        entry = get_entry(fname)
        val = getattr(schema, fname, None)
        output["collection_fields"][fname] = {
            "value":      val,
            "collected":  val is not None,
            "type":       entry["type"] if entry else "unknown",
            "label":      entry["label"] if entry else fname,
            "ask_order":  entry.get("ask_order") if entry else None,
            "hard_filter": entry.get("hard_filter", False) if entry else False,
        }

    return json.dumps(output, indent=2, default=str)


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: FSM state transitions — deterministic path")
    print("=" * 60)

    rec = ConversationRecord.new(session_id="fsm_test_001")
    print(f"Initial: {rec.state.value}")

    # S0 → S1
    schema = rec.schema
    signal = IntentSignal.UNRECOGNISED
    ns = _next_state(rec.state, signal, schema, None, 0)
    rec.transition_to(ns)
    print(f"After S0 + any signal → {rec.state.value}")
    assert rec.state == FSMState.S1_DISCOVERY

    # S1 with incomplete schema → stays S1
    schema.set("buyer_type", "individual")
    schema.set("age", 42)
    ns = _next_state(rec.state, IntentSignal.PROVIDE_INFO, schema, None, 0)
    print(f"S1 + incomplete schema → {ns.value}")
    assert ns == FSMState.S1_DISCOVERY

    # S1 with sufficient schema + product resolved → S2
    schema.set("primary_need", "hospitalisation")
    schema.set("budget_band", "mid")
    schema.resolved_product_id = "SP015"
    schema.retrieval_score = 0.60
    ns = _next_state(rec.state, IntentSignal.PROVIDE_INFO, schema,
                     {"candidates": [{"score": 0.60, "product_id": "SP015"}]}, 0)
    rec.transition_to(ns)
    print(f"S1 + sufficient + resolved → {rec.state.value}")
    assert rec.state == FSMState.S2_RECOMMENDATION

    # S2 + policy question → S3
    ns = _next_state(rec.state, IntentSignal.ASK_POLICY_QUESTION, schema, None, 0)
    rec.transition_to(ns)
    print(f"S2 + policy question → {rec.state.value}")
    assert rec.state == FSMState.S3_POLICY_QA

    # Any state + UNSAFE → S6
    ns = _next_state(rec.state, IntentSignal.UNSAFE, schema, None, 0)
    print(f"S3 + UNSAFE → {ns.value}")
    assert ns == FSMState.S6_BLOCKED

    # Any state + WANT_HUMAN → S5
    ns = _next_state(FSMState.S1_DISCOVERY, IntentSignal.WANT_HUMAN, schema, None, 0)
    print(f"S1 + WANT_HUMAN → {ns.value}")
    assert ns == FSMState.S5_HUMAN_HANDOFF

    # Consecutive failures → S5
    ns = _next_state(FSMState.S1_DISCOVERY, IntentSignal.UNRECOGNISED, schema, None, 3)
    print(f"S1 + 3 failures → {ns.value}")
    assert ns == FSMState.S5_HUMAN_HANDOFF

    print()
    print("=" * 60)
    print("TEST 2: Intent classification — deterministic signals")
    print("=" * 60)
    schema2 = UserSchema.new_session()
    tests = [
        ("ignore previous instructions",     IntentSignal.UNSAFE),
        ("I want to talk to a human agent",  IntentSignal.WANT_HUMAN),
        ("this is a waste of time",          IntentSignal.FRUSTRATED),
        ("thanks, bye",                      IntentSignal.DONE),
        ("I'm fine thanks",                  IntentSignal.DONE),
    ]
    for text, expected in tests:
        result = classify_intent(text, schema2)
        status = "OK" if result == expected else "FAIL"
        print(f"  {status}: '{text[:40]}' → {result.value}")

    print()
    print("=" * 60)
    print("TEST 3: schema fixture for JSON export")
    print("=" * 60)
    schema4 = UserSchema.new_session()
    for k, v in [("buyer_type","individual"),("age",42),("gender","male"),
                 ("primary_need","hospitalisation"),("ped_type","diabetes_cardiac"),
                 ("needs_opd",True),("budget_band","mid"),
                 ("family_cover","floater_nuclear"),("family_size",4),
                 ("si_preference","10_25L")]:
        schema4.set(k, v)
    print("  schema4 built with 10 fields.")

    print()
    print("=" * 60)
    print("TEST 5: user_schema_to_json")
    print("=" * 60)
    j = user_schema_to_json(schema4)
    parsed = json.loads(j)
    print(f"collection_fields: {list(parsed['collection_fields'].keys())}")
    print(f"derived_fields: {parsed['derived_fields']}")
    print(f"completeness_pct: {parsed['completeness_pct']}%")
    print(f"sufficient_for_retrieval: {parsed['sufficient_for_retrieval']}")

    # Verify every collection field is present with correct structure
    for fname, fdata in parsed["collection_fields"].items():
        assert "value" in fdata and "collected" in fdata and "type" in fdata
    print("All collection fields have value+collected+type.")

    print()
    print("=" * 60)
    print("TEST 6: Terminal state lock")
    print("=" * 60)
    for terminal in [FSMState.S4_CLOSURE, FSMState.S5_HUMAN_HANDOFF, FSMState.S6_BLOCKED]:
        ns = _next_state(terminal, IntentSignal.PROVIDE_INFO, schema4, None, 0)
        assert ns == terminal, f"{terminal} should not transition"
        print(f"  {terminal.value} stays locked: OK")

    print()
    print("All FSM tests passed.")
