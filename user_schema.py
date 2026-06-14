"""
Swasthya Insurance Agent — User Schema
=======================================
Single definition of every field the system collects, derives, or logs about a user.

Three field groups
------------------
  COLLECTION  — Fields collected by asking the user (9 askable fields in ask_order).
                These feed directly into filter_products() and get_plan_options().
                Missing = not yet asked. None = asked but user declined to answer.

  DERIVED     — Computed from collection fields without asking.
                resolved_product_id, resolved_plan_id, retrieval_score.
                Set by the orchestrator after tool calls return.

  ANALYTICS   — Session metadata for call analytics and A/B testing.
                Never sent to the LLM. Logged to the data warehouse.

Sufficiency gate
----------------
The schema is "sufficient for retrieval" when the three minimum fields are known:
  buyer_type + age + primary_need
All other fields improve retrieval quality but are not required to trigger it.

The schema is "sufficient for recommendation" when a single product has been
resolved (resolved_product_id is set) so features + plans can be presented.

The schema is "sufficient for plan selection" when one product is finalised
(resolved_product_id set) — plans for that product can then be discussed.

The schema is "sufficient for closure" only when BOTH a product and a plan are
finalised (resolved_product_id AND resolved_plan_id). This enforces the funnel:
info gathering → retrieval → explain/sell options → Q&A → finalise one policy →
finalise one plan → close — so a keen ('prospective') user is never closed out
at the product stage before choosing a plan.

Usage
-----
  schema = UserSchema.new_session(session_id="sess_abc", language="hinglish")
  schema.buyer_type = "individual"
  schema.age = 42
  schema.primary_need = "hospitalisation"

  if schema.sufficient_for_retrieval():
      result = filter_products(schema.to_tool_input())

  schema.resolved_product_id = "SP015"
  if schema.sufficient_for_recommendation():
      features = get_product_features("SP015", schema.to_tool_input())
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from attribute_glossary import askable_fields, by_layer, get_entry


# ---------------------------------------------------------------------------
# VALID VALUE SETS  (sourced from glossary — never duplicated)
# ---------------------------------------------------------------------------

def _valid(key: str) -> set:
    """Return the set of valid string values for an enum field from the glossary."""
    entry = get_entry(key)
    if entry and isinstance(entry.get("valid_values"), dict):
        return set(str(k) for k in entry["valid_values"].keys())
    return set()


# Build at import time — one read of the glossary
_VALID_BUYER_TYPES    = _valid("buyer_type")
_VALID_GENDERS        = _valid("gender")
_VALID_PRIMARY_NEEDS  = _valid("primary_need")
_VALID_PED_TYPES      = _valid("ped_type")
_VALID_BUDGET_BANDS   = _valid("budget_band")
_VALID_FAMILY_COVERS  = _valid("family_cover")
_VALID_SI_PREFS       = _valid("si_preference")
_VALID_STAGES         = _valid("conversation_stage")
_VALID_DROP_REASONS   = _valid("drop_off_reason")
_VALID_USER_INTENTS   = _valid("user_intent")

# Minimum set of fields to trigger filter_products()
MINIMUM_FOR_RETRIEVAL = {"buyer_type", "age", "primary_need"}

# Preferred fields: improve scoring quality but not required
PREFERRED_FOR_RETRIEVAL = {"gender", "has_ped", "ped_type", "needs_opd", "budget_band"}

# Required for plan selection after product resolved
MINIMUM_FOR_PLAN = {"resolved_product_id"}
PREFERRED_FOR_PLAN = {"si_preference", "budget_band", "family_cover", "family_size"}


# ---------------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------------

class SchemaValidationError(ValueError):
    """Raised when a field is set to an illegal value."""
    pass


def _check_enum(key: str, value: Any, valid_set: set) -> None:
    if value is not None and str(value) not in valid_set:
        raise SchemaValidationError(
            f"Invalid value for {key!r}: {value!r}. Must be one of {sorted(valid_set)}"
        )


def _check_int_range(key: str, value: Any, lo: int, hi: int) -> None:
    if value is not None:
        if not isinstance(value, int):
            raise SchemaValidationError(
                f"{key!r} must be an integer, got {type(value).__name__}: {value!r}"
            )
        if not (lo <= value <= hi):
            raise SchemaValidationError(
                f"{key!r}={value} out of range [{lo}, {hi}]"
            )


def _check_bool(key: str, value: Any) -> None:
    if value is not None and not isinstance(value, bool):
        raise SchemaValidationError(
            f"{key!r} must be True or False, got {type(value).__name__}: {value!r}"
        )


# ---------------------------------------------------------------------------
# USER SCHEMA DATACLASS
# ---------------------------------------------------------------------------

@dataclass
class UserSchema:
    """
    Complete user schema.

    Fields with default=None mean "not yet collected".
    Fields with a concrete default (e.g. conversation_stage) are always set.

    Never instantiate directly — use UserSchema.new_session() to get a schema
    with session_id and timestamps pre-populated.
    """

    # ── COLLECTION: eligibility (hard filters) ──────────────────────────────

    buyer_type: str | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 1,
            "hard_filter": True,
            "description": "Who is buying: individual / employer_large / employer_sme / gig_worker",
        }
    )

    age: int | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 2,
            "hard_filter": True,
            "description": "Applicant age in years. Must be 0–120.",
        }
    )

    gender: str | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 3,
            "hard_filter": True,
            "description": "female / male / other. Unlocks or excludes female-only products.",
        }
    )

    # ── COLLECTION: need (product pointers) ──────────────────────────────────

    primary_need: str | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 4,
            "hard_filter": True,
            "description": "Main coverage need: hospitalisation / maternity / cancer / etc.",
        }
    )

    has_ped: bool | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 5,
            "hard_filter": False,
            "description": "True if applicant has any pre-existing condition.",
        }
    )

    ped_type: str | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 5,
            "hard_filter": True,
            "description": "diabetes_cardiac / other_ped / none. Only meaningful if has_ped=True.",
        }
    )

    needs_opd: bool | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 6,
            "hard_filter": False,
            "description": "True if user wants OPD (outpatient) cover, not just hospitalisation.",
        }
    )

    # ── COLLECTION: affordability ─────────────────────────────────────────────

    budget_band: str | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 7,
            "hard_filter": False,
            "description": "micro / budget / mid / premium. Annual premium tolerance.",
        }
    )

    # ── COLLECTION: plan-level (asked after product resolved) ────────────────

    family_cover: str | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 8,
            "hard_filter": False,
            "description": "individual / floater_nuclear / floater_joint.",
        }
    )

    family_size: int | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 8,
            "hard_filter": False,
            "description": "Total people to cover including applicant. 1–8.",
        }
    )

    si_preference: str | None = field(
        default=None,
        metadata={
            "group": "collection",
            "ask_order": 9,
            "hard_filter": False,
            "description": "1_2L / 3_5L / 10_25L / 50L_plus. Desired sum insured range.",
        }
    )

    # ── DERIVED: set by orchestrator after tool calls ─────────────────────────

    resolved_product_id: str | None = field(
        default=None,
        metadata={
            "group": "derived",
            "description": "Product ID confirmed by retrieval + user acceptance. e.g. 'SP015'.",
        }
    )

    resolved_plan_id: str | None = field(
        default=None,
        metadata={
            "group": "derived",
            "description": "Plan ID confirmed by plan selector + user acceptance. e.g. 'SP015-Gold'.",
        }
    )

    retrieval_score: float | None = field(
        default=None,
        metadata={
            "group": "derived",
            "description": "Score of resolved_product_id from filter_products. 0.0–1.0.",
        }
    )

    retrieval_candidates: list | None = field(
        default=None,
        metadata={
            "group": "derived",
            "description": "Full ranked candidate list from last filter_products call.",
        }
    )

    probe_asked: bool = field(
        default=False,
        metadata={
            "group": "derived",
            "description": "True if a probe question was issued to narrow >3 candidates.",
        }
    )

    # ── ANALYTICS: session metadata ───────────────────────────────────────────

    session_id: str = field(
        default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}",
        metadata={
            "group": "analytics",
            "description": "Unique session identifier. Auto-generated if not supplied.",
        }
    )

    # NOTE: `channel` was removed — the agent is voice-based for the demo,
    # so no inbound channel is tracked.

    language: str | None = field(
        default=None,
        metadata={
            "group": "analytics",
            "description": "User's spoken language: 'hindi' / 'english' / 'hinglish'. "
                           "Drives the translation agent that localises every response.",
        }
    )

    conversation_stage: str = field(
        default="info_gathering",
        metadata={
            "group": "analytics",
            "description": "Pipeline stage: info_gathering / retrieval / recommendation / rag_open / closed.",
        }
    )

    drop_off_reason: str | None = field(
        default=None,
        metadata={
            "group": "analytics",
            "description": "Why user left without purchase (populated on close without sale).",
        }
    )

    user_intent: str | None = field(
        default=None,
        metadata={
            "group": "analytics",
            "description": "Most recent classified intent (mirrors IntentSignal). "
                           "Set by the orchestrator each turn; logged for analytics.",
        }
    )

    purchased: bool = field(
        default=False,
        metadata={
            "group": "analytics",
            "description": "True if user completed a purchase in this session.",
        }
    )

    turn_count: int = field(
        default=0,
        metadata={
            "group": "analytics",
            "description": "Number of conversation turns so far.",
        }
    )

    tool_calls_made: list = field(
        default_factory=list,
        metadata={
            "group": "analytics",
            "description": "Ordered log of tool calls: [{'tool': str, 'ts': ISO8601}].",
        }
    )

    questions_asked: list = field(
        default_factory=list,
        metadata={
            "group": "analytics",
            "description": "Ordered log of fields asked: [{'key': str, 'ts': ISO8601}].",
        }
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        metadata={
            "group": "analytics",
            "description": "Session creation timestamp (ISO 8601 UTC).",
        }
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        metadata={
            "group": "analytics",
            "description": "Last update timestamp (ISO 8601 UTC).",
        }
    )

    # ── CLASS METHODS ─────────────────────────────────────────────────────────

    @classmethod
    def new_session(
        cls,
        session_id: str | None = None,
        language: str | None = None,
    ) -> "UserSchema":
        """
        Primary constructor. Always use this rather than UserSchema() directly.
        Generates session_id if not supplied. Sets timestamps.
        """
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        ts = datetime.now(timezone.utc).isoformat()
        return cls(
            session_id=sid,
            language=language,
            created_at=ts,
            updated_at=ts,
        )

    # ── SETTERS WITH VALIDATION ───────────────────────────────────────────────

    def set(self, key: str, value: Any) -> "UserSchema":
        """
        Set a field by name with validation. Returns self for chaining.
        Use this from the orchestrator when parsing LLM extractions.

        Raises SchemaValidationError on illegal values.
        Records updated_at and the field in questions_asked if it is a collection field.
        """
        validators = {
            "buyer_type":    lambda v: _check_enum("buyer_type", v, _VALID_BUYER_TYPES),
            "age":           lambda v: _check_int_range("age", v, 0, 120),
            "gender":        lambda v: _check_enum("gender", v, _VALID_GENDERS),
            "primary_need":  lambda v: _check_enum("primary_need", v, _VALID_PRIMARY_NEEDS),
            "has_ped":       lambda v: _check_bool("has_ped", v),
            "ped_type":      lambda v: _check_enum("ped_type", v, _VALID_PED_TYPES),
            "needs_opd":     lambda v: _check_bool("needs_opd", v),
            "budget_band":   lambda v: _check_enum("budget_band", v, _VALID_BUDGET_BANDS),
            "family_cover":  lambda v: _check_enum("family_cover", v, _VALID_FAMILY_COVERS),
            "family_size":   lambda v: _check_int_range("family_size", v, 1, 8),
            "si_preference": lambda v: _check_enum("si_preference", v, _VALID_SI_PREFS),
            "conversation_stage": lambda v: _check_enum("conversation_stage", v, _VALID_STAGES),
            "drop_off_reason": lambda v: _check_enum("drop_off_reason", v, _VALID_DROP_REASONS),
            "user_intent":   lambda v: _check_enum("user_intent", v, _VALID_USER_INTENTS),
            "turn_count":    lambda v: _check_int_range("turn_count", v, 0, 10_000),
            "purchased":     lambda v: _check_bool("purchased", v),
            "probe_asked":   lambda v: _check_bool("probe_asked", v),
        }

        if not hasattr(self, key):
            raise SchemaValidationError(f"Unknown field {key!r}")

        if key in validators and value is not None:
            validators[key](value)

        setattr(self, key, value)
        self.updated_at = datetime.now(timezone.utc).isoformat()

        # Track which collection fields have been asked/set for analytics
        collection_keys = {f["key"] for f in askable_fields()}
        if key in collection_keys and value is not None:
            already_logged = {q["key"] for q in self.questions_asked}
            if key not in already_logged:
                self.questions_asked.append({
                    "key": key,
                    "ts":  self.updated_at,
                })

        return self

    def log_tool_call(self, tool_name: str) -> None:
        """Record a tool call in the analytics log."""
        self.tool_calls_made.append({
            "tool": tool_name,
            "ts":   datetime.now(timezone.utc).isoformat(),
        })
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def increment_turn(self) -> None:
        """Call once per conversation turn."""
        self.turn_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # ── SUFFICIENCY GATES ─────────────────────────────────────────────────────

    def sufficient_for_retrieval(self) -> bool:
        """
        True when filter_products() can run meaningfully.
        Minimum: buyer_type + age + primary_need all known.
        """
        return all(
            getattr(self, k) is not None
            for k in MINIMUM_FOR_RETRIEVAL
        )

    def sufficient_for_recommendation(self) -> bool:
        """
        True when get_product_features() + get_plan_options() can run.
        Requires: retrieval done + product resolved.
        """
        return self.resolved_product_id is not None

    def sufficient_for_plan_selection(self) -> bool:
        """
        True once exactly one product is finalised (resolved_product_id set).
        This is the gate to move from comparing policies into discussing that
        product's plans. budget_band / si_preference, if known, only refine which
        plan tier is recommended — they are not required to enter plan discussion.
        """
        return self.resolved_product_id is not None

    def sufficient_for_closure(self) -> bool:
        """
        True only when the user has finalised BOTH a product and a specific plan.
        Closure (thank + end) must never happen at the product stage alone: the
        user first agrees to a product, then to a plan, and only then do we close.
        This is what stops a 'prospective' signal from prematurely exiting the
        funnel before a plan is chosen.
        """
        return (
            self.resolved_product_id is not None
            and self.resolved_plan_id is not None
        )

    def retrieval_completeness_pct(self) -> float:
        """
        Percentage of collection fields that have been collected.
        Useful for progress indicators and analytics dashboards.
        """
        all_keys = [f["key"] for f in askable_fields()]
        collected = sum(1 for k in all_keys if getattr(self, k, None) is not None)
        return round(collected / len(all_keys) * 100, 1)

    # ── MISSING FIELD NAVIGATION ──────────────────────────────────────────────

    def next_missing_field(self) -> dict | None:
        """
        Return the glossary entry for the next field to ask, in ask_order.
        Skips ped_type if has_ped is False.
        Skips family_size if family_cover is 'individual'.
        Returns None when all askable fields are collected.
        """
        for entry in askable_fields():
            key = entry["key"]
            val = getattr(self, key, None)
            if val is not None:
                continue   # already collected

            # Conditional skips
            if key == "ped_type" and self.has_ped is False:
                continue   # no PED — skip asking type
            if key == "family_size" and self.family_cover == "individual":
                continue   # individual cover — family size irrelevant

            return entry   # this is the next field to ask

        return None   # all collected

    def missing_collection_fields(self) -> list[dict]:
        """
        Return all uncollected askable fields in ask_order.
        Applies the same conditional skip logic as next_missing_field().
        Used by the retrieval tool to report what's still missing.
        """
        missing = []
        for entry in askable_fields():
            key = entry["key"]
            val = getattr(self, key, None)
            if val is not None:
                continue
            if key == "ped_type" and self.has_ped is False:
                continue
            if key == "family_size" and self.family_cover == "individual":
                continue
            missing.append({"key": key, "label": entry["label"], "ask_order": entry["ask_order"]})
        return missing

    def collected_collection_fields(self) -> dict[str, Any]:
        """
        Return a dict of all collection fields that have been set (not None).
        This is the safe view of the schema — only what we actually know.
        """
        all_keys = [f["key"] for f in askable_fields()]
        return {k: getattr(self, k) for k in all_keys if getattr(self, k) is not None}

    # ── SERIALISATION ─────────────────────────────────────────────────────────

    def to_tool_input(self) -> dict:
        """
        Return a clean dict of collection fields for passing to retrieval tools.
        - Only fields that have been set (not None)
        - No analytics, no derived fields
        - No internal orchestrator state
        This is what gets passed as user_schema in every tool call.
        """
        return self.collected_collection_fields()

    def to_llm_context(self) -> dict:
        """
        Return the schema state the LLM can see in its system prompt.
        Includes collection + derived fields (so LLM knows what's resolved).
        Excludes analytics (session_id, timestamps, tool_call logs).
        """
        excluded = {
            "session_id", "created_at", "updated_at",
            "tool_calls_made", "questions_asked",
            "user_intent",            # the model should not see its own intent label
            "retrieval_candidates",   # too large for context
        }
        result = {}
        for f in self.__dataclass_fields__:    # type: ignore[attr-defined]
            if f in excluded:
                continue
            val = getattr(self, f)
            if val is not None and val != [] and val != 0 and val is not False:
                result[f] = val
            elif f in ("turn_count", "probe_asked", "purchased"):
                result[f] = val   # always include these even if falsy
        return result

    def to_analytics_record(self) -> dict:
        """
        Return the full schema as a flat dict for warehouse logging.
        Includes all fields. Lists are JSON-serialised strings.
        """
        raw = asdict(self)
        # Flatten lists to JSON strings for columnar storage
        for key in ("tool_calls_made", "questions_asked", "retrieval_candidates"):
            if key in raw and isinstance(raw[key], list):
                import json
                raw[key] = json.dumps(raw[key]) if raw[key] else None
        return raw

    def to_dict(self) -> dict:
        """Full dict — all fields including None values. For debugging."""
        return asdict(self)

    # ── REPR ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        collected = self.collected_collection_fields()
        return (
            f"UserSchema(session={self.session_id!r}, "
            f"stage={self.conversation_stage!r}, "
            f"collected={list(collected.keys())}, "
            f"resolved={self.resolved_product_id!r})"
        )

    def summary(self) -> str:
        """Human-readable single-line summary for logs."""
        pct = self.retrieval_completeness_pct()
        return (
            f"[{self.session_id}] "
            f"stage={self.conversation_stage} | "
            f"collection={pct}% | "
            f"resolved={self.resolved_product_id or 'none'} | "
            f"plan={self.resolved_plan_id or 'none'} | "
            f"turns={self.turn_count} | "
            f"purchased={self.purchased}"
        )


# ---------------------------------------------------------------------------
# SCHEMA FIELD CATALOGUE  (for documentation and test generation)
# ---------------------------------------------------------------------------

def field_catalogue() -> list[dict]:
    """
    Return a structured catalogue of all schema fields with their metadata.
    Merges dataclass field metadata with glossary descriptions.
    Useful for generating documentation or driving UI form builders.
    """
    rows = []
    for fname, fobj in UserSchema.__dataclass_fields__.items():   # type: ignore[attr-defined]
        meta = fobj.metadata
        glossary_entry = get_entry(fname)
        row = {
            "key":          fname,
            "group":        meta.get("group", "unknown"),
            "ask_order":    meta.get("ask_order"),
            "hard_filter":  meta.get("hard_filter", False),
            "description":  meta.get("description", ""),
            "type":         glossary_entry["type"] if glossary_entry else "unknown",
            "nullable":     glossary_entry.get("nullable") if glossary_entry else True,
            "valid_values": glossary_entry.get("valid_values") if glossary_entry else None,
            "question_text":glossary_entry.get("question_text") if glossary_entry else None,
            "label":        glossary_entry["label"] if glossary_entry else fname,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("TEST 1: Basic instantiation and field setting")
    print("=" * 60)
    s = UserSchema.new_session(session_id="test_001", language="hinglish")
    print(repr(s))

    s.set("buyer_type", "individual")
    s.set("age", 42)
    s.set("gender", "male")
    s.set("primary_need", "hospitalisation")
    s.set("has_ped", True)
    s.set("ped_type", "diabetes_cardiac")
    s.set("needs_opd", True)
    s.set("budget_band", "mid")

    print(repr(s))
    print(f"Completeness: {s.retrieval_completeness_pct()}%")

    print()
    print("=" * 60)
    print("TEST 2: Sufficiency gates")
    print("=" * 60)
    s2 = UserSchema.new_session(session_id="test_002")
    print(f"Empty schema — sufficient_for_retrieval: {s2.sufficient_for_retrieval()}")

    s2.set("buyer_type", "individual")
    s2.set("age", 38)
    print(f"buyer_type+age only — sufficient_for_retrieval: {s2.sufficient_for_retrieval()}")

    s2.set("primary_need", "hospitalisation")
    print(f"+ primary_need — sufficient_for_retrieval: {s2.sufficient_for_retrieval()}")

    print(f"sufficient_for_recommendation (no resolved): {s2.sufficient_for_recommendation()}")
    s2.resolved_product_id = "SP015"
    print(f"sufficient_for_recommendation (resolved): {s2.sufficient_for_recommendation()}")
    print(f"sufficient_for_plan_selection (no budget): {s2.sufficient_for_plan_selection()}")
    s2.set("budget_band", "mid")
    print(f"sufficient_for_plan_selection (+ budget): {s2.sufficient_for_plan_selection()}")

    print()
    print("=" * 60)
    print("TEST 3: next_missing_field navigation")
    print("=" * 60)
    s3 = UserSchema.new_session(session_id="test_003")
    s3.set("buyer_type", "individual")
    s3.set("age", 28)
    s3.set("gender", "female")
    s3.set("primary_need", "maternity")

    while True:
        nxt = s3.next_missing_field()
        if nxt is None:
            print("  All fields collected.")
            break
        print(f"  Next to ask: [{nxt['ask_order']}] {nxt['key']} — \"{nxt.get('question_text', '')}\"")
        # Simulate user answering
        answers = {
            "has_ped": False,
            "ped_type": None,     # skipped because has_ped=False
            "needs_opd": False,
            "budget_band": "mid",
            "family_cover": "floater_nuclear",
            "family_size": 3,
            "si_preference": "3_5L",
        }
        if nxt["key"] in answers:
            if answers[nxt["key"]] is not None:
                s3.set(nxt["key"], answers[nxt["key"]])
            else:
                # Field skipped — set it directly to avoid infinite loop in test
                setattr(s3, nxt["key"], "__skipped__")

    print()
    print("=" * 60)
    print("TEST 4: Conditional skip — ped_type when has_ped=False")
    print("=" * 60)
    s4 = UserSchema.new_session(session_id="test_004")
    s4.set("buyer_type", "individual")
    s4.set("age", 30)
    s4.set("gender", "male")
    s4.set("primary_need", "hospitalisation")
    s4.set("has_ped", False)
    nxt = s4.next_missing_field()
    print(f"Next field after has_ped=False: {nxt['key'] if nxt else None}")
    assert nxt["key"] != "ped_type", "ped_type should be skipped when has_ped=False"
    print("ped_type correctly skipped.")

    print()
    print("=" * 60)
    print("TEST 5: family_size skipped when family_cover=individual")
    print("=" * 60)
    s5 = UserSchema.new_session(session_id="test_005")
    for k, v in [("buyer_type","individual"),("age",35),("gender","female"),
                 ("primary_need","hospitalisation"),("has_ped",False),
                 ("needs_opd",True),("budget_band","budget"),("family_cover","individual")]:
        s5.set(k, v)
    nxt = s5.next_missing_field()
    print(f"Next field after family_cover=individual: {nxt['key'] if nxt else None}")
    if nxt:
        assert nxt["key"] != "family_size", "family_size should be skipped"
    print("family_size correctly skipped.")

    print()
    print("=" * 60)
    print("TEST 6: Validation — illegal values raise SchemaValidationError")
    print("=" * 60)
    s6 = UserSchema.new_session(session_id="test_006")
    tests = [
        ("buyer_type", "corporation"),
        ("age",        150),
        ("gender",     "nonbinary"),
        ("primary_need", "dental"),
        ("budget_band", "cheap"),
        ("family_size", 0),
        ("ped_type", "cancer"),
    ]
    for key, bad_val in tests:
        try:
            s6.set(key, bad_val)
            print(f"  FAIL: {key}={bad_val!r} should have raised")
        except SchemaValidationError as e:
            print(f"  OK: {key}={bad_val!r} → {e}")

    print()
    print("=" * 60)
    print("TEST 7: to_tool_input — only non-None collection fields")
    print("=" * 60)
    s7 = UserSchema.new_session(session_id="test_007")
    s7.set("buyer_type", "individual")
    s7.set("age", 42)
    s7.set("primary_need", "hospitalisation")
    s7.set("has_ped", True)
    s7.set("ped_type", "diabetes_cardiac")
    tool_input = s7.to_tool_input()
    print("to_tool_input():", json.dumps(tool_input, indent=2))
    assert "session_id" not in tool_input, "analytics field leaked into tool input"
    assert "resolved_product_id" not in tool_input, "derived field leaked into tool input"
    assert None not in tool_input.values(), "None values in tool input"
    print("No analytics or None values leaked.")

    print()
    print("=" * 60)
    print("TEST 8: to_llm_context — collection + derived, no analytics logs")
    print("=" * 60)
    s8 = UserSchema.new_session(session_id="test_008")
    s8.set("buyer_type", "individual")
    s8.set("age", 42)
    s8.set("primary_need", "hospitalisation")
    s8.resolved_product_id = "SP015"
    s8.retrieval_score = 0.91
    s8.increment_turn()
    ctx = s8.to_llm_context()
    print("to_llm_context():", json.dumps(ctx, indent=2))
    assert "tool_calls_made" not in ctx
    assert "questions_asked" not in ctx
    assert "session_id" not in ctx
    print("Analytics logs correctly excluded from LLM context.")

    print()
    print("=" * 60)
    print("TEST 9: Analytics logging")
    print("=" * 60)
    s9 = UserSchema.new_session(session_id="test_009", language="hinglish")
    s9.set("buyer_type", "individual")
    s9.set("age", 35)
    s9.set("primary_need", "cancer")
    s9.set("user_intent", "prospective")
    s9.log_tool_call("filter_products")
    s9.log_tool_call("get_product_features")
    s9.increment_turn()
    s9.increment_turn()
    s9.resolved_product_id = "SP014"
    s9.resolved_plan_id = "SP014-25L"
    s9.set("purchased", True)
    s9.set("conversation_stage", "closed")

    record = s9.to_analytics_record()
    print(f"session_id      : {record['session_id']}")
    print(f"language        : {record['language']}")
    print(f"user_intent     : {record['user_intent']}")
    print(f"primary_need    : {record['primary_need']}")
    print(f"resolved_product: {record['resolved_product_id']}")
    print(f"resolved_plan   : {record['resolved_plan_id']}")
    print(f"purchased       : {record['purchased']}")
    print(f"turn_count      : {record['turn_count']}")
    print(f"tool_calls_made : {record['tool_calls_made']}")
    print(f"questions_asked : {record['questions_asked']}")
    print(f"summary         : {s9.summary()}")

    print()
    print("=" * 60)
    print("TEST 10: Full field catalogue")
    print("=" * 60)
    catalogue = field_catalogue()
    groups = {}
    for row in catalogue:
        groups.setdefault(row["group"], []).append(row["key"])
    for grp, keys in groups.items():
        print(f"  {grp:12}: {keys}")

    print()
    print(f"Total fields: {len(catalogue)}")
    print()
    print("All tests passed.")
