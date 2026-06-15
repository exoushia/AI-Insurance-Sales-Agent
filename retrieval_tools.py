"""
Swasthya Insurance Agent — Retrieval Tool Implementations
==========================================================
These are the four functions the orchestrator calls when the LLM issues a tool_call.
The LLM never executes these directly — it only sees the tool signatures (in attribute_glossary.tool_schema()).
The LLM receives the return value as a tool_result message and uses it to compose its response.

Import chain:
  attribute_glossary  →  field metadata, hard_filter keys, askable_fields
  policy_feature_registry  →  REGISTRY, all accessor functions

No LLM calls here. Every function is pure Python — deterministic, unit-testable,
auditable. The LLM decides WHICH tool to call and WHEN; the code decides WHAT to return.

Tool return contract
--------------------
Every tool returns a dict. The orchestrator JSON-serialises it and sends it back
to the LLM as a tool_result content block. Keys beginning with _ are stripped
before sending (they are orchestrator-internal logging hints).

User schema contract
--------------------
user_schema is a plain dict with a subset of the keys defined in attribute_glossary.
Missing keys mean "not yet collected" — they are NOT the same as False or None.
Functions must handle missing keys gracefully (use .get() with a sentinel default).

Scoring note
------------
Scores are floats in [0, 1]. 1.0 = perfect match on all hard filters + all boosts.
Hard filter failure = product removed from results entirely (not scored at 0).
Soft scores determine ranking among products that passed hard filters.
"""

from __future__ import annotations
import json
import os
import re
from typing import Any

# Data assets live in <package>/data/ (regulations + treatment costs) and
# <package>/data/policies/ (per-product wording text + metadata).
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_POLICIES_DIR = os.path.join(_DATA_DIR, "policies")

from policy_feature_registry import (
    REGISTRY,
    all_product_ids,
    get_product_for_llm,
    get_plans,
    get_waiting,
    get_score_boosts,
    get_feature,
)
from attribute_glossary import (
    askable_fields,
    hard_filters as glossary_hard_filter_keys,
)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Dynamic hard-filter rules — (user_schema_key, registry_key, match_logic).
#
# Philosophy: we recommend using the MAXIMUM information the user has given us.
# Every attribute the user has actually provided (present in the schema) is treated
# as a HARD eligibility constraint — we only surface products that genuinely fit what
# they told us. Attributes the user has NOT provided are simply skipped here and
# instead feed the soft ranking that extracts the top 3 among eligible products.
#
# This is deliberately SEPARATE from the sufficiency gate (MINIMUM_FOR_RETRIEVAL):
# sufficiency decides whether we may retrieve at all; these rules decide which
# products are eligible given whatever the user has shared so far.
#
# Only attributes with a clear product-level eligibility meaning live here. Plan-level
# attributes (si_preference, family_size) and non-exclusionary / fuzzy ones
# (ped_type, family_cover) are intentionally left to soft scoring — promoting them to
# hard filters would wrongly eliminate products the user could still buy.
#
# match_logic:
#   "in_list"         → user value must be in the product's list field
#   "range_min"       → user value (age) must be ≥ product field
#   "range_max"       → user value (age) must be ≤ product field
#   "gender"          → female_only products require gender == "female"
#   "require_if_true" → if the user value is True, the product field must be truthy
#   "budget_ceiling"  → product's cheapest served band must be ≤ the user's budget band
_DYNAMIC_FILTER_RULES: list[tuple[str, str, str]] = [
    ("buyer_type",   "buyer_types",   "in_list"),
    ("age",          "entry_age_min", "range_min"),
    ("age",          "entry_age_max", "range_max"),
    ("gender",       "gender",        "gender"),
    ("primary_need", "primary_needs", "in_list"),
    ("needs_opd",    "opd_covered",   "require_if_true"),
    ("budget_band",  "budget_bands",  "budget_ceiling"),
]

# Budget bands ordered cheapest → priciest, for budget_ceiling comparisons.
_BUDGET_BAND_ORDER = ["micro", "budget", "mid", "premium"]

# Eligibility gates that can NEVER be relaxed — a product legally/structurally
# cannot cover the user outside these. Everything else in _DYNAMIC_FILTER_RULES
# (buyer_type, primary_need, needs_opd, budget_band) is a PREFERENCE: if no
# product satisfies every preference, we relax preferences (not eligibility) and
# return the nearest matches rather than dead-ending on an empty candidate set.
_ELIGIBILITY_KEYS = {"age", "gender"}
_RELAXABLE_KEYS = [
    user_key for user_key, _reg, _logic in _DYNAMIC_FILTER_RULES
    if user_key not in _ELIGIBILITY_KEYS
]

# Sufficiency gate — the agent MUST know these before any retrieval runs.
# Distinct from the dynamic hard filters above: this is the minimum to start,
# not a statement about which attributes constrain eligibility.
MINIMUM_FOR_RETRIEVAL = {"buyer_type", "age", "primary_need"}

# If more than this many products survive hard filters, ask a probe question
MAX_CANDIDATES_WITHOUT_PROBE = 3

# Map si_preference enum → SI range in INR for plan matching
_SI_RANGE_MAP = {
    "1_2L":    (100_000,   200_000),
    "3_5L":    (300_000,   500_000),
    "10_25L":  (1_000_000, 2_500_000),
    "50L_plus":(5_000_000, 999_999_999),
}

# Budget band → annual premium range in INR
_BUDGET_RANGE_MAP = {
    "micro":   (0,      2_000),
    "budget":  (2_000,  10_000),
    "mid":     (10_000, 30_000),
    "premium": (30_000, 999_999_999),
}


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _passes_hard_filters(product_id: str, schema: dict) -> tuple[bool, list[str]]:
    """
    Check whether a product is eligible given everything the user has told us.

    Every attribute present in the schema with a rule in _DYNAMIC_FILTER_RULES acts
    as a hard constraint; attributes the user has not provided are skipped (they
    inform soft ranking instead). Returns (passes, reasons) where reasons is
    populated only on failure — used for probe-question selection.
    """
    p = REGISTRY[product_id]
    reasons = []

    for user_key, reg_key, logic in _DYNAMIC_FILTER_RULES:
        user_val = schema.get(user_key)
        if user_val is None:
            continue   # not provided yet — not a constraint

        if logic == "in_list":
            reg_val = p.get(reg_key, [])
            if user_val not in reg_val:
                reasons.append(f"{user_key}={user_val!r} not in product {reg_key}={reg_val}")
                return False, reasons

        elif logic == "range_min":
            reg_val = p.get(reg_key)
            if reg_val is not None and isinstance(user_val, int) and user_val < reg_val:
                reasons.append(f"age {user_val} below entry_age_min {reg_val}")
                return False, reasons

        elif logic == "range_max":
            reg_val = p.get(reg_key)
            if reg_val is not None and isinstance(user_val, int) and user_val > reg_val:
                reasons.append(f"age {user_val} above entry_age_max {reg_val}")
                return False, reasons

        elif logic == "gender":
            reg_val = p.get(reg_key, "any")
            if reg_val == "female_only" and user_val != "female":
                reasons.append(f"product is female_only, user gender={user_val!r}")
                return False, reasons

        elif logic == "require_if_true":
            # Only constrains when the user explicitly wants the feature (True).
            if user_val is True and not p.get(reg_key, False):
                reasons.append(f"user needs {user_key} but product {reg_key} is absent")
                return False, reasons

        elif logic == "budget_ceiling":
            # Budget is an upper bound: keep products whose cheapest served band is
            # at or below it; eliminate products that only serve pricier bands.
            bands = [b for b in p.get(reg_key, []) if b in _BUDGET_BAND_ORDER]
            if bands and user_val in _BUDGET_BAND_ORDER:
                cheapest = min(_BUDGET_BAND_ORDER.index(b) for b in bands)
                if cheapest > _BUDGET_BAND_ORDER.index(user_val):
                    reasons.append(
                        f"product cheapest band {_BUDGET_BAND_ORDER[cheapest]!r} "
                        f"exceeds user budget {user_val!r}"
                    )
                    return False, reasons

    return True, []


def _soft_score(product_id: str, schema: dict) -> tuple[float, list[str]]:
    """
    Compute a soft match score in [0, 1] for a product that passed hard filters.
    Returns (score, matched_fields) where matched_fields lists what contributed.
    """
    p = REGISTRY[product_id]
    score = 0.0
    matched = []

    # 0. Base relevance — the product directly serves the user's stated primary
    #    need. Guarantees on-need products never collapse to 0.0 just because the
    #    user hasn't supplied many secondary attributes yet (e.g. a plain
    #    "hospitalisation" request), so ranking stays meaningful from turn one.
    primary_need = schema.get("primary_need")
    if primary_need and primary_need in p.get("primary_needs", []):
        score += 0.25
        matched.append("serves_primary_need")

    # 1. OPD match
    if "needs_opd" in schema:
        needs = schema["needs_opd"]
        has = p.get("opd_covered", False)
        if needs == has:
            score += 0.15
            matched.append("opd_covered")
        elif needs and not has:
            score -= 0.10   # penalty for missing a hard need

    # 2. Budget band match
    if "budget_band" in schema:
        bb = schema["budget_band"]
        if bb in p.get("budget_bands", []):
            score += 0.10
            matched.append("budget_band")

    # 3. PED type match — strongest soft signal
    ped_type = schema.get("ped_type")
    if ped_type == "diabetes_cardiac":
        # SP015 has reduced 12-month wait for diabetes/cardiac
        wait = get_waiting(product_id)
        if wait.get("ped_months_diabetes_cardiac") == 12:
            score += 0.35
            matched.append("ped_12mo_diabetes_cardiac")
        elif wait.get("ped_months", 48) <= 12:
            score += 0.25
            matched.append("ped_short_wait")
    elif ped_type == "other_ped":
        wait = get_waiting(product_id)
        ped_wait = wait.get("ped_months", 48)
        if ped_wait <= 12:
            score += 0.15
            matched.append("ped_short_wait")
        elif ped_wait <= 36:
            score += 0.05
            matched.append("ped_standard_wait")

    # 4. Maternity need (inferred from primary_need or family_cover)
    if schema.get("primary_need") == "maternity":
        if p.get("maternity_covered"):
            score += 0.20
            matched.append("maternity_covered")

    # 5. Family cover alignment
    fc = schema.get("family_cover")
    if fc == "floater_joint" and p.get("cover_type") == "hybrid_ISI_pool":
        score += 0.15
        matched.append("cover_type_3gen")
    elif fc == "floater_nuclear" and p.get("cover_type") in ("floater", "group"):
        score += 0.05
        matched.append("cover_type_floater")

    # 6. Age-specific product boosts
    age = schema.get("age")
    if age is not None:
        boosts = get_score_boosts(product_id)
        if age <= 35 and "age_lte_35" in boosts:
            score += boosts["age_lte_35"]
            matched.append("age_lte_35_boost")
        if age >= 60 and "age_gte_60" in boosts:
            score += boosts["age_gte_60"]
            matched.append("age_gte_60_boost")

    # 7. Buyer-type boosts (group products)
    bt = schema.get("buyer_type")
    if bt:
        boosts = get_score_boosts(product_id)
        bt_key = f"buyer_type_{bt}"
        if bt_key in boosts:
            score += boosts[bt_key]
            matched.append(f"buyer_type_boost_{bt}")

    # 8. Budget micro / aadhaar
    if schema.get("buyer_type") in ("gig_worker",) or schema.get("budget_band") == "micro":
        boosts = get_score_boosts(product_id)
        if "budget_band_micro" in boosts:
            score += boosts["budget_band_micro"]
            matched.append("micro_segment_boost")

    # 9. Primary need direct boosts
    pn = schema.get("primary_need")
    if pn:
        boosts = get_score_boosts(product_id)
        pn_key = f"primary_need_{pn}"
        if pn_key in boosts:
            score += boosts[pn_key]
            matched.append(f"primary_need_boost_{pn}")

    return max(0.0, min(score, 1.0)), matched   # floor at 0, cap at 1


def _select_probe_question(candidates: list[str], schema: dict) -> str | None:
    """
    Given more than MAX_CANDIDATES_WITHOUT_PROBE surviving products, pick the
    next schema field to ask that would eliminate the most candidates.
    Returns the question_text from the glossary for the best discriminating field.
    """
    # Fields already collected
    collected = set(schema.keys())

    # Candidate fields in ask_order, skipping already-collected ones
    askable = [
        f for f in askable_fields()
        if f["key"] not in collected and f.get("ask_order") is not None
    ]

    best_field = None
    best_discrimination = 0

    for field in askable:
        key = field["key"]
        vv = field.get("valid_values")
        if not vv or not isinstance(vv, dict):
            continue

        # Count how many valid values would eliminate at least one candidate
        splits = 0
        for val in vv.keys():
            test_schema = {**schema, key: val}
            surviving = [
                pid for pid in candidates
                if _passes_hard_filters(pid, test_schema)[0]
            ]
            if len(surviving) < len(candidates):
                splits += 1

        if splits > best_discrimination:
            best_discrimination = splits
            best_field = field

    if best_field:
        return best_field.get("question_text")
    return None


def _highlight_fields(product_id: str, schema: dict) -> list[dict]:
    """
    Return the top 3 feature keys most relevant to this user's schema,
    with their values and why they matter. The LLM leads the recommendation
    with these rather than reciting the full feature list.

    Priority order mirrors the soft scoring logic:
      PED reduction > OPD coverage > maternity > family type > budget fit > digital features
    """
    p = REGISTRY[product_id]
    highlights = []

    # Rule 1: PED reduction (most important for buyers with PED)
    if schema.get("ped_type") == "diabetes_cardiac":
        wait = get_waiting(product_id)
        dm_wait = wait.get("ped_months_diabetes_cardiac")
        if dm_wait:
            highlights.append({
                "key":   "waiting.ped_months_diabetes_cardiac",
                "label": "Diabetes/cardiac PED waiting period",
                "value": f"{dm_wait} months",
                "why":   f"Only {dm_wait} months before your diabetes/cardiac condition is covered — "
                         f"vs 36–48 months on standard products.",
            })
    elif schema.get("has_ped"):
        wait = get_waiting(product_id)
        ped = wait.get("ped_months", 48)
        if ped <= 12:
            highlights.append({
                "key":   "waiting.ped_months",
                "label": "PED waiting period",
                "value": f"{ped} months",
                "why":   f"Short {ped}-month wait before your pre-existing condition is covered.",
            })

    # Rule 2: OPD coverage (if user asked for it)
    if schema.get("needs_opd") and p.get("opd_covered"):
        opd = p.get("opd_limit_inr")
        highlights.append({
            "key":   "opd_covered",
            "label": "OPD cover included",
            "value": f"₹{opd:,}/year" if opd else "Yes",
            "why":   "Covers regular doctor visits and medicines — not just hospital stays.",
        })

    # Rule 3: Maternity (if primary need or female + family)
    if (schema.get("primary_need") == "maternity" or
            (schema.get("gender") == "female" and schema.get("family_cover") != "individual")):
        if p.get("maternity_covered"):
            normal = p.get("maternity_normal_delivery_inr") or p.get("maternity_normal_inr")
            day1 = p.get("maternity_day1", False)
            highlights.append({
                "key":   "maternity_covered",
                "label": "Maternity covered",
                "value": f"₹{normal:,} normal delivery" if normal else "Yes",
                "why":   "Day 1 — no waiting period." if day1 else "9-month waiting period applies.",
            })

    # Rule 4: No co-pay (meaningful vs products that have it)
    if p.get("copay_type") == "none":
        highlights.append({
            "key":   "copay_type",
            "label": "Zero co-pay",
            "value": "None",
            "why":   "Insurer pays 100% of every admissible claim — no mandatory out-of-pocket share.",
        })

    # Rule 5: Restoration (important for families)
    if schema.get("family_cover") in ("floater_nuclear", "floater_joint"):
        rt = p.get("restoration_type", "none")
        if rt == "unlimited_different_illness":
            highlights.append({
                "key":   "restoration_type",
                "label": "Unlimited restoration",
                "value": "Unlimited (different illness)",
                "why":   "SI refills unlimited times if exhausted — protects the whole family mid-year.",
            })
        elif rt == "once_different_illness":
            highlights.append({
                "key":   "restoration_type",
                "label": "Restoration included",
                "value": "Once per year (different illness)",
                "why":   "If one claim exhausts the SI, it refills once for the rest of the year.",
            })

    # Rule 6: Wearable / AI (tech-savvy signal)
    if p.get("wearable_discount") and schema.get("age", 99) <= 40:
        max_d = p.get("wearable_discount_max_pct", 0)
        highlights.append({
            "key":   "wearable_discount",
            "label": "Wearable health reward",
            "value": f"Up to {int(max_d*100)}% premium discount",
            "why":   "Wear a fitness tracker, hit your activity targets, pay less at renewal.",
        })

    # Rule 7: International cover
    if schema.get("primary_need") == "international" and p.get("international_cover"):
        highlights.append({
            "key":   "international_cover",
            "label": "International cover",
            "value": f"{p.get('international_hospitals', 0):,} hospitals worldwide",
            "why":   "Covers hospitalisation outside India — cashless at empanelled hospitals.",
        })

    # Rule 8: Daily cash (self-employed / gig signal)
    if p.get("hospital_daily_cash") and schema.get("buyer_type") in ("individual", "gig_worker"):
        highlights.append({
            "key":   "hospital_daily_cash",
            "label": "Fixed daily cash",
            "value": "No bills needed — discharge summary only",
            "why":   "Pays a fixed amount per hospital day regardless of the actual bill — "
                     "compensates for lost income.",
        })

    return highlights[:3]   # return top 3 only


def _budget_fits_plan(plan: dict, schema: dict) -> bool:
    """Return True if a plan's premium is at or below the user's budget ceiling.

    Budget is treated as an upper bound: cheaper-than-band plans always fit; only
    plans priced above the top of the user's band are flagged as out of budget.
    """
    bb = schema.get("budget_band")
    if not bb or bb not in _BUDGET_RANGE_MAP:
        return True   # unknown budget — include all
    _lo, hi = _BUDGET_RANGE_MAP[bb]
    prem = plan.get("annual_premium_inr", 0)
    return prem <= hi


def _si_fits_preference(plan: dict, schema: dict) -> bool:
    """Return True if a plan's SI matches the user's si_preference."""
    pref = schema.get("si_preference")
    if not pref or pref not in _SI_RANGE_MAP:
        return True   # no preference stated — include all
    lo, hi = _SI_RANGE_MAP[pref]
    si = plan.get("si_inr")
    if si is None:
        return True   # plans without SI (e.g. SP011 daily cash, SP020) — always include
    return lo <= si <= hi


# ---------------------------------------------------------------------------
# TOOL 1: filter_products
# ---------------------------------------------------------------------------

def filter_products(user_schema: dict) -> dict:
    """
    Tool implementation for filter_products(user_schema).

    Step 1: Hard filters — eliminate products that are impossible for this user.
    Step 2: Soft score — rank surviving products by how well they match.
    Step 3: Probe — if >3 candidates, identify the best next question to narrow further.

    Returns a dict the orchestrator serialises and sends to the LLM as tool_result.
    """
    candidates = []
    eliminated = []

    for pid in all_product_ids():
        passes, reasons = _passes_hard_filters(pid, user_schema)
        if passes:
            score, matched = _soft_score(pid, user_schema)
            candidates.append({
                "product_id":    pid,
                "name":          REGISTRY[pid]["name"],
                "score":         round(score, 3),
                "matched_on":    matched,
            })
        else:
            eliminated.append({"product_id": pid, "reason": reasons[0] if reasons else "filter"})

    # Sort by score descending
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Graceful no-match fallback — if every product was hard-filtered out, the
    # user's PREFERENCES (e.g. a micro-budget gig_worker wanting daily_cash) have
    # no exact product. Rather than dead-ending on an empty list (which leaves the
    # agent with nothing to do but loop on discovery), relax preference filters
    # while keeping eligibility gates (age/gender) and return the nearest matches.
    no_exact_match = False
    relaxed_constraints: list[str] = []
    if not candidates:
        no_exact_match = True
        candidates, relaxed_constraints = _relaxed_candidates(user_schema)

    # Determine if we need a probe question. Skip when we already relaxed: the
    # caller should pivot to the nearest match, not ask yet another question.
    top_candidates = [c for c in candidates if c["score"] > 0.10]
    probe_question = None
    if not no_exact_match and len(top_candidates) > MAX_CANDIDATES_WITHOUT_PROBE:
        top_ids = [c["product_id"] for c in top_candidates]
        probe_question = _select_probe_question(top_ids, user_schema)

    # Determine missing schema fields that would improve retrieval
    collected = set(user_schema.keys())
    missing_fields = [
        {"key": f["key"], "label": f["label"], "ask_order": f["ask_order"]}
        for f in askable_fields()
        if f["key"] not in collected
    ]

    return {
        "candidates":      candidates[:5],        # top 5 to show the LLM
        "eliminated_count": len(eliminated),
        "total_products":  len(all_product_ids()),
        "probe_question":  probe_question,         # None if ≤3 candidates
        "no_exact_match":  no_exact_match,         # True → candidates are nearest, not exact
        "relaxed_constraints": relaxed_constraints,  # which preferences were relaxed to find them
        "missing_fields":  missing_fields,
        "_eliminated":     eliminated,             # orchestrator-internal, stripped before LLM
    }


def _relaxed_candidates(user_schema: dict) -> tuple[list[dict], list[str]]:
    """Fallback ranking when hard filters eliminate every product.

    Keeps only the eligibility gates (age, gender) as hard constraints and drops
    the user's preference constraints, then ranks the survivors with the FULL
    schema so preferences still steer ordering (the best near-miss floats up).
    Returns (candidates, relaxed_constraints).
    """
    eligibility_only = {k: v for k, v in user_schema.items() if k in _ELIGIBILITY_KEYS}
    relaxed = [k for k in _RELAXABLE_KEYS if user_schema.get(k) is not None]

    nearest = []
    for pid in all_product_ids():
        passes, _ = _passes_hard_filters(pid, eligibility_only)
        if not passes:
            continue
        score, matched = _soft_score(pid, user_schema)   # full schema → ranking
        nearest.append({
            "product_id": pid,
            "name":       REGISTRY[pid]["name"],
            "score":      round(score, 3),
            "matched_on": matched,
        })
    nearest.sort(key=lambda x: x["score"], reverse=True)
    return nearest, relaxed



# ---------------------------------------------------------------------------
# TOOL 2: get_product_features
# ---------------------------------------------------------------------------

def get_product_features(product_id: str, user_schema: dict) -> dict:
    """
    Tool implementation for get_product_features(product_id, user_schema).

    Returns the full clean product dict from the registry (internal keys stripped)
    plus highlight_fields — the top 3 attributes most relevant to this user.

    The LLM uses highlight_fields to decide what to say first in the recommendation.
    It uses the full product dict to answer follow-up questions accurately.
    """
    if product_id not in REGISTRY:
        return {
            "error": f"Unknown product_id {product_id!r}. "
                     f"Valid IDs: {all_product_ids()}"
        }

    product = get_product_for_llm(product_id)   # strips _score_boosts
    highlights = _highlight_fields(product_id, user_schema)

    # Add a human-readable why_this_product string
    top_match = highlights[0]["why"] if highlights else "Strong match for your profile."

    return {
        "product":          product,
        "highlight_fields": highlights,
        "why_this_product": top_match,
    }


# ---------------------------------------------------------------------------
# TOOL 3: get_plan_options
# ---------------------------------------------------------------------------

def get_plan_options(product_id: str, user_schema: dict) -> dict:
    """
    Tool implementation for get_plan_options(product_id, user_schema).

    Ranks the plans within a product by:
      1. Whether the plan SI matches si_preference
      2. Whether the premium fits budget_band
      3. Whether maternity limits are sufficient (if maternity is a need)

    Returns ranked plans with a fit_reason string for each.
    """
    if product_id not in REGISTRY:
        return {"error": f"Unknown product_id {product_id!r}"}

    plans = get_plans(product_id)
    if not plans:
        return {"error": f"No plans found for {product_id}"}

    ranked = []
    for plan in plans:
        score = 0.0
        reasons = []

        # SI preference match
        if _si_fits_preference(plan, user_schema):
            score += 0.40
            si = plan.get("si_inr")
            if si:
                reasons.append(f"SI ₹{si:,} matches your preference")

        # Budget match
        if _budget_fits_plan(plan, user_schema):
            score += 0.30
            reasons.append(f"₹{plan['annual_premium_inr']:,}/yr fits your budget")
        else:
            # Still include but flag it
            prem = plan["annual_premium_inr"]
            bb = user_schema.get("budget_band", "unknown")
            reasons.append(f"₹{prem:,}/yr is outside {bb} band")

        # Family size — flag if plan SI seems low for family
        fs = user_schema.get("family_size", 1)
        si = plan.get("si_inr")
        if si and fs and fs >= 4 and si < 500_000:
            score -= 0.10
            reasons.append(f"SI ₹{si:,} may be low for {fs} members")

        # Maternity limits (if relevant)
        if user_schema.get("primary_need") == "maternity":
            norm = plan.get("maternity_normal_inr") or plan.get("maternity_normal_delivery_inr")
            if norm and norm >= 50_000:
                score += 0.20
                reasons.append(f"Normal delivery: ₹{norm:,}")

        ranked.append({
            "plan_id":            plan["plan_id"],
            "label":              plan["label"],
            "si_inr":             plan.get("si_inr"),
            "annual_premium_inr": plan["annual_premium_inr"],
            "daily_premium_inr":  plan.get("daily_benefit_inr"),   # SP011
            "deductible_inr":     plan.get("deductible_inr"),       # SP010
            "opd_limit_inr":      plan.get("opd_limit_inr"),
            "maternity_normal_inr": plan.get("maternity_normal_inr"),
            "maternity_csection_inr": plan.get("maternity_csection_inr"),
            "score":              round(min(score, 1.0), 3),
            "fit_reason":         " | ".join(reasons) if reasons else "General fit",
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    # Recommended plan = highest score
    recommended_id = ranked[0]["plan_id"] if ranked else None

    return {
        "product_id":       product_id,
        "product_name":     REGISTRY[product_id]["name"],
        "plans":            ranked,
        "recommended_plan": recommended_id,
        "recommendation_note": (
            f"Recommended {recommended_id} based on your SI preference "
            f"({user_schema.get('si_preference', 'not stated')}) and "
            f"budget ({user_schema.get('budget_band', 'not stated')})."
        ),
    }


# ---------------------------------------------------------------------------
# TOOL 4: search_regulations
# ---------------------------------------------------------------------------

def search_regulations(query: str, corpus_path: str | None = None) -> dict:
    """
    Tool implementation for search_regulations(query).

    In production: embed query and retrieve top-k chunks from a vector index
    built over the IRDAI regulations JSON corpus.

    In this MVP: keyword-based BM25-style search over the pre-chunked JSON,
    returning the most relevant sections by term overlap. No external dependencies.

    corpus_path: path to the regulations JSON file.
                 Defaults to <package>/data/policy_regulations_rag_ready.json.
    """
    default_path = os.path.join(_DATA_DIR, "policy_regulations_rag_ready.json")
    path = corpus_path or default_path

    if not os.path.exists(path):
        return {
            "error": "Regulations corpus not found. Provide corpus_path or ensure "
                     "the IRDAI JSON file is at the default location.",
            "query": query,
        }

    with open(path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Tokenise query — simple lower-case word split, stop-words removed
    _STOP = {
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "of",
        "to", "for", "and", "or", "with", "what", "how", "do", "does",
        "can", "i", "my", "about", "under", "by", "from", "be", "as",
    }
    query_terms = {
        w.lower().strip("?.,;:")
        for w in query.split()
        if w.lower().strip("?.,;:") not in _STOP and len(w) > 2
    }

    # Score each section by term overlap (normalised)
    scored_chunks = []
    for doc in corpus.get("documents", []):
        doc_name = doc.get("document_name", "")
        doc_type = doc.get("document_type", "")
        for chapter in doc.get("chapters", []):
            chapter_title = chapter.get("chapter_title", "")
            for section in chapter.get("sections", []):
                content = section.get("content", "")
                if len(content) < 80:
                    continue   # skip nearly-empty sections

                # Score = number of distinct query terms found in content
                content_lower = content.lower()
                hit_terms = {t for t in query_terms if t in content_lower}
                if not hit_terms:
                    continue

                # Boost for section title match
                title = section.get("section_title", "")
                title_hits = {t for t in query_terms if t in title.lower()}
                boost = 1.0 + 0.5 * len(title_hits)

                raw_score = len(hit_terms) / max(len(query_terms), 1)
                final_score = raw_score * boost

                scored_chunks.append({
                    "score":           round(final_score, 3),
                    "document":        doc_name[:60],
                    "document_type":   doc_type,
                    "chapter":         chapter_title[:60],
                    "section_number":  section.get("section_number", ""),
                    "section_title":   title,
                    "retrieval_category": section.get("retrieval_category", ""),
                    "content":         content[:800],   # truncate to avoid context bloat
                    "content_length":  len(content),
                })

    # Sort by score, return top 4
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:4]

    if not top_chunks:
        return {
            "query":   query,
            "results": [],
            "note":    "No matching sections found. Try rephrasing with more specific terms.",
        }

    return {
        "query":   query,
        "results": top_chunks,
        "note":    f"Found {len(scored_chunks)} matching sections; returning top {len(top_chunks)}.",
    }

def search_policy_wording(product_id: str, query: str, base_dir: str | None = None) -> dict:
    """
    Deterministic document parser for deep policy Q&A (S3_POLICY_QA).
    Enforces isolation by reading ONLY the file linked to the resolved product.

    base_dir defaults to <package>/data/policies/.
    """
    base_dir = base_dir or _POLICIES_DIR
    filename = f"swasthya_{product_id}.txt"
    filepath = os.path.join(base_dir, filename)

    if not os.path.exists(filepath):
        return {
            "error": f"Policy wording text document not found at reference path: {filepath}",
            "product_id": product_id,
            "chunks": []
        }

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Split document cleanly into clauses using markdown-style line structures
    clauses = re.split(r'(?=Clause \d+\.\d+|\bSECTION \d+:)|\n={2,}\n', raw_text)
    
    # Process text queries safely using simple normalized keyword matching
    stop_words = {"a", "an", "the", "is", "are", "what", "does", "cover", "under", "in", "of", "policy"}
    query_terms = {w.lower().strip("?.,:") for w in query.split() if w.lower().strip("?.,:") not in stop_words and len(w) > 2}

    matched_clauses = []
    for clause in clauses:
        clause_clean = clause.strip()
        if len(clause_clean) < 50:
            continue
            
        clause_lower = clause_clean.lower()
        score = sum(1 for term in query_terms if term in clause_lower)
        
        if score > 0:
            matched_clauses.append({
                "score": score,
                "text": clause_clean[:1200]  # Hard bracket length to prevent context bloat
            })

    # Sort results by keyword hit density
    matched_clauses.sort(key=lambda x: x["score"], reverse=True)

    return {
        "product_id": product_id,
        "query": query,
        "chunks": matched_clauses[:3]  # Return top 3 target clauses only
    }

# ---------------------------------------------------------------------------
# ORCHESTRATOR DISPATCH
# ---------------------------------------------------------------------------

def dispatch_tool_call(tool_name: str, tool_input: dict) -> dict:
    """
    Single entry point for the orchestrator.
    Receives the tool_name and tool_input from the LLM's tool_use block,
    routes to the correct implementation, strips internal keys, and returns
    the result as a dict ready to be JSON-serialised into a tool_result block.

    Usage in orchestrator:
        for block in llm_response.content:
            if block.type == "tool_use":
                result = dispatch_tool_call(block.name, block.input)
                # append result as tool_result to messages
    """
    _INTERNAL_RESULT_KEYS = {"_eliminated"}

    dispatch = {
        "filter_products":    lambda i: filter_products(i["user_schema"]),
        "get_product_features": lambda i: get_product_features(
            i["product_id"], i.get("user_schema", {})
        ),
        "get_plan_options":   lambda i: get_plan_options(
            i["product_id"], i.get("user_schema", {})
        ),
        "search_regulations": lambda i: search_regulations(i["query"]),
    }

    if tool_name not in dispatch:
        return {"error": f"Unknown tool: {tool_name!r}. Valid tools: {list(dispatch.keys())}"}

    result = dispatch[tool_name](tool_input)

    # Strip orchestrator-internal keys before sending to LLM
    return {k: v for k, v in result.items() if k not in _INTERNAL_RESULT_KEYS}


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("TEST 1: filter_products — individual female, 34yr, maternity")
    print("=" * 60)
    schema_1 = {
        "buyer_type":   "individual",
        "age":          34,
        "gender":       "female",
        "primary_need": "maternity",
        "budget_band":  "mid",
    }
    r1 = filter_products(schema_1)
    print(f"Candidates ({len(r1['candidates'])}):")
    for c in r1["candidates"]:
        print(f"  {c['product_id']:6} score={c['score']:.3f}  matched={c['matched_on']}")
    print(f"Probe question: {r1['probe_question']}")
    print(f"Eliminated: {r1['eliminated_count']}")

    print()
    print("=" * 60)
    print("TEST 2: filter_products — 42yr diabetic, needs OPD, mid budget")
    print("=" * 60)
    schema_2 = {
        "buyer_type":   "individual",
        "age":          42,
        "gender":       "male",
        "primary_need": "hospitalisation",
        "has_ped":      True,
        "ped_type":     "diabetes_cardiac",
        "needs_opd":    True,
        "budget_band":  "mid",
    }
    r2 = filter_products(schema_2)
    print(f"Candidates ({len(r2['candidates'])}):")
    for c in r2["candidates"]:
        print(f"  {c['product_id']:6} score={c['score']:.3f}  matched={c['matched_on']}")
    print(f"Probe question: {r2['probe_question']}")

    print()
    print("=" * 60)
    print("TEST 3: get_product_features — SP015 for schema_2")
    print("=" * 60)
    r3 = get_product_features("SP015", schema_2)
    print("Highlights:")
    for h in r3["highlight_fields"]:
        print(f"  [{h['key']}] {h['label']} = {h['value']}")
        print(f"    → {h['why']}")
    print(f"Why this product: {r3['why_this_product']}")
    print(f"Product keys returned: {list(r3['product'].keys())[:10]} ...")

    print()
    print("=" * 60)
    print("TEST 4: get_plan_options — SP015, mid budget, 10_25L SI")
    print("=" * 60)
    schema_4 = {**schema_2, "si_preference": "10_25L", "family_cover": "floater_nuclear", "family_size": 4}
    r4 = get_plan_options("SP015", schema_4)
    print(f"Recommended: {r4['recommended_plan']}")
    for plan in r4["plans"]:
        print(f"  {plan['plan_id']:20} score={plan['score']:.3f}  "
              f"₹{plan['annual_premium_inr']:,}/yr  reason: {plan['fit_reason']}")

    print()
    print("=" * 60)
    print("TEST 5: filter_products — employer_large (should → SP016 only)")
    print("=" * 60)
    schema_5 = {"buyer_type": "employer_large", "primary_need": "hospitalisation"}
    r5 = filter_products(schema_5)
    print(f"Candidates: {[c['product_id'] for c in r5['candidates']]}")
    print(f"Eliminated: {r5['eliminated_count']}")

    print()
    print("=" * 60)
    print("TEST 6: filter_products — 65yr senior (should → SP006 strongly)")
    print("=" * 60)
    schema_6 = {
        "buyer_type":   "individual",
        "age":          65,
        "gender":       "male",
        "primary_need": "hospitalisation",
        "budget_band":  "mid",
    }
    r6 = filter_products(schema_6)
    print(f"Candidates ({len(r6['candidates'])}):")
    for c in r6["candidates"]:
        print(f"  {c['product_id']:6} score={c['score']:.3f}")

    print()
    print("=" * 60)
    print("TEST 7: search_regulations — PED waiting period")
    print("=" * 60)
    r7 = search_regulations("PED waiting period pre-existing disease")
    print(f"Query: {r7['query']}")
    print(f"Note: {r7['note']}")
    for chunk in r7.get("results", []):
        print(f"  score={chunk['score']:.3f}  [{chunk['document_type']}] "
              f"{chunk['section_title'][:50]}")
        print(f"    {chunk['content'][:120]}...")

    print()
    print("=" * 60)
    print("TEST 8: dispatch_tool_call (orchestrator entry point)")
    print("=" * 60)
    result = dispatch_tool_call("filter_products", {"user_schema": schema_2})
    print(f"Dispatched filter_products → {len(result['candidates'])} candidates")
    assert "_eliminated" not in result, "_eliminated should be stripped"
    print("Internal keys stripped correctly.")

    print()
    print("TEST 9: _highlight_fields NOT leaking _score_boosts to LLM")
    r9 = get_product_features("SP015", schema_2)
    assert "_score_boosts" not in r9["product"], "_score_boosts leaked to LLM"
    print("Registry internal keys correctly stripped.")

    print()
    print("All tests passed.")
