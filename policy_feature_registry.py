"""
Swasthya Health Insurance — Policy Feature Registry
====================================================
Single source of truth for all product attributes.
Used by: retrieval scorer, plan selector, feature presenter, RAG escalation checker.

Type conventions
----------------
bool        : feature present (True) or absent (False)
int         : concrete count or amount in INR (where unambiguous)
float       : ratio / percentage expressed as 0.0–1.0
str         : enum value — always from the defined set, never free-form
list[str]   : multi-value enum (e.g. multiple PED types accepted)
dict        : nested sub-attributes (plans, waiting periods, benefit tiers)
None        : attribute not applicable to this product (different from False)

Naming conventions
------------------
All keys snake_case. Monetary values in INR (int). Percentages as float (0.25 = 25%).
Plan tiers keyed by plan_id string matching what appears in metadata JSON.
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# TYPE ALIASES (for readability — not enforced at runtime in this MVP)
# ---------------------------------------------------------------------------
INR       = int          # Indian Rupees
Pct       = float        # 0.0–1.0
Months    = int          # waiting period duration
Days      = int          # pre/post hospitalisation window
BoolOrNA  = bool | None  # True = present, False = absent, None = not applicable


# ---------------------------------------------------------------------------
# ENUM SETS  (document every valid value for each enum field)
# ---------------------------------------------------------------------------

BUYER_TYPES      = {"individual", "employer_large", "employer_sme", "gig_worker"}
GENDERS          = {"any", "female_only"}
SEGMENTS         = {
    "entry", "mid", "premium", "standardised", "micro",
    "senior", "maternity", "women", "youth", "supplement",
    "critical_illness", "cancer", "accident", "chronic_disease",
    "group_large", "group_sme", "ai_tech", "multi_gen"
}
BUDGET_BANDS     = {"micro", "budget", "mid", "premium"}
PRIMARY_NEEDS    = {
    "hospitalisation", "critical_illness", "cancer",
    "accident", "maternity", "top_up", "international", "daily_cash"
}
COVER_TYPES      = {"individual", "floater", "group", "hybrid_ISI_pool"}
NCB_TYPES        = {"si_increase", "premium_discount", "none"}
RESTORATION_TYPES= {"none", "once_different_illness", "unlimited_different_illness", "once_any_illness"}
COPAY_TYPES      = {"none", "flat_pct", "age_based_pct"}


# ---------------------------------------------------------------------------
# HELPER: plan dict constructor  (keeps plan definitions DRY)
# ---------------------------------------------------------------------------

def plan(
    plan_id: str,
    label: str,
    si_inr: INR | None,
    annual_premium_inr: INR,
    *,
    sample_profile: str,
    deductible_inr: INR | None = None,
    daily_benefit_inr: INR | None = None,
    opd_limit_inr: INR | None = None,
    maternity_normal_inr: INR | None = None,
    maternity_csection_inr: INR | None = None,
    room_rent_cap_pct_si: Pct | None = None,
    room_rent_cap_inr_day: INR | None = None,
) -> dict:
    return {k: v for k, v in {
        "plan_id":                plan_id,
        "label":                  label,
        "si_inr":                 si_inr,
        "annual_premium_inr":     annual_premium_inr,
        "sample_profile":         sample_profile,
        "deductible_inr":         deductible_inr,
        "daily_benefit_inr":      daily_benefit_inr,
        "opd_limit_inr":          opd_limit_inr,
        "maternity_normal_inr":   maternity_normal_inr,
        "maternity_csection_inr": maternity_csection_inr,
        "room_rent_cap_pct_si":   room_rent_cap_pct_si,
        "room_rent_cap_inr_day":  room_rent_cap_inr_day,
    }.items() if v is not None}


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

REGISTRY: dict[str, dict[str, Any]] = {

    # ════════════════════════════════════════════════════════════════════════
    "SP001": {
        # ── identity ──────────────────────────────────────────────────────
        "product_id":       "SP001",
        "name":             "Swasthya Protect",
        "uin":              "SWIHLIP24001V012024",
        "segment":          "entry",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["budget", "mid"],

        # ── eligibility ───────────────────────────────────────────────────
        "entry_age_min":    18,
        "entry_age_max":    65,
        "lifelong_renewal": True,
        "medical_exam_required": False,   # waived ≤55 yrs / ≤10L SI

        # ── core indemnity ────────────────────────────────────────────────
        "cover_type":              "floater",          # also sold individual
        "no_room_rent_cap":        False,              # cap applies at lower SI tiers
        "room_rent_cap_note":      "2% SI/day at 3L; uncapped at 10L",
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     False,
        "pre_hosp_days":           60,
        "post_hosp_days":          90,
        "daycare_procedures":      541,
        "domiciliary_covered":     False,
        "domiciliary_limit_inr":   None,
        "road_ambulance_inr":      3000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      12000,

        # ── restoration ───────────────────────────────────────────────────
        "restoration_type":        "once_different_illness",
        "restoration_pct":         1.0,

        # ── waiting periods ───────────────────────────────────────────────
        "waiting": {
            "initial_days":              30,
            "ped_months":                48,
            "specific_disease_months":   24,
            "accident_exempt":           True,
        },

        # ── NCB ───────────────────────────────────────────────────────────
        "ncb": {
            "type":              "si_increase",
            "pct_per_year":      0.10,
            "max_pct":           0.50,
        },

        # ── OPD / outpatient ──────────────────────────────────────────────
        "opd_covered":             False,
        "opd_limit_inr":           None,
        "telemedicine":            False,

        # ── benefit / fixed-pay ───────────────────────────────────────────
        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               False,
        "mental_health_opd_limit_inr":     None,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        # ── admin / digital ───────────────────────────────────────────────
        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        # ── plans ─────────────────────────────────────────────────────────
        "plans": [
            plan("SP001-3L",  "Protect 3L",  si_inr=300_000,   annual_premium_inr=4_400,
                 sample_profile="25yr individual", room_rent_cap_pct_si=0.02),
            plan("SP001-5L",  "Protect 5L",  si_inr=500_000,   annual_premium_inr=6_200,
                 sample_profile="25yr individual", room_rent_cap_pct_si=0.02),
            plan("SP001-10L", "Protect 10L", si_inr=1_000_000, annual_premium_inr=26_500,
                 sample_profile="35yr family-4"),
        ],

        # ── retrieval scoring weights (used by scorer) ────────────────────
        "_score_boosts": {
            "first_time_buyer": 0.15,
            "budget_band_budget": 0.10,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP002": {
        "product_id":       "SP002",
        "name":             "Swasthya Protect Plus",
        "uin":              "SWIHLIP24002V012024",
        "segment":          "mid",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["mid"],

        "entry_age_min":    18,
        "entry_age_max":    65,
        "lifelong_renewal": True,
        "medical_exam_required": False,

        "cover_type":              "floater",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     True,
        "pre_hosp_days":           60,
        "post_hosp_days":          180,
        "daycare_procedures":      541,
        "domiciliary_covered":     True,
        "domiciliary_limit_inr":   50_000,
        "road_ambulance_inr":      5_000,
        "air_ambulance_covered":   True,
        "air_ambulance_inr":       500_000,
        "international_cover":     False,
        "ayush_pct_si":            0.50,
        "cashless_hospitals":      14_000,

        "restoration_type":        "unlimited_different_illness",
        "restoration_pct":         1.0,

        "waiting": {
            "initial_days":            30,
            "ped_months":              36,
            "specific_disease_months": 24,
            "accident_exempt":         True,
        },

        "ncb": {
            "type":         "si_increase",
            "pct_per_year": 0.10,
            "max_pct":      0.50,
        },

        "opd_covered":             True,
        "telemedicine":            True,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               True,
        "mental_health_opd_limit_inr":     10_000,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             0.968,

        "plans": [
            plan("SP002-5L",  "Protect Plus 5L",  si_inr=500_000,   annual_premium_inr=7_800,
                 sample_profile="25yr individual", opd_limit_inr=15_000),
            plan("SP002-10L", "Protect Plus 10L", si_inr=1_000_000, annual_premium_inr=29_800,
                 sample_profile="35yr family-4",  opd_limit_inr=20_000),
            plan("SP002-25L", "Protect Plus 25L", si_inr=2_500_000, annual_premium_inr=48_000,
                 sample_profile="35yr individual", opd_limit_inr=25_000),
            plan("SP002-50L", "Protect Plus 50L", si_inr=5_000_000, annual_premium_inr=82_000,
                 sample_profile="45yr individual", opd_limit_inr=25_000),
        ],

        "_score_boosts": {
            "needs_opd": 0.15,
            "budget_band_mid": 0.10,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP003": {
        "product_id":       "SP003",
        "name":             "Swasthya Protect Global",
        "uin":              "SWIHLIP24003V012024",
        "segment":          "premium",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation", "international"],
        "budget_bands":     ["premium"],

        "entry_age_min":    18,
        "entry_age_max":    65,
        "lifelong_renewal": True,
        "medical_exam_required": False,

        "cover_type":              "floater",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     True,
        "pre_hosp_days":           60,
        "post_hosp_days":          180,
        "daycare_procedures":      541,
        "domiciliary_covered":     True,
        "domiciliary_limit_inr":   100_000,
        "road_ambulance_inr":      5_000,
        "air_ambulance_covered":   True,
        "air_ambulance_inr":       1_000_000,   # per hospitalisation
        "international_cover":     True,
        "international_hospitals": 12_000,
        "international_countries": ["USA","UK","Singapore","Thailand","UAE","Germany","Australia"],
        "claims_currency":         "INR_at_SBI_TT",
        "ayush_pct_si":            1.00,
        "cashless_hospitals":      14_000,

        "restoration_type":        "unlimited_different_illness",
        "restoration_pct":         1.0,

        "waiting": {
            "initial_days":            30,
            "ped_months":              36,
            "specific_disease_months": 24,
            "accident_exempt":         True,
        },

        "ncb": {
            "type":         "si_increase",
            "pct_per_year": 0.10,
            "max_pct":      0.50,
        },

        "opd_covered":             True,
        "opd_limit_inr":           50_000,
        "telemedicine":            True,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               True,
        "mental_health_opd_limit_inr":     20_000,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             0.945,

        "plans": [
            plan("SP003-50L", "Global 50L",  si_inr=5_000_000,  annual_premium_inr=32_000,
                 sample_profile="25yr individual"),
            plan("SP003-1Cr", "Global 1Cr",  si_inr=10_000_000, annual_premium_inr=58_000,
                 sample_profile="35yr individual"),
            plan("SP003-2Cr", "Global 2Cr",  si_inr=20_000_000, annual_premium_inr=98_000,
                 sample_profile="35yr family-4"),
        ],

        "_score_boosts": {
            "primary_need_international": 0.40,  # near-deterministic pointer
            "budget_band_premium": 0.10,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP004": {
        "product_id":       "SP004",
        "name":             "Swasthya Arogya Sanjeevani",
        "uin":              "SWIHLIP24004V012024",
        "segment":          "standardised",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["budget"],

        "entry_age_min":    18,
        "entry_age_max":    65,
        "lifelong_renewal": True,
        "medical_exam_required": False,
        "regulatory_note":  "IRDAI standardised product — terms identical across all insurers",

        "cover_type":              "floater",
        "no_room_rent_cap":        False,
        "room_rent_cap_pct_si":    0.02,   # 2% SI/day general ward; 5% ICU
        "icu_cap_pct_si":          0.05,
        "icu_covered":             True,
        "no_disease_sublimits":    False,
        "copay_type":              "flat_pct",
        "copay_pct":               0.05,
        "consumables_covered":     False,
        "pre_hosp_days":           30,
        "post_hosp_days":          60,
        "daycare_procedures":      541,
        "domiciliary_covered":     False,
        "domiciliary_limit_inr":   None,
        "road_ambulance_inr":      2_500,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      12_000,

        "restoration_type":        "none",
        "restoration_pct":         None,

        "waiting": {
            "initial_days":            30,
            "ped_months":              48,
            "specific_disease_months": 24,
            "accident_exempt":         True,
        },

        "ncb": {
            "type":         "si_increase",
            "pct_per_year": 0.05,
            "max_pct":      0.50,
        },

        "opd_covered":             False,
        "opd_limit_inr":           None,
        "telemedicine":            False,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               False,
        "mental_health_opd_limit_inr":     None,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        "plans": [
            plan("SP004-1L", "Sanjeevani 1L", si_inr=100_000,  annual_premium_inr=1_800,
                 sample_profile="25yr individual"),
            plan("SP004-3L", "Sanjeevani 3L", si_inr=300_000,  annual_premium_inr=3_400,
                 sample_profile="25yr individual"),
            plan("SP004-5L", "Sanjeevani 5L", si_inr=500_000,  annual_premium_inr=5_100,
                 sample_profile="25yr individual"),
        ],

        "_score_boosts": {
            "wants_irdai_standard": 0.20,
            "price_sensitive": 0.10,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP005": {
        "product_id":       "SP005",
        "name":             "Swasthya Micro Bima",
        "uin":              "SWIHLIP24005V012024",
        "segment":          "micro",
        "buyer_types":      ["individual", "gig_worker"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["micro"],

        "entry_age_min":    18,
        "entry_age_max":    60,
        "lifelong_renewal": False,
        "medical_exam_required": False,
        "regulatory_framework": "IRDAI (Micro-Insurance) Regulations, 2005",

        "cover_type":              "individual",
        "no_room_rent_cap":        False,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     False,
        "pre_hosp_days":           30,
        "post_hosp_days":          60,
        "daycare_procedures":      100,            # simplified list
        "domiciliary_covered":     False,
        "domiciliary_limit_inr":   None,
        "road_ambulance_inr":      1_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      5_000,
        "government_hospitals_accepted": True,

        "restoration_type":        "none",
        "restoration_pct":         None,

        "waiting": {
            "initial_days":            30,
            "ped_months":              12,
            "specific_disease_months": 12,
            "accident_exempt":         True,
        },

        "ncb": {"type": "none", "pct_per_year": None, "max_pct": None},

        "opd_covered":             False,
        "opd_limit_inr":           None,
        "telemedicine":            False,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         False,
        "mental_health_opd":               False,
        "mental_health_opd_limit_inr":     None,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         True,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,
        "languages_supported":             11,
        "enrollment_methods":              ["app", "whatsapp", "ivr", "csc"],

        "plans": [
            plan("SP005-1L", "Micro 1L", si_inr=100_000, annual_premium_inr=799,
                 sample_profile="any individual"),
            plan("SP005-2L", "Micro 2L", si_inr=200_000, annual_premium_inr=1_199,
                 sample_profile="any individual"),
        ],

        "_score_boosts": {
            "budget_band_micro": 0.40,
            "aadhaar_only_enrollment": 0.20,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP006": {
        "product_id":       "SP006",
        "name":             "Swasthya Senior Shield",
        "uin":              "SWIHLIP24006V012024",
        "segment":          "senior",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["mid", "premium"],

        "entry_age_min":    60,
        "entry_age_max":    80,
        "lifelong_renewal": True,
        "medical_exam_required": True,

        "cover_type":              "floater",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "age_based_pct",
        "copay_schedule":          {"60_69": 0.10, "70_80": 0.20},
        "copay_pct":               None,           # see copay_schedule
        "consumables_covered":     False,
        "pre_hosp_days":           30,
        "post_hosp_days":          90,
        "daycare_procedures":      541,
        "domiciliary_covered":     True,
        "domiciliary_limit_inr":   50_000,
        "road_ambulance_inr":      3_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.50,
        "cashless_hospitals":      12_000,

        "restoration_type":        "once_different_illness",
        "restoration_pct":         1.0,

        "waiting": {
            "initial_days":            30,
            "ped_months":              12,           # reduced for seniors (IRDAI 2024)
            "specific_disease_months": 12,
            "accident_exempt":         True,
        },

        "ncb": {
            "type":         "si_increase",
            "pct_per_year": 0.05,
            "max_pct":      0.25,
        },

        "opd_covered":             False,
        "opd_limit_inr":           None,
        "telemedicine":            True,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               False,
        "mental_health_opd_limit_inr":     None,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            True,
        "cdmp_conditions":                 ["diabetes","hypertension","COPD","CKD","CVD"],
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           True,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        "plans": [
            plan("SP006-3L",  "Senior 3L",  si_inr=300_000,   annual_premium_inr=18_000,
                 sample_profile="60-64yr individual"),
            plan("SP006-5L",  "Senior 5L",  si_inr=500_000,   annual_premium_inr=28_500,
                 sample_profile="60-64yr individual"),
            plan("SP006-10L", "Senior 10L", si_inr=1_000_000, annual_premium_inr=52_000,
                 sample_profile="60-64yr individual"),
            plan("SP006-25L", "Senior 25L", si_inr=2_500_000, annual_premium_inr=98_000,
                 sample_profile="60-64yr individual"),
        ],

        "_score_boosts": {
            "age_gte_60": 0.50,   # hard near-requirement
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP007": {
        "product_id":       "SP007",
        "name":             "Swasthya Maternity Suraksha",
        "uin":              "SWIHLIP24007V012024",
        "segment":          "maternity",
        "buyer_types":      ["individual"],
        "gender":           "female_only",
        "primary_needs":    ["maternity"],
        "budget_bands":     ["mid"],

        "entry_age_min":    21,
        "entry_age_max":    45,
        "lifelong_renewal": False,
        "medical_exam_required": False,

        "cover_type":              "individual",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     True,
        "pre_hosp_days":           60,
        "post_hosp_days":          60,
        "daycare_procedures":      541,
        "domiciliary_covered":     False,
        "domiciliary_limit_inr":   None,
        "road_ambulance_inr":      3_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.0,
        "cashless_hospitals":      14_000,

        "restoration_type":        "none",
        "restoration_pct":         None,

        "waiting": {
            "initial_days":            30,
            "maternity_months":        9,
            "ped_months":              36,
            "specific_disease_months": 24,
            "accident_exempt":         True,
        },

        "ncb": {"type": "none", "pct_per_year": None, "max_pct": None},

        "opd_covered":             True,   # pre/post-natal OPD
        "prenatal_opd_inr":        10_000,
        "postnatal_opd_inr":       10_000,
        "telemedicine":            False,

        "maternity_covered":               True,
        "maternity_normal_delivery_inr":   50_000,
        "maternity_csection_inr":          75_000,
        "newborn_covered":                 True,
        "newborn_cover_days":              90,
        "nicu_limit_inr":                  100_000,
        "newborn_vaccines":                6,
        "ivf_covered":                     False,    # Plan A; see plans
        "mental_health_inpatient":         False,
        "mental_health_opd":               False,
        "mental_health_opd_limit_inr":     None,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        "plans": [
            plan("SP007-A", "Maternity Plan A", si_inr=None, annual_premium_inr=13_800,
                 sample_profile="26-30yr female",
                 maternity_normal_inr=50_000, maternity_csection_inr=75_000),
            plan("SP007-B", "Maternity Plan B (IVF)", si_inr=None, annual_premium_inr=20_200,
                 sample_profile="26-30yr female",
                 maternity_normal_inr=50_000, maternity_csection_inr=75_000),
            # Plan B adds IVF — captured in plan-level flag below
        ],
        "plan_B_ivf_limit_inr":    200_000,
        "plan_B_ivf_max_attempts": 3,

        "_score_boosts": {
            "primary_need_maternity": 0.50,
            "gender_female": 0.10,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP008": {
        "product_id":       "SP008",
        "name":             "Swasthya Women Wellness",
        "uin":              "SWIHLIP24008V012024",
        "segment":          "women",
        "buyer_types":      ["individual"],
        "gender":           "female_only",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["mid"],

        "entry_age_min":    25,
        "entry_age_max":    55,
        "lifelong_renewal": True,
        "medical_exam_required": False,

        "cover_type":              "individual",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     True,
        "pre_hosp_days":           60,
        "post_hosp_days":          90,
        "daycare_procedures":      541,
        "domiciliary_covered":     False,
        "domiciliary_limit_inr":   None,
        "road_ambulance_inr":      5_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      14_000,

        "restoration_type":        "once_different_illness",
        "restoration_pct":         1.0,

        "waiting": {
            "initial_days":            30,
            "ped_months":              36,
            "women_cancers_day1":      True,   # breast, cervical, ovarian, uterine
            "specific_disease_months": 24,
            "accident_exempt":         True,
        },

        "ncb": {"type": "si_increase", "pct_per_year": 0.10, "max_pct": 0.50},

        "opd_covered":                    True,
        "opd_limit_inr":                  None,    # see plan-specific
        "cancer_screening_inr_year":      15_000,
        "hrt_covered_inr_year":           12_000,
        "pcos_opd_inr_year":              10_000,
        "telemedicine":                   True,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               True,
        "mental_health_opd_limit_inr":     50_000,
        "mental_health_sessions":          12,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      True,
        "cancer_screening_voucher":        True,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           True,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        "plans": [
            plan("SP008-5L",  "Women Wellness 5L",  si_inr=500_000,   annual_premium_inr=13_200,
                 sample_profile="30-34yr female", opd_limit_inr=25_000),
            plan("SP008-10L", "Women Wellness 10L", si_inr=1_000_000, annual_premium_inr=24_500,
                 sample_profile="30-34yr female", opd_limit_inr=40_000),
        ],

        "_score_boosts": {
            "gender_female": 0.15,
            "women_specific_cancer_day1": 0.15,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP009": {
        "product_id":       "SP009",
        "name":             "Swasthya Young Star",
        "uin":              "SWIHLIP24009V012024",
        "segment":          "youth",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["budget"],

        "entry_age_min":    18,
        "entry_age_max":    35,
        "lifelong_renewal": False,
        "transition_at_age": 35,
        "transition_to":    ["SP001", "SP002"],
        "medical_exam_required": False,

        "cover_type":              "individual",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     False,
        "pre_hosp_days":           30,
        "post_hosp_days":          90,
        "daycare_procedures":      541,
        "domiciliary_covered":     False,
        "domiciliary_limit_inr":   None,
        "road_ambulance_inr":      3_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      12_000,
        "sports_injuries_day1":    True,

        "restoration_type":        "once_different_illness",
        "restoration_pct":         1.0,

        "waiting": {
            "initial_days":            30,
            "ped_months":              24,
            "specific_disease_months": 12,
            "accident_exempt":         True,
        },

        "ncb": {"type": "si_increase", "pct_per_year": 0.10, "max_pct": 0.50},

        "opd_covered":             True,
        "telemedicine":            True,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               True,
        "mental_health_opd_limit_inr":     10_000,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               True,
        "wearable_discount_max_pct":       0.10,
        "wearable_step_target_5k":         0.05,
        "wearable_step_target_10k":        0.10,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        "plans": [
            plan("SP009-3L", "Young Star 3L", si_inr=300_000, annual_premium_inr=3_400,
                 sample_profile="22-25yr individual", opd_limit_inr=5_000),
            plan("SP009-5L", "Young Star 5L", si_inr=500_000, annual_premium_inr=5_800,
                 sample_profile="22-25yr individual", opd_limit_inr=8_000),
        ],

        "_score_boosts": {
            "age_lte_35": 0.30,
            "budget_band_budget": 0.10,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP010": {
        "product_id":       "SP010",
        "name":             "Swasthya Super Top-Up",
        "uin":              "SWIHLIP24010V012024",
        "segment":          "supplement",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["top_up"],
        "budget_bands":     ["budget", "mid"],

        "entry_age_min":    18,
        "entry_age_max":    65,
        "lifelong_renewal": True,
        "medical_exam_required": False,
        "aggregate_deductible":  True,   # NOT per-event — key differentiator
        "primary_policy_required": False,

        "cover_type":              "floater",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     False,
        "pre_hosp_days":           60,
        "post_hosp_days":          90,
        "daycare_procedures":      541,
        "domiciliary_covered":     True,
        "domiciliary_limit_inr":   50_000,
        "road_ambulance_inr":      5_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      12_000,

        "restoration_type":        "none",   # deductible resets each year, not restoration
        "restoration_pct":         None,

        "waiting": {
            "initial_days":            30,
            "ped_months":              36,
            "specific_disease_months": 24,
            "accident_exempt":         True,
        },

        "ncb": {
            "type":         "premium_discount",
            "pct_per_year": 0.05,
            "max_pct":      0.30,
        },

        "opd_covered":             False,
        "opd_limit_inr":           None,
        "telemedicine":            False,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               False,
        "mental_health_opd_limit_inr":     None,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        # Plans: (product, deductible) combinations
        "plans": [
            plan("SP010-Silver-2L", "STU Silver, ₹2L ded.", si_inr=2_500_000,
                 annual_premium_inr=6_800, sample_profile="Age 35",
                 deductible_inr=200_000),
            plan("SP010-Silver-5L", "STU Silver, ₹5L ded.", si_inr=2_500_000,
                 annual_premium_inr=3_600, sample_profile="Age 35",
                 deductible_inr=500_000),
            plan("SP010-Gold-5L",   "STU Gold, ₹5L ded.",  si_inr=5_000_000,
                 annual_premium_inr=6_700, sample_profile="Age 35",
                 deductible_inr=500_000),
            plan("SP010-Gold-10L",  "STU Gold, ₹10L ded.", si_inr=5_000_000,
                 annual_premium_inr=4_100, sample_profile="Age 35",
                 deductible_inr=1_000_000),
            plan("SP010-Plat-10L",  "STU Plat, ₹10L ded.", si_inr=10_000_000,
                 annual_premium_inr=8_200, sample_profile="Age 35",
                 deductible_inr=1_000_000),
            plan("SP010-Plat-25L",  "STU Plat, ₹25L ded.", si_inr=10_000_000,
                 annual_premium_inr=3_400, sample_profile="Age 35",
                 deductible_inr=2_500_000),
        ],

        "_score_boosts": {
            "primary_need_top_up": 0.50,
            "has_existing_policy": 0.15,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP011": {
        "product_id":       "SP011",
        "name":             "Swasthya Hospital Daily Cash",
        "uin":              "SWIHLIP24011V012024",
        "segment":          "supplement",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["daily_cash"],
        "budget_bands":     ["budget"],

        "entry_age_min":    18,
        "entry_age_max":    65,
        "lifelong_renewal": True,
        "medical_exam_required": False,

        "cover_type":              "individual",
        "no_room_rent_cap":        None,    # N/A — fixed benefit, no indemnity
        "icu_covered":             True,    # 2× daily rate for ICU
        "no_disease_sublimits":    None,    # N/A
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     None,    # N/A
        "pre_hosp_days":           None,    # N/A
        "post_hosp_days":          None,    # N/A
        "daycare_procedures":      None,    # 50% of daily benefit for 6+hr
        "daycare_pct_daily":       0.50,
        "domiciliary_covered":     False,
        "road_ambulance_inr":      None,
        "air_ambulance_covered":   False,
        "international_cover":     False,
        "ayush_pct_si":            None,
        "cashless_hospitals":      0,       # no cashless — fixed benefit, discharge summary only
        "bills_required":          False,   # KEY differentiator
        "simultaneous_claim_allowed": True, # can claim alongside any indemnity policy

        "restoration_type":        None,
        "restoration_pct":         None,

        "waiting": {
            "initial_days":            30,
            "ped_months":              24,
            "specific_disease_months": 12,
            "accident_exempt":         True,
        },

        "ncb": {
            "type":         "premium_discount",
            "pct_per_year": 0.05,
            "max_pct":      0.20,
        },

        "opd_covered":             False,
        "telemedicine":            False,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               False,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               True,   # implicit — daily cash replaces income
        "hospital_daily_cash":             True,   # core feature
        "icu_benefit_multiplier":          2,
        "convalescence_benefit":           True,
        "convalescence_11_20_days_x":      5,
        "convalescence_21plus_days_x":     10,
        "settlement_days":                 7,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        "plans": [
            plan("SP011-Bronze",   "Daily Cash Bronze",   si_inr=None, annual_premium_inr=1_500,
                 sample_profile="Age 35", daily_benefit_inr=1_000),
            plan("SP011-Silver",   "Daily Cash Silver",   si_inr=None, annual_premium_inr=2_800,
                 sample_profile="Age 35", daily_benefit_inr=2_000),
            plan("SP011-Gold",     "Daily Cash Gold",     si_inr=None, annual_premium_inr=4_600,
                 sample_profile="Age 35", daily_benefit_inr=3_000),
            plan("SP011-Platinum", "Daily Cash Platinum", si_inr=None, annual_premium_inr=7_200,
                 sample_profile="Age 35", daily_benefit_inr=5_000),
        ],

        "_score_boosts": {
            "primary_need_daily_cash": 0.50,
            "self_employed": 0.15,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP012": {
        "product_id":       "SP012",
        "name":             "Swasthya Personal Accident Shield",
        "uin":              "SWIHLIP24012V012024",
        "segment":          "accident",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["accident"],
        "budget_bands":     ["budget"],

        "entry_age_min":    18,
        "entry_age_max":    60,
        "lifelong_renewal": True,
        "medical_exam_required": False,

        "cover_type":              "individual",
        "accident_only":           True,    # no illness cover
        "worldwide_cover":         True,
        "no_room_rent_cap":        None,    # N/A
        "icu_covered":             None,    # N/A — not indemnity
        "no_disease_sublimits":    None,    # N/A
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     None,
        "pre_hosp_days":           None,
        "post_hosp_days":          None,
        "daycare_procedures":      None,
        "domiciliary_covered":     False,
        "road_ambulance_inr":      None,
        "air_ambulance_covered":   False,
        "international_cover":     True,   # worldwide accident cover
        "ayush_pct_si":            None,
        "cashless_hospitals":      0,      # no cashless — fixed benefit on docs
        "section_80d":             False,  # NOT health insurance per Income Tax Act

        "waiting": {
            "initial_days":  0,    # Day 1 — accidents only
            "ped_months":    None, # N/A
            "accident_exempt": True,
        },

        "ncb": {
            "type":         "premium_discount",
            "pct_per_year": 0.05,
            "max_pct":      0.20,
        },

        # Benefit schedule
        "accidental_death_pct_csi":   1.00,
        "ptd_pct_csi":                1.00,
        "ppd_schedule":               "per_disability_table",   # 5%–70% of CSI
        "ttd_pct_csi_per_week":       0.01,
        "ttd_max_weeks":              104,
        "ttd_max_weekly_inr":         25_000,

        # Built-in sub-benefits
        "accidental_hosp_daily_inr":  2_000,
        "accidental_hosp_max_days":   30,
        "education_benefit_per_child_inr": 100_000,
        "education_benefit_max_children":  2,
        "family_transport_inr":       15_000,
        "funeral_expenses_inr":       20_000,

        "occupation_categories": {
            "A": {"examples": "office, IT, teachers, bankers", "loading": 0.0},
            "B": {"examples": "shop owners, delivery managers", "loading": 0.25},
            "C": {"examples": "construction, truck drivers, factory floor", "loading": 0.75},
            "D": {"examples": "stunt, deep-sea, armed forces combat", "loading": 1.50},
        },

        "opd_covered":             False,
        "telemedicine":            False,
        "maternity_covered":       False,
        "newborn_covered":         False,
        "ivf_covered":             False,
        "mental_health_inpatient": False,
        "mental_health_opd":       False,
        "critical_illness_lumpsum":False,
        "cancer_stage_benefit":    False,
        "income_protection":       True,   # TTD weekly benefit
        "hospital_daily_cash":     False,
        "accidental_death_benefit":True,
        "disability_benefit":      True,
        "women_specific_cancer_day1":False,
        "cancer_screening_voucher":False,

        "wearable_discount":       False,
        "ai_health_engine":        False,
        "genetic_testing":         False,
        "cdmp":                    False,
        "aadhaar_only_enrollment": False,
        "pmjay_coordination":      False,
        "hr_portal":               False,
        "annual_health_checkup":   False,
        "free_look_days":          15,
        "csr":                     0.972,

        "plans": [
            plan("SP012-10L", "PA 10L CSI",  si_inr=1_000_000, annual_premium_inr=550,
                 sample_profile="Age 18-45, Cat A"),
            plan("SP012-25L", "PA 25L CSI",  si_inr=2_500_000, annual_premium_inr=1_200,
                 sample_profile="Age 18-45, Cat A"),
            plan("SP012-50L", "PA 50L CSI",  si_inr=5_000_000, annual_premium_inr=2_200,
                 sample_profile="Age 18-45, Cat A"),
            plan("SP012-1Cr", "PA 1Cr CSI",  si_inr=10_000_000, annual_premium_inr=4_000,
                 sample_profile="Age 18-45, Cat A"),
        ],

        "_score_boosts": {
            "primary_need_accident": 0.50,
            "physical_occupation": 0.15,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP013": {
        "product_id":       "SP013",
        "name":             "Swasthya Critical Guard",
        "uin":              "SWIHLIP24013V012024",
        "segment":          "critical_illness",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["critical_illness"],
        "budget_bands":     ["mid"],

        "entry_age_min":    21,
        "entry_age_max":    60,
        "lifelong_renewal": True,
        "medical_exam_required": True,   # all applicants above 40 or SI ≥ 25L

        "cover_type":              "individual",
        "fixed_benefit_only":      True,
        "no_room_rent_cap":        None,  # N/A
        "icu_covered":             None,  # N/A
        "no_disease_sublimits":    None,  # N/A
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     None,
        "pre_hosp_days":           None,
        "post_hosp_days":          None,
        "daycare_procedures":      None,
        "domiciliary_covered":     False,
        "road_ambulance_inr":      None,
        "air_ambulance_covered":   False,
        "international_cover":     False,
        "ayush_pct_si":            None,
        "cashless_hospitals":      0,
        "survival_period_days":    30,

        "conditions_total":        20,
        "tier_a_conditions":       10,
        "tier_a_payout_pct_si":    1.00,
        "tier_b_conditions":       10,
        "tier_b_payout_pct_si":    0.50,
        "policy_continues_after_tier_b": True,
        "premium_waiver_after_tier_b_years": 2,

        "key_conditions_tier_a": [
            "cancer_specified_severity", "heart_attack", "CABG",
            "stroke_permanent", "kidney_failure_dialysis",
            "major_organ_transplant", "permanent_paralysis",
            "multiple_sclerosis", "open_heart_valve", "motor_neurone"
        ],
        "key_conditions_tier_b": [
            "angioplasty", "aorta_graft", "pulmonary_hypertension",
            "coma", "loss_of_sight", "loss_of_hearing",
            "loss_of_speech", "severe_burns", "bacterial_meningitis", "aplastic_anaemia"
        ],

        "waiting": {
            "initial_days":   90,
            "ped_months":     48,
            "accident_exempt": True,
        },

        "ncb": {"type": "none", "pct_per_year": None, "max_pct": None},

        "smoker_loading_pct":  0.40,
        "female_discount_pct": 0.10,

        "opd_covered":             False,
        "telemedicine":            False,
        "maternity_covered":       False,
        "newborn_covered":         False,
        "ivf_covered":             False,
        "mental_health_inpatient": False,
        "mental_health_opd":       False,
        "critical_illness_lumpsum": True,
        "cancer_stage_benefit":    False,
        "income_protection":       False,
        "hospital_daily_cash":     False,
        "accidental_death_benefit":False,
        "disability_benefit":      False,
        "women_specific_cancer_day1":False,
        "cancer_screening_voucher": True,
        "cancer_screening_inr_year": 3_500,
        "second_opinion_service":  True,
        "oncology_counselling_sessions": 3,

        "wearable_discount":       False,
        "ai_health_engine":        False,
        "genetic_testing":         False,
        "cdmp":                    False,
        "aadhaar_only_enrollment": False,
        "pmjay_coordination":      False,
        "hr_portal":               False,
        "annual_health_checkup":   False,
        "section_80d":             True,
        "free_look_days":          15,
        "csr":                     0.936,

        "plans": [
            plan("SP013-10L", "Critical Guard 10L", si_inr=1_000_000,  annual_premium_inr=5_800,
                 sample_profile="35yr male non-smoker"),
            plan("SP013-25L", "Critical Guard 25L", si_inr=2_500_000,  annual_premium_inr=13_500,
                 sample_profile="35yr male non-smoker"),
            plan("SP013-50L", "Critical Guard 50L", si_inr=5_000_000,  annual_premium_inr=25_800,
                 sample_profile="35yr male non-smoker"),
        ],

        "_score_boosts": {
            "primary_need_critical_illness": 0.50,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP014": {
        "product_id":       "SP014",
        "name":             "Swasthya Cancer Protect",
        "uin":              "SWIHLIP24014V012024",
        "segment":          "cancer",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["cancer"],
        "budget_bands":     ["mid"],

        "entry_age_min":    21,
        "entry_age_max":    60,
        "lifelong_renewal": True,
        "medical_exam_required": True,

        "cover_type":              "individual",
        "fixed_benefit_only":      True,
        "no_room_rent_cap":        None,
        "icu_covered":             None,
        "no_disease_sublimits":    None,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     None,
        "pre_hosp_days":           None,
        "post_hosp_days":          None,
        "daycare_procedures":      None,
        "domiciliary_covered":     False,
        "road_ambulance_inr":      None,
        "air_ambulance_covered":   False,
        "international_cover":     False,
        "ayush_pct_si":            None,
        "cashless_hospitals":      0,
        "survival_period_days":    30,

        "stage_payouts": {
            "CIS":        0.25,
            "stage_I":    0.50,
            "stage_II_III": 1.00,
            "stage_IV":   1.00,
        },
        "stage_escalation_benefit":  True,
        "recurrence_benefit_pct_si": 0.50,
        "second_primary_benefit_pct_si": 0.50,
        "income_protection_monthly": {
            "10L_SI": 10_000,
            "25L_SI": 20_000,
            "50L_SI": 30_000,
        },
        "income_protection_max_months": 24,

        "waiting": {
            "initial_days":   90,
            "ped_months":     None,   # prior cancer excluded, not time-waited
            "accident_exempt": False,
        },

        "ncb": {"type": "none", "pct_per_year": None, "max_pct": None},

        "smoker_loading_pct":  0.50,

        "opd_covered":             False,
        "telemedicine":            False,
        "maternity_covered":       False,
        "newborn_covered":         False,
        "ivf_covered":             False,
        "mental_health_inpatient": False,
        "mental_health_opd":       False,
        "critical_illness_lumpsum": True,
        "cancer_stage_benefit":    True,
        "income_protection":       True,
        "hospital_daily_cash":     False,
        "accidental_death_benefit":False,
        "disability_benefit":      False,
        "women_specific_cancer_day1":False,
        "cancer_screening_voucher": True,
        "oncology_navigation":     True,
        "genetic_counselling":     True,
        "survivorship_support":    True,

        "wearable_discount":       False,
        "ai_health_engine":        False,
        "genetic_testing":         False,
        "cdmp":                    False,
        "aadhaar_only_enrollment": False,
        "pmjay_coordination":      False,
        "hr_portal":               False,
        "annual_health_checkup":   False,
        "section_80d":             True,
        "free_look_days":          15,
        "csr":                     None,

        "plans": [
            plan("SP014-10L", "Cancer Protect 10L", si_inr=1_000_000,  annual_premium_inr=11_800,
                 sample_profile="35yr male non-smoker"),
            plan("SP014-25L", "Cancer Protect 25L", si_inr=2_500_000,  annual_premium_inr=13_400,
                 sample_profile="35yr female non-smoker"),
            plan("SP014-50L", "Cancer Protect 50L", si_inr=5_000_000,  annual_premium_inr=24_600,
                 sample_profile="35yr female non-smoker"),
        ],

        "_score_boosts": {
            "primary_need_cancer": 0.60,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP015": {
        "product_id":       "SP015",
        "name":             "Swasthya Heart + Diabetes Care",
        "uin":              "SWIHLIP24015V012024",
        "segment":          "chronic_disease",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["mid", "premium"],

        "entry_age_min":    25,
        "entry_age_max":    65,
        "lifelong_renewal": True,
        "medical_exam_required": True,

        "cover_type":              "floater",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     True,
        "pre_hosp_days":           60,
        "post_hosp_days":          120,
        "daycare_procedures":      541,
        "domiciliary_covered":     False,
        "domiciliary_limit_inr":   None,
        "road_ambulance_inr":      3_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      12_000,

        "restoration_type":        "once_different_illness",
        "restoration_pct":         1.0,

        "waiting": {
            "initial_days":                  30,
            "ped_months_diabetes_cardiac":   12,   # KEY: reduced vs standard 36-48
            "ped_months_other":              36,
            "specific_disease_months":       24,
            "accident_exempt":               True,
        },

        "ncb": {"type": "si_increase", "pct_per_year": 0.05, "max_pct": 0.30},

        "opd_covered":               True,
        "opd_includes_medications":  True,
        "opd_medications": ["insulin", "oral_hypoglycaemics", "statins", "antihypertensives"],
        "cgm_sensors_inr_year":      8_000,
        "glucometer_strips_inr_year":5_000,
        "dietitian_sessions_year":   6,
        "podiatry_covered":          True,
        "telemedicine":              True,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               False,
        "mental_health_opd_limit_inr":     None,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            True,
        "cdmp_conditions":                 ["diabetes","hypertension","CAD","CKD"],
        "health_milestone_discount_max":   0.20,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           True,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,

        "risk_tiers": {
            "tier_1": {"criteria": "HbA1c<7.5%, no cardiac, BP controlled", "loading": 0.0},
            "tier_2": {"criteria": "HbA1c 7.5-9% or uncontrolled BP",       "loading": 0.25},
            "tier_3": {"criteria": "HbA1c>9% or CAD history or prior MI",   "loading": 0.50},
            "tier_4": {"criteria": "post-CABG, EF<40%, dialysis",           "loading": 0.80},
        },

        "plans": [
            plan("SP015-Silver",   "HDC Silver 3L",   si_inr=300_000,   annual_premium_inr=18_500,
                 sample_profile="Age 40, Tier 1", opd_limit_inr=15_000),
            plan("SP015-Gold",     "HDC Gold 5L",     si_inr=500_000,   annual_premium_inr=28_000,
                 sample_profile="Age 40, Tier 1", opd_limit_inr=25_000),
            plan("SP015-Platinum", "HDC Platinum 10L",si_inr=1_000_000, annual_premium_inr=44_000,
                 sample_profile="Age 40, Tier 1", opd_limit_inr=40_000),
        ],

        "_score_boosts": {
            "ped_type_diabetes_cardiac": 0.45,  # near-deterministic
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP016": {
        "product_id":       "SP016",
        "name":             "Swasthya Corporate Care",
        "uin":              "SWIHLIP24016V012024",
        "segment":          "group_large",
        "buyer_types":      ["employer_large"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["mid", "premium"],

        "min_group_size":   50,
        "max_group_size":   None,
        "lifelong_renewal": True,
        "medical_exam_required": False,   # no individual underwriting

        "cover_type":              "group",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     True,
        "pre_hosp_days":           60,
        "post_hosp_days":          90,
        "daycare_procedures":      541,
        "domiciliary_covered":     True,
        "domiciliary_limit_inr":   50_000,
        "road_ambulance_inr":      5_000,
        "air_ambulance_covered":   True,
        "air_ambulance_inr":       200_000,
        "international_cover":     False,
        "ayush_pct_si":            0.50,
        "cashless_hospitals":      14_000,

        "restoration_type":        "once_different_illness",
        "restoration_pct":         1.0,

        "waiting": {
            "initial_days":  0,      # group — Day 1 for everything
            "ped_months":    0,      # Day 1
            "maternity_months": 0,   # Day 1
            "mental_health_days": 0, # Day 1
            "accident_exempt": True,
        },

        "ncb": {"type": "none", "pct_per_year": None, "max_pct": None},

        "opd_covered":             True,
        "telemedicine":            True,

        "maternity_covered":           True,
        "maternity_day1":              True,
        "newborn_covered":             True,
        "nicu_limit_inr":              100_000,
        "ivf_covered":                 False,  # rider only
        "mental_health_inpatient":     True,
        "mental_health_day1":          True,
        "mental_health_opd":           True,
        "mental_health_opd_limit_inr": 10_000,
        "mha_2017_compliant":          True,
        "critical_illness_lumpsum":    False,  # rider only
        "cancer_stage_benefit":        False,
        "income_protection":           False,
        "hospital_daily_cash":         False,
        "accidental_death_benefit":    False,
        "disability_benefit":          False,
        "women_specific_cancer_day1":  False,
        "cancer_screening_voucher":    False,

        "wearable_discount":           False,
        "ai_health_engine":            False,
        "genetic_testing":             False,
        "cdmp":                        False,
        "aadhaar_only_enrollment":     False,
        "pmjay_coordination":          False,
        "hr_portal":                   True,
        "hrms_integrations":           ["Darwinbox","SAP SuccessFactors","BambooHR","Keka"],
        "portability_on_separation":   True,
        "annual_health_checkup":       True,
        "section_80d":                 True,   # employee-paid portion
        "free_look_days":              15,
        "csr":                         None,

        "optional_riders": [
            "parent_cover", "dental_vision_opd", "ivf_art",
            "critical_illness_5L", "personal_accident_25L", "daily_cash_1K"
        ],

        "plans": [
            plan("SP016-Basic",    "CC Basic 3L",    si_inr=300_000,   annual_premium_inr=12_000,
                 sample_profile="per employee incl. family", opd_limit_inr=5_000,
                 maternity_normal_inr=50_000, maternity_csection_inr=75_000),
            plan("SP016-Standard", "CC Standard 5L", si_inr=500_000,   annual_premium_inr=18_500,
                 sample_profile="per employee incl. family", opd_limit_inr=10_000,
                 maternity_normal_inr=75_000, maternity_csection_inr=112_500),
            plan("SP016-Premium",  "CC Premium 10L", si_inr=1_000_000, annual_premium_inr=28_000,
                 sample_profile="per employee incl. family", opd_limit_inr=15_000,
                 maternity_normal_inr=100_000, maternity_csection_inr=150_000),
            plan("SP016-Elite",    "CC Elite 25L",   si_inr=2_500_000, annual_premium_inr=52_000,
                 sample_profile="per employee incl. family", opd_limit_inr=25_000,
                 maternity_normal_inr=200_000, maternity_csection_inr=300_000),
        ],

        "_score_boosts": {
            "buyer_type_employer_large": 0.70,  # hard filter effectively
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP017": {
        "product_id":       "SP017",
        "name":             "Swasthya SME Suraksha",
        "uin":              "SWIHLIP24017V012024",
        "segment":          "group_sme",
        "buyer_types":      ["employer_sme"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["budget", "mid"],

        "min_group_size":   7,
        "max_group_size":   50,
        "flat_rate_pricing":True,
        "lifelong_renewal": True,
        "medical_exam_required": False,
        "digital_onboarding_minutes": 30,

        "cover_type":              "group",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     False,
        "pre_hosp_days":           30,
        "post_hosp_days":          60,
        "daycare_procedures":      541,
        "domiciliary_covered":     True,
        "domiciliary_limit_inr":   10_000,
        "road_ambulance_inr":      2_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      12_000,

        "restoration_type":        "once_different_illness",
        "restoration_pct":         0.50,  # SP017-Standard; full in SP017-Plus

        "waiting": {
            "initial_days":     30,
            "ped_months":       0,   # Day 1 for group
            "maternity_months": 0,   # Day 1 for group
            "accident_exempt":  True,
        },

        "ncb": {"type": "none", "pct_per_year": None, "max_pct": None},

        "opd_covered":             True,
        "telemedicine":            True,
        "languages_supported":     11,

        "maternity_covered":           True,
        "maternity_day1":              True,
        "newborn_covered":             True,
        "nicu_limit_inr":              50_000,
        "ivf_covered":                 False,
        "mental_health_inpatient":     True,
        "mental_health_opd":           False,
        "critical_illness_lumpsum":    False,
        "cancer_stage_benefit":        False,
        "income_protection":           False,
        "hospital_daily_cash":         False,
        "accidental_death_benefit":    False,
        "disability_benefit":          False,
        "women_specific_cancer_day1":  False,
        "cancer_screening_voucher":    False,

        "wearable_discount":           False,
        "ai_health_engine":            False,
        "genetic_testing":             False,
        "cdmp":                        False,
        "aadhaar_only_enrollment":     False,
        "pmjay_coordination":          False,
        "hr_portal":                   True,
        "hr_portal_simplified":        True,
        "portability_on_separation":   True,
        "annual_health_checkup":       False,
        "section_80d":                 True,
        "free_look_days":              15,
        "csr":                         None,

        "group_size_discounts": {
            "16_30": 0.05,
            "31_50": 0.10,
        },

        "plans": [
            plan("SP017-Basic",    "SME Basic 2L",    si_inr=200_000, annual_premium_inr=1_200,
                 sample_profile="per employee", opd_limit_inr=5_000,
                 maternity_normal_inr=25_000, maternity_csection_inr=37_500),
            plan("SP017-Standard", "SME Standard 3L", si_inr=300_000, annual_premium_inr=1_800,
                 sample_profile="per employee", opd_limit_inr=8_000,
                 maternity_normal_inr=40_000, maternity_csection_inr=60_000),
            plan("SP017-Plus",     "SME Plus 5L",     si_inr=500_000, annual_premium_inr=2_800,
                 sample_profile="per employee", opd_limit_inr=12_000,
                 maternity_normal_inr=60_000, maternity_csection_inr=90_000),
        ],

        "_score_boosts": {
            "buyer_type_employer_sme": 0.70,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP018": {
        "product_id":       "SP018",
        "name":             "Swasthya AI Health Companion",
        "uin":              "SWIHLIP24018V012024",
        "segment":          "ai_tech",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["mid", "premium"],

        "entry_age_min":    21,
        "entry_age_max":    55,
        "lifelong_renewal": True,
        "medical_exam_required": False,
        "smartphone_required":   True,

        "cover_type":              "individual",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     True,
        "pre_hosp_days":           60,
        "post_hosp_days":          180,
        "daycare_procedures":      541,
        "domiciliary_covered":     True,
        "domiciliary_limit_inr":   50_000,
        "road_ambulance_inr":      5_000,
        "air_ambulance_covered":   True,
        "air_ambulance_inr":       500_000,
        "international_cover":     False,
        "ayush_pct_si":            0.50,
        "cashless_hospitals":      14_000,

        "restoration_type":        "unlimited_different_illness",
        "restoration_pct":         1.0,

        "waiting": {
            "initial_days":            30,
            "ped_months":              36,
            "specific_disease_months": 24,
            "accident_exempt":         True,
        },

        "ncb": {"type": "si_increase", "pct_per_year": 0.10, "max_pct": 0.50},

        "opd_covered":             True,
        "telemedicine":            True,
        "international_telemedicine": True,
        "international_tele_sessions": 2,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         True,
        "mental_health_opd":               True,
        "mental_health_opd_limit_inr":     15_000,
        "mental_health_sessions":          12,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        False,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               True,
        "wearable_discount_max_pct":       0.30,
        "wearable_tiers": {
            "platinum": {"score_range": [90,100], "discount": 0.30},
            "gold":     {"score_range": [75,89],  "discount": 0.20},
            "silver":   {"score_range": [60,74],  "discount": 0.10},
            "bronze":   {"score_range": [45,59],  "discount": 0.05},
        },
        "wearable_compatible": ["Apple Watch","Fitbit","Garmin","Samsung Galaxy Watch","Mi Band","Whoop"],
        "ai_health_engine":                True,
        "ai_engine_name":                  "SwasthyaAI",
        "ai_health_score_weekly":          True,
        "genetic_testing":                 True,
        "genetic_panel_free":              1,
        "genetic_data_used_for_underwriting": False,
        "advanced_diagnostics_per_year":   1,
        "advanced_diagnostics_options":    ["CAC_scan","whole_body_MRI","liquid_biopsy"],
        "longevity_consult_per_year":      1,
        "longevity_consult_inr":           5_000,
        "second_opinion_specialists":      500,
        "remote_patient_monitoring":       True,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         False,
        "pmjay_coordination":              False,
        "hr_portal":                       False,
        "annual_health_checkup":           True,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             0.971,
        "privacy_compliance":              ["PDPB_2023","DISHA","IRDAI_data_security"],

        "plans": [
            plan("SP018-10L", "AI Companion 10L", si_inr=1_000_000,  annual_premium_inr=11_800,
                 sample_profile="30yr individual", opd_limit_inr=20_000),
            plan("SP018-25L", "AI Companion 25L", si_inr=2_500_000,  annual_premium_inr=24_600,
                 sample_profile="30yr individual", opd_limit_inr=35_000),
            plan("SP018-50L", "AI Companion 50L", si_inr=5_000_000,  annual_premium_inr=46_000,
                 sample_profile="30yr individual", opd_limit_inr=50_000),
        ],

        "_score_boosts": {
            "wearable_user": 0.20,
            "health_conscious": 0.15,
            "budget_band_mid": 0.05,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP019": {
        "product_id":       "SP019",
        "name":             "Swasthya Gig Worker Shield",
        "uin":              "SWIHLIP24019V012024",
        "segment":          "micro",
        "buyer_types":      ["individual", "gig_worker"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["micro"],

        "entry_age_min":    18,
        "entry_age_max":    60,
        "lifelong_renewal": False,
        "medical_exam_required": False,

        "cover_type":              "individual",
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "copay_type":              "none",
        "copay_pct":               None,
        "consumables_covered":     False,
        "pre_hosp_days":           30,
        "post_hosp_days":          60,
        "daycare_procedures":      100,
        "domiciliary_covered":     False,
        "domiciliary_limit_inr":   None,
        "road_ambulance_inr":      1_000,
        "air_ambulance_covered":   False,
        "air_ambulance_inr":       None,
        "international_cover":     False,
        "ayush_pct_si":            0.25,
        "cashless_hospitals":      12_000,
        "government_hospitals_accepted": True,

        "restoration_type":        "none",
        "restoration_pct":         None,

        "waiting": {
            "initial_illness_days":    30,
            "opd_days":                0,   # OPD from Day 1 — KEY differentiator
            "accident_days":           0,   # accident from Day 1
            "ped_months_inpatient":    12,
            "maternity_months":        9,
        },

        "ncb": {"type": "none", "pct_per_year": None, "max_pct": None},

        "opd_covered":             True,
        "opd_day1":                True,
        "opd_accident_limit_inr":  2_000,
        "telemedicine":            True,
        "languages_supported":     11,

        "maternity_covered":               False,
        "newborn_covered":                 False,
        "ivf_covered":                     False,
        "mental_health_inpatient":         False,
        "mental_health_opd":               False,
        "critical_illness_lumpsum":        False,
        "cancer_stage_benefit":            False,
        "income_protection":               False,
        "hospital_daily_cash":             False,
        "accidental_death_benefit":        True,
        "accidental_death_inr":            100_000,
        "disability_benefit":              False,
        "women_specific_cancer_day1":      False,
        "cancer_screening_voucher":        False,

        "wearable_discount":               False,
        "ai_health_engine":                False,
        "genetic_testing":                 False,
        "cdmp":                            False,
        "aadhaar_only_enrollment":         True,
        "pmjay_coordination":              True,
        "hr_portal":                       False,
        "annual_health_checkup":           False,
        "section_80d":                     True,
        "free_look_days":                  15,
        "csr":                             None,
        "settlement_days":                 7,
        "enrollment_methods":              ["app","whatsapp","ivr","csc"],
        "platform_partnerships":           ["Swiggy","Zomato","Ola","Porter","Urban Company"],
        "platform_subsidised_rate_inr_day": 2,

        "plans": [
            plan("SP019-Lite",    "GWS Lite 1L",    si_inr=100_000, annual_premium_inr=1_095,
                 sample_profile="any gig worker", opd_limit_inr=3_000, daily_benefit_inr=3),
            plan("SP019-Plus",    "GWS Plus 2L",    si_inr=200_000, annual_premium_inr=1_799,
                 sample_profile="any gig worker", opd_limit_inr=5_000, daily_benefit_inr=5),
            plan("SP019-Max",     "GWS Max 3L",     si_inr=300_000, annual_premium_inr=2_499,
                 sample_profile="any gig worker", opd_limit_inr=8_000, daily_benefit_inr=7),
            plan("SP019-Family",  "GWS Family 2L",  si_inr=200_000, annual_premium_inr=2_199,
                 sample_profile="family floater",  opd_limit_inr=5_000),
            plan("SP019-FamilyPlus","GWS Family+ 3L",si_inr=300_000,annual_premium_inr=3_299,
                 sample_profile="family floater",  opd_limit_inr=8_000),
        ],

        "_score_boosts": {
            "buyer_type_gig_worker": 0.35,
            "budget_band_micro": 0.35,
            "aadhaar_only_enrollment": 0.15,
        },
    },


    # ════════════════════════════════════════════════════════════════════════
    "SP020": {
        "product_id":       "SP020",
        "name":             "Swasthya Family Legacy",
        "uin":              "SWIHLIP24020V012024",
        "segment":          "multi_gen",
        "buyer_types":      ["individual"],
        "gender":           "any",
        "primary_needs":    ["hospitalisation"],
        "budget_bands":     ["premium"],

        "entry_age_min_proposer":  25,
        "entry_age_max_proposer":  55,
        "lifelong_renewal":        True,
        "medical_exam_required":   False,   # per generation

        "cover_type":              "hybrid_ISI_pool",
        "max_members":             8,
        "generations": {
            "G1": {
                "relationship":       "grandparents / parents-in-law",
                "max_members":        4,
                "copay_60_69":        0.10,
                "copay_70_plus":      0.20,
                "ped_months":         12,
                "domiciliary_inr":    50_000,
            },
            "G2": {
                "relationship":       "proposer and/or spouse",
                "max_members":        2,
                "copay":              0,
                "ped_months":         36,
            },
            "G3": {
                "relationship":       "dependent children",
                "max_members":        4,
                "max_age":            25,
                "copay":              0,
                "ped_months":         36,
                "transition_to":      ["SP001","SP002"],
            },
        },

        "auto_si_increase_pct":    0.05,
        "auto_si_increase_no_underwriting": True,
        "no_room_rent_cap":        True,
        "icu_covered":             True,
        "no_disease_sublimits":    True,
        "consumables_covered":     True,
        "pre_hosp_days_G2_G3":     60,
        "pre_hosp_days_G1":        30,
        "post_hosp_days_G2_G3":    120,
        "post_hosp_days_G1":       90,
        "daycare_procedures":      541,
        "road_ambulance_inr":      5_000,
        "air_ambulance_covered":   True,
        "air_ambulance_inr":       300_000,
        "international_cover":     False,
        "ayush_pct_si":            0.50,
        "cashless_hospitals":      14_000,

        "restoration_type":        "once_different_illness",
        "restoration_pct":         1.0,

        "ncb": {
            "type":         "si_increase",
            "pct_per_year": 0.10,
            "max_pct":      0.50,
            "per_member":   True,   # does NOT affect other members
        },

        "opd_covered":             True,
        "telemedicine":            True,

        "maternity_covered":           True,
        "maternity_waiting_months":    9,
        "maternity_normal_inr":        75_000,
        "maternity_csection_inr":      112_500,
        "newborn_covered":             True,
        "newborn_cover_days":          90,
        "ivf_covered":                 False,
        "mental_health_inpatient":     True,
        "mental_health_opd":           True,
        "mental_health_opd_limit_inr": 15_000,
        "critical_illness_lumpsum":    False,
        "cancer_stage_benefit":        False,
        "income_protection":           False,
        "hospital_daily_cash":         False,
        "accidental_death_benefit":    False,
        "disability_benefit":          False,
        "women_specific_cancer_day1":  False,
        "cancer_screening_voucher":    False,

        "wearable_discount":           False,
        "ai_health_engine":            False,
        "genetic_testing":             False,
        "cdmp":                        True,
        "cdmp_for":                    "G1_members",
        "aadhaar_only_enrollment":     False,
        "pmjay_coordination":          False,
        "hr_portal":                   False,
        "annual_health_checkup":       True,
        "section_80d":                 True,
        "section_80d_max_inr":         75_000,   # ₹25K self + ₹50K senior parents
        "free_look_days":              15,
        "csr":                         0.965,

        "plans": [
            plan("SP020-Essential", "FL Essential", si_inr=None, annual_premium_inr=62_000,
                 sample_profile="6-member family (estimate)", opd_limit_inr=10_000),
            plan("SP020-Comfort",   "FL Comfort",   si_inr=None, annual_premium_inr=110_400,
                 sample_profile="6-member family (estimate)", opd_limit_inr=15_000),
            plan("SP020-Premium",   "FL Premium",   si_inr=None, annual_premium_inr=220_000,
                 sample_profile="6-member family (estimate)", opd_limit_inr=25_000),
            plan("SP020-Legacy",    "FL Legacy",    si_inr=None, annual_premium_inr=420_000,
                 sample_profile="6-member family (estimate)", opd_limit_inr=40_000),
        ],

        "_score_boosts": {
            "family_cover_floater_joint": 0.40,
            "has_parents_to_insure": 0.20,
            "budget_band_premium": 0.10,
        },
    },
}


# ---------------------------------------------------------------------------
# ACCESSOR FUNCTIONS  (used by retrieval scorer, feature presenter, plan selector)
# ---------------------------------------------------------------------------

def get_product(product_id: str) -> dict:
    """Return the full registry entry for a product."""
    if product_id not in REGISTRY:
        raise KeyError(f"Unknown product_id: {product_id!r}")
    return REGISTRY[product_id]


def get_feature(product_id: str, feature_key: str, default=None):
    """
    Return a single feature value, with None-safe default.
    Usage: get_feature("SP002", "opd_covered")  → True
    """
    return REGISTRY.get(product_id, {}).get(feature_key, default)


def get_plans(product_id: str) -> list[dict]:
    """Return the plan list for a product."""
    return REGISTRY.get(product_id, {}).get("plans", [])


def get_plan(product_id: str, plan_id: str) -> dict | None:
    """Return a specific plan by plan_id."""
    for p in get_plans(product_id):
        if p["plan_id"] == plan_id:
            return p
    return None


def get_waiting(product_id: str) -> dict:
    """Return the waiting period sub-dict."""
    return REGISTRY.get(product_id, {}).get("waiting", {})


def get_ncb(product_id: str) -> dict:
    """Return the NCB sub-dict."""
    return REGISTRY.get(product_id, {}).get("ncb", {})


def get_score_boosts(product_id: str) -> dict:
    """Return retrieval scoring boosts for this product."""
    return REGISTRY.get(product_id, {}).get("_score_boosts", {})


# Internal keys the LLM should never see — scoring weights are orchestrator-only
_INTERNAL_KEYS = {"_score_boosts"}


def get_product_for_llm(product_id: str) -> dict:
    """
    Return a clean product dict safe to return as a tool call result.
    Strips internal keys (prefixed with _) that are orchestrator-only.
    The LLM uses this to compose recommendation responses — it should
    only see typed feature values, not scoring machinery.
    """
    product = get_product(product_id)   # raises KeyError if not found
    return {k: v for k, v in product.items() if k not in _INTERNAL_KEYS}


def all_product_ids() -> list[str]:
    """Return every product_id in the registry (SP001–SP020)."""
    return list(REGISTRY.keys())


def products_by_segment(segment: str) -> list[str]:
    """Return product_ids whose `segment` equals the given value (e.g. 'micro', 'senior')."""
    return [pid for pid, p in REGISTRY.items() if p.get("segment") == segment]


def products_by_buyer_type(buyer_type: str) -> list[str]:
    """Return product_ids that accept the given buyer_type in their `buyer_types` list."""
    return [pid for pid, p in REGISTRY.items() if buyer_type in p.get("buyer_types", [])]


def products_with_feature(feature_key: str, value=True) -> list[str]:
    """Find all products where feature_key == value (useful for boolean features)."""
    return [
        pid for pid, p in REGISTRY.items()
        if p.get(feature_key) == value
    ]


def feature_matrix(feature_keys: list[str]) -> dict[str, dict]:
    """
    Return a matrix of {product_id: {feature: value}} for the given keys.
    Used by the feature presenter to build comparison views.
    """
    return {
        pid: {k: p.get(k) for k in feature_keys}
        for pid, p in REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# QUICK SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Registry loaded: {len(REGISTRY)} products\n")

    print("=== Accessor smoke tests ===")
    print(f"SP002 opd_covered       : {get_feature('SP002', 'opd_covered')}")
    print(f"SP015 ped diabetes wait  : {get_waiting('SP015').get('ped_months_diabetes_cardiac')} months")
    print(f"SP013 tier_a_conditions  : {get_feature('SP013', 'tier_a_conditions')}")
    print(f"SP019 opd_day1           : {get_feature('SP019', 'opd_day1')}")
    print(f"SP010 aggregate_deductible: {get_feature('SP010', 'aggregate_deductible')}")
    print(f"SP020 auto_si_increase   : {get_feature('SP020', 'auto_si_increase_pct')}")

    print("\n=== Products with maternity day-1 ===")
    for pid in all_product_ids():
        if REGISTRY[pid].get("waiting", {}).get("maternity_months") == 0:
            print(f"  {pid}: {REGISTRY[pid]['name']}")

    print("\n=== Products with OPD ===")
    for pid in products_with_feature("opd_covered", True):
        print(f"  {pid}: {REGISTRY[pid]['name']}")

    print("\n=== Female-only products ===")
    for pid in all_product_ids():
        if REGISTRY[pid].get("gender") == "female_only":
            print(f"  {pid}: {REGISTRY[pid]['name']}")

    print("\n=== Group buyer products ===")
    for bt in ["employer_large", "employer_sme"]:
        pids = products_by_buyer_type(bt)
        print(f"  {bt}: {pids}")

    print("\n=== SP016 plans ===")
    for p in get_plans("SP016"):
        print(f"  {p['plan_id']}: ₹{p['annual_premium_inr']:,}/yr, SI ₹{p['si_inr']:,}")

    print("\n=== Feature matrix (opd + maternity + mental_health_opd) ===")
    matrix = feature_matrix(["opd_covered", "maternity_covered", "mental_health_opd"])
    for pid, feats in matrix.items():
        row = " | ".join(f"{k}={v}" for k, v in feats.items())
        print(f"  {pid}: {row}")
