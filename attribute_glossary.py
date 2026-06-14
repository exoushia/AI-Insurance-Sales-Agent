"""
Swasthya Insurance Agent — Unified Attribute Glossary
======================================================
Single source of truth for every named attribute in the system.
Used by:
  - policy_feature_registry.py   (policy + plan attributes)
  - user_schema.py               (user attributes — artifact 2)
  - retrieval_scorer.py          (knows which fields are hard filters)
  - feature_presenter.py         (knows display labels + descriptions)
  - next_question_generator.py   (knows question text + valid values)
  - LLM system prompts           (injected so the model understands field semantics)

Structure of each entry
-----------------------
Each attribute is a dict with:
  key          : the canonical snake_case field name used everywhere in code
  layer        : "policy" | "plan" | "user" | "shared"
  type         : "bool" | "int" | "float" | "str_enum" | "list_enum" | "dict" | "int_inr"
  label        : short human-readable display name (for UI + LLM prompts)
  description  : one-sentence explanation of what this attribute means
                 Written for the LLM — no jargon, no assumed insurance knowledge.
  valid_values : for enums — the exhaustive list of allowed values with meaning
  unit         : for numerics — "INR" | "months" | "days" | "pct_0_1" | "count" | "years"
  nullable     : True if None is a valid value (= not applicable, not unknown)
  hard_filter  : True if this field eliminates products when mismatched (used by scorer)
  ask_order    : int — position in question priority queue (1 = ask first); None = don't ask
  question_text: the natural-language question the agent asks to collect this value
  example      : a concrete example value to help the LLM understand the type
"""

from __future__ import annotations

GLOSSARY: list[dict] = [

    # ═══════════════════════════════════════════════════════════════════════
    # USER ATTRIBUTES — collected during info-gathering stage
    # ═══════════════════════════════════════════════════════════════════════

    {
        "key":          "buyer_type",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Buyer type",
        "description":  "Whether the person is buying insurance for themselves as an individual, "
                        "or on behalf of employees as a business owner. This is the first and most "
                        "important question because group products (SP016, SP017) are only available "
                        "to employers, and individual products cannot be bought by an employer for staff.",
        "valid_values": {
            "individual":     "Buying for themselves or their family",
            "employer_large": "Buying for 50 or more employees",
            "employer_sme":   "Buying for 7 to 50 employees",
            "gig_worker":     "Self-employed delivery, cab, domestic or daily-wage worker",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    1,
        "question_text": "Are you looking for health insurance for yourself and your family, "
                         "or for employees at your company?",
        "example":      "individual",
    },

    {
        "key":          "age",
        "layer":        "user",
        "type":         "int",
        "label":        "Age",
        "description":  "The applicant's age in years at the time of purchase. "
                        "Determines which products are available: Young Star (SP009) only accepts "
                        "18–35, Senior Shield (SP006) only accepts 60–80, most others accept 18–65. "
                        "Also affects premium pricing within a product.",
        "valid_values": None,
        "unit":         "years",
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    2,
        "question_text": "How old are you?",
        "example":      38,
    },

    {
        "key":          "gender",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Gender",
        "description":  "The applicant's gender. Only relevant because two products — "
                        "Maternity Suraksha (SP007) and Women Wellness (SP008) — are exclusively "
                        "for female applicants. Also affects premium on cancer and critical illness "
                        "products (SP013, SP014) where female applicants get a 10% discount on "
                        "SP013 due to lower cardiac incidence.",
        "valid_values": {
            "female": "Female applicant — unlocks SP007 and SP008",
            "male":   "Male applicant — SP007 and SP008 are excluded",
            "other":  "Any other gender identity — treated as male for product eligibility",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    3,
        "question_text": "What is your gender?",
        "example":      "female",
    },

    {
        "key":          "primary_need",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Primary insurance need",
        "description":  "The single most important thing the person wants their insurance to cover. "
                        "This is the strongest product-pointing signal in the schema — seven of the "
                        "values map near-deterministically to a specific product. 'Hospitalisation' "
                        "is the default for standard health cover; the others are specialist needs.",
        "valid_values": {
            "hospitalisation":   "Standard in-patient hospital cover — the most common need",
            "critical_illness":  "Lump-sum payout on diagnosis of serious illness (cancer, heart attack, stroke) — SP013",
            "cancer":            "Cancer-specific stage-based payout + income protection — SP014",
            "accident":          "Protection against accidental death and disability — SP012",
            "maternity":         "Pregnancy, delivery, and newborn cover — SP007",
            "top_up":            "Extra cover on top of an existing policy using an aggregate deductible — SP010",
            "international":     "Hospital cover outside India — SP003",
            "daily_cash":        "Fixed cash per day of hospitalisation regardless of bills — SP011",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    4,
        "question_text": "What is the main thing you want your health insurance to cover — "
                         "everyday hospital stays, a specific illness like cancer or heart disease, "
                         "accident protection, maternity, or something else?",
        "example":      "hospitalisation",
    },

    {
        "key":          "has_ped",
        "layer":        "user",
        "type":         "bool",
        "label":        "Has pre-existing condition",
        "description":  "Whether the applicant has any diagnosed medical condition that existed "
                        "before buying this policy. Pre-existing diseases (PED) are subject to "
                        "waiting periods — typically 36–48 months before they are covered. "
                        "This flag triggers the follow-up question about PED type.",
        "valid_values": {
            True:  "Has one or more pre-existing conditions",
            False: "No known pre-existing conditions",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    5,
        "question_text": "Do you have any existing health conditions — like diabetes, heart disease, "
                         "high blood pressure, or anything else — that you'd want to be covered for?",
        "example":      True,
    },

    {
        "key":          "ped_type",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Pre-existing condition type",
        "description":  "The category of pre-existing disease. This is the single most important "
                        "differentiator among the hospitalisation products. Diabetes or cardiac PED "
                        "points strongly to SP015 (Heart + Diabetes Care), which has only a 12-month "
                        "wait vs 36–48 months elsewhere, and covers insulin, CGM sensors, and dietitian "
                        "OPD. Only asked if has_ped is True.",
        "valid_values": {
            "diabetes_cardiac": "Diabetes (Type 1 or 2), heart disease, hypertension, or metabolic syndrome",
            "other_ped":        "Any other pre-existing condition (e.g. thyroid, arthritis, kidney disease)",
            "none":             "No pre-existing conditions — same as has_ped=False",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  True,
        "ask_order":    5,     # same ask_order as has_ped — asked as follow-up in same turn
        "question_text": "Is it diabetes, heart disease, or high blood pressure — or something else?",
        "example":      "diabetes_cardiac",
    },

    {
        "key":          "needs_opd",
        "layer":        "user",
        "type":         "bool",
        "label":        "Needs OPD cover",
        "description":  "Whether the person wants their policy to cover routine outpatient visits — "
                        "doctor consultations, diagnostics, and medicines — not just hospitalisation. "
                        "OPD coverage eliminates SP001, SP004, SP005, SP006, SP010, SP011, SP012, "
                        "SP013, SP014 from consideration. Significantly raises the premium.",
        "valid_values": {
            True:  "Wants cover for regular doctor visits and medicines (OPD)",
            False: "Only needs hospitalisation cover; OPD not required",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,     # soft filter — eliminates but not exclusively
        "ask_order":    6,
        "question_text": "Do you also want cover for regular doctor visits and medicines — "
                         "not just hospital stays?",
        "example":      True,
    },

    {
        "key":          "budget_band",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Annual budget band",
        "description":  "The approximate annual premium the person is willing to pay. "
                        "Used as a guardrail — if the best-matched product is outside the budget, "
                        "the agent offers the closest affordable alternative. "
                        "Micro = under ₹2,000/yr; Budget = ₹2,000–₹10,000/yr; "
                        "Mid = ₹10,000–₹30,000/yr; Premium = above ₹30,000/yr.",
        "valid_values": {
            "micro":   "Under ₹2,000 per year (₹3–₹5/day) — SP005, SP019",
            "budget":  "₹2,000–₹10,000 per year — SP001, SP004, SP009, SP010, SP011, SP012",
            "mid":     "₹10,000–₹30,000 per year — SP002, SP006, SP013, SP014, SP015, SP018",
            "premium": "Above ₹30,000 per year — SP003, SP020",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    7,
        "question_text": "Roughly how much are you comfortable spending on health insurance per year?",
        "example":      "mid",
    },

    {
        "key":          "family_cover",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Family coverage type",
        "description":  "Whether the person wants to cover just themselves, or include family members. "
                        "A 'floater' policy gives one shared sum insured to a group of family members. "
                        "A 'joint' or multi-generational cover (SP020) uses individual SI per member "
                        "plus a shared pool — best for large joint families with elderly parents.",
        "valid_values": {
            "individual":      "Cover for the applicant only",
            "floater_nuclear":  "Cover for applicant + spouse + up to 3 children (2–4 members)",
            "floater_joint":    "Cover for applicant + spouse + children + parents (5–8 members, 3 generations)",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    8,
        "question_text": "Is this just for you, or would you like to cover family members too — "
                         "spouse, children, or parents?",
        "example":      "floater_nuclear",
    },

    {
        "key":          "family_size",
        "layer":        "user",
        "type":         "int",
        "label":        "Number of family members to cover",
        "description":  "The total count of people (including the applicant) to be covered under "
                        "one policy. Affects floater premium and whether SP020 (max 8 members, "
                        "3 generations) is the right product.",
        "valid_values": None,
        "unit":         "count",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    8,     # asked in same turn as family_cover
        "question_text": "How many people in total would you like to cover?",
        "example":      4,
    },

    {
        "key":          "si_preference",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Sum insured preference",
        "description":  "The amount of financial cover the person wants per policy year. "
                        "This is a plan-level input — it selects the tier within the already-resolved "
                        "product, not the product itself. Should be asked after product is confirmed.",
        "valid_values": {
            "1_2L":    "₹1–₹2 lakhs — micro products (SP005, SP019)",
            "3_5L":    "₹3–₹5 lakhs — standard individual cover",
            "10_25L":  "₹10–₹25 lakhs — comprehensive cover",
            "50L_plus":"₹50 lakhs or more — premium and global products",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    9,
        "question_text": "How much cover would you like — ₹3 lakhs, ₹5 lakhs, ₹10 lakhs, or more?",
        "example":      "10_25L",
    },

    # ── Analytics-only user fields (never asked; inferred or logged) ─────────

    {
        "key":          "session_id",
        "layer":        "user",
        "type":         "str",
        "label":        "Session ID",
        "description":  "Unique identifier for this conversation session. Used for analytics "
                        "and to reconstruct the conversation for quality review.",
        "valid_values": None,
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "sess_20240613_abc123",
    },

    {
        "key":          "language",
        "layer":        "user",
        "type":         "str",
        "label":        "Preferred language",
        "description":  "The language the user is speaking in. The agent is voice-based and every "
                        "response is routed through a translation agent, so this drives the output "
                        "language. Expected values: 'hindi', 'english', or 'hinglish' (code-mixed "
                        "Hindi-English). Also used to highlight multilingual products (SP019 supports "
                        "11 regional languages).",
        "valid_values": None,
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "hinglish",
    },

    {
        "key":          "conversation_stage",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Conversation stage",
        "description":  "The current stage of the conversation pipeline. Controls which modules "
                        "are active: info_gathering feeds the next_question_generator; retrieval "
                        "runs the scorer; recommendation activates the feature_presenter; "
                        "rag_open handles free-form queries.",
        "valid_values": {
            "info_gathering":  "Collecting schema fields — next_question_generator is active",
            "retrieval":       "Schema sufficiency gate passed — scorer running",
            "recommendation":  "Product resolved — feature_presenter composing response",
            "rag_open":        "User asked something outside structured flow — RAG active",
            "closed":          "Conversation ended (purchase, drop-off, or transfer)",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "info_gathering",
    },

    {
        "key":          "drop_off_reason",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "Drop-off reason",
        "description":  "Why the user left without purchasing. Captured for analytics. "
                        "Only populated when conversation_stage = closed without purchase.",
        "valid_values": {
            "price":          "User found premium too expensive",
            "waiting_period": "User objected to waiting period length",
            "coverage_gap":   "Product didn't cover what user needed",
            "competitor":     "User chose another insurer",
            "not_ready":      "User wanted more time to decide",
            "technical":      "Call dropped or technical failure",
            "unknown":        "Reason not captured",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "waiting_period",
    },

    {
        "key":          "user_intent",
        "layer":        "user",
        "type":         "str_enum",
        "label":        "User intent",
        "description":  "The most recent intent classified for the user's turn. Logged for analytics "
                        "and call review. Mirrors the IntentSignal enum in fsm.py — keep the two in "
                        "sync. Set by the orchestrator after intent classification; never asked.",
        "valid_values": {
            "prospective":         "Keen to buy / ready to proceed",
            "inquiry":             "Asking a clarification or specific question",
            "exploratory":         "Unsure / just looking, low commitment",
            "provide_info":        "Answered a discovery question",
            "ask_policy_question": "Deep policy-text question once a product is resolved",
            "want_human":          "Explicit escalation request",
            "done":                "Finished / satisfied, no purchase intent",
            "frustrated":          "Repeated dissatisfaction",
            "unsafe":              "Prompt-injection or unsafe instruction",
            "explore_more":        "Wants to see other options after a recommendation",
            "unrecognised":        "Could not be classified confidently",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "prospective",
    },


    # ═══════════════════════════════════════════════════════════════════════
    # POLICY ATTRIBUTES — keys used in policy_feature_registry.py
    # ═══════════════════════════════════════════════════════════════════════

    {
        "key":          "product_id",
        "layer":        "policy",
        "type":         "str",
        "label":        "Product code",
        "description":  "The internal identifier for the product (e.g. SP001). Matches the top-level "
                        "key in the registry. Referenced in all inter-module calls.",
        "valid_values": None,
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "SP015",
    },

    {
        "key":          "buyer_types",
        "layer":        "policy",
        "type":         "list_enum",
        "label":        "Eligible buyer types",
        "description":  "The types of buyers who can purchase this product. A product may accept "
                        "multiple buyer types (e.g. SP005 accepts both 'individual' and 'gig_worker'). "
                        "Matched against user.buyer_type as a hard filter — if the user's type is not "
                        "in this list, the product is immediately excluded.",
        "valid_values": {
            "individual":     "Individual or family buyer",
            "employer_large": "Employer with 50+ employees",
            "employer_sme":   "Employer with 7–50 employees",
            "gig_worker":     "Gig economy or informal sector worker",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    None,
        "question_text": None,
        "example":      ["individual", "gig_worker"],
    },

    {
        "key":          "gender",
        "layer":        "policy",
        "type":         "str_enum",
        "label":        "Gender restriction",
        "description":  "Whether this product is restricted to a specific gender. 'any' means no "
                        "restriction. 'female_only' means only female applicants may purchase it.",
        "valid_values": {
            "any":         "No gender restriction",
            "female_only": "Female applicants only (SP007, SP008)",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    None,
        "question_text": None,
        "example":      "any",
    },

    {
        "key":          "entry_age_min",
        "layer":        "policy",
        "type":         "int",
        "label":        "Minimum entry age",
        "description":  "The youngest age at which a new applicant can purchase this product. "
                        "If the user's age is below this, the product is excluded. "
                        "Most products start at 18; SP006 (Senior Shield) starts at 60.",
        "valid_values": None,
        "unit":         "years",
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    None,
        "question_text": None,
        "example":      18,
    },

    {
        "key":          "entry_age_max",
        "layer":        "policy",
        "type":         "int",
        "label":        "Maximum entry age",
        "description":  "The oldest age at which a new applicant can first purchase this product. "
                        "After this age, only renewal of an existing policy is permitted. "
                        "SP009 (Young Star) caps at 35; most others cap at 60 or 65.",
        "valid_values": None,
        "unit":         "years",
        "nullable":     True,
        "hard_filter":  True,
        "ask_order":    None,
        "question_text": None,
        "example":      65,
    },

    {
        "key":          "primary_needs",
        "layer":        "policy",
        "type":         "list_enum",
        "label":        "Primary needs served",
        "description":  "The insurance needs this product is designed to meet. Matched against "
                        "user.primary_need. If the user's primary need is not in this list, "
                        "the product scores very low (may still be valid as a supplement).",
        "valid_values": {
            "hospitalisation": "In-patient hospital cover",
            "critical_illness": "Lump-sum on illness diagnosis",
            "cancer":           "Cancer-specific cover",
            "accident":         "Accidental death and disability",
            "maternity":        "Pregnancy and delivery",
            "top_up":           "Aggregate deductible top-up",
            "international":    "Hospital cover outside India",
            "daily_cash":       "Fixed daily cash on hospitalisation",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    None,
        "question_text": None,
        "example":      ["hospitalisation"],
    },

    {
        "key":          "budget_bands",
        "layer":        "policy",
        "type":         "list_enum",
        "label":        "Budget bands served",
        "description":  "The premium budget bands this product's plans fall into. Matched against "
                        "user.budget_band. A mismatch reduces score but does not hard-exclude "
                        "(the agent may still present it with a note about affordability).",
        "valid_values": {
            "micro":   "Under ₹2,000/year",
            "budget":  "₹2,000–₹10,000/year",
            "mid":     "₹10,000–₹30,000/year",
            "premium": "Over ₹30,000/year",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      ["mid"],
    },

    {
        "key":          "opd_covered",
        "layer":        "policy",
        "type":         "bool",
        "label":        "OPD cover included",
        "description":  "Whether the policy covers outpatient department (OPD) expenses — doctor "
                        "consultations, diagnostics, and prescription medicines — without requiring "
                        "hospitalisation. Absent in entry-level products (SP001, SP004) and all "
                        "fixed-benefit products (SP011, SP012, SP013, SP014).",
        "valid_values": {
            True:  "OPD expenses covered within annual limit",
            False: "No OPD cover — hospitalisation only",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      True,
    },

    {
        "key":          "opd_limit_inr",
        "layer":        "policy",
        "type":         "int",
        "label":        "OPD annual limit",
        "description":  "The maximum amount reimbursed for outpatient expenses in a policy year. "
                        "Ranges from ₹5,000 (SP017 Basic) to ₹50,000 (SP003, SP018 Platinum). "
                        "NULL if opd_covered is False.",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      20000,
    },

    {
        "key":          "maternity_covered",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Maternity cover included",
        "description":  "Whether the policy covers pregnancy-related expenses: normal delivery, "
                        "C-section, pre-natal and post-natal OPD, and newborn hospitalisation. "
                        "Individual policies (SP007) have a 9-month waiting period; group policies "
                        "(SP016, SP017) cover maternity from Day 1.",
        "valid_values": {
            True:  "Maternity expenses covered (delivery, pre/post-natal, newborn)",
            False: "No maternity cover",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      True,
    },

    {
        "key":          "maternity_day1",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Maternity covered from Day 1",
        "description":  "Whether maternity is covered without any waiting period. True only for "
                        "group products (SP016, SP017) where Day 1 PED and maternity cover is "
                        "an IRDAI-mandated feature of group insurance. Individual policies require "
                        "a 9-month wait.",
        "valid_values": {
            True:  "No waiting period — maternity covered from first day of policy",
            False: "Standard 9-month waiting period applies",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "ivf_covered",
        "layer":        "policy",
        "type":         "bool",
        "label":        "IVF / fertility treatment covered",
        "description":  "Whether the policy covers in-vitro fertilisation (IVF) and other assisted "
                        "reproductive technology (ART). Only SP007 Plan B covers this (up to ₹2 lakhs, "
                        "3 attempts). SP016 Elite offers it as an optional rider.",
        "valid_values": {
            True:  "IVF and ART covered (SP007 Plan B only as standard)",
            False: "No fertility treatment cover",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "no_room_rent_cap",
        "layer":        "policy",
        "type":         "bool",
        "label":        "No room rent cap",
        "description":  "Whether the policy places a daily limit on the hospital room rent it will pay. "
                        "This matters because if the actual room rent exceeds the cap, the insurer "
                        "applies a proportional deduction to the entire bill — not just the room. "
                        "True = no cap (room at any rate covered); False = cap applies.",
        "valid_values": {
            True:  "No daily room rent cap — any room type covered",
            False: "Room rent capped (usually 1–2% of SI per day)",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      True,
    },

    {
        "key":          "copay_type",
        "layer":        "policy",
        "type":         "str_enum",
        "label":        "Co-pay type",
        "description":  "Whether the insured must pay a percentage of every claim themselves. "
                        "A co-pay reduces the premium but means the policyholder always pays "
                        "something out-of-pocket on every claim. 'none' = insurer pays 100%.",
        "valid_values": {
            "none":          "No co-pay — insurer covers 100% of admissible claim",
            "flat_pct":      "Fixed percentage on every claim (e.g. 5% on SP004)",
            "age_based_pct": "Co-pay percentage increases with age (SP006: 10% at 60–69, 20% at 70+)",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "none",
    },

    {
        "key":          "copay_pct",
        "layer":        "policy",
        "type":         "float",
        "label":        "Co-pay percentage",
        "description":  "The flat co-pay percentage applied to every claim. "
                        "0.05 = 5% co-pay (insured pays 5%, insurer pays 95%). "
                        "NULL if copay_type is 'none' or 'age_based_pct'.",
        "valid_values": None,
        "unit":         "pct_0_1",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      0.05,
    },

    {
        "key":          "consumables_covered",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Consumables covered",
        "description":  "Whether single-use items charged by the hospital — gloves, syringes, "
                        "PPE kits, IV sets — are covered. These are excluded in many policies and "
                        "can add 10–20% to a hospital bill. Covered in SP002, SP003, SP016, SP018, SP020.",
        "valid_values": {
            True:  "Hospital consumables included in the claim",
            False: "Consumables excluded — patient pays these separately",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      True,
    },

    {
        "key":          "pre_hosp_days",
        "layer":        "policy",
        "type":         "int",
        "label":        "Pre-hospitalisation days covered",
        "description":  "The number of days before hospital admission for which related medical "
                        "expenses (consultations, tests, medicines) are reimbursed, provided the "
                        "hospitalisation happens and is covered. Ranges from 30 (SP004, SP019) "
                        "to 60 (SP002, SP003, SP015, SP018).",
        "valid_values": None,
        "unit":         "days",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      60,
    },

    {
        "key":          "post_hosp_days",
        "layer":        "policy",
        "type":         "int",
        "label":        "Post-hospitalisation days covered",
        "description":  "The number of days after discharge for which follow-up medical expenses "
                        "are covered — doctor visits, medicines, physiotherapy. SP002 leads at "
                        "180 days; SP015 (Heart+Diabetes) provides 120 days for chronic conditions.",
        "valid_values": None,
        "unit":         "days",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      90,
    },

    {
        "key":          "domiciliary_covered",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Domiciliary hospitalisation covered",
        "description":  "Whether treatment taken at home — when the patient cannot be moved to a "
                        "hospital for medical reasons — is counted as hospitalisation and covered. "
                        "Important for elderly or immobile patients. Covered in SP006 (seniors), "
                        "SP010, SP018, SP020.",
        "valid_values": {
            True:  "Home treatment qualifies as hospitalisation if medically certified",
            False: "Only in-patient hospital admissions qualify",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      True,
    },

    {
        "key":          "domiciliary_limit_inr",
        "layer":        "policy",
        "type":         "int",
        "label":        "Domiciliary cover limit",
        "description":  "The maximum amount per year claimable for domiciliary (home) treatment. "
                        "NULL if domiciliary_covered is False.",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      50000,
    },

    {
        "key":          "international_cover",
        "layer":        "policy",
        "type":         "bool",
        "label":        "International cover",
        "description":  "Whether the policy covers hospitalisation outside India. "
                        "Only SP003 (Protect Global) provides full planned and emergency "
                        "international cover. SP012 (Personal Accident) provides worldwide "
                        "accident cover only.",
        "valid_values": {
            True:  "Hospitalisation outside India covered (cashless or reimbursement)",
            False: "Cover limited to India only",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  True,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "restoration_type",
        "layer":        "policy",
        "type":         "str_enum",
        "label":        "Restoration type",
        "description":  "How the sum insured is replenished if exhausted mid-year by a large claim. "
                        "Without restoration, once the SI is used up, the insured is unprotected "
                        "for the rest of the year. 'unlimited_different_illness' (SP002, SP003) is "
                        "the strongest form.",
        "valid_values": {
            "none":                          "No restoration — SI exhaustion = no further cover that year",
            "once_different_illness":        "SI refilled once per year, only for a different illness",
            "unlimited_different_illness":   "SI refilled unlimited times per year for different illnesses",
            "once_any_illness":              "SI refilled once per year for any illness (same or different)",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "once_different_illness",
    },

    {
        "key":          "ayush_pct_si",
        "layer":        "policy",
        "type":         "float",
        "label":        "AYUSH cover (% of SI)",
        "description":  "The proportion of the sum insured available for treatments under Ayurveda, "
                        "Yoga, Unani, Siddha, and Homeopathy in AYUSH-registered hospitals. "
                        "Ranges from 0 (SP007) to 1.0 = 100% (SP003).",
        "valid_values": None,
        "unit":         "pct_0_1",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      0.25,
    },

    {
        "key":          "cashless_hospitals",
        "layer":        "policy",
        "type":         "int",
        "label":        "Cashless hospital network size",
        "description":  "The number of empanelled hospitals where the insured can get treatment "
                        "without paying upfront — the insurer settles directly with the hospital. "
                        "12,000 for most products; 14,000 for premium products. 0 for fixed-benefit "
                        "products (SP011, SP012) where cashless is not applicable.",
        "valid_values": None,
        "unit":         "count",
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      14000,
    },

    {
        "key":          "air_ambulance_covered",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Air ambulance covered",
        "description":  "Whether the cost of helicopter or air evacuation to a hospital is covered. "
                        "Relevant for buyers in remote areas or frequent travellers. "
                        "Available in SP002, SP003, SP016 Elite, SP018, SP020.",
        "valid_values": {
            True:  "Air ambulance / helicopter evacuation costs covered",
            False: "Air ambulance not covered",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "mental_health_inpatient",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Mental health in-patient cover",
        "description":  "Whether psychiatric hospitalisation is covered on par with physical illness. "
                        "Mandated by the Mental Healthcare Act 2017 for most products. "
                        "Absent only in micro products (SP005, SP019) and purely accident/benefit products.",
        "valid_values": {
            True:  "In-patient psychiatric treatment covered within SI",
            False: "Mental health hospitalisation not covered",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      True,
    },

    {
        "key":          "mental_health_opd",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Mental health OPD cover",
        "description":  "Whether outpatient therapy sessions (psychologist, psychiatrist) are covered "
                        "within the OPD benefit. Present in SP002, SP008, SP009, SP016, SP018, SP020. "
                        "SP008 (Women Wellness) leads at ₹50,000/year for 12 sessions.",
        "valid_values": {
            True:  "Therapy and psychiatry OPD sessions covered",
            False: "Mental health OPD not covered",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "critical_illness_lumpsum",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Critical illness lump-sum benefit",
        "description":  "Whether the policy pays a fixed lump sum upon diagnosis of a covered "
                        "critical illness, regardless of actual treatment costs. The payout can be "
                        "used for anything — treatment, loan repayment, income replacement. "
                        "Core to SP013 (20 conditions) and SP014 (cancer-specific).",
        "valid_values": {
            True:  "Lump-sum paid on diagnosis (no bills needed)",
            False: "No fixed benefit on critical illness diagnosis",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "cancer_stage_benefit",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Cancer stage-based benefit",
        "description":  "Whether the payout varies by the stage of cancer at diagnosis. "
                        "Unique to SP014: CIS = 25% SI, Stage I = 50% SI, Stage II–IV = 100% SI, "
                        "plus monthly income protection. Allows payouts at early stages before "
                        "cancer becomes advanced.",
        "valid_values": {
            True:  "Payout scales with cancer stage at diagnosis (SP014 only)",
            False: "No stage-based cancer benefit",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "income_protection",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Income protection benefit",
        "description":  "Whether the policy pays a monthly amount during treatment or disability. "
                        "SP014 pays ₹10–30K/month for up to 24 months during cancer treatment. "
                        "SP012 pays 1% of CSI per week (TTD benefit) for up to 104 weeks after accident.",
        "valid_values": {
            True:  "Monthly income replacement during illness or disability",
            False: "No income protection — lump sum or indemnity only",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "hospital_daily_cash",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Hospital daily cash benefit",
        "description":  "Whether the policy pays a fixed daily amount for every day of hospitalisation, "
                        "regardless of actual bills. Unique to SP011. No receipts or itemised bills "
                        "needed — only a discharge summary. Can be claimed alongside any indemnity policy.",
        "valid_values": {
            True:  "Fixed cash per hospitalisation day regardless of bills (SP011 only)",
            False: "No daily cash benefit",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "accidental_death_benefit",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Accidental death benefit",
        "description":  "Whether the policy pays the full Capital Sum Insured to the nominee if "
                        "the insured dies as a direct result of an accident. Core to SP012 (Personal "
                        "Accident Shield). Also built into SP019 at ₹1 lakh.",
        "valid_values": {
            True:  "100% of CSI paid to nominee on accidental death",
            False: "No accidental death benefit",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "disability_benefit",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Permanent disability benefit",
        "description":  "Whether the policy pays for permanent loss of a limb, organ, or function "
                        "due to an accident. PTD (total disability) = 100% CSI. PPD (partial) = "
                        "percentage of CSI per a defined schedule. TTD (temporary) = weekly income "
                        "replacement. Only in SP012.",
        "valid_values": {
            True:  "Covers permanent total, partial, and temporary total disability from accidents",
            False: "No disability benefit",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "women_specific_cancer_day1",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Women-specific cancers from Day 1",
        "description":  "Whether breast, cervical, ovarian, and uterine cancers are covered without "
                        "waiting period. Unique to SP008 (Women Wellness), which also includes "
                        "annual mammogram, Pap smear, and HPV test as part of the cancer screening benefit.",
        "valid_values": {
            True:  "Breast, cervical, ovarian, uterine cancers covered from Day 1 (SP008 only)",
            False: "Standard waiting periods apply for cancer",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "wearable_discount",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Wearable health reward discount",
        "description":  "Whether wearing a fitness tracker and meeting activity goals earns a "
                        "premium discount at renewal. SP018 (AI Health Companion) offers up to 30% "
                        "off for users scoring 90–100 on the weekly health score. SP009 (Young Star) "
                        "offers up to 10% for step targets.",
        "valid_values": {
            True:  "Premium discount earned through wearable health data",
            False: "No wearable-linked discount",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "ai_health_engine",
        "layer":        "policy",
        "type":         "bool",
        "label":        "AI health engine",
        "description":  "Whether the product includes an AI-powered health monitoring platform "
                        "(SwasthyaAI™) that ingests wearable data, computes a weekly health score, "
                        "and generates personalised nudges. Unique to SP018.",
        "valid_values": {
            True:  "SwasthyaAI™ platform included — weekly health score, personalised coaching",
            False: "No AI health engine",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "genetic_testing",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Genetic testing included",
        "description":  "Whether a genomic risk panel (cardiovascular, cancer, pharmacogenomics, "
                        "diabetes markers) is provided free at enrolment. Unique to SP018. "
                        "Genetic data is never used for claims adjudication or premium revision.",
        "valid_values": {
            True:  "One free genomic panel at enrolment (SP018 only)",
            False: "No genetic testing benefit",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "cdmp",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Chronic Disease Management Programme",
        "description":  "Whether the policy includes a structured CDMP: a dedicated care coordinator, "
                        "medication reminders, quarterly biometric reviews, and health coaching for "
                        "chronic conditions. Present in SP006 (seniors), SP015 (diabetes/cardiac), "
                        "SP020 (G1 generation).",
        "valid_values": {
            True:  "CDMP care coordinator assigned from Day 1",
            False: "No structured chronic disease management programme",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "aadhaar_only_enrollment",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Aadhaar-only enrollment",
        "description":  "Whether the policy can be purchased using only an Aadhaar card — no bank "
                        "statements, salary slips, or medical forms needed. Critical for the "
                        "informally employed. Applies to SP005 and SP019.",
        "valid_values": {
            True:  "Aadhaar eKYC sufficient — no other documents needed",
            False: "Standard documentation required (PAN, address proof, etc.)",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "pmjay_coordination",
        "layer":        "policy",
        "type":         "bool",
        "label":        "PM-JAY coordination",
        "description":  "Whether the policy is designed to supplement Ayushman Bharat (PM-JAY) by "
                        "filling its gaps — specifically OPD cover and primary care, which PM-JAY "
                        "does not cover. Unique to SP019 (Gig Worker Shield).",
        "valid_values": {
            True:  "Complements PM-JAY — covers OPD and gaps PM-JAY doesn't (SP019 only)",
            False: "No PM-JAY coordination",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "hr_portal",
        "layer":        "policy",
        "type":         "bool",
        "label":        "HR portal / employer admin",
        "description":  "Whether the policy includes an employer-facing admin portal for bulk "
                        "employee onboarding, e-card generation, claims dashboard, and HRMS integration. "
                        "Available in SP016 (full HRMS API) and SP017 (simplified owner dashboard).",
        "valid_values": {
            True:  "HR portal with bulk onboarding and claims dashboard",
            False: "No employer portal",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "aggregate_deductible",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Aggregate deductible (top-up)",
        "description":  "Whether this policy uses an aggregate annual deductible — meaning the "
                        "insured absorbs all expenses up to a threshold across the whole year "
                        "(not per event), and the policy pays everything above it. "
                        "Unique to SP010. The key distinction from a simple top-up, which resets "
                        "the deductible at each hospitalisation.",
        "valid_values": {
            True:  "Annual aggregate deductible — accumulates across all hospitalisations (SP010 only)",
            False: "No aggregate deductible structure",
        },
        "unit":         None,
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      False,
    },

    {
        "key":          "auto_si_increase_pct",
        "layer":        "policy",
        "type":         "float",
        "label":        "Auto SI increase (% per year)",
        "description":  "The percentage by which each member's individual sum insured automatically "
                        "increases each year, without medical examination or fresh underwriting. "
                        "Protects against medical inflation. Unique to SP020 (Family Legacy) at 5%/year. "
                        "NULL for all other products.",
        "valid_values": None,
        "unit":         "pct_0_1",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      0.05,
    },

    {
        "key":          "section_80d",
        "layer":        "policy",
        "type":         "bool",
        "label":        "Section 80D tax deduction eligible",
        "description":  "Whether the premium paid is deductible from taxable income under Section 80D "
                        "of the Indian Income Tax Act. All health insurance products qualify. "
                        "SP012 (Personal Accident) does NOT qualify as it is not classified as health "
                        "insurance under the Act.",
        "valid_values": {
            True:  "Premium deductible under Section 80D (up to ₹25K self; ₹50K senior parents)",
            False: "Not eligible for Section 80D deduction (SP012 only)",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      True,
    },


    # ═══════════════════════════════════════════════════════════════════════
    # WAITING PERIOD ATTRIBUTES (sub-dict of policy, shown as flat here)
    # ═══════════════════════════════════════════════════════════════════════

    {
        "key":          "waiting.initial_days",
        "layer":        "policy",
        "type":         "int",
        "label":        "Initial waiting period (days)",
        "description":  "The number of days from policy inception during which all illness-related "
                        "claims are rejected. Accidents are always exempt from this waiting period. "
                        "Standard is 30 days; critical illness products use 90 days. "
                        "Group products (SP016, SP017) have zero initial wait.",
        "valid_values": None,
        "unit":         "days",
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      30,
    },

    {
        "key":          "waiting.ped_months",
        "layer":        "policy",
        "type":         "int",
        "label":        "PED waiting period (months)",
        "description":  "The number of months from inception before a pre-existing disease (any "
                        "condition diagnosed or treated in the 48 months before buying the policy) "
                        "is covered. Standard is 36–48 months. SP015 reduces this to 12 months "
                        "for diabetes and cardiac conditions. Group products waive it entirely (0).",
        "valid_values": None,
        "unit":         "months",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      36,
    },

    {
        "key":          "waiting.ped_months_diabetes_cardiac",
        "layer":        "policy",
        "type":         "int",
        "label":        "PED waiting period — diabetes/cardiac (months)",
        "description":  "The reduced PED waiting period specifically for diabetes, hypertension, "
                        "and cardiac conditions in SP015 (Heart + Diabetes Care). 12 months vs the "
                        "standard 36–48. Only present on SP015.",
        "valid_values": None,
        "unit":         "months",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      12,
    },


    # ═══════════════════════════════════════════════════════════════════════
    # NCB ATTRIBUTES (sub-dict of policy)
    # ═══════════════════════════════════════════════════════════════════════

    {
        "key":          "ncb.type",
        "layer":        "policy",
        "type":         "str_enum",
        "label":        "NCB (No Claim Bonus) type",
        "description":  "How the policy rewards claim-free years. Two models: SI increase means "
                        "the sum insured grows by a percentage each claim-free year (more cover). "
                        "Premium discount means the renewal premium reduces. SI increase is generally "
                        "better for long-term value; discount is better for short-term cashflow.",
        "valid_values": {
            "si_increase":      "Sum insured increases by X% per claim-free year (max cap applies)",
            "premium_discount": "Renewal premium discounted by X% per claim-free year",
            "none":             "No NCB — no reward for claim-free years",
        },
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "si_increase",
    },

    {
        "key":          "ncb.pct_per_year",
        "layer":        "policy",
        "type":         "float",
        "label":        "NCB rate per claim-free year",
        "description":  "The percentage increase (for SI increase type) or discount (for premium "
                        "discount type) earned per consecutive claim-free year. SP001/SP002 earn "
                        "10% SI per year; SP006/SP015 earn 5%.",
        "valid_values": None,
        "unit":         "pct_0_1",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      0.10,
    },

    {
        "key":          "ncb.max_pct",
        "layer":        "policy",
        "type":         "float",
        "label":        "NCB maximum accumulation",
        "description":  "The ceiling on NCB accumulation. SI increase products cap at 50% "
                        "(meaning SI can grow at most 50% above the original through NCB). "
                        "Premium discount products cap at 20–30%.",
        "valid_values": None,
        "unit":         "pct_0_1",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      0.50,
    },


    # ═══════════════════════════════════════════════════════════════════════
    # PLAN ATTRIBUTES
    # ═══════════════════════════════════════════════════════════════════════

    {
        "key":          "plan_id",
        "layer":        "plan",
        "type":         "str",
        "label":        "Plan identifier",
        "description":  "The unique code for a specific plan tier within a product. "
                        "Format: PRODUCT_CODE-TIER (e.g. SP002-10L, SP010-Gold-5L). "
                        "The plan_id is resolved by the plan_selector after the product is confirmed.",
        "valid_values": None,
        "unit":         None,
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      "SP002-10L",
    },

    {
        "key":          "si_inr",
        "layer":        "plan",
        "type":         "int",
        "label":        "Sum insured",
        "description":  "The maximum amount the insurer will pay in a policy year under this plan. "
                        "For SP020 (Family Legacy) SI is split into individual ISI per member plus "
                        "a shared pool — plan-level SI is NULL and the generation-specific values "
                        "apply. For fixed-benefit products (SP011, SP012) SI is the Capital Sum "
                        "Insured used to calculate benefit payouts.",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      1000000,
    },

    {
        "key":          "annual_premium_inr",
        "layer":        "plan",
        "type":         "int",
        "label":        "Annual premium",
        "description":  "The base annual premium for this plan at the sample profile stated. "
                        "Actual premium varies with age, zone, family composition, occupation "
                        "category (SP012), risk tier (SP015), and smoker status (SP013, SP014). "
                        "All premiums are before GST (18% applicable).",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     False,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      29800,
    },

    {
        "key":          "deductible_inr",
        "layer":        "plan",
        "type":         "int",
        "label":        "Annual aggregate deductible",
        "description":  "The total amount of medical expenses the insured must absorb in a policy "
                        "year before SP010 (Super Top-Up) begins paying. This is aggregate across "
                        "all hospitalisations — not per event. A lower deductible = higher premium "
                        "but lower out-of-pocket threshold. NULL for all products except SP010.",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      500000,
    },

    {
        "key":          "daily_benefit_inr",
        "layer":        "plan",
        "type":         "int",
        "label":        "Daily cash benefit",
        "description":  "The fixed amount paid per day of hospitalisation under SP011 (Hospital "
                        "Daily Cash). ICU stays pay 2× this amount. The actual hospital bill is "
                        "irrelevant — this is a fixed benefit. NULL for all products except SP011.",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      3000,
    },

    {
        "key":          "opd_limit_inr",
        "layer":        "plan",
        "type":         "int",
        "label":        "OPD limit for this plan",
        "description":  "The plan-specific annual OPD reimbursement limit. Higher SI plans generally "
                        "carry higher OPD limits. NULL for plans in products where OPD is not covered.",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      20000,
    },

    {
        "key":          "maternity_normal_inr",
        "layer":        "plan",
        "type":         "int",
        "label":        "Normal delivery benefit",
        "description":  "The lump-sum or reimbursement limit for normal vaginal delivery under "
                        "this plan. NULL for plans where maternity is not covered.",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      75000,
    },

    {
        "key":          "maternity_csection_inr",
        "layer":        "plan",
        "type":         "int",
        "label":        "C-section delivery benefit",
        "description":  "The lump-sum or reimbursement limit for caesarean section delivery. "
                        "Typically 150% of the normal delivery limit. NULL if maternity not covered.",
        "valid_values": None,
        "unit":         "INR",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      112500,
    },

    {
        "key":          "room_rent_cap_pct_si",
        "layer":        "plan",
        "type":         "float",
        "label":        "Room rent cap (% of SI per day)",
        "description":  "The maximum daily room rent covered as a fraction of the sum insured. "
                        "E.g. 0.02 means 2% of SI per day is the cap. If the actual room costs "
                        "more, proportional deduction applies to the whole bill. NULL if no cap.",
        "valid_values": None,
        "unit":         "pct_0_1",
        "nullable":     True,
        "hard_filter":  False,
        "ask_order":    None,
        "question_text": None,
        "example":      0.02,
    },
]


# ---------------------------------------------------------------------------
# LOOKUP HELPERS
# ---------------------------------------------------------------------------

def get_entry(key: str, layer: str | None = None) -> dict | None:
    """
    Return the glossary entry for a given attribute key.

    A few keys exist in more than one layer (``gender`` lives in both the user
    and policy layers; ``opd_limit_inr`` in both policy and plan). Pass ``layer``
    ("user", "policy", or "plan") to disambiguate. When ``layer`` is None the
    first matching entry is returned, preserving the original behaviour.
    """
    for entry in GLOSSARY:
        if entry["key"] == key and (layer is None or entry["layer"] == layer):
            return entry
    return None


def get_label(key: str) -> str:
    """Return the human-readable label for a key."""
    entry = get_entry(key)
    return entry["label"] if entry else key


def get_description(key: str) -> str:
    """Return the LLM-ready description for a key."""
    entry = get_entry(key)
    return entry["description"] if entry else ""


def get_valid_values(key: str) -> dict | None:
    """Return valid values for enum fields."""
    entry = get_entry(key)
    return entry["valid_values"] if entry else None


def get_question(key: str) -> str | None:
    """Return the question text for a user-facing attribute."""
    entry = get_entry(key)
    return entry["question_text"] if entry else None


def by_layer(layer: str) -> list[dict]:
    """Return all entries for a given layer: 'user', 'policy', 'plan'."""
    return [e for e in GLOSSARY if e["layer"] == layer]


def hard_filters() -> list[str]:
    """Return keys of all hard-filter attributes (used by retrieval scorer)."""
    return [e["key"] for e in GLOSSARY if e.get("hard_filter")]


def askable_fields() -> list[dict]:
    """Return user-layer fields that should be asked, sorted by ask_order."""
    return sorted(
        [e for e in GLOSSARY if e["layer"] == "user" and e.get("ask_order") is not None],
        key=lambda e: e["ask_order"]
    )


def llm_context_block(keys: list[str] | None = None) -> str:
    """
    Generate a compact text block suitable for injection into an LLM system prompt.
    If keys is None, returns context for all user-layer fields.
    If keys is a list, returns only those keys.
    """
    entries = GLOSSARY if keys else by_layer("user")
    if keys:
        entries = [e for e in GLOSSARY if e["key"] in keys]

    lines = ["ATTRIBUTE DEFINITIONS", "=" * 40]
    for e in entries:
        lines.append(f"\n[{e['key']}] {e['label']} ({e['type']})")
        lines.append(f"  {e['description']}")
        if e.get("valid_values") and isinstance(e["valid_values"], dict):
            lines.append("  Valid values:")
            for val, meaning in e["valid_values"].items():
                lines.append(f"    {val!r}: {meaning}")
        if e.get("question_text"):
            lines.append(f"  Question: \"{e['question_text']}\"")
    return "\n".join(lines)


def tool_schema() -> list[dict]:
    """
    Return the four LLM tool definitions in Anthropic API format.
    Pass this list directly to the `tools` parameter of the API call.
    These are the ONLY tools the LLM sees — the implementations live
    in retrieval_tools.py and are never exposed to the model.
    """
    # Build the user_schema JSON schema from glossary user-layer fields
    user_schema_properties = {}
    for entry in by_layer("user"):
        key = entry["key"]
        typ = entry["type"]
        prop: dict = {"description": entry["description"]}
        if typ == "bool":
            prop["type"] = "boolean"
        elif typ == "int":
            prop["type"] = "integer"
        elif typ == "float":
            prop["type"] = "number"
        elif typ in ("str_enum", "str"):
            prop["type"] = "string"
            if entry.get("valid_values") and isinstance(entry["valid_values"], dict):
                prop["enum"] = [str(v) for v in entry["valid_values"].keys()]
        elif typ == "list_enum":
            prop["type"] = "array"
            prop["items"] = {"type": "string"}
        else:
            prop["type"] = "string"
        user_schema_properties[key] = prop

    user_schema_object = {
        "type": "object",
        "description": "The current state of the user schema. Include all fields collected "
                       "so far; omit fields not yet known (do not send null — just omit).",
        "properties": user_schema_properties,
    }

    return [
        {
            "name": "filter_products",
            "description": (
                "Filter and rank all 20 Swasthya insurance products based on the user's "
                "collected attributes. Run hard eligibility filters first (buyer_type, age, "
                "gender, primary_need), then soft-score remaining candidates. "
                "Call this as soon as buyer_type, age, and primary_need are known — "
                "you don't need all 9 schema fields before calling. "
                "Returns a ranked list of (product_id, score, matched_on, missing_fields). "
                "If more than 3 products score above 0.4, the result includes a "
                "probe_question field — ask it before calling get_product_features."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "user_schema": user_schema_object
                },
                "required": ["user_schema"],
            },
        },
        {
            "name": "get_product_features",
            "description": (
                "Retrieve the full feature set for a specific product by product_id. "
                "Always call this before describing any product to the user — "
                "never state product features from memory. "
                "Returns a typed dict of all product attributes: coverage, waiting periods, "
                "OPD, maternity, NCB, co-pay, cashless network, and more. "
                "Also returns highlight_fields: the top 3 attributes most relevant to "
                "the current user schema, which you should lead your recommendation with."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product code, e.g. 'SP015'. Get this from filter_products.",
                    },
                    "user_schema": user_schema_object,
                },
                "required": ["product_id", "user_schema"],
            },
        },
        {
            "name": "get_plan_options",
            "description": (
                "Get the available plan tiers for a confirmed product, ranked by fit "
                "to the user's SI preference, family size, and budget band. "
                "Call this after the user has confirmed they want the product, or when "
                "they ask about pricing or coverage amounts. "
                "Returns a ranked list of plans with SI, annual premium, and a "
                "fit_reason string explaining why each plan was ranked where it was."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product code from filter_products, e.g. 'SP002'",
                    },
                    "user_schema": user_schema_object,
                },
                "required": ["product_id", "user_schema"],
            },
        },
        {
            "name": "search_regulations",
            "description": (
                "Search IRDAI regulatory documents for answers about insurance rights, "
                "standard terms, legal requirements, or policy regulations. "
                "Use when the user asks about: portability rights, what IRDAI mandates, "
                "standard exclusion lists, free look periods, ombudsman process, "
                "grievance procedures, or anything not answered by product features. "
                "Do NOT use this to look up product-specific features — use "
                "get_product_features for that. "
                "Returns the top matching regulatory passages with source citations."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query, e.g. 'what is the standard PED waiting period'",
                    }
                },
                "required": ["query"],
            },
        },
    ]


def tool_descriptions() -> str:
    """
    Return a compact plain-text summary of all tools for use in the system prompt.
    Tells the LLM what tools exist and when to use each one.
    """
    return """
AVAILABLE TOOLS — use these to retrieve all product and regulatory information.
Never state product features, premiums, or waiting periods from memory.

1. filter_products(user_schema)
   → Call once buyer_type + age + primary_need are known.
   → Returns ranked product_ids. If >3 candidates, ask the probe_question first.

2. get_product_features(product_id, user_schema)
   → Call before describing any product. Returns full feature dict + highlight_fields.
   → highlight_fields tells you which 3 attributes to lead the recommendation with.

3. get_plan_options(product_id, user_schema)
   → Call after product is confirmed, or when user asks about price/coverage amount.
   → Returns ranked plans with premiums and fit_reason.

4. search_regulations(query)
   → Call for questions about IRDAI rules, rights, portability, exclusion lists.
   → Do NOT use for product-specific features.

RULE: If you have not called get_product_features for a product, you cannot describe it.
"""


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_fields = by_layer("user")
    policy_fields = by_layer("policy")
    plan_fields = by_layer("plan")

    print(f"Total entries : {len(GLOSSARY)}")
    print(f"User layer    : {len(user_fields)}")
    print(f"Policy layer  : {len(policy_fields)}")
    print(f"Plan layer    : {len(plan_fields)}")

    print(f"\nHard filters: {hard_filters()}")

    print("\nAskable fields in order:")
    for f in askable_fields():
        print(f"  [{f['ask_order']}] {f['key']} — {f['label']}")

    print("\nLLM context block (user fields only):\n")
    print(llm_context_block())

