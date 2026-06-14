# AI-Insurance-Sales-Agent
This is for sarvam's assignment

# Swasthya AI Insurance Conversational Core — MVP Architectural Blueprint

Welcome, AI Developer / Coding Assistant. This repository contains a state-driven, semi-deterministic health insurance onboarding conversational engine. It is purpose-built as an MVP for a low-latency voice/text framework. 

The defining architecture of this system is the absolute decoupling of **State Machine Tracking/Data Queries (Deterministic Python)** from **Conversational Phrasing/Wording Generation (Stochastic LLM)**.

---

## Quick Start (Environment + OpenAI)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create your local environment file:
```bash
cp .env.example .env
```

3. Set your API key in `.env`:
```env
OPENAI_API_KEY=your_key_here
SARVAM_API_KEY=your_key_here
```

4. Run the text-mode loop:
```bash
python main.py
```

### Centralized Model Selection
- All model names are controlled from one place: `settings.py` (`ModelCatalog`).
- You can switch models either by editing `settings.py` defaults or by setting these `.env` keys:
      - `OPENAI_MODEL_INTENT`
      - `OPENAI_MODEL_RESPONSE`
      - `OPENAI_MODEL_VALIDATOR`
      - `OPENAI_MODEL_TRANSLATION`

### Fallback Mode
- If `OPENAI_API_KEY` is missing, the app still runs in deterministic fallback mode for current text testing.

### Sarvam Key
- `SARVAM_API_KEY` is now part of app config and will be used in upcoming voice STT/TTS integration.

---

## 1. Core Project Intent & Guardrails

The goal of this system is to onboard customers, qualify their profiles against a 20-product suite, handle underwriting logic deterministically, and present an optimized, legally compliant health insurance proposal without conversational rambling[cite: 1, 11].

### Core Directives for Code Generation:
*   **No Guessing / Hallucinating Data:** All premiums, sums insured, room rent caps, and coverage parameters are hard-coded inside `policy_feature_registry.py`[cite: 13, 15]. The LLM must *never* deduce numbers from memory.
*   **Statically Encapsulated State Machine:** Transitions between conversational phases are written in explicit Python logic inside `fsm.py` [_next_state()][cite: 12, 13]. The LLM can *never* decide when to transition states or bypass schema validations[cite: 12, 16].
*   **Isolated Data Mutations:** Every user profile update must travel exclusively through `UserSchema.set()`[cite: 16]. This ensures runtime type checking, format compliance, and programmatic error capturing[cite: 16].

---

## 2. Directory & Component Structure

Keep your file layout flat and modularized exactly as mapped below so context stays fully scannable:

```text
swasthya-agent-mvp/
├── data/
│   ├── policy_regulations_rag_ready.json # IRDAI Legal & Portability text dataset[cite: 1]
│   └── treatment_costs.json              # Surgery & Daily Ward financial matrix[cite: 11]
├── attribute_glossary.py                 # Single source of truth for schema variables & enums[cite: 11]
├── user_schema.py                        # Dataclass tracking user state, validation, completeness[cite: 11, 16]
├── policy_feature_registry.py            # Static specifications for all 20 health plans
├── retrieval_tools.py                    # Multi-tier algorithmic ranking, sorting & RAG engines[cite: 11, 15]
├── fsm.py                                # State Machine declarations and sub-agent phrasers
├── prompts_template.py                   # Centralized repository for all system prompts[cite: 11]
├── orchestrator.py                       # Main transaction operational loop and sync gate[cite: 11]
└── main.py                               # Non-voice interactive CLI test and scenario harness[cite: 11]

## 3. Finite State Machine (FSM) LayoutThe session runtime maps strictly to the 7 Conversational States declared inside FSMState:  

S0_START: Initial entry pipeline configuration zone.  
S1_DISCOVERY: Information Gathering loop. Evaluates UserSchema gaps turn-by-turn based on the glossary priority queue (ask_order).  
S2_RECOMMENDATION: Consultation phase triggered automatically when sufficiency_score benchmarks pass. Runs the mathematical match matrix and presents a tightly grounded proposal text.  
S3_POLICY_QA: Targeted contextual RAG parsing. Active when a user asks complex follow-ups regarding the recommended plan.  
S4_CLOSURE: End of cycle. Generates document pack summaries and finalizes the CRM sync.  S5_HUMAN_HANDOFF: Fail-safe exit block. Triggers human intervention if the user requests an agent, displays severe frustration, or fails validation gates 3 consecutive times.  S6_BLOCKED: Anti-jailbreak wall[cite: 12]. Immediately isolates and locks the environment upon detection of prompt injection strings[cite: 11, 12].

## 4. End-to-End System Turn Execution Flow

[User String Input]
       │
       ▼
 1. Security Check      ──► Scans for injection strings -> Immediately drops to S6_BLOCKED[cite: 12]
       │
       ▼
 2. M0.1 Extractor Pass ──► Calls Claude Core to process Intent & parse glossary variables[cite: 11]
       │
       ▼
 3. Validation Sync     ──► Passes variables to schema.set(). If SchemaValidationError pops[cite: 16]:
       │                     └─► Rolls back turn, increments error count -> Activates ClarifyingAgent[cite: 11]
       │
       ▼ (If Schema Valid)
 4. Retrieval Scoring   ──► If mandatory entries exist, executes algorithmic filter_products()[cite: 15]
       │                     └─► Updates top_candidates & sets resolved_product_id[cite: 12, 16]
       │
       ▼
 5. FSM Transition Gate ──► Compares state against _next_state() logic. Mutates state if valid
       │
       ▼
 6. Response Synthesis  ──► Renders language output based on active state criteria:
                             ├─► S1: Run NextQuestionAgent using glossary question blueprints[cite: 12]
                             ├─► S2: Run PolicySummaryAgent matching registry database snippets[cite: 12]
                             └─► S3: Run Local RAG tool locked ONLY to resolved_product_id.txt


## 5. Coding Guidelines for Copilot/Cursor Agents
When writing additions or debugging modules inside this project workspace, you must adhere to these structural rules:

Do Not Introduce External Frameworks: Do not install LangChain, LlamaIndex, CrewAI, or additional state machine libraries. The core engine runs on native Python variables, simple dictionaries, and standard data frames[cite: 12, 13]. Unless excplictly mentioned by me.

Prompts Must Stay Isolated: Never place raw string prompts inside logical functions. Every new instruction block must be saved inside prompts_template.py as an uppercase variable string and explicitly imported[cite: 11].

Enforce Enum Ingestion Rules: Never append loose string configurations to the schema. All allowed types must map perfectly to the literals declared inside attribute_glossary.py and policy_feature_registry.py[cite: 11, 13].

Maintain Snippet Grounding on RAG Looks: When writing file reading paths for deep Q&A passes, ensure the script isolates execution exclusively to swasthya_{resolved_product_id}.txt. Never loop over unverified asset lists globally.