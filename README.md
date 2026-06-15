# Swasthya AI Insurance Sales Agent

**What I built:** A voice-first, multilingual health insurance sales conversational agent that discovers customer needs, recommends policies from a 20-product suite, answers grounded questions, handles price objections with real treatment cost data, and closes sales — in English, Hindi, or Hinglish. If the user is turned to a prospective buyer, he/she is sent a thank you WhatsApp note.

---

## Overview — What I Do

I am a sales agent, not an FAQ bot. My job is to:
- **Discover** the customer's needs in a minimal number of turns (age, who's covered, primary need, budget).
- **Recommend** the best-fit policy from Swasthya's suite via algorithmic / deterministic filtering on the discovered profile.
- **Ground everything** in real data: product features, plan tiers/premiums, treatment costs, IRDAI regulations and never invent a number.
- **Handle objections** (price, waiting periods) with empathy and evidence, then pivot to close.
- **Close** by confirming product + plan tier, recording the purchase, and notifying the customer by WhatsApp.
- **Speak naturally** in the customer's language (text or voice) and keep replies short and spoken-friendly.
- **Escalate to Human** in any state when user turns frustrated or insists on talking to a human. I reassure the user that his issues will be dealt with. 

## Architecture — How I Work

I run in **one of two modes** selected by the `ORCHESTRATION_MODE` environment variable:

### Mode 1: Deterministic FSM (default)
- **Control:** Explicit finite-state machine (S0–S6) with 15 sub-agents (M_01–M_15).
- **Strength:** Predictable, auditable, every tool call and state transition is logged.
- **Weakness:** Rigid routing; open follow-ups like "explain that more" sometimes go to the wrong state.
- **Run:** `python main.py` (text) or `./run_demo.sh` (voice).

### Mode 2: Agentic LLM (headline demos)
- **Control:** OpenAI native tool-calling — the LLM drives the conversation, calling tools as needed.
- **Strength:** Natural flow from discovery through recommendation to close; handles open questions well.
- **Weakness:** Less auditable; requires guardrails to prevent hallucination.
- **Run:** `ORCHESTRATION_MODE=agentic python demo_agentic.py` (text) or `ORCHESTRATION_MODE=agentic ./run_demo.sh` (voice).

**The tradeoff:** Start deterministic to prove the funnel, then layer agentic on top behind a feature flag. Both engines share one conversation record and one set of retrieval tools, so product data is consistent. If the LLM is unavailable or the tool loop errors, the FSM automatically takes over.

### Shared guarantees (both modes)
- Tools are the **only source of truth** — no number is ever stated from the model's memory.
- **M_15 NumericGuardrail** re-checks every figure in the reply against the tool outputs the model was given.
- **M_11 TTS normalization** prepares text for voice (₹ → "rupees", acronyms → spaced letters, strip markup).
- **M_10 Translator** renders replies in the customer's detected language.

---

## Conversation Flow — Finite States (FSM mode)

I track conversation state explicitly. Here are the seven states and how I move through them:

| State | I am doing | Entry triggers | Exit triggers | Sub-agents |
|-------|-----------|---|---|---|
| **S0_START** | Initializing session | Incoming message | Always → S1 | None |
| **S1_DISCOVERY** | Asking for missing profile fields | Session start or no product resolved yet | Enough fields to recommend → S2; repeated failures → S5; customer done → S4; escalation → S5 | M_05 ProbeQuestion, M_01 Understand, M_04 Schema extract |
| **S2_RECOMMENDATION** | Presenting a product and plan options | Sufficient profile fields collected | Customer agrees → S4; asks a question → S3; escalation → S5 | M_06 Retrieval, M_07 PolicySummary, M_15 Guardrail |
| **S3_POLICY_QA** | Answering detailed questions about the recommended product | Customer asks product-specific questions | Customer ready to buy → S4; ask more discovery questions → S1; escalation → S5 | M_08 PolicyQA, M_09 RAG (vector search), M_15 Guardrail |
| **S4_CLOSURE** | Recording the purchase and saying goodbye | Intent=DONE or PROSPECTIVE with sufficient fields | Terminal | M_03 Closure, M_14 WhatsApp notify |
| **S5_HUMAN_HANDOFF** | Handing off to a human agent | Customer frustration, 3 failures, explicit request, or unsafe content | Terminal | M_02 Escalation, M_03 Closure |
| **S6_BLOCKED** | Rejecting the interaction | Unsafe input detected (prompt injection) | Terminal | None |

---

## User Intents — How I Classify What the Customer Wants

Every turn, I classify the customer's intent and route accordingly:

| Intent | Meaning | Action |
|--------|---------|--------|
| **PROSPECTIVE** | Customer is ready to buy or close | Move to S4 (closure) if product + plan are confirmed |
| **INQUIRY** | Asking a specific question | Answer from grounded data (M_08 / M_09) |
| **EXPLORATORY** | Just looking, low commitment | Keep discovering profile or show more options |
| **PROVIDE_INFO** | Answering one of my questions | Extract and store the data, continue discovery |
| **ASK_POLICY_QUESTION** | Asking about product coverage / features | Ground answer in policy wording and features |
| **WANT_HUMAN** | Explicit request for a human | Escalate immediately (S5) |
| **DONE** | Customer is satisfied / done | Close (S4) |
| **FRUSTRATED** | Repeated dissatisfaction | Escalate (S5) after 3 failures |
| **UNSAFE** | Prompt injection / jailbreak attempt | Block (S6) |
| **EXPLORE_MORE** | Customer wants to see other options | Rerun recommendation; show alternatives |
| **UNRECOGNISED** | I couldn't classify | Ask for clarification; don't escalate on this alone |

---

## Sub-Agents — My 16 Brains

I am powered by 16 specialized sub-agents, each with one responsibility:

| ID | Name | Mode(s) | What it does | Guarantees |
|----|------|---------|---|---|
| **M_01** | Intent Classifier | Both | Calls OpenAI to classify intent + extract confidence | Every safety-critical intent (UNSAFE, WANT_HUMAN, FRUSTRATED) is always validated deterministically; LLM is advisory only |
| **M_04** | Schema Extractor | Both | Regex + LLM pass to extract missing user profile fields | Only updates fields that are missing; validates against attribute glossary enum; deterministic pass always runs first |
| **M_05** | Probing Agent | FSM | Asks the next discovery question based on missing fields | Uses glossary-aligned question templates; max 8 discovery turns |
| **M_06** | Policy Retrieval | Both | Algorithmic ranking: filters 20 products by profile, scores by SI preference + budget fit + family size | Returns top 3 candidates + recommended tier; resolves one product if a clear winner exists |
| **M_07** | Policy Summary | FSM | Presents the recommended product and its plan tiers in spoken language | LLM composes grounded summary; M_15 guardrail checks all numbers; fallback to deterministic template if guardrail fails |
| **M_08** | Policy QA | FSM | Answers customer questions about product coverage | First checks policy wording for confident clauses; if found, answers with LLM; otherwise hands to M_09 (RAG) |
| **M_09** | Agentic RAG | Both | Vector semantic search over policy + regulation + treatment cost corpora | Returns top 6 vectors; LLM synthesizes answer; M_15 guardrail; deterministic fallback |
| **M_10** | Translator | Both | Renders replies in customer's detected language (English / Hindi / Hinglish) | Sarvam API; Hinglish stays code-mixed (roman numerals + Hindi); deterministic passthrough if no key |
| **M_11** | Response Queue | Both | Assembles and normalizes the final spoken text | Converts ₹ → "rupees", acronyms → spaced letters (IRDAI → "I R D A I"), strips markup/brackets/clauses |
| **M_12** | Response Validator | Both | Checks that the reply is safe and grounded | Blocks denylisted over-promise phrases; optionally checks if reply addresses the user's latest query |
| **M_13** | Analytics Logger | Both | Logs every turn to `logs/conversation_<session_id>.json` | Captures full user message, agent response, agents fired, schema snapshot, flags |
| **M_14** | WhatsApp Agent | FSM | Sends purchase confirmation via Twilio WhatsApp sandbox | Reads `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `DEMO_WA_NUMBER`; always logs attempt even if send fails |
| **M_15** | Numeric Guardrail | Both | Re-checks every number in the reply against tool outputs | If a number appears without being in the grounding context, flags it (advisory; doesn't block) |
| **M_16** | Sales Agent (Agentic) | Agentic only | OpenAI tool-calling orchestrator — calls tools to discover, recommend, explain, and close | Native tool-calling; 7 tools reuse retrieval implementations; falls back to FSM if unavailable; M_15 guardrail advisory |
| **M_02** | Escalation | Both | Canned escalation messages (want_human / frustrated) | Terminal; hands off to M_03 |
| **M_03** | Closure | Both | Canned closure messages (thank you / apology); hands off to M_14 if purchased | Terminal |

---

## Data & Retrieval — Sources of Truth

### Product Registry
- **20 health plans** (SP001–SP020: Swasthya Young Star, Maternity Suraksha, Heart + Diabetes, etc.)
- Each plan has: name, description, sum insured, annual premium, waiting periods, coverage limits (maternity, critical illness, room rent cap, OPD, day-care, etc.).
- Hard-coded in `policy_feature_registry.py`; retrieved by M_06 (filtering) and M_07/M_08 (feature details).

### User Profile (Schema)
- **Fields:** buyer_type, age, family_size, primary_need, budget_band, pre_existing_conditions, family_cover, sum_insured_preference, language, resolved_product_id, resolved_plan_id, purchased.
- All validated against `attribute_glossary.py` (enum values are the source of truth).
- Tracked in `UserSchema` dataclass; mutations only via `schema.set()`.

### Vector RAG (LlamaIndex + ChromaDB)
- **1,825 vectors** persisted in `data/vector_store/` (collection `swasthya_rag`), built once via `python -m rag.ingest`.
- **Corpora:**
  - **Policy wording** (881 vectors): chunked product texts (SP001–SP020) with metadata (product_id, section, section_title).
  - **Regulations** (898 vectors): IRDAI policy holder rights, waiting periods, portability, free-look, tax, exclusions.
  - **Treatment costs** (46 vectors): hospitalization per-day rates, common procedures (delivery, surgery, dialysis), chronic management.
- **Embedding:** OpenAI `text-embedding-3-small` (1536-dim).
- **Retrieval:** M_09 calls `get_rag_index().query()` with top_k=6; dedupes by text; LLM synthesizes answer.
- **Ingestion:** Idempotent; drops and rebuilds collection on re-run. Requires `OPENAI_API_KEY`.

### Treatment Costs (data/treatment_costs.json)
- Python dict (not JSON) with hospitalization rates and procedure costs (govt / private / metro_private).
- M_16 tool `estimate_value_vs_cost` maps by concern keyword (delivery, surgery, dialysis, etc.) → real costs.

---

## Voice Stack — How I Speak

I use **Sarvam APIs** for voice I/O and **Pipecat** for the real-time pipeline:

| Component | Config | Why |
|-----------|--------|-----|
| **STT** | Sarvam Saaras saaras:v3 in `transcribe` mode | Preserves language, numbers, code-mixing (essential for Hinglish); no translate-mode flattening |
| **Language Detection** | Local regex (Devanagari \u0900–\u097F → hindi; Hinglish markers → hinglish; else english) | No extra network hop; instant |
| **LLM Brain** | OpenAI (gpt-4.1-mini for agentic; gpt-4.1-mini/nano for FSM sub-agents) | Fast, reliable for tool-calling and NER |
| **TTS** | Sarvam Bulbul bulbul:v3, voice shreya | Steady Indian-English/Hindi delivery; 24 kHz; temperature 0.5 for consistency |
| **TTS Pre-processing** | M_11 normalization (₹ → "rupees", acronyms → spaced letters, strip markup) | Ensures TTS reads numbers and acronyms correctly before Bulbul sees them |
| **Transport** | SmallWebRTCTransport (Pipecat) with Silero VAD | Browser mic/speaker UI at localhost:7860; local VAD with stop_secs=0.8 to avoid over-segmentation |
| **Latency tracking** | Per-turn elapsed time (STT-final → reply-ready, ms) | Logged in orchestrator_processor.py for performance profiling |

**Pipeline:** Browser mic → WebRTC → Silero VAD → Sarvam STT → Orchestrator (FSM or Agentic M_16) in executor → M_10 translate → M_11 normalize → Sarvam TTS → Browser speaker.

---

## Database / Persistence

- **Conversation logs:** `logs/conversation_<session_id>.json` (per-turn user message, assistant reply, agents fired, schema snapshot).
- **LLM call logs:** `logs/llm_calls.log` (every OpenAI / Sarvam API call with prompt, response, cost).
- **Vector store:** `data/vector_store/` (ChromaDB persisted indexes; dropped and rebuilt on ingest re-run).
- **WhatsApp outbox:** `wa_outbox.json` (records of WhatsApp sends for audit).

---

## Tech Stack

**Language & Runtime:**
- Python 3.13 (venv at `.venv/bin/python`)

**LLM & Embeddings:**
- OpenAI (GPT-4 Turbo, GPT-4 Mini; text-embedding-3-small for vectors)
- Sarvam APIs (Saaras v3 STT, Bulbul v3 TTS, Translate)

**Voice & Real-time:**
- Pipecat 1.3.0 (real-time voice pipeline framework)
- Silero (local VAD, Pipecat-built)
- FastAPI + Uvicorn (WebRTC signaling server)

**Data & Retrieval:**
- LlamaIndex 0.11+ (vector index orchestration)
- ChromaDB 0.5+ (vector store)
- Sarsvamai 0.1+ (Sarvam SDK)

**Messaging:**
- Twilio 9.0+ (WhatsApp sandbox sends)

**Dependencies:** See `requirements.txt`.

---

## Limitations & Forward Tasks

**Current constraints (by design):**
- **No streaming TTS:** Each reply waits for full synthesis before audio plays. Next: Sarvam streaming API.
- **Single session mode:** One customer at a time per voice server instance. Next: Concurrent call handling + pooling.
- **Hinglish only on code-mix:** Not a full Hindi native app; Hinglish is code-mixed English-Hindi. Next: Full Hindi intent classification + entity extraction if demand grows.
- **Waiting period determinism:** FSM asks waiting-period questions naively; agentic M_16 is smarter. Next: A/B the two engines on real calls to measure close-rate delta.

**Known minor issues (not blocking):**
- M_15 guardrail sometimes flags inline plan-tier option numbers (false positive, advisory only).
- finalize_purchase plan_id map is heuristic (substring match on label); mismatches are logged but don't fail the purchase.

**Next priorities:**
1. **Record and analyze** real voice demos (price_anxiety + hinglish) for latency and quality.
2. **Run A/B test** FSM vs Agentic M_16 on close rates (both modes fully functional).
3. **Implement streaming TTS** to cut end-to-end latency in half.
4. **Profiling:** Measure per-turn breakdown (STT, LLM, TTS, orchestrator) and optimize slowest path.
5. **Sarvam pronunciation dictionary:** Once exposed through Pipecat, add custom acronym pronunciations (IRDAI, NCB, PED, etc.).
6. **Closer tie-in with Sarvam sales team:** Integrate real policy pricing, underwriting rules, and live premium quotes from Sarvam's platform APIs.

---

## Takeaways

What I learned building this:

- **One source of truth is everything.** Keeping product data, schema validation, and retrieval logic in one place (not scattered in prompts or LLM memory) is the difference between a hallucination-prone chatbot and a sales agent.
- **Guardrails cost less than you think.** M_15 numeric guardrail + M_11 TTS normalization catch most problems at the source; you don't need an expensive validation LLM pass for every reply.
- **Multilingual from the start.** Adding Hinglish support was cheap (language detection + Sarvam translate) and it doubles the addressable market; worth building in day 1.
- **Deterministic + Agentic = best of both worlds.** Start deterministic, layer agentic on top behind a flag. If the LLM gets slow or unavailable, you don't go dark; the FSM takes over.
- **Voice latency kills UX.** Even a 2-second delay between STT and TTS feels robotic. Every 100ms saved in the orchestrator is perceptible to the customer.
- **Operators care about logs.** Full conversation logs + LLM call logs + guardrail flags let you debug live customer issues without replaying them. Build logging in, not as an afterthought.

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