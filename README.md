# Swasthya AI Insurance Sales Agent



Video walkthrough of the architecture : https://drive.google.com/file/d/15SuntDuTjuN7K-Bl5P8V_BDTeAZDszG0/view?usp=sharing


Demo 1: https://drive.google.com/file/d/1S2VXazfHXp5NFaaCMZarcZxNxblLC83G/view?usp=sharing


Demo 2: https://drive.google.com/file/d/19p5UfnTSThwMfvBA7uMelhvGTUMomtbg/view?usp=sharing

**What I built:** A voice-first, multilingual health insurance sales conversational agent that discovers customer needs, recommends policies from a 20-product (dummy created for this project) suite, answers grounded questions about a specific policy or health insurance in general, recommends most suitable policies, handles price objections with real treatment-cost data, and escalates any conversation to human if the user is frustrated — in English, Hindi, or Hinglish. On a confirmed purchase it records the sale and sends a thank-you WhatsApp confirmation via Twilio.

---

## Overview

The sales agent:
- **Discover** the customer's needs in a minimal number of turns (age, who's covered, primary need, budget). It is restricted to recommend before knowing "minimum attributes" of a user, defined in the user schema.
- **Recommend** the best-fit policy from Swasthya's suite via algorithmic filtering on the discovered profile. The LLM never decides which policy to sell, it extracts the NER from user's natural language into a structured format that is used to update specific attributes in the User schema, call a deterministic function to hard filter + soft-score the 20 policies and extract the best one.  
- **Ground everything** in real data: product features, plan tiers/premiums, treatment costs, IRDAI regulations — and never invent a number.
- **Handle objections** (price, waiting periods) with empathy and evidence, then pivot to close.
- **Close** by confirming product + plan tier, recording the purchase, and sending a WhatsApp confirmation.
- **Speak naturally** in the customer's language (text or voice) and keep replies short and spoken-friendly.
- **Escalate to a human** whenever the customer turns frustrated or asks for one, reassuring them their issue will be handled.

## Architecture — How I Work
![alt text](<tentative worflow.jpeg>)
The live pipeline is an **agentic LLM orchestrator**: OpenAI native tool-calling drives the
whole conversation. The model decides, turn by turn, when to discover, recommend, explain,
price-justify, or close — by calling tools. It never invents facts; every number comes from a
tool result.

```
customer turn
  └─► M_16 Sales Agent (OpenAI Agents SDK tool-calling loop)
          ├─ save_profile ............ store discovered profile fields
          ├─ recommend_products ...... rank the 20-policy suite for this profile
          ├─ explain_product ......... product features / coverage details
          ├─ show_plan_options ....... plan tiers + premiums for a product
          ├─ estimate_value_vs_cost .. real treatment costs vs. premium (objection handling)
          ├─ answer_general_question . IRDAI rights / portability / tax via RAG
          └─ finalize_purchase ....... record the sale once product + plan are confirmed
   └─► M_11 Response Queue ........... normalise the reply for voice (numbers, acronyms, length)
   └─► M_13 Analytics Logger ......... persist the turn (intent, tool sequence, schema snapshot)
   └─► M_14 WhatsApp Agent ........... send purchase confirmation (on finalize_purchase turn only)
```

All seven tools wrap the **same retrieval implementations** used everywhere else
(`retrieval_tools.py` + the product registry), so there is exactly one source of truth for
product facts. **M_15 NumericGuardrail** re-checks every figure in the reply against the tool
outputs the model was given (advisory). For voice, **M_10 Translator** renders the reply in the
customer's detected language before **M_11** normalises it.

### Agentic backend

M_16 now runs on the OpenAI Agents SDK (`SalesAgentAgentsSDK`) by default.
Set `OPENAI_AGENTS_TRACE=1` to emit real traces to the OpenAI Traces UI.
The `sdk_trace_id` is written to each turn in `logs/conversation_<session_id>.json`
for cross-reference with the trace entry.

- **Run (text):** `ORCHESTRATION_MODE=agentic python demo_agentic.py`
- **Run (text + traces):** `ORCHESTRATION_MODE=agentic OPENAI_AGENTS_TRACE=1 python demo_agentic.py`
- **Run (voice):** `ORCHESTRATION_MODE=agentic ./run_demo.sh` → browser UI at `localhost:7860`

### Fallback: deterministic FSM
A rule-based finite-state machine is kept purely as a **safety net**. If the LLM is unavailable
or the tool loop errors, the orchestrator transparently hands off to the FSM (`fsm_fallback`),
sharing the same conversation record and retrieval tools so product data stays consistent. The
FSM can also be forced (`ORCHESTRATION_MODE=fsm python main.py`) for fully deterministic,
auditable runs. It adds a few fallback-only agents — probing (M_05), templated policy summaries
(M_07), and policy QA routing (M_08).

---

## User Intents — How to classify what the customer wants

Each turn is classified into one intent (deterministic safety check first, LLM advisory second).
The agentic pipeline uses this for routing emphasis, analytics, and the safety-critical exits
(human handoff, frustration, unsafe input); the FSM fallback uses it for explicit state transitions.

| Intent | Meaning | How I respond |
|--------|---------|---------------|
| **PROSPECTIVE** | Ready to buy or close | Confirm product + plan, then record the purchase |
| **INQUIRY** | A specific question | Answer from grounded tool data |
| **EXPLORATORY** | Just looking, low commitment | Keep discovering the profile or show more options |
| **PROVIDE_INFO** | Answering one of my questions | Extract and store the field, continue discovery |
| **ASK_POLICY_QUESTION** | About product coverage / features | Ground the answer in policy wording and features |
| **WANT_HUMAN** | Explicit request for a human | Escalate immediately (safety exit) |
| **DONE** | Satisfied / finished | Close gracefully |
| **FRUSTRATED** | Repeated dissatisfaction | Escalate after repeated failures |
| **UNSAFE** | Prompt injection / jailbreak attempt | Block the interaction |
| **EXPLORE_MORE** | Wants to see other options | Re-rank and present alternatives |
| **UNRECOGNISED** | Couldn't classify | Ask for clarification; don't escalate on this alone |

---

## Sub-agents — the Specialists

The agentic pipeline runs a small set of specialists; the rest are deterministic fallbacks the
FSM uses when the LLM is unavailable. Each has exactly one responsibility.

| ID | Name | Used in | What it does | Guarantees |
|----|------|---------|---|---|
| **M_01** | Intent Classifier | **Agentic + FSM** | OpenAI intent + confidence classification | Safety-critical intents (UNSAFE, WANT_HUMAN, FRUSTRATED) are always validated deterministically; LLM is advisory only |
| **M_02** | Escalation | FSM fallback | Canned escalation messages (want_human / frustrated) | Terminal; hands off to M_03 |
| **M_03** | Closure | FSM fallback | Canned closure messages; hands off to M_14 if purchased | Terminal |
| **M_04** | Schema Extractor | FSM fallback | Regex + LLM pass to extract missing profile fields | Validates against the attribute glossary enum; deterministic pass first (agentic uses the `save_profile` tool instead) |
| **M_05** | Probing Agent | FSM fallback | Asks the next discovery question from missing fields | Glossary-aligned templates; max 8 discovery turns |
| **M_06** | Policy Retrieval | **Agentic + FSM** | Algorithmic ranking behind the `recommend_products` / `show_plan_options` tools: filters the 20 products by profile, scores by SI preference + budget fit + family size | Returns top candidates + recommended tier; relaxes preference filters (keeping eligibility gates) when nothing matches exactly, so the agent never loops |
| **M_07** | Policy Summary | FSM fallback | Templated spoken summary of the recommended product + tiers | M_15 guardrail on all numbers; deterministic template if guardrail fails |
| **M_08** | Policy QA | FSM fallback | Routes product questions to policy wording, then RAG | Confident-clause check first; otherwise hands to M_09 |
| **M_09** | Agentic RAG | **Agentic + FSM** | Vector search behind `answer_general_question` (regulations) and `explain_product` (policy wording) | Top-6 retrieval; dedupes; LLM synthesises; M_15 guardrail; deterministic fallback |
| **M_10** | Translator | **Agentic + FSM** | Renders replies in the customer's detected language (English / Hindi / Hinglish) | Sarvam API; Hinglish stays code-mixed; deterministic passthrough if no key |
| **M_11** | Response Queue | **Agentic + FSM** | Assembles and normalises the final spoken text | ₹ → "rupees", acronyms → spaced letters (IRDAI → "I R D A I"), Indian-format digit grouping (300000 → "3,00,000"), strips markup, caps at 2,500 chars on a sentence boundary |
| **M_12** | Response Validator | FSM fallback | Checks the reply is safe and grounded | Blocks denylisted over-promise phrases |
| **M_13** | Analytics Logger | **Agentic + FSM** | Logs every turn to `logs/conversation_<session_id>.json` | Captures user message, reply, intent, tool sequence/trace, agents fired, schema snapshot |
| **M_14** | WhatsApp Agent | **Agentic + FSM** | Sends purchase confirmation via Twilio WhatsApp sandbox | Fired on the `finalize_purchase` turn (agentic) or M_03 handoff (FSM); reads `Account_SID` / `Auth_token` / `DEMO_WA_NUMBER`; always logs to `logs/wa_outbox.json` even if the send fails |
| **M_15** | Numeric Guardrail | **Agentic + FSM** | Re-checks every number in the reply against the tool outputs the model was given | Flags ungrounded figures (advisory; doesn't block) |
| **M_16** | Sales Agent (Agents SDK) | **Agentic** | OpenAI Agents SDK tool-calling orchestrator — drives discovery, recommendation, objection handling, and close via 7 tools | Emits real OpenAI Traces (`OPENAI_AGENTS_TRACE=1`); tools reuse shared retrieval implementations; falls back to FSM on error |

### The 7 agentic tools (M_16)
All wrap the same retrieval code in `retrieval_tools.py` / the product registry — one source of truth:

| Tool | Purpose |
|------|---------|
| `save_profile` | Store validated profile fields (buyer_type, age, primary_need, budget, etc.) |
| `recommend_products` | Rank the 20-policy suite for the current profile |
| `explain_product` | Product features / coverage details (policy-wording RAG) |
| `show_plan_options` | Plan tiers + premiums for a chosen product |
| `estimate_value_vs_cost` | Real treatment-cost ranges to contrast with the premium (objection handling) |
| `answer_general_question` | IRDAI rights / portability / free-look / tax via regulations RAG |
| `finalize_purchase` | Record the sale once a product **and** plan tier are confirmed |

---

## Data & Retrieval — Sources of Truth

The agent draws on **four data modalities**, each consumed the way it's best suited to —
structured lookups for facts that must be exact, vector RAG for open-ended questions.

| Modality | Source | Shape | How it's used |
|----------|--------|-------|---------------|
| **Product registry** (structured) | `policy_feature_registry.py` | 20 plans (SP001–SP020) with typed fields: name, sum insured, annual premium, waiting periods, coverage limits (maternity, critical illness, room rent, OPD, day-care) | Direct lookup + algorithmic ranking by `recommend_products` / `show_plan_options` / `explain_product` — no model guessing |
| **Policy wording** (unstructured text) | `data/policies/swasthya_SP001–020.txt` (20 files) | Full policy documents, chunked by section | Vector RAG behind `explain_product` for clause-level coverage questions |
| **Regulations** (semi-structured) | `data/policy_regulations_rag_ready.json` | 5 IRDAI source documents (e.g. Insurance Act 1938 + amendments) organised as chapters → sections | Vector RAG behind `answer_general_question` for rights, portability, free-look, tax, grievance/ombudsman, exclusions |
| **Treatment costs** (structured) | `data/treatment_costs.json` (Python literal dict) | `hospitalization_per_day`, `common_procedures`, `annual_chronic_management` with govt / private / metro-private bands | Direct lookup by `estimate_value_vs_cost` to put a premium in real-rupee perspective |

### User Profile (Schema)
- **Fields:** buyer_type, age, family_size, primary_need, budget_band, pre_existing_conditions, family_cover, sum_insured_preference, language, resolved_product_id, resolved_plan_id, purchased.
- All validated against `attribute_glossary.py` (enum values are the source of truth).
- Tracked in the `UserSchema` dataclass; mutated only via `schema.set()` — in the agentic pipeline via the `save_profile` tool, which rejects any value outside the glossary.

### Vector RAG (LlamaIndex + ChromaDB)
- **1,825 vectors** in `data/vector_store/` (collection `swasthya_rag`), built once via `python -m rag.ingest`.
- **Breakdown by `source_type`:** policy wording **881** · regulations **898** · treatment costs **46**.
- **Embedding:** OpenAI `text-embedding-3-small` (1536-dim).
- **Retrieval:** M_09 queries with top_k=6, dedupes by text, and the LLM synthesises the answer (M_15 guardrail checks any numbers).
- **Ingestion:** Idempotent — drops and rebuilds the collection on re-run. Requires `OPENAI_API_KEY`.

---

## Voice Stack — How I Speak

I use **Sarvam APIs** for voice I/O and **Pipecat** for the real-time pipeline:

| Component | Config | Why |
|-----------|--------|-----|
| **STT** | Sarvam Saaras saaras:v3 in `codemix` mode, auto-detect language | `codemix` returns natural Roman-script Hinglish ("mujhe cover chahiye") instead of Devanagari or translate-mode flattening; leaving language unset lets v3 detect Hindi/English/Hinglish per utterance |
| **Language Detection** | Local regex (Devanagari \u0900–\u097F → hindi; Hinglish markers → hinglish; else english) | No extra network hop; instant |
| **LLM Brain** | OpenAI (gpt-4.1-mini for agentic; gpt-4.1-mini/nano for FSM sub-agents) | Fast, reliable for tool-calling and NER |
| **TTS** | Sarvam Bulbul bulbul:v3, voice priya, language set per turn | `priya` is a low-error v3 female voice; M_11 drives the TTS language per turn via `tts_language_for()`, so Hindi/Hinglish replies are voiced in hi-IN; 24 kHz; temperature 0.5 for consistency |
| **TTS Pre-processing** | M_11 normalization (₹ → "rupees", acronyms → spaced letters, **Indian-format digit grouping** 300000 → "3,00,000", strip markup, 2,500-char cap at sentence boundary) | Ensures Bulbul reads amounts and acronyms correctly and never exceeds the per-request character limit |
| **Transport** | SmallWebRTCTransport (Pipecat) with Silero VAD | Browser mic/speaker UI at localhost:7860; local VAD with stop_secs=0.8 to merge paused phrases into one turn while still allowing barge-in |
| **Latency tracking** | Per-turn elapsed time (STT-final → reply-ready, ms) | Logged in orchestrator_processor.py for performance profiling |

**Pipeline:** Browser mic → WebRTC → Silero VAD → Sarvam STT → Agentic orchestrator (M_16 tool-calling; FSM on fallback) in executor → M_10 translate → M_11 normalize → Sarvam TTS → Browser speaker.

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
- OpenAI (gpt-4.1-mini for the agentic orchestrator; gpt-4.1-mini/nano for sub-agents; text-embedding-3-small for vectors)
- openai-agents 0.17.5 (OpenAI Agents SDK backend, real OpenAI Traces)
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

**Tests:**
- `tests/test_sales_agent_backend_parity.py` — historical migration check that compares legacy and SDK behavior. Keep for reference while old parity logs exist; for current runtime validation, run agentic demos and analytics directly.

**Analytics:**
- `tools/session_analytics.py` — post-session trace analyser. Reads `logs/openai_events.jsonl` and `logs/conversation_<session>.json` and renders three charts into `logs/analytics_<session>.png`:
  - **Chart 1** — Tokens per sub-agent (prompt + completion stacked bar).
  - **Chart 2** — LLM latency per sub-agent (avg ms per call) + per-turn wall-clock latency.
  - **Chart 3** — Conversation flow: each turn's user intent (colour-coded) alongside every tool called / agent fired.

  ```bash
  # latest session
  python tools/session_analytics.py
  # specific session + open window
  python tools/session_analytics.py --session demo_price_anxiety --show
  ```

---

## Limitations & Forward Tasks

**Current constraints (by design):**
- **No streaming TTS:** Each reply waits for full synthesis before audio plays.
- **Single session mode:** One customer at a time per voice server instance.
- **Hinglish only on code-mix:** Not a full Hindi native app; Hinglish is code-mixed English-Hindi.
- **Sarvam STT prompt not supported:** `saaras:v3` does not accept a custom STT prompt parameter; the Pipecat 1.3.0 `SarvamSTTService` would raise `ValueError` if one is passed.
- **No pronunciation dictionary in Pipecat 1.3.0:** `SarvamTTSSettings` does not expose a `pronunciation_dict_id` field in the installed version; acronym pronunciation is handled instead by M_11's text normalization (IRDAI → "I R D A I").

**Known minor issues (not blocking):**
- M_15 guardrail sometimes flags inline plan-tier option numbers (false positive, advisory only).
- `finalize_purchase` plan_id resolution is heuristic (substring match on label); mismatches are logged but don't block the purchase record.

---

## Takeaways

What I learned building this:

- **One source of truth is everything.** Keeping product data, schema validation, and retrieval logic in one place (not scattered in prompts or LLM memory) is the difference between a hallucination-prone chatbot and a sales agent.
- **Guardrails cost less than you think.** M_15 numeric guardrail + M_11 TTS normalization catch most problems at the source; you don't need an expensive validation LLM pass for every reply.
- **Multilingual from the start.** Adding Hinglish support was cheap (language detection + Sarvam translate) and it doubles the addressable market; worth building in day 1.
- **Keep a deterministic fallback.** The agentic tool-calling pipeline is the product; the FSM exists so that if the LLM gets slow or unavailable, you don't go dark — it takes over on the same record and tools.
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

4. Run the agentic pipeline:
```bash
# text demos
ORCHESTRATION_MODE=agentic python demo_agentic.py

# text demos with OpenAI Traces
ORCHESTRATION_MODE=agentic OPENAI_AGENTS_TRACE=1 python demo_agentic.py

# voice (browser UI at localhost:7860)
ORCHESTRATION_MODE=agentic ./run_demo.sh

# optional historical parity test from migration phase
ORCHESTRATION_MODE=agentic python tests/test_sales_agent_backend_parity.py
```

### Centralized Model Selection
- All model names are controlled from one place: `settings.py` (`ModelCatalog`).
- You can switch models either by editing `settings.py` defaults or by setting these `.env` keys:
  - `OPENAI_MODEL_INTENT`
  - `OPENAI_MODEL_RESPONSE`
  - `OPENAI_MODEL_VALIDATOR`
  - `OPENAI_MODEL_TRANSLATION`

### Fallback Mode
- If `OPENAI_API_KEY` is missing or the tool loop errors, the app drops to the deterministic FSM (`ORCHESTRATION_MODE=fsm python main.py` forces it), so it never goes dark.

### Sarvam Key
- `SARVAM_API_KEY` powers the voice STT/TTS (Saaras + Bulbul) and the M_10 translator.
