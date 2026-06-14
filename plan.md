# AI Insurance Sales Agent — Implementation Plan

## Goal
A deterministic, auditable health-insurance sales agent built as a `subagents/` package
of single-responsibility modules (M_01–M_15) behind ONE uniform contract
(`run(ctx: AgentContext) -> AgentResult`), driven by a deterministic FSM and orchestrator.
LLM reasoning is layered **on top of** the deterministic core, never replacing it: every
LLM-bound agent keeps its Phase 0 deterministic behavior as a fallback.

## Working Rules
- Keep edits targeted and surgical; explain every change.
- Do not break schema keys, ask order, or FSM state/transition invariants without migration notes.
- Keep business logic (state, retrieval, scoring, numeric grounding) in deterministic Python.
- The LLM only does **reasoning + phrasing**; it never owns state. The orchestrator applies
  all `schema_updates` / transitions. M_15 numeric guardrail stays 100% deterministic.
- Centralize all prompt text in `prompts_template.py`. Ground every generation in retrieved data.
- No complex error handling — a failed/empty LLM call falls back to the deterministic stub and logs it.

## Status Legend
- [ ] Not started   - [-] In progress   - [x] Completed

---

# ✅ Phase 0 — Deterministic Sub-Agent Harness (COMPLETE)

Goal: extract every responsibility into an independent `subagents/*.py` module behind a uniform
contract, with a deterministic stub for each so the whole pipeline runs offline end-to-end.
**Done & validated** — a scripted transcript flows `S0→S1→S2→S3→S4` (and `S0→S1→S5`) with the
correct agents firing and a `logs/conversation_<id>.json` log produced.

### What was built
- [x] **`subagents/base.py`** — uniform contract: `AgentContext`, `AgentResult`,
  `BaseSubAgent` protocol, `AgentID` (M_01–M_15) + `AGENT_NAMES`. Agents never mutate
  state directly; they return `schema_updates` / `handoff_to` / `next_state_hint`.
- [x] **`subagents/__init__.py`** — `AGENT_REGISTRY` (15 singleton instances) + `get_agent()`.
- [x] **`fsm.py` refactor** — deterministic core only: `FSMState` (S0–S6), `ConversationRecord`,
  `IntentSignal`, `INTENT_DEFINITIONS`, `classify_intent()`, `_next_state()`. Moved the two old
  agent classes + templates out into the package. All FSM tests pass.
- [x] **`retrieval_tools.py` path fixes** — `_DATA_DIR` / `_POLICIES_DIR`; `search_regulations`
  and `search_policy_wording` now resolve files relative to the package.
- [x] **`orchestrator.py` rewrite** — deterministic per-turn router (see pipeline below).
- [x] **`scenarios.py`** — two scripted, offline E2E transcripts asserting state path,
  which M_ fired, and that the analytics log was written. Both pass.

### Sub-agent registry (Phase 0 deterministic behavior)
| ID | Agent | Trigger | Phase-0 deterministic behavior |
|----|-------|---------|--------------------------------|
| M_01 | IntentClassifier | every turn | wraps `fsm.classify_intent`; conf 1.0 keyword hit else 0.3 |
| M_02 | EscalationAgent | state S5 | canned text by reason (want_human/frustrated/failsafe); handoff M_03 |
| M_03 | ClosureAgent | state S4 / after M_02 | closing text; sets `drop_off_reason`; handoff M_14 if purchased |
| M_04 | SchemaExtractor (NER) | every turn | regex/keyword extraction via glossary `valid_values` |
| M_05 | ProbingAgent | state S1 | `next_missing_field()` + question templates / probe |
| M_06 | PolicyRetrievalAgent | enter S2 | `filter_products`; auto-resolve clear winner; handoff M_07 |
| M_07 | PolicySummaryAgent | M_06 handoff | template `_speech()` grounded; M_15-validated |
| M_08 | PolicyQAAgent | state S3 | clause echo from `search_policy_wording`; M_15-validated; else handoff M_09 |
| M_09 | AgenticRAGAgent | M_08 handoff | keyword search policy/regs/treatment-costs; concat; M_15-validated |
| M_10 | TranslatorAgent | outbound, non-English | identity passthrough stub |
| M_11 | ResponseQueue | outbound | deterministic FIFO join of segments |
| M_12 | ResponseValidator | outbound | denylist scan + grounding token-overlap |
| M_13 | AnalyticsLogger | end of turn | append turn to `logs/conversation_<id>.json` (only I/O sink) |
| M_14 | WhatsAppAgent | M_03 handoff if purchased | append to `logs/wa_outbox.json` + print |
| M_15 | NumericGuardrail | inside M_07/08/09 | every number in response must appear in context |

### Orchestrator pipeline (deterministic, `process_message`)
1. `schema.increment_turn()`; record user message; `S0→S1` bootstrap.
2. **M_01** intent → `register_intent_outcome`; set `user_intent`.
3. **M_04** schema extraction → orchestrator applies `schema_updates`.
4. **M_06** retrieval while unresolved + `sufficient_for_retrieval` → caches `retrieval_result`.
5. `fsm._next_state(...)` deterministic transition; mark `purchased` at S4 when prospective+closure.
6. Route to the state's entry agent and follow its handoff chain
   (`S1:M_05 · S2:M_07 · S3:M_08 · S4:M_03 · S5:M_02`, `S6` = blocked constant).
7. Outbound: **M_12** validate → **M_10** translate → **M_11** assemble.
8. **M_13** logs the turn; append assistant message.

---

# ✅ Phase 1 — LLM Reasoning Layer (OpenAI + Sarvam) — COMPLETE

Goal: add real LLM calls **only where reasoning improves on the deterministic stub**, one
sub-agent at a time. Each LLM-bound agent: (a) try the LLM path, (b) fall back to its existing
Phase 0 deterministic stub on empty/failed output, (c) log which path was taken. No complex
error handling. M_15 grounding and all FSM/transition logic stay deterministic.

> **Status: shipped.** M_01, M_04, M_05, M_07, M_08, M_09 use OpenAI; M_10 uses the Sarvam SDK;
> M_02/M_03/M_06/M_11/M_13/M_14/M_15 stay deterministic by design. `llm_gateway.py` has
> `complete_json()` + `generate_response()` + per-call logging and a `SarvamGateway`. All prompts
> live in `prompts_template.py`. `scenarios.py` passes both LLM-enabled and in forced-fallback
> mode (keys removed). Safety-critical intents, numeric grounding (M_15), and every state
> transition remain 100% deterministic.

### Models & providers (from `.env`, via `settings.py`)
- Reasoning: `OPENAI_API_KEY` + `OPENAI_MODEL_INTENT` / `_RESPONSE` / `_VALIDATOR` / `_TRANSLATION`.
- Translation: **Sarvam SDK** when the user's language is Hindi/Hinglish; otherwise OpenAI/identity.
- ⚠️ Rotate the keys currently committed in `.env`; ensure `.env` is git-ignored.

### Which agents get an LLM (and which stay deterministic)
- **LLM reasoning (OpenAI):** M_01 (intent), M_04 (extraction), M_05 (next-question phrasing),
  M_07 (grounded recommendation), M_08 (grounded clause answer), M_09 (grounded synthesis).
- **LLM phrasing (light, optional):** M_02, M_03 (empathetic closing/handoff wording).
- **Translation (Sarvam/OpenAI):** M_10.
- **Stay fully deterministic (NO LLM):** M_06 (scoring/retrieval), M_11 (FIFO),
  M_13 (I/O log), M_14 (I/O outbox), M_15 (numeric grounding). M_12 keeps deterministic
  denylist + overlap; an LLM relevance verdict is optional and advisory only.

### Implementation order (one agent at a time, no per-stage tests)

#### 1.0 LLM infrastructure
Status: [ ]
- Extend `llm_gateway.py`: add a `complete_json(system, user, model, *, temperature)` helper
  (uses `response_format={"type":"json_object"}`) and keep `generate_response` for prose.
- Add a small `_log(agent_id, path, model, ok)` line per call (path = "llm" | "fallback").
- Add `SarvamGateway` (Sarvam SDK) with `translate(text, target_lang)` + `is_available`.
- Pass `config` + `llm` (+ Sarvam) handles into `AgentContext` so agents can reason.
- Every agent base pattern: `text = llm_try(...); if not text: text = self._deterministic(...)`.

#### 1.1 M_01 IntentClassifier (OpenAI)
Status: [ ]
- Use `INTENT_CLASSIFIER_SYSTEM` / `_USER_TEMPLATE` (already centralized) via `complete_json`.
- Parse `{"intent","confidence"}`; safety-critical intents stay deterministic (never overridden).
- Fallback: `fsm.classify_intent`. Log path + confidence.

#### 1.2 M_04 SchemaExtractor (OpenAI)
Status: [ ]
- New strict-JSON extraction prompt in `prompts_template.py` constrained to glossary
  `valid_values` (enum-guarded). Orchestrator still validates via `schema.set()`.
- Fallback: the Phase 0 regex/keyword extractor. Never overwrite already-set fields.

#### 1.3 M_05 ProbingAgent (OpenAI phrasing)
Status: [ ]
- Deterministic core still chooses WHICH field/probe (the gate logic is reasoning we trust);
  LLM only rephrases the chosen question naturally (one question, voice-friendly).
- Fallback: existing question/probe templates.

#### 1.4 M_07 PolicySummaryAgent (OpenAI, grounded)
Status: [ ]
- LLM writes the recommendation **only from** the `_build_context()` block (features + plans).
- Run M_15 numeric guardrail on the LLM output; on guardrail failure → deterministic `_speech()`.
- Fallback: Phase 0 template speech.

#### 1.5 M_08 PolicyQAAgent (OpenAI, grounded)
Status: [ ]
- Use `POLICY_QA_ANSWER_ENGINE_PROMPT` with retrieved clauses only; require clause citation.
- M_15-validate; on weak/empty grounding → handoff M_09 (unchanged). Fallback: clause echo.

#### 1.6 M_09 AgenticRAGAgent (OpenAI, grounded synthesis)
Status: [ ]
- LLM synthesizes from the multi-source context (policy wording + regulations + treatment costs).
- M_15-validate; no grounded context → handoff M_02. Fallback: Phase 0 concatenation.

#### 1.7 M_02 / M_03 closing phrasing (OpenAI, light, optional)
Status: [ ]
- Optionally soften canned escalation/closure text via LLM; same structure, handoffs unchanged.
- Fallback: the canned templates.

#### 1.8 M_12 ResponseValidator (optional LLM verdict)
Status: [ ]
- Keep deterministic denylist + overlap as the gate. Optionally add an advisory LLM relevance
  check (`OPENAI_MODEL_VALIDATOR`); it can flag but never silently rewrite grounded numbers.

#### 1.9 M_10 TranslatorAgent (Sarvam SDK)
Status: [ ]
- If `schema.language` ∈ {hindi, hinglish} → translate the final English response via Sarvam SDK.
- Preserve canonical numbers (₹/SI/waiting period) — re-run M_15 on the translated text.
- Other non-English → OpenAI translation; English → identity passthrough. Fallback: identity.

#### 1.10 Orchestrator + tooling integration
Status: [ ]
- Wire `config` + `llm` + Sarvam into the per-turn `AgentContext`.
- Integrate all tooling functions through `retrieval_tools.dispatch_tool_call` so tool calls are
  uniform and logged (M_13 already records `tool_calls`).
- Keep the orchestrator the single owner of sequencing, schema writes, and transitions.

#### 1.11 Centralized, grounded prompt templates
Status: [ ]
- Consolidate all new prompts (extraction, probing, summary, RAG, validator, translation) in
  `prompts_template.py` with strict output contracts, anti-ramble, and grounding constraints.
- Single source of truth for intent labels (`fsm.INTENT_DEFINITIONS`) and product/plan context.

#### 1.12 Validate transition logic + failure/recovery policy
Status: [ ]
- Re-run `scenarios.py` (happy path + escalation) with LLM enabled AND with key removed
  (forces deterministic fallback) — both must still produce a valid `S0→…` path and a log.
- Verify: no illegal transitions from terminal states; `consecutive_failures ≥ MAX_FAILURES`
  → S5 handoff; unsafe → S6; recovery preserves schema/context across the fallback path.

---

## Risks to Watch
- LLM output drift breaking strict-JSON parsing (extraction/intent) → schema set failures.
- Ungrounded generation introducing numbers not in context → caught by M_15, but log + fall back.
- Translation altering policy numbers/meaning → mitigated by re-running M_15 post-translation.
- Latency / cost from per-turn LLM calls → keep temperature low, prompts tight, fallback ready.
- Committed API keys in `.env` → rotate and git-ignore.

## Definition of Done (Phase 1)
- Each LLM-bound agent has: an OpenAI/Sarvam path, a deterministic fallback, and per-call logging.
- All prompts centralized in `prompts_template.py` and grounded in retrieved data.
- Orchestrator integrates tooling via `dispatch_tool_call`; owns all state writes/transitions.
- `scenarios.py` passes both with LLM enabled and in forced-fallback mode.
- Transition logic + failure/recovery policy validated; M_15 grounding intact.

---

# ✅ Phase 2 — Agentic RAG + Multilingual Voice Integration (COMPLETE)

Goal: make the agent (a) answer deep policy questions from a real vector knowledge base and
(b) run as a live, multilingual **voice** agent the user can talk to in a browser. End state:
**everything is integrated and works end-to-end** (English / Hindi / Hinglish, discovery →
recommendation → Q&A → closure → WhatsApp), even if not perfect on every turn.

> Note on numbering: this "Phase 2" bundles the vector-RAG upgrade and the voice layer — the two
> pieces of work that took the LLM-enabled text agent (Phase 1) to a working spoken demo.

## 2A — Vector RAG (M_09 rewrite)
- [x] **`rag/` package** — `config.py` (all constants / metadata keys), `ingest.py`
  (run-once `python -m rag.ingest`), `index_store.py` (lazy singleton `get_rag_index()` with
  product-filtered query).
- [x] **Ingestion** — **1,825 vectors** persisted to `data/vector_store/` (policy 881,
  regulation 898, treatment_cost 46), `text-embedding-3-small`, ChromaDB collection
  `swasthya_rag`. Numbers (₹ / commas) preserved verbatim in cost chunks.
- [x] **M_09 AgenticRAGAgent rewrite** — vector-RAG only (no keyword fallback). Merges a
  product-filtered query + a general query, `top_k=6`, dedupe by text, sort by score. LLM
  synthesis (`RAG_SYNTHESIS_*`) + M_15 numeric guard + deterministic-concat fallback. If the
  index is unavailable or returns nothing → handoff to escalation. M_06 / M_08 / `retrieval_tools`
  unchanged; the rest of the pipeline is untouched (purely an M_09 internals swap).
- [x] `requirements.txt` += `llama-index-core` / `-embeddings-openai` / `-vector-stores-chroma`,
  `chromadb`. `.gitignore` += `data/vector_store/`. All self-tests + `scenarios.py` pass.

## 2B — Voice layer (Pipecat + Sarvam)
- [x] **`voice/` package** — single, isolated voice surface that wraps the existing text
  orchestrator (the orchestrator stays the brain; no business logic moved into voice):
  - `voice_config.py` — **one place for every knob**: `STTConfig`, `TTSConfig`,
    `TransportConfig`, `ConversationConfig`, bundled in `VoiceConfig` / `VOICE_CONFIG`;
    `tts_language_for()` maps the schema language → Bulbul `Language`.
  - `orchestrator_processor.py` — `OrchestratorProcessor(FrameProcessor)`: on a final
    `TranscriptionFrame`, runs `orchestrator.process_message(text)` in a thread executor (the
    orchestrator does blocking network calls), then pushes a `TTSUpdateSettingsFrame`
    (per-turn language) + `TTSSpeakFrame(reply)`. Errors → spoken apology.
  - `run_voice.py` — builds the Pipecat pipeline `[transport.input(), stt, brain, tts,
    transport.output()]`, greeting on connect, one orchestrator (one `ConversationRecord`)
    per browser connection.
- [x] **STT** — `SarvamSTTService`, **Saaras `saaras:v3`**, `mode="transcribe"` (keeps the
  user's language + numbers + code-mix). Pipecat-local **Silero VAD** (`vad_signals=False`).
- [x] **TTS** — `SarvamTTSService`, **Bulbul `bulbul:v2`**, voice **`anushka`**, 22050 Hz.
  Settings (model/voice/prosody) via the non-deprecated `Settings` API; language switched
  per-turn via `TTSUpdateSettingsFrame`.
- [x] **Transport** — **`SmallWebRTCTransport`** via the Pipecat dev runner; browser mic/speaker
  UI served at **http://localhost:7860** (`/` → `/client/`). Fixed the frontend 404 by installing
  the correct prebuilt package **`pipecat-ai-prebuilt`** (the runner imports
  `pipecat_ai_prebuilt.frontend`, *not* the similarly-named small-webrtc prebuilt).
- [x] **Per-turn language flow** — keep `fsm.detect_input_language` + M_10 Sarvam output
  translation; STT transcribe preserves the user's language, the multilingual LLM understands,
  M_10 renders the reply in the detected language, and the TTS language follows each turn.
- [x] `requirements.txt` += `pipecat-ai[webrtc,silero,sarvam]`, `pipecat-ai-prebuilt`,
  `fastapi`, `uvicorn`, `python-multipart`.

## 2C — Latency / quality pass (the "fast path" is now the only path)
- [x] **Parallelized the independent per-turn LLM calls** — intent (M_01) + schema (M_04) run
  concurrently in a thread pool (they only read the schema; side-effects applied on the main
  thread afterward). Cuts one full round-trip per turn.
- [x] **Removed two cosmetic/non-blocking LLM hops** — M_05's question-rephrase pass (templates
  are already conversational) and M_12's "does this address the query?" advisory check. Deleted
  the dead `_maybe_rephrase` / `_check_addresses` code, their prompts, and the `fast_mode` flag
  (the lean path is unconditional now).
- [x] **Smaller models per role** (via `.env` / `settings.ModelCatalog`) — `gpt-4.1-nano` for the
  cheap structured tasks (intent, schema NER, validator, translation), `gpt-4.1-mini` kept for
  customer-facing copy + RAG synthesis (`response_generator`). Added `OPENAI_MODEL_SCHEMA`.
- [x] **Rewrote the discovery questions** to open, conversational phrasing (no IVR-style "press 1"
  option menus) and added few-shot examples + confidence calibration to the intent & schema
  prompts so the smaller models classify reliably.
- [x] **Gateway hardening** — `complete_json` / `generate_response` now retry once on *transient*
  OpenAI errors (rate-limit / timeout / connection blip) with a short backoff, so a single blip
  no longer silently degrades a turn to the deterministic fallback.
- Result: a text turn dropped from **~6.5 s → ~2.5–3 s**. Offline `scenarios.py` + all self-tests
  still pass.

## Known limitations carried into Phase 3
- **Audio feels slow.** The pipeline is request/response and largely **sequential / non-streaming**:
  STT waits for end-of-utterance, the whole LLM turn runs, then TTS speaks. No token-streaming into
  TTS, limited overlap of stages.
- **Hand-rolled orchestration.** Our own `AgentContext` / `AGENT_REGISTRY` / handoff-chain / retry
  logic is great for learning and control but is bespoke, hard to scale to new states/edge cases,
  and has **no built-in observability**.
- **Correctness varies turn-to-turn** (intent/extraction occasionally wrong), and there is no
  tracing UI to see *why* a given turn went the way it did.
- **Burst rate-limiting** under rapid automated testing (not an issue at human speaking cadence).

---

# 🔜 Phase 3 — Framework Migration, Orchestration Paradigm & Real-Time Voice (PLAN)

Goal: move off the bespoke harness onto a maintained framework for **abstraction + observability**,
decide how much control we hand to the LLM vs keep deterministic, and fix the **audio latency** with
a streaming / async pipeline. Testing remains live, by voice, by the user. This phase is about
**architecture decisions** as much as code — each workstream below ends in a locked decision.

## Why move (motivation)
- The custom context/registry/handoff/retry code is bespoke and bespoke-to-debug. A framework gives
  us **standard abstractions** (nodes/edges or agent+tools, typed state, streaming) and **first-class
  observability** (traces, latency per step, token cost) — which also makes the **demo** far easier
  to narrate and debug.
- Defining *every* FSM state by hand does not scale and we don't have time to cover every edge case.

## Workstream A — Framework adoption + observability
Status: [ ]
- Pick the orchestration framework (decided in Workstream B) and stand it up alongside the current
  code so we can port one slice at a time without a big-bang rewrite.
- Wire **observability/tracing** from day one: per-turn trace, per-node latency, token + cost,
  and the inputs/outputs of each agent/tool call. Candidates: **LangSmith** (native to LangGraph),
  **Langfuse** (framework-agnostic, OSS), or OpenTelemetry export. Goal: open one trace and *see*
  why a turn behaved as it did (great for the demo).
- Keep the deterministic safety rails (M_15 numeric grounding, unsafe→S6, terminal-state locks)
  as explicit nodes/tools regardless of framework — they are non-negotiable.

## Workstream B — Orchestration paradigm (the key decision)
Status: [ ]
The question: how much sequencing do we keep deterministic vs let the LLM drive?

| Option | What it is | Pros | Cons |
|--------|-----------|------|------|
| **B1 — Port FSM to LangGraph** | Translate our states/transitions into a LangGraph state graph (nodes = our agents, edges = our transition rules), typed shared state replacing `AgentContext`. | Keeps our deterministic control + grounding; gains typed state, streaming, checkpoints, LangSmith tracing; lowest behavior risk. | Still have to define states/edges (the scaling pain remains, just better organized). |
| **B2 — LLM + tool-calling loop** | Give one LLM a goal + a toolbox (retrieve, recommend, answer, quote, send WhatsApp, escalate) and let it decide which tool to call, in what order, and when it's done. Orchestrator just runs the loop. | Deletes most of our hand-written state-transition code; handles edge cases the model can reason about; very common pattern today; fastest to extend. | Less deterministic; harder to guarantee the sales funnel order, grounding, and safety; needs strong guardrails + eval. |
| **B3 — Hybrid (recommended starting point)** | LangGraph as the backbone with a **few high-level deterministic phases** (discovery / recommend / Q&A / closure / escalate), but **within** a phase the LLM uses tool-calling freely. | Best of both: deterministic funnel + safety where it matters, LLM flexibility where edge cases live; incremental migration from B1. | Two paradigms to reason about; need clear boundaries between "graph decides" and "LLM decides". |

- **Decision criteria:** funnel-order guarantees, grounding/safety preservation, edge-case coverage,
  lines-of-code removed, demo-ability, time budget.
- **Leaning:** start with **B1 (port to LangGraph, low risk)** to get the framework + observability,
  then selectively open up tool-calling **inside** nodes (→ B3) where deterministic states are
  painful. Avoid a full B2 rewrite under time pressure unless guardrails + eval are in place.
- Map our 15 sub-agents to tools/nodes; keep M_15 + safety as deterministic tools the LLM cannot
  bypass.

## Workstream C — Real-time voice: latency, streaming & async
Status: [ ]
Diagnose and fix the slow audio. The current pipeline is sequential and non-streaming.

- **What "async" means here:** `async`/`await` lets one process handle many I/O waits (network calls
  to STT/LLM/TTS) **concurrently** instead of blocking on each in turn. Pipecat is already an async
  framework, but **we block it**: `orchestrator.process_message` makes synchronous OpenAI/Sarvam
  calls (we currently offload it to a thread executor). The win is to make the orchestrator itself
  async and **stream** stages so audio starts sooner.
- **Streaming is the big lever:** today STT waits for end-of-utterance → the *entire* LLM reply is
  generated → only then TTS speaks. Stream **LLM tokens → TTS** sentence-by-sentence so the user
  hears the first words while the rest is still being generated. Also overlap STT/VAD with
  processing and allow barge-in.
- **Transport / platform: Vapi vs Pipecat.**

  | | **Pipecat (current)** | **Vapi** |
  |---|---|---|
  | Model | Self-hosted async pipeline, full control of frames/STT/TTS/LLM | Managed voice-agent platform (handles telephony, turn-taking, streaming) |
  | Latency | Depends on *our* wiring (currently not streaming) | Tuned end-to-end for low latency out of the box |
  | Control | Total (custom processors, our orchestrator inline) | Higher-level config; less low-level control |
  | Sarvam STT/TTS | Native services already integrated | Need to confirm Sarvam (Saaras/Bulbul) support / custom-LLM webhook |
  | Effort | Already working; needs streaming + async tuning | New integration; may need to expose our orchestrator as a custom-LLM endpoint |

- **Plan:** first try to fix Pipecat (cheaper, already integrated) — make the orchestrator async,
  stream LLM→TTS, confirm Silero VAD/barge-in tuning, measure each stage. Benchmark Vapi in parallel
  **only if** Pipecat streaming can't hit an acceptable spoken-latency target *and* Vapi can drive
  Sarvam STT/TTS + our orchestrator. Decision driven by measured end-to-end latency and Sarvam
  compatibility.
- **Instrument the audio path:** log per-stage time (VAD close → STT final → LLM first token →
  LLM done → TTS first audio) so "slow" becomes a number we can attack.

## Workstream D — Code & system-design optimization (fallbacks, LLM-call budget)
Status: [ ]
Reduce cost, latency, and brittleness in *how* we call the LLM and fall back — independent of the
framework choice. These are mostly mechanical wins we can land incrementally.

### D1 — Cut the number of LLM calls per turn
- **Baseline today:** a turn can fire up to ~4–5 LLM calls (intent, schema, response/RAG, validator,
  translation). We already parallelized intent+schema and deleted the rephrase/addressing hops.
- **Merge intent + extraction into one call** — one strict-JSON prompt returning
  `{intent, confidence, fields:{…}}`. Halves the discovery round-trips; the orchestrator still
  applies `schema_updates` deterministically. (Trade: one bigger prompt vs two parallel small ones —
  benchmark both for latency *and* accuracy with the nano model.)
- **Make the validator (M_12) and translator (M_10) conditional, not unconditional** — skip M_12's
  LLM verdict entirely when the deterministic denylist/overlap already passes (it's advisory); skip
  M_10 when `language == english`. Only spend a call when it changes the output.
- **Gate expensive agents behind cheap deterministic checks** — only invoke M_09 RAG synthesis when
  M_08's clause echo is genuinely insufficient; only call the response generator when there's
  something new to say (don't re-generate canned closings/escalations).
- **Target:** typical discovery turn = **1 LLM call** (merged intent+schema), recommendation/Q&A
  turn = **1–2** (generate + optional RAG), closing turns = **0** where templated.

### D2 — Caching & reuse
- **Embedding/RAG cache** — cache `(normalized_query → retrieved_chunks)` and reuse within a session;
  the embedding for a repeated/near-identical question shouldn't be recomputed.
- **Prompt prefix / context reuse** — keep the long, static system prompts + product catalog constant
  so they're cacheable (OpenAI prompt caching), and pass only the small per-turn delta as the user
  message. Don't rebuild the full product/plan context string every turn — build once, memoize.
- **Schema-derived shortcuts** — once a field is set, never re-extract it; once a product is resolved,
  stop running M_06. (Mostly already true — audit for stragglers.)

### D3 — Fallback logic, hardened and observable
- **Make fallback a first-class, typed outcome, not a silent `if not text`.** Standardize a small
  result wrapper: `path ∈ {llm, fallback, cache}`, `reason`, `latency_ms`. Today fallbacks are
  scattered `if not text: text = self._deterministic(...)` checks — centralize the pattern so every
  agent reports *why* it fell back (empty output, JSON parse fail, transient error, guardrail reject).
- **Distinguish transient vs permanent failures** — transient (429/timeout/5xx) → the existing single
  retry with backoff; permanent (bad JSON, schema-invalid, M_15 reject) → go straight to deterministic
  fallback, don't waste a retry. (We added transient retry in Phase 2; formalize the taxonomy.)
- **Budget/timeout guard** — per-call timeout + a per-turn LLM-call ceiling; if exceeded, fall back
  deterministically rather than stacking latency. Surfaces in the trace (Workstream A).
- **Fallback-rate as a metric** — log fallback frequency per agent; a spiking fallback rate is the
  early-warning signal that a prompt/model regressed. Wire it into observability.
- **Keep the invariant:** M_15 numeric grounding + safety rails run on **every** path (llm, fallback,
  cache, translated) — a fallback must never bypass grounding.

### D4 — Concurrency & async correctness (ties into Workstream C)
- **Replace the thread-pool offload with native async** once the orchestrator is async — `asyncio`
  gather for genuinely-independent calls (e.g. a merged-intent call alongside a speculative retrieval)
  instead of `ThreadPoolExecutor`. Keep the rule: parallel agents may only **read** shared state;
  the orchestrator applies all writes on one thread/coroutine to avoid races.
- **Stream where it helps** — token-stream the response generator into TTS (Workstream C); for
  structured calls (intent/schema) streaming gives nothing, so keep them unary.
- **Cancellation/barge-in** — when the user barges in, cancel in-flight LLM/TTS work so a stale reply
  isn't spoken over the new turn.

### D5 — Coding-hygiene cleanups
- **Single LLM call-site** — route every call through `llm_gateway` (no ad-hoc OpenAI calls in
  agents) so retries, logging, caching, and timeouts are enforced in exactly one place.
- **Centralize model selection** — already in `settings.ModelCatalog`/`.env`; audit that no agent
  hardcodes a model. Document the nano-vs-mini split as a cost policy.
- **Typed prompt contracts** — validate structured outputs against a schema (e.g. Pydantic) at the
  gateway boundary so a malformed field triggers a clean fallback instead of a downstream crash.

## Open decisions to lock in Phase 3

1. Framework: **LangGraph** (assumed) — confirm vs alternatives.
2. Paradigm: **B1 → B3** (port then selectively open tool-calling) — confirm vs full B2.
3. Observability tool: **LangSmith** vs **Langfuse** vs OTel.
4. Voice platform: **fix Pipecat (streaming/async)** first; **Vapi** only if needed and Sarvam-compatible.
5. How much of M_01–M_15 becomes tools vs nodes vs deterministic guards.
6. LLM-call budget: merge intent+schema into one call? make validator/translator conditional? — confirm.

## Definition of Done (Phase 3)- Orchestration runs on the chosen framework with the deterministic safety rails (M_15, unsafe→S6,
  terminal locks) intact and the full funnel (discovery→recommendation→Q&A→closure→WhatsApp) working
  by voice in English / Hindi / Hinglish.
- A trace/observability dashboard shows per-turn, per-step latency + token cost for any conversation.
- Voice latency materially reduced via streaming LLM→TTS + an async orchestrator; first-word latency
  measured and acceptable for a live demo, with barge-in working.
- A short written rationale for each locked decision (framework, paradigm, observability, voice
  platform), plus a measured before/after latency comparison.
- LLM-call budget enforced: typical discovery turn ≤1 call, validator/translator conditional,
  fallbacks typed + observable (path/reason/latency logged), and M_15 grounding runs on every path.
