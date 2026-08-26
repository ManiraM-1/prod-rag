# Prod_RAG — Project Journal

A detailed technical record of what this project is, why it's built the way it is, and every significant problem encountered and how it was resolved. Written as reference material for documentation, a GitHub README, and resume/interview prep.

---

## 1. What This Project Is

**Prod_RAG** is a production-style Retrieval-Augmented Generation (RAG) system: a FastAPI backend that answers technical questions (Kubernetes, Intel hardware, enterprise networking) by retrieving relevant chunks from a vector database and synthesizing answers with an LLM, wrapped in a full production stack:

- **Backend**: FastAPI + LangGraph (multi-node agent pipeline: planner → retriever → responder)
- **Retrieval**: Qdrant vector database, embeddings via `sentence-transformers`
- **LLM Gateway**: Portkey (unified routing, fallback, caching, observability across all LLM calls)
- **LLM Provider**: Groq (hosting `openai/gpt-oss-120b` / `gpt-oss-20b`)
- **Guardrails**: custom-built content-safety/topic-scoping gate (originally NeMo Guardrails, replaced — see §3)
- **UI**: Streamlit chat interface
- **Observability**: Pydantic Logfire (distributed tracing across UI → backend → LLM calls)
- **Evaluation**: a separate Streamlit eval suite using RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision/Recall, Answer Correctness) plus a custom guardrails precision/recall harness

The project is a learning vehicle for building and operating a *real* production RAG stack — not just "call an LLM with some context," but the full set of concerns that come with it: gateway routing, observability, safety gates, graceful degradation, and rigorous evaluation.

---

## 2. Architecture Overview

```
User → Streamlit UI → FastAPI (/query)
                          │
                          ├─► Gate 1: Guardrails (deterministic + LLM scope-check)
                          │     fires? → return canned response, stop here
                          │
                          └─► Gate 2: LangGraph RAG pipeline
                                planner_node   (Portkey → gpt-oss-120b: decide CONVERSATIONAL vs technical search query)
                                     │
                                retrieve_node  (Qdrant vector search, only if technical)
                                     │
                                generate_node  (Portkey → gpt-oss-120b: synthesize final answer)
```

All LLM traffic — RAG pipeline *and* guardrails — routes through a single Portkey gateway factory (`app/gateway/client.py`), so nothing bypasses centralized cost tracking, failover, or logging.

---

## 3. Major Problem Areas

### 3.1 — The Model Deprecation Crisis

**What happened:** The project originally ran on `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`. Groq deprecated both for free/developer-tier accounts on **2026-08-16**, migrating recommended usage to `openai/gpt-oss-120b` / `openai/gpt-oss-20b`. This wasn't discovered from documentation — it surfaced as a live `404 model_not_found` error the first time the app was run this session.

**How it was diagnosed:** Fetched Groq's live deprecations page directly rather than trusting stale knowledge; confirmed the shutdown date and official replacement models. Later in the conversation, when a third-party AI tool claimed the old model was "still active," re-verified by making a **live API call** with the project's actual key — the ground-truth test that settled the question (404, confirmed dead for this account, despite the model still existing for Enterprise/contact-sales tiers).

**Resolution:** Migrated every reference to the old model names across `app/config.py`, `app/gateway/client.py`, `app/guardrails/rails.py`, and later `evals/metrics.py` (a second instance of the same bug was caught during the evals audit, with a missing `openai/` prefix).

**Skill demonstrated:** Not trusting documentation or third-party claims over direct, reproducible verification against the real system.

---

### 3.2 — Portkey Gateway Integration

**Problem 1 — `block_inline_config`:** Portkey's org-level policy rejected the app's inline `GATEWAY_CONFIG` (fallback/cache/retry rules) sent via request headers, with the error *"Inline config is not allowed... reference a saved config by its 'pc-...' slug instead."* Multiple wrong theories were tested and discarded (account tier, timing of the account creation, API key permissions) before finding the real, community-confirmed cause: this org's key type requires configs to be pre-saved in the Portkey dashboard, not sent inline.

**Resolution:** Created a saved Config in the Portkey dashboard mirroring `GATEWAY_CONFIG`; the code's `GATEWAY_CONFIG` dict stays as documentation/source-of-truth (commented out where it would otherwise be sent), while the dashboard Config is what's actually attached to the API key.

**Problem 2 — Missing `_user` metadata:** Only one of two LLM call paths (`get_langchain_llm()`, used by the planner) attached `_user`/`feature`/`environment` metadata to Portkey requests. The other path (`responder.py`'s raw native `portkey_client.chat.completions.create()` call) sent none, showing as `user: not-set` in Portkey's dashboard.

**Resolution:** Instead of patching each call site, bound `.with_options(metadata={...})` once on the shared `portkey_client` instance at construction time in `client.py` — every consumer inherits it automatically, zero per-call duplication.

**Problem 3 — Guardrails bypassing Portkey entirely:** The guardrails LLM call was built directly on `langchain_groq.ChatGroq`, talking straight to Groq's API — invisible to Portkey's cost tracking, failover, and logs. Root architectural principle established: *a call that bypasses the gateway is an architecture gap, not a reasonable tradeoff* — production systems route every LLM call through the gateway, specifically because the safety-critical calls are the ones that most need centralized failover.

**Resolution:** Extended `get_langchain_llm()` to accept model/temperature/extra-params overrides, and routed the guardrails scope-check through it instead of a separate direct client.

---

### 3.3 — Guardrails: Full Rewrite (NeMo Colang v1 → Custom)

This was the single largest architectural change in the project.

**Root cause of the original failure:** NeMo Guardrails' Colang v1 classifies user intent via a **few-shot completion trick** — it shows the LLM a fake conversation transcript ending in the current user turn and expects the model to *continue the pattern* with a line like `User intent: express greeting`, rather than actually replying. This relies on the model being a compliant pattern-completer rather than a conversational assistant. It worked with the old Llama model. It broke completely with `gpt-oss`, for two distinct, stacked reasons:

1. **Reasoning leakage**: `gpt-oss` is a reasoning model — it emits a `<think>...</think>` chain-of-thought block before answering. Even with Groq's `include_reasoning: false` flag set, this leaked into `.content` (a documented Groq/community bug, not unique to this project), producing rambling reasoning text instead of a clean classification label. Directly observed: an 800+ word looping reasoning trace for the simple input "tell me a joke," never converging to a usable answer.

2. **Instruction-following override**: even after stripping the `<think>` block, `gpt-oss` still answered conversationally ("Hello! How can I help you today?") instead of emitting NeMo's expected `User intent: express greeting` line — because modern, strongly RLHF-tuned models prioritize "be a helpful assistant" over blindly completing a meta-pattern, even when instructed otherwise via few-shot examples.

**Resolution:** Replaced NeMo Guardrails entirely with a hand-rolled `guard()` function:
- **Deterministic exact/substring phrase matching** for jailbreak, greeting, farewell, and capabilities intents — zero LLM calls, zero flakiness, instant.
- **One direct yes/no LLM call** for open-ended off-topic detection — a direct question ("is this in scope, yes or no"), not NeMo's fragile transcript-completion trick.
- Removed `colang_rules.py`, `nemoguardrails`, and `langchain-nvidia-ai-endpoints` from the project entirely.

**Follow-up bugs found and fixed after the rewrite:**
- **Substring false-positives**: the loose "does this phrase appear anywhere" check matched "hi" inside "hi, whats kubernetes," wrongly firing the greeting rail on real technical questions. Fixed by splitting into a strict exact-match check (for greeting/farewell/capabilities) and a loose substring check (kept only for jailbreak phrases, which are legitimately meant to be caught mid-sentence).
- **Off-topic scope-check too narrow**: "what's my previous question" (a legitimate memory-based question the RAG pipeline is designed to handle) got blocked as off-topic, since the scope-check only recognized Kubernetes/Intel/networking as in-scope. First fix attempt (widening the LLM prompt to also describe meta-questions) caused a real regression — verified directly, before/after, using the same model call — it made the classifier more conservative and started blocking genuine technical questions like "what are autopods." Reverted the prompt; fixed properly with a separate deterministic phrase-list check for memory-questions, keeping the LLM prompt untouched.
- **Reasoning leakage found to be a *systemic* risk, not guardrails-specific**: while fixing this, discovered `planner.py` had the *exact same* latent vulnerability — an unprotected exact-match check (`decision == "CONVERSATIONAL"`) against raw `gpt-oss` output. Rather than fix guardrails alone, centralized `<think>`-stripping into `client.py` itself (a `ChatOpenAI` subclass + a shared `strip_reasoning()` utility), so it protects every LLM consumer automatically — including the RAG pipeline nodes — instead of relying on each new piece of code remembering to add its own protection.

---

### 3.4 — RAG Response Quality

Technical answers were factually correct but verbose ("writing long stories"), which would make automated eval scoring (relevancy/conciseness metrics, similarity-to-ground-truth) noisy. Root cause: the prompts in `responder.py` had zero instruction about length or directness. Fixed by adding explicit conciseness constraints (no restating the question, no padding, prefer 2-5 sentences unless the question genuinely needs more) to both the conversational and technical-answer prompts.

---

### 3.5 — UI Bugs

- **Guardrail-blocked responses "invisible" in the UI**: the thought-process dropdown showing "Guardrails Fired / Retrieval Skipped" appeared to never render. Root cause: guardrail-fired responses are now near-instant (no LLM call for deterministic matches), so the Streamlit `st.status(...)` box opened and auto-collapsed faster than the browser could visually register it — not a data bug, a rendering-speed artifact. Fixed by keeping the box expanded specifically for blocked responses, with a distinct label.
- **Redundant/scattered logging**: a guardrails-blocked event was logged twice — once inside `guard()`'s Logfire span, once again in `main.py` *after* that span had already closed — producing a stray top-level log entry disconnected from the actual trace. Consolidated to one log call per `guard()` invocation, with `❌`/error-level logging for anything that gets intercepted and `✅`/info-level only for genuine pass-through.
- **`BACKEND_URL=""` (set but empty) in `.env`**: `os.getenv("BACKEND_URL", "http://localhost:8000")` only falls back to the default when the key is *missing*, not when it's present-but-empty — so the UI was POSTing to `"/query"` with no scheme at all (`requests.exceptions.ConnectionError` / "No scheme supplied"). A subtle but common Python footgun.

---

### 3.6 — Evals Pipeline: Dependency Hell

Getting the evaluation suite running surfaced a cascade of environment/dependency issues, each diagnosed and fixed in turn rather than guessed at:

- **`ragas` broken on import**: `ragas==0.4.3`'s own code unconditionally imports `ChatVertexAI` from a `langchain_community` submodule that no longer exists (removed upstream in favor of a separate `langchain-google-vertexai` package) — crashing on import for *any* user, regardless of which LLM provider they actually use. Confirmed as a known, multiply-reported upstream bug (not fixed in a released version at time of use). Initially tried downgrading `ragas` per a community "fix" — verified directly that this specific claim was wrong (the bug exists in the older version too). Real fix: a `sys.modules` shim registered in the project's own code (`evals/_compat.py`), imported before anything that imports `ragas` — resolves the stale import path to the real, working class rather than either a fake stub or a slow (2+ minute first-import) real SDK load.
- **`sentence-transformers` too old**: installed version (`2.2.2`) called a `huggingface_hub` function (`cached_download`) removed in the current `huggingface_hub` release. Upgraded rather than downgrading the hub package (lower blast radius, since `huggingface_hub` is more central to the dependency tree).
- **Windows encoding bug**: `open(golden_dataset.json)` with no explicit encoding defaults to the Windows system locale (`cp1252`) instead of UTF-8, crashing on any non-ASCII byte in the file. Fixed on both the read and write paths.
- **Missing `sentence-transformers` install entirely**, a stale `JUDGE_MODEL = "gpt-oss-20b"` missing the required `openai/` prefix (same class of bug as §3.1, caught via direct verification, not just review — every fix in this project was tested against the real API where feasible, not assumed correct from reading code).

**Methodology note**: for this whole phase, verification leaned heavily on Streamlit's own official test harness (`streamlit.testing.v1.AppTest`), which actually executes the app's script logic end-to-end without needing a browser — far stronger evidence than static code reading, and used to catch a real deprecation warning (`use_container_width` → `width=`) affecting 8 call sites.

---

### 3.7 — Evals Model Reliability: The Structured-Output Problem

The deepest debugging arc of the project. Running the real evaluation metrics against Groq's `gpt-oss-20b` produced widespread failures — Context Precision failing on nearly every sample.

**Diagnosis process (not guesswork — each hypothesis was tested against the real API before being accepted or discarded):**
1. Inspected `ragas`'s actual source (`ContextPrecision.ascore`) — found it makes *one LLM call per retrieved context* (not one per sample), roughly doubling exposure to any per-call failure risk for that specific metric.
2. Traced the real error payloads (not just the exception class): found `instructor` (the library `ragas` uses for structured-output extraction) forces `tool_choice: "required"` on Groq — and the raw `failed_generation` fields showed `gpt-oss` frequently producing *perfectly correct* JSON as plain text content, but failing to wrap it as an actual tool invocation. Groq's API then hard-rejects this — not a validation error `instructor`'s retry logic can catch and correct, since it's an API-level rejection of a well-formed-but-wrongly-shaped response.
3. Tested `instructor`'s `strict=False` mode directly — made things *worse* (0/4 vs. baseline), and confirmed via logs that these API-level 400s aren't retried at all despite raising `max_retries`, since `instructor`'s retry mechanism only engages on validation-catchable failures.
4. Tested `instructor.Mode.JSON_SCHEMA` (asks for JSON as regular response content, no forced tool call) against the *exact same failing scenarios* — succeeded 4/4, then 10/10 across three different metrics in follow-up tests. This is the real fix: the model already reliably produces correct JSON content; it just doesn't reliably wrap it as a tool call. Sidestep the tool-calling requirement entirely.

**Second-order finding:** even after the mode fix, a live run still showed some failures — this time `json_validate_failed` and, critically, one explicit `IncompleteOutputException: output incomplete due to max_tokens limit`. Root cause: `ragas`'s default `max_tokens=1024` was too small for a full statement-list-plus-verdicts JSON response on realistic (not toy-example) content — confirmed by reproducing the failure with realistic-length synthetic data after initial toy-example tests had misleadingly passed. Fixed by raising `max_tokens` to 4096.

**Resilience layer, independent of root-cause fixes:** since `ragas`/`instructor`'s exposed configuration doesn't allow tuning retries or mode through the official API, and some residual flakiness is inherent to a free-tier reasoning model doing forced structured extraction, `_batched_score()` was rewritten to catch a failure on any single sample, log it, and skip just that one sample — rather than letting one bad sample crash the entire multi-experiment run. Verified via a mock test that the samples/scores arrays stay correctly aligned even when some are skipped (avoiding a subtle off-by-one bug that would have mismatched questions to scores).

**Tertiary finding, discovered while validating the above:** the dedicated evaluation judge key (`JUDGE_GROQ`, intentionally separated from the production `GROQ_API_KEY` specifically so eval runs never starve the production app's quota — a design already present in the code's own docstring) hit Groq's **200,000 tokens/day** hard limit, from the cumulative cost of the extensive real-API validation testing plus the actual run. Not a bug — a reminder that thorough empirical validation against a free-tier API has a real, finite budget, and a good reason to design in quota separation up front (which this project already had).

---

### 3.8 — Miscellaneous Bugs Fixed

- `app/main.py`: imported a module (`app.agents.graph`) that hadn't been written yet.
- `app/agents/nodes/planner.py`: imported `app.gateway`, also not yet written at that point.
- `app/services/retrieval/qdrant_service.py`: imported from `app.services.retrieval.embedding` (singular) when the actual module is `embeddings.py` (plural) — caught by cross-referencing how the rest of the codebase already imported it correctly elsewhere, confirming which side of the mismatch was the typo.

---

## 4. Key Engineering Principles Demonstrated

Worth pulling into a resume/interview narrative directly:

- **Empirical verification over assumption**: nearly every fix in this project was tested against the real system (live API calls, real file loads, Streamlit's actual test harness) before being declared correct — including catching cases where an initial "logical" fix (widening a prompt, disabling strict mode) made things measurably worse when actually tested.
- **Root-cause diagnosis over symptom-patching**: the guardrails rewrite and the structured-output fix both came from tracing failures down to their actual mechanism (reading library source, inspecting raw API error payloads) rather than trial-and-error parameter tweaking.
- **Architectural consistency**: the "everything through one gateway" principle was applied even when it meant more work (routing guardrails through Portkey) and even when it surfaced its own new problems (quota sharing) — a real tradeoff, made and documented rather than glossed over.
- **Defense in depth, not either/or**: the final guardrails/evals design combines a root-cause fix (correct extraction mode) *and* a resilience layer (skip-and-continue) — not relying on either alone, because the underlying model's flakiness is real and only partially addressable.
- **Debugging under real-world constraints**: working around free-tier rate limits, deprecated models, and org-level gateway policies — the kind of constraints production systems actually operate under, not idealized conditions.

---

## 5. Current State (as of last session)

- Model migration: complete, verified.
- Portkey gateway: fully integrated, all LLM calls routed through it, metadata/observability correct.
- Guardrails: fully rewritten, bug-fixed, verified against a battery of real and adversarial test cases.
- RAG response quality: conciseness constraints applied.
- UI: bugs fixed, verified via real test harness.
- Evals pipeline: all known import/dependency/config bugs fixed and verified; structured-output reliability fix applied and partially validated (8/8 and 10/10 real-call success in isolated tests) but **not yet confirmed on a full real end-to-end run**, due to hitting the daily token quota on the dedicated judge key during validation.

**Next step**: run the full Phase 1 (live pipeline) + Phase 2 (RAGAS metrics) evaluation for real once the judge key's daily quota resets, to get the first fully-validated metrics report for this project.
