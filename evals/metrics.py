"""
Phase 2: RAGAS + Tool Correctness metrics.
Uses JUDGE_GROQ key so production GROQ_API_KEY is never exhausted by eval runs.
All LLM-based metrics run one sample at a time (GENERAL_BATCH_SIZE=1), with a
COOLDOWN_MINI-second pause between samples and a COOLDOWN_STANDARD-second pause
between experiments, calibrated for Groq's 6,000 TPM on_demand tier.
Contexts are truncated to CONTEXT_TRUNCATE chars per chunk (CONTEXT_LIMIT chunks max)
to keep requests within that budget without cutting a chunk off mid-thought.
"""


import os
import asyncio
import instructor
import logfire
import pandas as pd
from openai import AsyncOpenAI


from ragas.llms.base import InstructorLLM, InstructorModelArgs
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "openai/gpt-oss-20b"
COOLDOWN_STANDARD = 62
COOLDOWN_MINI = 40       # between individual samples - lets sliding TPM window recover (~2,800 tok/sample)
GENERAL_BATCH_SIZE = 1  # one sample at a time: abatch_score fires calls concurrently per sample,
                         # so batch>1 stacks multiple samples' async calls inside the same second
CONTEXT_TRUNCATE = 1510  # chars per context chunk. Sources come back from retriever.py prefixed with
                          # "CONTENT: " (9 chars) ahead of the real chunk text, which itself is up to 1500
                          # chars (app/ingestion/chunking/splitter.py's chunk_size default) — 1510 covers the
                          # full prefix + full chunk with a touch of margin, instead of clipping the last few
                          # characters of a max-length chunk. The old value of 300 was hiding most of a
                          # chunk's actual content from the judge, artificially deflating
                          # Faithfulness/Context Precision/Context Recall scores regardless of true RAG quality.
CONTEXT_LIMIT = 3        # number of context chunks passed to RAGAS per sample (was 2 — still below the 5
                          # pipeline.py actually captures, but closer to what the RAG system really retrieved).
                          # Cooldowns already space samples ~40-60s apart (see COOLDOWN_MINI/STANDARD), so the
                          # per-minute TPM budget has room for a fuller request; this was over-conservative.


def _build_judge():
    """
    Groq's gpt-oss models are unreliable with forced tool-calling (the default
    extraction mode ragas.llms.llm_factory hardcodes via instructor.Mode.TOOLS) —
    confirmed directly: the model often produces perfectly correct JSON but fails
    to wrap it as an actual tool invocation, which Groq's API then hard-rejects
    (tool_use_failed / output_parse_failed), not something instructor's own
    retry-on-validation-error logic catches since it's an API-level rejection,
    not a parseable-but-wrong response.

    instructor.Mode.JSON_SCHEMA sidesteps this entirely — it asks for JSON as
    regular response content instead of forcing a tool call, which is what the
    model already does reliably. Verified directly: 10/10 real calls across
    Faithfulness/AnswerRelevancy/ContextPrecision succeeded with this mode on
    scenarios that failed repeatedly under the default TOOLS mode.

    llm_factory() doesn't expose mode control, so InstructorLLM is built
    directly here instead of going through it.

    max_tokens=1024 (ragas's default) is too small: on a real run it produced
    an IncompleteOutputException (output cut off mid-JSON), which also explains
    some of the json_validate_failed errors — truncated JSON isn't valid JSON.
    Raised to 4096 so a full statement list + verdicts has room to complete.
    """
    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")
    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    patched_client = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
    llm = InstructorLLM(
        client=patched_client,
        model=JUDGE_MODEL,
        provider="openai",
        model_args=InstructorModelArgs(max_tokens=4096),
    )
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return llm, embeddings

async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (Groq TPM buffer)..."
    if status_cb:
        status_cb(msg)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
    if status_cb:
        status_cb(f"✅ Ready - starting next experiment.")
        
        
def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated AND actual_contexts populated
    (no fallback to relevant_contexts — an empty actual_contexts means the system
    genuinely didn't retrieve anything for this sample, which is real signal, not
    something to paper over with the golden reference contexts).
    Truncates contexts to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks —
    matches the real ~1500-char chunk size rather than cutting a chunk off mid-thought.
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        # No fallback to relevant_contexts (the golden/reference contexts) here on purpose:
        # if actual_contexts is empty, that's real signal — the system didn't retrieve anything
        # for this sample (misrouted to conversational, or Qdrant found nothing) — and silently
        # substituting the "correct" reference contexts would mask that failure by scoring it
        # as if retrieval had worked perfectly. Skip it instead, same as an empty response.
        raw_contexts = s.get("actual_contexts") or []
        if not raw_contexts:
            continue
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    return pd.DataFrame([
        {"question": s["question"][:65], metric_key: round(float(r.value), 3)}
        for s, r in zip(samples, scores)
    ])


def _safe_avg(df: pd.DataFrame, metric_key: str):
    """
    df[metric_key].mean() blows up with KeyError if every sample in this
    experiment failed (e.g. a quota/rate-limit wall hit mid-run) — _score_df
    on an empty samples/scores pair produces a DataFrame with no columns at
    all, not just no rows. Returns None instead of crashing the whole run
    over a logging statement.
    """
    if metric_key not in df.columns:
        return None
    return round(df[metric_key].mean(), 3)


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> tuple[list, list]:
    """
    Runs abatch_score in chunks of GENERAL_BATCH_SIZE with cooldowns between chunks.
    Keeps each burst under 6,000 TPM on Groq's on_demand tier.

    A single sample failing (e.g. Groq's gpt-oss occasionally not calling the
    required tool for structured extraction — a known flakiness, not a hard
    incompatibility) is logged and skipped rather than crashing the whole
    6-experiment run. Returns (samples, scores) as parallel lists containing
    only the samples that actually produced a score, so callers stay aligned
    even when some were skipped.
    """
    ok_samples = []
    ok_scores = []
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if b_idx > 0:
            await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
        try:
            scores = await metric.abatch_score(batch)
        except Exception as e:
            logfire.error(f"⚠️ {label} sample {b_idx + 1} failed, skipping | {type(e).__name__}: {e}")
            if status_cb:
                status_cb(f"⚠️ {label} sample {b_idx + 1} failed (skipped) — continuing...")
            continue
        batch_samples = samples[b_idx * GENERAL_BATCH_SIZE : b_idx * GENERAL_BATCH_SIZE + len(batch)]
        ok_samples.extend(batch_samples)
        ok_scores.extend(scores)
    return ok_samples, ok_scores

async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs all 6 experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    """
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    results = {}

    with logfire.span("🧪 Eval Phase 2 - All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 - Faithfulness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 1 - Faithfulness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            used_samples, scores = await _batched_score(Faithfulness(llm=judge_llm), inputs, samples, status_cb, "Faithfulness")
            df = _score_df("faithfulness", used_samples, scores)
            results["faithfulness"] = df
            logfire.info("🧪 Faithfulness done", avg=_safe_avg(df, "faithfulness"))

        await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 - Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 2 - Answer Relevancy"):
            inputs = [
                {"user_input": s["question"], "response": s["actual_response"]}
                for s in samples
            ]
            used_samples, scores = await _batched_score(
                AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Relevancy"
            )
            df = _score_df("answer_relevancy", used_samples, scores)
            results["answer_relevancy"] = df
            logfire.info("🧪 Answer Relevancy done", avg=_safe_avg(df, "answer_relevancy"))

        await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 - Context Precision ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 3 - Context Precision"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            used_samples, scores = await _batched_score(ContextPrecision(llm=judge_llm), inputs, samples, status_cb, "Context Precision")
            df = _score_df("context_precision", used_samples, scores)
            results["context_precision"] = df
            logfire.info("🧪 Context Precision done", avg=_safe_avg(df, "context_precision"))

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 - Context Recall ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 4 - Context Recall"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            used_samples, scores = await _batched_score(ContextRecall(llm=judge_llm), inputs, samples, status_cb, "Context Recall")
            df = _score_df("context_recall", used_samples, scores)
            results["context_recall"] = df
            logfire.info("🧪 Context Recall done", avg=_safe_avg(df, "context_recall"))

        await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        # ── Exp 5: Answer Correctness (split into batches) ────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 5/6 - Answer Correctness batch 1/2...")
        with logfire.span("🧪 Exp 5 - Answer Correctness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]
            used_samples, all_scores = await _batched_score(
                AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Correctness"
            )
            df = _score_df("answer_correctness", used_samples, all_scores)
            results["answer_correctness"] = df
            logfire.info("🧪 Answer Correctness done", avg=_safe_avg(df, "answer_correctness"))

        await _cooldown(COOLDOWN_STANDARD, "Answer Correctness", status_cb)

        # ── Exp 6: Tool Correctness (no LLM - Jaccard) ───────────────────────
        if status_cb:
            status_cb("⚡ Exp 6/6 - Tool Correctness (zero LLM calls)...")
        with logfire.span("🧪 Exp 6 - Tool Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                union = len(called | expected)
                score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("🧪 Tool Correctness done", avg=_safe_avg(df, "tool_correctness"))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results
