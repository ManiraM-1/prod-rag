"""
Standalone script: runs the full evaluation flow (Phase 1 live pipeline +
guardrails eval + Phase 2 RAGAS metrics) directly, outside Streamlit's
session state, saving all results to testing/results/ for inspection.

Reuses the same tested functions the Streamlit app calls — no duplicated logic.
Requires: uvicorn running on localhost:8000 (Phase 1 and guardrails eval hit it).
"""
import asyncio
import json
import os
import sys

# Windows console defaults to cp1252, which can't encode the emoji Logfire's
# console exporter and this script's own progress prints use — reconfigure to
# UTF-8 before anything else prints, or the run crashes on the first emoji.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="testing-script")

from evals.pipeline import run_pipeline, load_golden_dataset, save_results
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def pipeline_progress(i, total, question, stage, response=""):
    if stage == "calling":
        print(f"[Phase 1] [{i + 1}/{total}] Calling: {question[:70]}")
    else:
        status = "OK" if response else "EMPTY"
        print(f"[Phase 1] [{i + 1}/{total}] {status}: {response[:80]}")


def guardrails_progress(i, total, input_text):
    print(f"[Guardrails] [{i + 1}/{total}] Testing: {input_text[:70]}")


def metrics_status(msg):
    print(f"[Phase 2] {msg}")


def main():
    print("=" * 70)
    print("PHASE 1: Live Pipeline")
    print("=" * 70)
    golden = load_golden_dataset()
    enriched = run_pipeline(golden, progress_callback=pipeline_progress)
    save_results(enriched, os.path.join(RESULTS_DIR, "enriched_dataset.json"))
    print(f"Saved enriched dataset -> testing/results/enriched_dataset.json")

    print()
    print("=" * 70)
    print("GUARDRAILS EVAL")
    print("=" * 70)
    g_results = run_guardrails_eval(enriched["guardrails_samples"], progress_callback=guardrails_progress)
    g_metrics = compute_guardrails_metrics(g_results)
    print("Guardrails metrics:", json.dumps(g_metrics, indent=2))
    with open(os.path.join(RESULTS_DIR, "guardrails_results.json"), "w", encoding="utf-8") as f:
        json.dump({"results": g_results, "metrics": g_metrics}, f, indent=2)

    print()
    print("=" * 70)
    print("PHASE 2: RAGAS Metrics")
    print("=" * 70)
    metric_results = asyncio.run(run_all_metrics(enriched, status_cb=metrics_status))

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    summary = {}
    for key, df in metric_results.items():
        df.to_csv(os.path.join(RESULTS_DIR, f"{key}.csv"), index=False)
        avg = df[key].mean() if key in df.columns else None
        summary[key] = None if avg is None else round(float(avg), 3)
        print(f"{key}: avg={summary[key]} (n={len(df)})")

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print()
    print("All results saved under testing/results/")


if __name__ == "__main__":
    main()
