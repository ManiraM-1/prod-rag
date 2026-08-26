"""
Runs Phase 2 (RAGAS metrics) only, against the enriched_dataset.json already
saved from today's Phase 1 run. Avoids redundantly re-running Phase 1 / the
guardrails eval, which already succeeded and don't need to change.
"""
import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(override=True)

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="testing-script")

from evals.metrics import run_all_metrics

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def metrics_status(msg):
    print(f"[Phase 2] {msg}")


def main():
    with open(os.path.join(RESULTS_DIR, "enriched_dataset.json"), encoding="utf-8") as f:
        enriched = json.load(f)

    print("=" * 70)
    print("PHASE 2: RAGAS Metrics (using existing enriched_dataset.json)")
    print("=" * 70)
    metric_results = asyncio.run(run_all_metrics(enriched, status_cb=metrics_status))

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    summary = {}
    for key, df in metric_results.items():
        df.to_csv(os.path.join(RESULTS_DIR, f"{key}.csv"), index=False)
        avg = df[key].mean() if key in df.columns and len(df) else None
        summary[key] = None if avg is None else round(float(avg), 3)
        print(f"{key}: avg={summary[key]} (n={len(df)})")

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print()
    print("All results saved under testing/results/")


if __name__ == "__main__":
    main()
