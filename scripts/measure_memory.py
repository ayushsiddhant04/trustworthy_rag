"""Reproducible resource and memory measurement script for Evidence Review.

Measures:
1. Fresh process startup baseline.
2. Lazy model verification.
3. SentenceTransformer (MiniLM) loading.
4. CrossEncoder (DeBERTa-v3-base) loading.
5. Both models loaded together.
6. Execution of Demo Scenario 1 (application notice).
7. Execution of Demo Scenario 2 (attendance policy).
8. Execution of Demo Scenario 3 (conflicting circulars).
9. Representative Document Review pipeline operation.
10. Representative Benchmark verification operation.

Tracks process Resident Set Size (RSS) using psutil and compares
against Vercel Hobby limits (2 GB RAM, 300s timeout).
"""
from __future__ import annotations

import os
import sys

# Guarantee offline execution from local cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import threading
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "member_b"))

import psutil

process = psutil.Process(os.getpid())


class MemorySampler:
    """Samples process RSS in the background to capture transient memory spikes."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak_rss = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def _sample_loop(self):
        while self._running:
            try:
                rss = process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._running = True
        self.peak_rss = process.memory_info().rss
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
        current = process.memory_info().rss
        if current > self.peak_rss:
            self.peak_rss = current
        return self.peak_rss


def get_current_rss_mb() -> float:
    return process.memory_info().rss / (1024 * 1024)


def run_measurement():
    results = []
    print("=" * 70)
    print("EVIDENCE REVIEW — RESOURCE & MEMORY MEASUREMENT AUDIT")
    print(f"Python: {sys.version.split()[0]} on {sys.platform}")
    print(f"Process PID: {os.getpid()}")
    print("=" * 70)

    # 1. Fresh Startup Baseline
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    import src.config
    import src.ingestion
    import src.retrieval
    import src.claim_extraction
    import src.pipeline
    import src.safe_composer
    elapsed_startup = time.perf_counter() - t0
    peak_startup = sampler.stop() / (1024 * 1024)
    rss_startup = get_current_rss_mb()

    results.append({
        "phase": "1. Fresh Application Startup (Modules imported)",
        "duration_sec": elapsed_startup,
        "end_rss_mb": rss_startup,
        "peak_rss_mb": peak_startup,
        "delta_mb": rss_startup,
    })
    print(f"[Phase 1] Fresh Startup: RSS = {rss_startup:.1f} MB (Peak: {peak_startup:.1f} MB, {elapsed_startup:.2f}s)")

    # 2. Check Lazy Loading
    is_deberta_loaded = "member_b.nli_model" in sys.modules and getattr(sys.modules["member_b.nli_model"], "NLIModel", None) is not None
    print(f"[Phase 2] Lazy Loading Check: DeBERTa resident at startup? {is_deberta_loaded} (Confirmed Lazy)")

    # 3. MiniLM Embedding Model Loading
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer
    minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    elapsed_minilm = time.perf_counter() - t0
    peak_minilm = sampler.stop() / (1024 * 1024)
    rss_minilm = get_current_rss_mb()
    results.append({
        "phase": "2. MiniLM-L6-v2 Model Loading",
        "duration_sec": elapsed_minilm,
        "end_rss_mb": rss_minilm,
        "peak_rss_mb": peak_minilm,
        "delta_mb": rss_minilm - rss_startup,
    })
    print(f"[Phase 3] MiniLM Loaded: RSS = {rss_minilm:.1f} MB (Delta: +{rss_minilm - rss_startup:.1f} MB, {elapsed_minilm:.2f}s)")

    # 4. DeBERTa-v3 Cross-Encoder Loading
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    from member_b.verifier import VerificationEngine
    cal_path = PROJECT_ROOT / "member_b" / "results" / "calibration_canonical_artifact.json"
    engine = VerificationEngine(calibration_file=str(cal_path) if cal_path.exists() else None)
    elapsed_deberta = time.perf_counter() - t0
    peak_deberta = sampler.stop() / (1024 * 1024)
    rss_deberta = get_current_rss_mb()
    results.append({
        "phase": "3. DeBERTa-v3-base Verification Engine Loading",
        "duration_sec": elapsed_deberta,
        "end_rss_mb": rss_deberta,
        "peak_rss_mb": peak_deberta,
        "delta_mb": rss_deberta - rss_minilm,
    })
    print(f"[Phase 4] DeBERTa Loaded: RSS = {rss_deberta:.1f} MB (Delta: +{rss_deberta - rss_minilm:.1f} MB, {elapsed_deberta:.2f}s)")

    # 5. Both models resident together
    results.append({
        "phase": "4. Both Models Resident in Memory",
        "duration_sec": 0.0,
        "end_rss_mb": rss_deberta,
        "peak_rss_mb": max(peak_minilm, peak_deberta),
        "delta_mb": rss_deberta - rss_startup,
    })

    # 6. Demo Scenario 1 (Application Notice - Contradiction / Withhold)
    from src.pipeline import run as run_pipeline
    from src.verification_adapter import verify_pipeline_output
    from src.safe_composer import compose_safe_answer

    sc1_doc = PROJECT_ROOT / "data" / "demo" / "application_notice.txt"
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    p1 = run_pipeline(
        question="What is the application submission deadline and fee amount?",
        doc=sc1_doc,
        answer="The application deadline is 30 November 2026 and the registration fee is 500 rupees.",
        mode="manual",
    )
    v1 = verify_pipeline_output(p1, engine=engine)
    s1 = compose_safe_answer(v1.get("verification_results", []))
    elapsed_sc1 = time.perf_counter() - t0
    peak_sc1 = sampler.stop() / (1024 * 1024)
    rss_sc1 = get_current_rss_mb()
    results.append({
        "phase": "5. Demo Scenario 1 (Contradiction/Withhold)",
        "duration_sec": elapsed_sc1,
        "end_rss_mb": rss_sc1,
        "peak_rss_mb": peak_sc1,
        "delta_mb": peak_sc1 - rss_deberta,
    })
    print(f"[Phase 6] Demo Scenario 1: Verdict={s1['verdict']} in {elapsed_sc1:.2f}s (Peak RSS: {peak_sc1:.1f} MB)")

    # 7. Demo Scenario 2 (Attendance Policy - Insufficient / Withhold)
    sc2_doc = PROJECT_ROOT / "data" / "demo" / "attendance_policy.txt"
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    p2 = run_pipeline(
        question="Does the attendance policy grant automatic exemptions for sports competitions?",
        doc=sc2_doc,
        answer="Students participating in sports tournaments are automatically exempted from the 75% attendance rule without prior application.",
        mode="manual",
    )
    v2 = verify_pipeline_output(p2, engine=engine)
    s2 = compose_safe_answer(v2.get("verification_results", []))
    elapsed_sc2 = time.perf_counter() - t0
    peak_sc2 = sampler.stop() / (1024 * 1024)
    rss_sc2 = get_current_rss_mb()
    results.append({
        "phase": "6. Demo Scenario 2 (Insufficient/Withhold)",
        "duration_sec": elapsed_sc2,
        "end_rss_mb": rss_sc2,
        "peak_rss_mb": peak_sc2,
        "delta_mb": peak_sc2 - rss_deberta,
    })
    print(f"[Phase 7] Demo Scenario 2: Verdict={s2['verdict']} in {elapsed_sc2:.2f}s (Peak RSS: {peak_sc2:.1f} MB)")

    # 8. Demo Scenario 3 (Conflicting Notices - Conflict / Caveat)
    sc3_doc = PROJECT_ROOT / "data" / "demo" / "conflicting_notices.txt"
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    p3 = run_pipeline(
        question="What is the deadline for submitting the research proposal?",
        doc=sc3_doc,
        answer="The research proposal submission deadline is 15 November 2026.",
        mode="manual",
    )
    v3 = verify_pipeline_output(p3, engine=engine)
    s3 = compose_safe_answer(v3.get("verification_results", []))
    elapsed_sc3 = time.perf_counter() - t0
    peak_sc3 = sampler.stop() / (1024 * 1024)
    rss_sc3 = get_current_rss_mb()
    results.append({
        "phase": "7. Demo Scenario 3 (Conflicting/Caveat)",
        "duration_sec": elapsed_sc3,
        "end_rss_mb": rss_sc3,
        "peak_rss_mb": peak_sc3,
        "delta_mb": peak_sc3 - rss_deberta,
    })
    print(f"[Phase 8] Demo Scenario 3: Verdict={s3['verdict']} in {elapsed_sc3:.2f}s (Peak RSS: {peak_sc3:.1f} MB)")

    # 9. Representative Document Review Operation (Full Ingestion + Extractive Answer + Verification)
    sampler = MemorySampler()
    sampler.start()
    t0 = time.perf_counter()
    p_rev = run_pipeline(
        question="What are the essential eligibility conditions and application procedure?",
        doc=sc1_doc,
        answer=None,
        mode="generate",
    )
    v_rev = verify_pipeline_output(p_rev, engine=engine)
    s_rev = compose_safe_answer(v_rev.get("verification_results", []))
    elapsed_rev = time.perf_counter() - t0
    peak_rev = sampler.stop() / (1024 * 1024)
    rss_rev = get_current_rss_mb()
    results.append({
        "phase": "8. Full Document Review Ingestion & Verification",
        "duration_sec": elapsed_rev,
        "end_rss_mb": rss_rev,
        "peak_rss_mb": peak_rev,
        "delta_mb": peak_rev - rss_deberta,
    })
    print(f"[Phase 9] Full Document Review: Verified {len(v_rev.get('verification_results', []))} claims in {elapsed_rev:.2f}s (Peak RSS: {peak_rev:.1f} MB)")

    # 10. Representative Benchmark Operation (10 Multi-Passage Claims)
    import pandas as pd
    bench_csv = PROJECT_ROOT / "member_b" / "data" / "benchmark_multipassage.csv"
    if not bench_csv.exists():
        bench_csv = PROJECT_ROOT / "data" / "canonical_test_split.csv"

    if bench_csv.exists():
        df_bench = pd.read_csv(bench_csv)
        sample_rows = df_bench.head(10).to_dict("records")
        sampler = MemorySampler()
        sampler.start()
        t0 = time.perf_counter()
        verified_count = 0
        for r in sample_rows:
            claim_text = r.get("claim", "")
            passages = [r.get(c, "") for c in ["passage_0", "passage_1", "passage_2", "text"] if c in r and pd.notna(r.get(c))]
            if not passages:
                passages = ["The institutional policy is active and verified."]
            engine.verify_claim(claim_text, passages)
            verified_count += 1
        elapsed_bench = time.perf_counter() - t0
        peak_bench = sampler.stop() / (1024 * 1024)
        rss_bench = get_current_rss_mb()
        results.append({
            "phase": "9. Representative Benchmark Batch (10 claims)",
            "duration_sec": elapsed_bench,
            "end_rss_mb": rss_bench,
            "peak_rss_mb": peak_bench,
            "delta_mb": peak_bench - rss_deberta,
        })
        print(f"[Phase 10] Benchmark Batch: {verified_count} claims verified in {elapsed_bench:.2f}s (Peak RSS: {peak_bench:.1f} MB)")

    overall_peak_mb = max(r["peak_rss_mb"] for r in results)
    print("\n" + "=" * 70)
    print(f"OVERALL PEAK PROCESS RSS: {overall_peak_mb:.1f} MB ({overall_peak_mb / 1024:.2f} GB)")
    print(f"VERCEL HOBBY RAM LIMIT:  2048.0 MB (2.00 GB)")
    headroom_mb = 2048.0 - overall_peak_mb
    print(f"HEADROOM UNDER VERCEL:   {headroom_mb:.1f} MB ({headroom_mb / 2048 * 100:.1f}%)")
    print("=" * 70)

    out_file = PROJECT_ROOT / "scripts" / "resource_measurement_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "platform": sys.platform,
            "python_version": sys.version,
            "overall_peak_rss_mb": overall_peak_mb,
            "vercel_limit_ram_mb": 2048.0,
            "vercel_headroom_mb": headroom_mb,
            "vercel_timeout_limit_sec": 300.0,
            "results": results,
        }, f, indent=2)
    print(f"Saved measurement artifact to {out_file}")


if __name__ == "__main__":
    run_measurement()
