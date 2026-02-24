"""
End-to-end test of the HealthGuard AI pipeline.
Run: python test_pipeline.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.corpus import chunk_corpus, build_faiss_index
from agents.agents import extract_symptoms, retrieve_literature, generate_differential, validate_output
from orchestrator import run_healthguard

# --- Test vignettes ---
VIGNETTES = [
    {
        "name": "Pneumonia",
        "notes": (
            "45-year-old male presents with 3 days of productive cough with yellow sputum, "
            "fever (38.9C), pleuritic chest pain on the right side, and shortness of breath. "
            "He is a current smoker with 20 pack-year history. Vitals: HR 102, RR 24, BP 130/85, "
            "SpO2 93% on room air. Exam reveals crackles in the right lower lobe."
        ),
    },
    {
        "name": "Pulmonary Embolism",
        "notes": (
            "65-year-old obese female presents with acute onset dyspnea and pleuritic chest pain. "
            "She had right knee replacement surgery 10 days ago and has been relatively immobile. "
            "She reports mild cough without sputum and one episode of hemoptysis. "
            "Vitals: HR 115, RR 28, BP 138/88, SpO2 89% on room air. Exam reveals tachycardia, "
            "clear lung fields, and mild right calf swelling."
        ),
    },
    {
        "name": "Heart Failure",
        "notes": (
            "72-year-old female with known heart failure and diabetes presents with worsening "
            "dyspnea on exertion, orthopnea, and paroxysmal nocturnal dyspnea. "
            "Weight gain of 4 kg in 5 days. Bilateral lower extremity edema. "
            "Vitals: HR 98, RR 22, BP 100/65, SpO2 91%. Exam: elevated JVP, S3 gallop, "
            "bilateral crackles, 2+ pitting edema."
        ),
    },
    {
        "name": "COVID-19",
        "notes": (
            "35-year-old male with 5 days of high fever, dry cough, severe fatigue, "
            "myalgia, headache, and loss of smell and taste. Progressive shortness of breath. "
            "Vitals: HR 100, RR 24, SpO2 92%. Chest X-ray: bilateral ground-glass opacities."
        ),
    },
]


def main():
    print("=" * 70)
    print("HealthGuard AI – Pipeline Test")
    print("=" * 70)

    # 1. Load model
    print("\n[1] Loading embedding model...")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"    Model loaded in {time.time() - t0:.1f}s")

    # 2. Build index
    print("\n[2] Building FAISS index...")
    t0 = time.time()
    chunks = chunk_corpus(chunk_size=500, chunk_overlap=100)
    index, chunk_list, _ = build_faiss_index(chunks, model)
    print(f"    Index built: {index.ntotal} vectors, {len(chunk_list)} chunks in {time.time() - t0:.1f}s")

    # 3. Run each vignette
    for vig in VIGNETTES:
        print(f"\n{'=' * 70}")
        print(f"TEST: {vig['name']}")
        print(f"{'=' * 70}")
        print(f"Notes: {vig['notes'][:100]}...")

        result = run_healthguard(
            notes=vig["notes"],
            embedding_model=model,
            faiss_index=index,
            chunks=chunk_list,
            top_k=8,
        )

        if result.error:
            print(f"  ERROR: {result.error}")
            continue

        print(f"\n  Findings extracted: {len(result.findings)}")
        for f in result.findings[:5]:
            print(f"    - {f['finding']}: {f.get('value', '')}")
        if len(result.findings) > 5:
            print(f"    ... and {len(result.findings) - 5} more")

        print(f"\n  Literature chunks retrieved: {len(result.literature)}")
        for c in result.literature[:3]:
            print(f"    - [{c['chunk_id']}] score={c['relevance_score']:.3f} | {c['title'][:50]}...")

        print(f"\n  Differential (first 500 chars):")
        print(f"    {result.differential[:500].replace(chr(10), chr(10) + '    ')}")

        print(f"\n  Validation: valid={result.validation.get('valid')}, "
              f"citations={result.validation.get('citations_found', 0)}")
        if result.validation.get("issues"):
            for issue in result.validation["issues"]:
                print(f"    WARNING: {issue}")

        print(f"\n  Timings: {result.timings}")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
