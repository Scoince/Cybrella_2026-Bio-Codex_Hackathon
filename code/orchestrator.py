"""
HealthGuard AI – Orchestrator
Coordinates the multi-agent pipeline:
  1. Extract symptoms/findings
  2. Retrieve relevant medical literature
  3. Generate differential diagnosis
  4. Validate output citations
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from agents.agents import (
    extract_symptoms,
    generate_differential,
    retrieve_literature,
    validate_output,
)


@dataclass
class PipelineResult:
    """Container for the full pipeline output."""
    findings: list[dict] = field(default_factory=list)
    literature: list[dict] = field(default_factory=list)
    differential: str = ""
    validation: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    error: str | None = None


def run_healthguard(
    notes: str,
    embedding_model,
    faiss_index,
    chunks: list[dict],
    top_k: int = 8,
) -> PipelineResult:
    """
    Main orchestrator: run the full HealthGuard AI pipeline.

    Parameters
    ----------
    notes : str
        Free-text clinical notes.
    embedding_model : SentenceTransformer
        The sentence-transformer model for query embedding.
    faiss_index : faiss.Index
        Pre-built FAISS index of medical literature chunks.
    chunks : list[dict]
        Chunk metadata aligned with the FAISS index.
    top_k : int
        Number of literature chunks to retrieve.

    Returns
    -------
    PipelineResult
    """
    result = PipelineResult()

    try:
        # --- Agent 1: Symptom Extraction ---
        t0 = time.time()
        result.findings = extract_symptoms(notes)
        result.timings["extract_symptoms"] = round(time.time() - t0, 2)

        if not result.findings:
            result.error = (
                "No clinical findings could be extracted. "
                "Please provide more detailed clinical notes."
            )
            return result

        # --- Agent 2: Literature Retrieval ---
        t0 = time.time()
        result.literature = retrieve_literature(
            result.findings, embedding_model, faiss_index, chunks, top_k=top_k
        )
        result.timings["retrieve_literature"] = round(time.time() - t0, 2)

        # --- Agent 3: Differential Diagnosis ---
        t0 = time.time()
        result.differential = generate_differential(
            result.findings, result.literature
        )
        result.timings["generate_differential"] = round(time.time() - t0, 2)

        # --- Agent 4: Validation ---
        t0 = time.time()
        result.validation = validate_output(
            result.differential, result.literature
        )
        result.timings["validate_output"] = round(time.time() - t0, 2)

    except Exception as e:
        result.error = f"Pipeline error: {e}"

    return result
