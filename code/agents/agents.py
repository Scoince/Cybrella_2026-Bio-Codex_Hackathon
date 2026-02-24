"""
Agent functions for HealthGuard AI.

Each agent is a Python function with a clear responsibility.
They use the OpenAI API when a key is available, otherwise fall back
to a deterministic rule-based implementation for demo purposes.
"""

from __future__ import annotations
import json
import os
import re
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Return an OpenAI client if a key is configured, else None."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def _llm_chat(system_prompt: str, user_prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 0.2) -> str | None:
    """Call the OpenAI chat API. Returns None if no key is available."""
    client = _get_openai_client()
    if client is None:
        return None
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# =========================================================================
# AGENT 1 – Symptom / Finding Extractor
# =========================================================================

_EXTRACT_SYSTEM = (
    "You are a clinical NLP system. Extract all clinical findings "
    "(symptoms, signs, demographics, risk factors) from the following "
    "clinical notes. Return ONLY a JSON array where each element has keys: "
    '"finding", "value" (if any, else null), and "context" (the sentence '
    "it appeared in). Do not include any other text."
)


def _extract_symptoms_llm(notes: str) -> list[dict]:
    raw = _llm_chat(_EXTRACT_SYSTEM, notes)
    if raw is None:
        return []
    # strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    return json.loads(raw)


def _extract_symptoms_fallback(notes: str) -> list[dict]:
    """
    Rule-based extraction: look for known medical keywords.
    Good enough for a demo without API keys.
    """
    KNOWN_FINDINGS = [
        # Symptoms
        "fever", "cough", "dyspnea", "shortness of breath", "chest pain",
        "pleuritic chest pain", "wheezing", "hemoptysis", "sputum",
        "orthopnea", "paroxysmal nocturnal dyspnea", "palpitations",
        "edema", "leg swelling", "lower extremity edema", "fatigue",
        "weight loss", "weight gain", "nausea", "vomiting", "diarrhea",
        "abdominal pain", "headache", "dizziness", "syncope", "confusion",
        "altered mental status", "diaphoresis", "night sweats",
        "polyuria", "polydipsia", "blurred vision",
        "anosmia", "ageusia", "myalgia", "joint pain", "back pain",
        "rash", "pruritus", "dysuria", "hematuria", "oliguria",
        "anorexia", "tachycardia", "tachypnea", "hypotension",
        "hypertension", "hypoxemia", "hypoxia", "crackles", "rales",
        "jugular venous distension", "murmur", "gallop",
        "hemiparesis", "aphasia", "ataxia", "vertigo", "diplopia",
        # Risk factors / demographics
        "smoking", "smoker", "diabetes", "diabetic", "hypertensive",
        "obese", "obesity", "alcohol", "sedentary",
        "immunosuppressed", "immunocompromised",
        "male", "female", "elderly",
    ]

    notes_lower = notes.lower()
    sentences = re.split(r'(?<=[.!?])\s+', notes)
    findings = []
    seen = set()

    # Extract age
    age_match = re.search(r'(\d{1,3})[\s-]*year[\s-]*old', notes_lower)
    if age_match:
        findings.append({
            "finding": "age",
            "value": age_match.group(1),
            "context": age_match.group(0),
        })

    # Extract gender
    if re.search(r'\b(male|man)\b', notes_lower):
        findings.append({"finding": "sex", "value": "male", "context": ""})
    elif re.search(r'\b(female|woman)\b', notes_lower):
        findings.append({"finding": "sex", "value": "female", "context": ""})

    for finding in KNOWN_FINDINGS:
        pattern = r'\b' + re.escape(finding) + r'\b'
        if re.search(pattern, notes_lower) and finding not in seen:
            seen.add(finding)
            # find containing sentence
            ctx = ""
            for s in sentences:
                if re.search(pattern, s.lower()):
                    ctx = s.strip()
                    break
            findings.append({
                "finding": finding,
                "value": None,
                "context": ctx,
            })

    return findings


def extract_symptoms(notes: str) -> list[dict]:
    """
    Agent 1: Extract clinical findings from free-text notes.
    Uses LLM if available, otherwise rule-based fallback.
    """
    result = _extract_symptoms_llm(notes)
    if result:
        return result
    return _extract_symptoms_fallback(notes)


# =========================================================================
# AGENT 2 – Literature Retriever
# =========================================================================

def retrieve_literature(findings: list[dict], model, faiss_index, chunks, top_k: int = 8) -> list[dict]:
    """
    Agent 2: Retrieve the most relevant medical literature chunks
    for the given clinical findings.
    """
    # Build query from findings
    query_parts = []
    for f in findings:
        query_parts.append(f["finding"])
        if f.get("value"):
            query_parts.append(str(f["value"]))
    query_text = " ".join(query_parts)

    # Embed query
    q_emb = model.encode([query_text], convert_to_numpy=True).astype("float32")
    import faiss as _faiss
    _faiss.normalize_L2(q_emb)

    # Search
    scores, indices = faiss_index.search(q_emb, top_k)

    results = []
    seen_ids = set()
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        if chunk["chunk_id"] in seen_ids:
            continue
        seen_ids.add(chunk["chunk_id"])
        results.append({
            **chunk,
            "relevance_score": float(score),
        })

    return results


# =========================================================================
# AGENT 3 – Differential Diagnosis Generator
# =========================================================================

_DIFF_SYSTEM = (
    "You are a clinical decision support system. Based on the patient findings "
    "and the following medical literature excerpts, generate a ranked differential "
    "diagnosis. For each condition:\n"
    "1. State the condition name\n"
    "2. Give a brief justification linking patient findings to the condition\n"
    "3. Cite the exact literature chunk(s) that support it using the format "
    "[Source: <article title>]\n"
    "4. Assign a confidence level (High / Moderate / Low)\n\n"
    "Return the results as a numbered list. Be thorough but concise."
)


def _generate_differential_llm(findings: list[dict], literature: list[dict], model_name: str = "gpt-4o") -> str | None:
    # Fall back to 3.5 if 4o isn't available
    findings_text = json.dumps(findings, indent=2)
    lit_text = "\n\n---\n\n".join(
        f"[Chunk: {c['chunk_id']}] (Source: {c['title']})\n{c['text']}"
        for c in literature
    )
    user_prompt = (
        f"## Patient Findings\n{findings_text}\n\n"
        f"## Retrieved Medical Literature\n{lit_text}"
    )
    # Try gpt-4o first, then fall back
    for m in [model_name, "gpt-4o-mini", "gpt-3.5-turbo"]:
        try:
            result = _llm_chat(_DIFF_SYSTEM, user_prompt, model=m, temperature=0.3)
            if result:
                return result
        except Exception:
            continue
    return None


def _score_condition(findings_set: set, keywords: list[str]) -> int:
    """Count how many finding keywords match a condition's keyword list."""
    return sum(1 for kw in keywords if kw in findings_set)


def _generate_differential_fallback(findings: list[dict], literature: list[dict]) -> str:
    """
    Rule-based differential generator. Maps findings to conditions
    using keyword matching against the retrieved literature.
    """
    findings_set = set()
    for f in findings:
        findings_set.add(f["finding"].lower())
        if f.get("value"):
            findings_set.add(str(f["value"]).lower())

    # Condition → (keywords, typical findings, article IDs that discuss it)
    CONDITIONS = {
        "Community-Acquired Pneumonia (CAP)": {
            "keywords": ["fever", "cough", "dyspnea", "shortness of breath",
                         "pleuritic chest pain", "sputum", "tachypnea",
                         "tachycardia", "crackles", "rales"],
            "description": "Infection of the lung parenchyma presenting with respiratory symptoms and systemic inflammation.",
            "article_key": "pneumonia",
        },
        "Acute Heart Failure / Decompensated Heart Failure": {
            "keywords": ["dyspnea", "shortness of breath", "orthopnea",
                         "paroxysmal nocturnal dyspnea", "edema",
                         "lower extremity edema", "leg swelling", "fatigue",
                         "crackles", "rales", "jugular venous distension",
                         "gallop", "tachycardia", "weight gain"],
            "description": "Inability of the heart to pump adequately, causing fluid overload and congestion.",
            "article_key": "heart_failure",
        },
        "Acute Coronary Syndrome (ACS)": {
            "keywords": ["chest pain", "diaphoresis", "nausea", "dyspnea",
                         "shortness of breath", "palpitations", "syncope",
                         "tachycardia", "hypertension", "hypotension",
                         "diabetes", "diabetic", "smoking", "smoker"],
            "description": "Spectrum including unstable angina, NSTEMI, and STEMI due to coronary artery occlusion.",
            "article_key": "acute_coronary",
        },
        "COPD Exacerbation": {
            "keywords": ["cough", "dyspnea", "shortness of breath", "wheezing",
                         "sputum", "smoking", "smoker", "tachypnea",
                         "hypoxemia", "hypoxia"],
            "description": "Acute worsening of COPD symptoms beyond normal day-to-day variation.",
            "article_key": "copd",
        },
        "Asthma Exacerbation": {
            "keywords": ["wheezing", "cough", "dyspnea", "shortness of breath",
                         "chest tightness", "tachypnea", "tachycardia",
                         "hypoxemia"],
            "description": "Acute worsening of airway inflammation and bronchospasm.",
            "article_key": "asthma",
        },
        "Pulmonary Embolism (PE)": {
            "keywords": ["dyspnea", "shortness of breath", "pleuritic chest pain",
                         "chest pain", "tachycardia", "tachypnea", "hemoptysis",
                         "hypoxemia", "hypoxia", "syncope", "hypotension",
                         "leg swelling", "edema"],
            "description": "Obstruction of pulmonary vasculature by thrombus, typically from DVT.",
            "article_key": "pulmonary_embolism",
        },
        "Sepsis": {
            "keywords": ["fever", "tachycardia", "tachypnea", "hypotension",
                         "confusion", "altered mental status", "dyspnea",
                         "shortness of breath", "cough", "dysuria",
                         "abdominal pain", "hypoxemia"],
            "description": "Life-threatening organ dysfunction from dysregulated host response to infection.",
            "article_key": "sepsis",
        },
        "Acute Ischemic Stroke": {
            "keywords": ["hemiparesis", "aphasia", "confusion",
                         "altered mental status", "headache", "dizziness",
                         "vertigo", "ataxia", "diplopia", "hypertension",
                         "hypertensive", "diabetes", "diabetic"],
            "description": "Acute cerebrovascular occlusion causing neurological deficits.",
            "article_key": "stroke",
        },
        "Type 2 Diabetes – Acute Complications": {
            "keywords": ["polyuria", "polydipsia", "weight loss", "fatigue",
                         "blurred vision", "nausea", "vomiting", "confusion",
                         "diabetes", "diabetic", "obese", "obesity"],
            "description": "Hyperglycemic emergencies (DKA/HHS) or symptomatic uncontrolled diabetes.",
            "article_key": "diabetes",
        },
        "COVID-19": {
            "keywords": ["fever", "cough", "fatigue", "myalgia", "headache",
                         "anosmia", "ageusia", "dyspnea", "shortness of breath",
                         "hypoxemia", "hypoxia", "diarrhea"],
            "description": "SARS-CoV-2 infection ranging from mild to critical illness.",
            "article_key": "covid",
        },
    }

    # Score each condition
    scored = []
    for name, info in CONDITIONS.items():
        score = _score_condition(findings_set, info["keywords"])
        if score > 0:
            # Find supporting literature
            supporting = []
            for chunk in literature:
                if info["article_key"] in chunk.get("article_id", "").lower():
                    supporting.append(chunk)
            # Also find chunks that mention any matching keyword
            if not supporting:
                for chunk in literature:
                    chunk_lower = chunk["text"].lower()
                    if any(kw in chunk_lower for kw in info["keywords"] if kw in findings_set):
                        supporting.append(chunk)
                        if len(supporting) >= 2:
                            break

            scored.append({
                "condition": name,
                "score": score,
                "max_possible": len(info["keywords"]),
                "description": info["description"],
                "supporting": supporting[:3],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Build output
    if not scored:
        return "No matching conditions found for the provided findings. Please provide more clinical detail."

    lines = ["# Differential Diagnosis\n"]
    for rank, item in enumerate(scored[:7], 1):
        pct = round(item["score"] / max(item["max_possible"], 1) * 100)
        if pct >= 60:
            confidence = "High"
        elif pct >= 35:
            confidence = "Moderate"
        else:
            confidence = "Low"

        lines.append(f"## {rank}. {item['condition']}")
        lines.append(f"**Confidence:** {confidence} ({item['score']}/{item['max_possible']} key findings matched)\n")
        lines.append(f"**Description:** {item['description']}\n")

        # Justification
        matched = [kw for kw in CONDITIONS[item["condition"]]["keywords"] if kw in findings_set]
        lines.append(f"**Matching findings:** {', '.join(matched)}\n")

        # Citations
        if item["supporting"]:
            lines.append("**Supporting Evidence:**")
            for chunk in item["supporting"]:
                # Extract a relevant sentence
                snippet = chunk["text"][:300].replace("\n", " ")
                lines.append(f'> "{snippet}..."')
                lines.append(f'> — *[Source: {chunk["title"]}]({chunk["url"]})*\n')
        else:
            lines.append("*No directly matching literature chunk retrieved for this condition.*\n")

        lines.append("")

    return "\n".join(lines)


def generate_differential(findings: list[dict], literature: list[dict]) -> str:
    """
    Agent 3: Generate a ranked differential diagnosis with citations.
    Uses LLM if available, otherwise rule-based fallback.
    """
    result = _generate_differential_llm(findings, literature)
    if result:
        return result
    return _generate_differential_fallback(findings, literature)


# =========================================================================
# AGENT 4 – Output Validator
# =========================================================================

def validate_output(differential: str, literature: list[dict]) -> dict:
    """
    Agent 4: Validate that cited sources exist in the retrieved literature.
    Returns a dict with 'valid' bool and 'issues' list.
    """
    issues = []
    # Check that article titles mentioned in the differential exist in literature
    lit_titles = {c["title"].lower() for c in literature}
    lit_ids = {c["chunk_id"].lower() for c in literature}

    # Find all [Source: ...] citations
    cited = re.findall(r'\[Source:\s*([^\]]+)\]', differential, re.IGNORECASE)
    for cite in cited:
        cite_lower = cite.strip().lower()
        if not any(cite_lower in t or t in cite_lower for t in lit_titles):
            issues.append(f"Citation not found in retrieved literature: '{cite.strip()}'")

    # Find all [Chunk: ...] references
    chunk_refs = re.findall(r'\[Chunk:\s*([^\]]+)\]', differential, re.IGNORECASE)
    for ref in chunk_refs:
        if ref.strip().lower() not in lit_ids:
            issues.append(f"Chunk reference not found: '{ref.strip()}'")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "citations_found": len(cited) + len(chunk_refs),
    }
