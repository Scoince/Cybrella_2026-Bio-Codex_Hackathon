# HealthGuard AI

Multi-agent clinical decision support system that accepts clinical notes, extracts symptoms, retrieves relevant medical literature, and outputs a traceable differential diagnosis with citations.

> **Disclaimer:** This is a prototype for educational and demonstration purposes only. It is NOT a certified medical device and must NOT be used for actual clinical decision-making.

## Architecture

```
Clinical Notes
     │
     ▼
┌─────────────────────┐
│  Agent 1: Symptom    │  NLP extraction of clinical findings
│  Extractor          │  (LLM or rule-based fallback)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 2: Literature │  Semantic search over medical corpus
│  Retriever          │  (FAISS + all-MiniLM-L6-v2)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 3: Differential│  Ranked diagnosis with evidence
│  Generator           │  (LLM or rule-based fallback)
└────────┬─────────────┘
         │
         ▼
┌─────────────────────┐
│  Agent 4: Output     │  Citation integrity check
│  Validator          │
└─────────────────────┘
```

## Features

- **Multi-agent pipeline** with 4 specialized agents
- **Semantic literature retrieval** using FAISS vector search + sentence-transformers
- **Built-in medical corpus** of 10 review articles covering pneumonia, heart failure, diabetes, COPD, ACS, asthma, PE, sepsis, stroke, CKD, and COVID-19
- **Traceable citations** — every diagnosis references specific literature chunks
- **Dual mode**: works with OpenAI API keys (GPT-4o/3.5) or standalone with rule-based fallback
- **Streamlit UI** with example clinical vignettes

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch (smaller download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. (Optional) Set OpenAI API key

For LLM-powered extraction and diagnosis generation, set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Or enter it in the sidebar of the Streamlit app. Without an API key, the system uses a built-in rule-based engine that still produces meaningful differential diagnoses.

## Project Structure

```
healthguard_ai/
├── app.py                  # Streamlit UI
├── orchestrator.py         # Pipeline coordinator
├── agents/
│   ├── __init__.py
│   └── agents.py           # 4 agent functions
├── data/
│   ├── __init__.py
│   └── corpus.py           # Medical corpus + FAISS index builder
├── test_pipeline.py        # End-to-end tests
├── requirements.txt
└── .streamlit/
    └── config.toml         # Streamlit theme
```

## Agents

| Agent | Function | Description |
|-------|----------|-------------|
| 1 | `extract_symptoms()` | Extracts clinical findings, demographics, and risk factors from free-text notes |
| 2 | `retrieve_literature()` | Embeds findings as a query and performs semantic search over the FAISS index |
| 3 | `generate_differential()` | Produces a ranked differential diagnosis with supporting evidence and citations |
| 4 | `validate_output()` | Checks that all cited sources exist in the retrieved literature |

## Extending the Corpus

To add real PubMed Central articles:

1. Download open-access articles as PDFs or text
2. Add them to the `ARTICLES` list in `data/corpus.py` with `id`, `title`, `url`, and `text` fields
3. The FAISS index rebuilds automatically on next app start

## Upgrading to Production

- **Replace FAISS with Pinecone** for persistent, scalable vector search
- **Use OpenAI API** for all agents (set `OPENAI_API_KEY`)
- **Add more articles** (20-30+) for broader condition coverage
- **Implement LangGraph** for complex multi-turn agent orchestration
- **Add user authentication** and audit logging for clinical use

## Tech Stack

- Python 3.10+
- Streamlit (UI)
- FAISS (vector search)
- sentence-transformers / all-MiniLM-L6-v2 (embeddings)
- OpenAI API (optional, for LLM-powered agents)
- langchain-text-splitters (document chunking)
