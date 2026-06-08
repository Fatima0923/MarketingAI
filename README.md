# AI Advertising Pipeline

A multi-agent AI system for advertising research. Uses synthetic consumer personas — to evaluate advertising stimuli on validated marketing constructs, with optional comparison against human survey responses.

Built with **CrewAI**, **FAISS RAG**, **Langfuse observability**, and a **Gradio** interface.

---

## Overview

The pipeline operates in two modes:

| Mode | Input | Output |
|------|-------|--------|
| **Mode A** | Ad sets only | Synthetic responses CSV + descriptive analytics |
| **Mode B** | Ad sets + human responses | Mode A + AI-human statistical comparison (Cohen's d, equivalence testing, alignment verdict) |

---

## Features

- Upload ad creatives directly (JPG, PNG, WEBP, PDF) — vision AI extracts headline, copy, and visual description automatically
- Generate N synthetic consumer personas using stratified Big Five + demographic sampling
- Evaluate ads on validated Likert scales (Brand Attitude, Purchase Intention, Brand Fit, Ad Credibility)
- FAISS RAG indexes uploaded brand briefs for grounded evaluation
- Risk Auditor Agent flags outlier and biased responses for human review (HITL)
- Supports DeepSeek, OpenAI, and Google Gemini — switchable from the UI
- Langfuse observability traces every agent call, LLM call, and tool call
- Outputs: responses CSV, comparison CSV, bar charts, full JSON

---

## Project Structure

```
project/
│
├── app.py                        # Gradio UI — all inputs from UI, nothing hardcoded
├── crew.py                       # Pipeline orchestration (Mode A / Mode B)
├── agents.py                     # 5 CrewAI agents
├── tasks.py                      # 5 CrewAI tasks with expected outputs
├── config.py                     # Provider registry, scales, paths
│
├── tools/
│   ├── __init__.py
│   └── tools.py                  # All tools in one file (8 sections)
│                                 #   Shared LLM call · Document ingestion
│                                 #   FAISS RAG store · Persona generator
│                                 #   Survey response generator · Analytics
│                                 #   CSV export · ALL_TOOLS list
│
├── fallback/
│   ├── __init__.py
│   └── fallback_handler.py       # 6 failure modes handled
│
├── monitoring/
│   ├── __init__.py
│   └── langfuse_config.py        # Langfuse observability setup
│
├── data/
│   ├── uploads/                  # Uploaded ad files and brief documents
│   └── stimuli/                  # (optional) pre-loaded ad JSON sets
│
├── outputs/                      # Generated CSVs, JSON, charts
│   └── examples/                 # Example outputs for reference
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.12 (required — CrewAI 1.x does not support Python 3.14)
- An API key for at least one supported provider: DeepSeek, OpenAI, or Google Gemini

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 3. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment

```bash
cp .env.example .env
```

Open `.env` and add your API key(s):

```
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
```

You only need one. All three are optional — the UI lets you enter keys directly.

### 6. Run the application

```bash
python app.py
```

Open your browser at: **http://localhost:7860**

---

## Usage

### Study Setup tab
- Select your LLM provider (DeepSeek / OpenAI / Gemini)
- Enter your API key
- Choose measurement constructs (Brand Attitude, Purchase Intention, Brand Fit, Ad Credibility)
- Set persona parameters: age range, n personas, personality variance
- Click **Save Setup**

### Ad Inputs tab
- **Upload ad files** — drag and drop JPG/PNG/WEBP/PDF ad creatives. Click Analyse to extract ad details using vision AI.
- **Manual entry** — type ad details directly for text-based ads.
- **JSON upload** — batch upload multiple ads as a JSON array.
- **Brand brief upload** — upload PDF/DOCX brand documents. These are indexed into FAISS and retrieved by agents during evaluation.
- **Human responses (optional)** — upload a CSV of human survey data to trigger Mode B comparison.

### Run Pipeline tab
- Click **Check inputs** to verify setup
- Click **Run Pipeline** — pipeline runs fully unattended
- Watch real-time log output

### Results tab
- View construct means and standard deviations per ad
- Download response CSVs
- Review flagged responses in the HITL queue

### AI vs Human tab (Mode B only)
- View four-level statistical comparison
- Descriptive comparison, Cohen's d, equivalence verdict, alignment verdict per construct

---

## Measurement Scales

All constructs use validated 3-item 7-point Likert scales

---

## Persona Construction

Personas are generated using stratified sampling across:
- **Big Five personality dimensions**
Parameters are fully configurable from the UI — age range, number of personas, personality variance, included genders and income levels.

---

## Fallback Mechanisms

| Failure Mode | Fallback Behaviour |
|---|---|
| LLM API failure | Exponential backoff (3s → 6s → 12s), then neutral midpoint |
| JSON parse failure | Recovery parser → structured default |
| Acquiescence bias | Detect all-same ratings → regenerate once |
| Statistical outlier | Flag for HITL review queue |
| Tool crash | Graceful error + neutral response, pipeline continues |
| Missing construct | Fill with midpoint 4.0, flag automatically |

---

## Supported Providers

| Provider | Models | Vision |
|---|---|---|
| DeepSeek | deepseek-chat, deepseek-reasoner | Limited |
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo | ✅ Full |
| Google Gemini | gemini-2.0-flash, gemini-1.5-pro | ✅ Full |

For image/PDF ad analysis, OpenAI gpt-4o or Gemini gemini-2.0-flash is recommended.

---

## Observability (Langfuse)

To enable tracing, add Langfuse keys to your `.env`:

```
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Traces are generated for every agent execution, LLM call, tool call, and error. View at [cloud.langfuse.com](https://cloud.langfuse.com).

---

## Output Files

All outputs are saved to the `outputs/` directory:

| File | Description |
|---|---|
| `responses_{ad_id}.csv` | All synthetic responses for one ad |
| `ai_human_comparison.csv` | Mode B statistical comparison |
| `results_full.json` | Full pipeline results including metadata |

---

## Human Response CSV Format (Mode B)

To enable AI-human comparison, upload a CSV with one of these column formats:

**Option 1 — Mean columns:**
```
brand_attitude_mean, purchase_intention_mean, brand_fit_mean
```

**Option 2 — Item columns:**
```
brand_attitude_item1, brand_attitude_item2, brand_attitude_item3,
purchase_intention_item1, ...
```

See `outputs/examples/human_responses_template.csv` for a template.

---

## MCP Discussion

This pipeline could benefit from Model Context Protocol (MCP) integration in four areas:

1. **Brand Asset Library** — brand documents and positioning exposed as a retrievable MCP resource, eliminating manual upload
2. **Ad Stimuli Repository** — versioned library of ad creatives accessible by campaign or brand
3. **Survey Platform Integration** — Qualtrics/SurveyMonkey API as MCP for real-time human data ingestion
4. **Persona Registry** — curated persona libraries reusable across studies and brands

The `PersonaGeneratorTool` could be packaged as an MCP server, exposing a `generate_personas(n, seed, constraints)` endpoint usable by any downstream agent.

---

## Requirements

See `requirements.txt` for the full dependency list. Key packages:

```
crewai>=1.0.0
langchain-openai>=0.1.0
gradio>=4.0.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
langfuse>=2.0.0
PyMuPDF>=1.23.0
```

---

## License

This project is for research purposes.
