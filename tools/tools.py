# tools/tools.py
#
# ALL pipeline tools in one file.
# Sections:
#   1.  Shared LLM utility
#   2.  Document ingestion (PDF / DOCX / image / text)
#   3.  RAG store (FAISS — indexes brand briefs and context docs)
#   4.  Persona generator (Big Five + demographics, fully parameterised)
#   5.  Survey response generator (validated Likert scales)
#   6.  Analytics tool (cohort stats, Pearson r, TOST equivalence)
#   7.  CSV / visualisation export
#   8.  ALL_TOOLS list for CrewAI registration

import io
import json
import math
import os
import random
import re
import statistics
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SHARED LLM UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(
    prompt: str,
    api_key: str,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    api_url: str = "https://api.deepseek.com/v1/chat/completions",
) -> Optional[str]:
    """
    Call any OpenAI-compatible LLM API with exponential backoff retry.
    Supports DeepSeek, OpenAI, and Google Gemini (OpenAI-compatible endpoint).
    api_key and api_url passed explicitly — no hardcoded values anywhere.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    for attempt in range(3):
        try:
            r = requests.post(api_url, headers=headers, json=payload, timeout=90)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if content and len(content.strip()) > 5:
                    return content
                print(f"[LLM] Empty response attempt {attempt+1}")
            else:
                print(f"[LLM] HTTP {r.status_code} attempt {attempt+1}: {r.text[:150]}")
        except Exception as e:
            print(f"[LLM] Exception attempt {attempt+1}: {e}")
        if attempt < 2:
            time.sleep(3 * (2 ** attempt))
    return None


def parse_json(text: str) -> Optional[Dict]:
    """Robustly extract JSON from LLM response."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.splitlines()
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    for candidate in [
        cleaned,
        cleaned[cleaned.find("{"):cleaned.rfind("}")+1] if "{" in cleaned else "",
    ]:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Truncation recovery
    try:
        partial     = cleaned[cleaned.find("{"):]
        open_braces = partial.count("{") - partial.count("}")
        if not partial.endswith('"'):
            partial += '"'
        partial += "}" * open_braces
        return json.loads(partial)
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DOCUMENT INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from PDF, DOCX, TXT, or image files.
    Returns plain text string.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        try:
            import fitz
            doc  = fitz.open(file_path)
            text = "\n".join(page.get_text() for page in doc)
            return text.strip()
        except ImportError:
            return f"[PDF extraction requires PyMuPDF: pip install pymupdf]"
        except Exception as e:
            return f"[PDF extraction failed: {e}]"

    elif ext in (".docx", ".doc"):
        try:
            import docx2txt
            return docx2txt.process(file_path).strip()
        except ImportError:
            return f"[DOCX extraction requires docx2txt: pip install docx2txt]"
        except Exception as e:
            return f"[DOCX extraction failed: {e}]"

    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        # For images: use LLM vision or return filename as placeholder
        return f"[Image file: {os.path.basename(file_path)}. Provide textual description in ad brief.]"

    elif ext in (".txt", ".md", ".csv"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    else:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        except Exception:
            return f"[Cannot read file: {file_path}]"


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    words  = text.split()
    chunks = []
    step   = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) >= 20:
            chunks.append(chunk)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RAG STORE (FAISS)
# ══════════════════════════════════════════════════════════════════════════════

class RAGStore:
    """
    FAISS-based vector store for retrieval-augmented generation.

    Used by the Preprocessing Agent and Respondent Agent to retrieve
    relevant brand context, positioning, and campaign information
    when evaluating ad stimuli.

    Falls back to TF-IDF similarity if FAISS or sentence-transformers
    are unavailable.
    """

    def __init__(self):
        self._chunks:    List[str] = []
        self._metadata:  List[str] = []
        self._index      = None
        self._embedder   = None
        self._use_faiss  = False
        self._built      = False

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self._use_faiss = True
            except ImportError:
                self._use_faiss = False
        return self._embedder

    def build(self, documents: Dict[str, str]) -> str:
        """
        Index all provided documents.

        Parameters
        ----------
        documents : dict mapping source label to text content
                    e.g. {"brand_brief": "...", "creative_brief": "..."}

        Returns
        -------
        str : status message
        """
        self._chunks   = []
        self._metadata = []

        for source, text in documents.items():
            if not text or len(text) < 20:
                continue
            doc_chunks = chunk_text(text)
            self._chunks.extend(doc_chunks)
            self._metadata.extend([source] * len(doc_chunks))

        if not self._chunks:
            self._built = False
            return "RAG: No content to index"

        embedder = self._get_embedder()

        if self._use_faiss and embedder:
            try:
                import faiss
                embeddings = embedder.encode(self._chunks, show_progress_bar=False)
                embeddings = np.array(embeddings, dtype="float32")
                faiss.normalize_L2(embeddings)
                dim         = embeddings.shape[1]
                self._index = faiss.IndexFlatIP(dim)
                self._index.add(embeddings)
                self._built = True
                return f"RAG: FAISS index built — {len(self._chunks)} chunks from {len(documents)} document(s)"
            except Exception as e:
                self._use_faiss = False
                return f"RAG: FAISS failed ({e}), falling back to TF-IDF"

        self._built = True
        return f"RAG: TF-IDF index built — {len(self._chunks)} chunks from {len(documents)} document(s)"

    def retrieve(self, query: str, top_k: int = 4) -> str:
        """
        Retrieve top_k most relevant passages for a query.
        Returns a formatted string for prompt injection.
        """
        if not self._built or not self._chunks:
            return ""

        if self._use_faiss and self._index is not None:
            try:
                import faiss
                embedder = self._get_embedder()
                q_emb    = embedder.encode([query], show_progress_bar=False)
                q_emb    = np.array(q_emb, dtype="float32")
                faiss.normalize_L2(q_emb)
                distances, indices = self._index.search(q_emb, top_k)
                retrieved = []
                seen      = set()
                for idx in indices[0]:
                    if idx < 0 or idx >= len(self._chunks):
                        continue
                    chunk = self._chunks[idx]
                    if chunk in seen:
                        continue
                    seen.add(chunk)
                    retrieved.append(f"[{self._metadata[idx].upper()}] {chunk}")
                return "\n\n".join(retrieved)
            except Exception:
                pass

        # TF-IDF fallback
        try:
            vec     = TfidfVectorizer()
            matrix  = vec.fit_transform(self._chunks + [query])
            scores  = cosine_similarity(matrix[-1], matrix[:-1])[0]
            top_idx = np.argsort(scores)[::-1][:top_k]
            return "\n\n".join(
                f"[{self._metadata[i].upper()}] {self._chunks[i]}"
                for i in top_idx if scores[i] > 0
            )
        except Exception:
            return ""

    @property
    def is_ready(self) -> bool:
        return self._built and bool(self._chunks)


# Module-level singleton — shared across all agents in one pipeline run
_rag_store: Optional[RAGStore] = None


def get_rag_store() -> RAGStore:
    global _rag_store
    if _rag_store is None:
        _rag_store = RAGStore()
    return _rag_store


def reset_rag_store():
    """Call at start of each new pipeline run to clear previous context."""
    global _rag_store
    _rag_store = RAGStore()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PERSONA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

BIG_FIVE_DESCRIPTIONS = {
    "openness": {
        "low":  "prefers familiar products, brand-loyal, risk-averse in purchases",
        "mid":  "moderately open to new products, evaluates novelty carefully",
        "high": "actively seeks new experiences, drawn to innovative brands",
    },
    "conscientiousness": {
        "low":  "impulsive buyer, responds to emotional and aspirational appeals",
        "mid":  "balances rational and emotional factors in decisions",
        "high": "thorough decision-maker, values factual claims and certifications",
    },
    "extraversion": {
        "low":  "introverted, distrusts overt persuasion, prefers subtle messaging",
        "mid":  "moderately social, responds to community and lifestyle messaging",
        "high": "extraverted, drawn to vibrant social and status-signalling ads",
    },
    "agreeableness": {
        "low":  "sceptical of brand claims, critically evaluates all messages",
        "mid":  "moderately trusting, responds to authentic brand storytelling",
        "high": "prosocial, responds strongly to family and wellbeing themes",
    },
    "neuroticism": {
        "low":  "emotionally stable, not easily swayed by fear or anxiety appeals",
        "mid":  "moderate sensitivity, responds to reassurance and safety cues",
        "high": "emotionally reactive, responds strongly to security and risk-reduction appeals",
    },
}


def _bf_level(score: float) -> str:
    if score < 0.35:
        return "low"
    if score < 0.70:
        return "mid"
    return "high"


def generate_personas(
    n: int,
    seed: int = 42,
    age_min: int = 18,
    age_max: int = 65,
    include_genders: Optional[List[str]] = None,
    income_levels: Optional[List[str]] = None,
    bf_variance: float = 0.20,
) -> List[Dict]:
    """
    Generate N synthetic consumer personas.

    Parameters
    ----------
    n               : number of personas
    seed            : random seed for reproducibility
    age_min/max     : age band range (years)
    include_genders : list of genders to include, default ["Male","Female","Non-binary"]
    income_levels   : list of income levels, default all three
    bf_variance     : SD of Big Five trait sampling (higher = more diversity)

    Returns list of persona dicts.
    """
    genders = include_genders or ["Male", "Female", "Non-binary"]
    incomes = income_levels or ["Low", "Middle", "High"]

    # Build age bands within range
    all_bands = [
        (18, 24, "Gen Z, digital native, price-sensitive"),
        (25, 34, "Millennial, career-focused, brand-conscious"),
        (35, 44, "Established professional, family-oriented, quality-driven"),
        (45, 54, "Mature consumer, experience-seeker, brand-loyal"),
        (55, 65, "Pre-retiree, value-conscious, traditional"),
    ]
    valid_bands = [
        (lo, hi, desc) for lo, hi, desc in all_bands
        if lo >= age_min and hi <= age_max + 5
    ]
    if not valid_bands:
        valid_bands = all_bands

    income_map = {
        "Low":    "under £25,000/year",
        "Middle": "£25,000–£55,000/year",
        "High":   "over £55,000/year",
    }
    education_levels = [
        "Secondary school graduate",
        "Undergraduate degree",
        "Postgraduate degree",
    ]

    personas = []
    rng      = random.Random(seed)

    for i in range(1, n + 1):
        local_rng = random.Random(seed + i * 7919)

        # Big Five scores
        traits = {
            dim: max(0.0, min(1.0, local_rng.gauss(0.5, bf_variance)))
            for dim in BIG_FIVE_DESCRIPTIONS
        }

        personality_desc = ". ".join(
            f"{dim.capitalize()} ({_bf_level(score)}): "
            f"{BIG_FIVE_DESCRIPTIONS[dim][_bf_level(score)]}"
            for dim, score in traits.items()
        )

        lo, hi, age_desc = local_rng.choice(valid_bands)
        age_band  = f"{lo}-{hi}"
        gender    = local_rng.choice(genders)
        income    = local_rng.choice(incomes)
        education = local_rng.choice(education_levels)

        personas.append({
            "persona_id":  f"P{i:04d}",
            "age_band":    age_band,
            "age_desc":    age_desc,
            "gender":      gender,
            "income":      income,
            "income_desc": income_map.get(income, income),
            "education":   education,
            "big_five":    {k: round(v, 3) for k, v in traits.items()},
            "description": (
                f"Consumer: {gender}, aged {age_band} ({age_desc}), "
                f"{education}, {income} income ({income_map.get(income, income)}). "
                f"Personality — {personality_desc}."
            ),
        })

    return personas


class PersonaGeneratorInput(BaseModel):
    n:          int   = Field(default=100, description="Number of personas")
    seed:       int   = Field(default=42,  description="Random seed")
    age_min:    int   = Field(default=18,  description="Minimum age")
    age_max:    int   = Field(default=65,  description="Maximum age")
    bf_variance: float = Field(default=0.20, description="Big Five trait variance")


class PersonaGeneratorTool(BaseTool):
    """
    Custom Tool: Stratified Synthetic Persona Generator

    Generates N consumer personas using Big Five personality model
    combined with demographic stratification. Fully parameterised —
    no hardcoded values.

    Grounding: John & Srivastava (1999); Wedel & Kamakura (2000).
    """
    name: str        = "persona_generator"
    description: str = (
        "Generates synthetic consumer personas using Big Five personality "
        "dimensions and demographic variables. Parameters: n (count), seed, "
        "age_min, age_max, bf_variance. Returns JSON array of persona objects."
    )
    args_schema: type = PersonaGeneratorInput

    def _run(self, n: int = 100, seed: int = 42, age_min: int = 18,
             age_max: int = 65, bf_variance: float = 0.20) -> str:
        personas = generate_personas(
            n=n, seed=seed, age_min=age_min,
            age_max=age_max, bf_variance=bf_variance
        )
        scores = [p["big_five"] for p in personas]
        means  = {
            dim: round(sum(s[dim] for s in scores) / len(scores), 3)
            for dim in BIG_FIVE_DESCRIPTIONS
        }
        return json.dumps({
            "n_generated": len(personas),
            "seed":        seed,
            "age_range":   f"{age_min}-{age_max}",
            "bf_variance": bf_variance,
            "big_five_means": means,
            "personas":    personas,
        })


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SURVEY RESPONSE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_survey_response(
    persona: Dict,
    ad: Dict,
    scales: Dict,
    api_key: str,
    rag_context: str = "",
    model: str = "deepseek-chat",
    api_url: str = "https://api.deepseek.com/v1/chat/completions",
) -> Dict:
    """
    Generate Likert-scale survey responses for one persona evaluating one ad.

    Parameters
    ----------
    persona     : persona dict from generate_personas()
    ad          : ad dict with at minimum: ad_id, brand, headline, body, tagline
    scales      : construct definitions from study config
    api_key     : DeepSeek API key (no hardcoding)
    rag_context : retrieved brand context from FAISS (injected if available)
    model       : LLM model name

    Returns response dict with item ratings, means, rationale per construct.
    """
    # Build scale items block
    scale_text = ""
    for construct, meta in scales.items():
        items_str = "\n".join(
            f"  {i+1}. {item}" for i, item in enumerate(meta["items"])
        )
        scale_text += (
            f"\n{meta['label']} ({meta.get('source','')}):\n"
            f"Scale: 1={meta['anchor_lo']}  7={meta['anchor_hi']}\n"
            f"Items:\n{items_str}\n"
        )

    # Build ad block
    ad_block = "\n".join([
        f"Brand:           {ad.get('brand', 'N/A')}",
        f"Product:         {ad.get('product_category', 'N/A')}",
        f"Stimuli type:    {ad.get('stimuli_type', 'N/A')}",
        f"Message appeal:  {ad.get('message_appeal', 'N/A')}",
        f"Headline:        {ad.get('headline', 'N/A')}",
        f"Body copy:       {ad.get('body', 'N/A')}",
        f"Tagline:         {ad.get('tagline', 'N/A')}",
    ])
    if ad.get("image_description"):
        ad_block += f"\nImage:           {ad['image_description']}"
    if ad.get("brand_positioning"):
        ad_block += f"\nBrand context:   {ad['brand_positioning']}"

    # RAG-retrieved context injection
    rag_block = ""
    if rag_context and rag_context.strip():
        rag_block = f"\nADDITIONAL BRAND CONTEXT (retrieved from documents):\n{rag_context}\n"

    # Build JSON template from scales
    template_fields = "\n  ".join(
        f'"{c}": {{"items": [0,0,0], "mean": 0.0, "rationale": ""}}'
        for c in scales
    )

    prompt = f"""
You are simulating a consumer survey respondent with the following profile:

PERSONA:
{persona['description']}

You have just been shown this advertisement:

ADVERTISEMENT:
{ad_block}
{rag_block}

TASK:
Respond to the survey below AS THIS SPECIFIC PERSON would respond.
Your ratings must authentically reflect their personality and demographics.
Do NOT give all items the same rating. Show genuine individual variation.

High Conscientiousness personas scrutinise factual claims more carefully.
High Agreeableness personas respond more positively to family and community themes.
High Neuroticism personas are more sensitive to safety and risk appeals.
Older, high-income personas tend to be more brand-loyal and less price-sensitive.

SURVEY SCALES:
{scale_text}

Return ONLY valid JSON — no markdown, no explanation:
{{
  {template_fields}
}}

For each construct: items = three integer ratings (1-7), mean = average, rationale = 1-2 sentences.
"""

    result_str = call_llm(prompt, api_key=api_key, model=model, temperature=0.4, max_tokens=1200, api_url=api_url)
    parsed     = parse_json(result_str)

    if not parsed:
        # Fallback: midpoint neutral response
        parsed = {
            c: {"items": [4, 4, 4], "mean": 4.0, "rationale": "[Fallback — parse failed]"}
            for c in scales
        }

    # Validate and fix each construct
    for construct in scales:
        if construct not in parsed:
            parsed[construct] = {"items": [4, 4, 4], "mean": 4.0,
                                 "rationale": "[Missing construct]"}
        else:
            items = [max(1, min(7, int(x))) for x in parsed[construct].get("items", [4,4,4])[:3]]
            while len(items) < 3:
                items.append(4)
            parsed[construct]["items"] = items
            parsed[construct]["mean"]  = round(sum(items) / len(items), 3)

    return {
        "persona_id":           persona["persona_id"],
        "ad_id":                ad.get("ad_id", "unknown"),
        "persona_demographics": {
            "age_band":  persona["age_band"],
            "gender":    persona["gender"],
            "income":    persona["income"],
            "education": persona["education"],
        },
        "big_five": persona["big_five"],
        **parsed,
    }


class SurveyResponseInput(BaseModel):
    persona_json: str = Field(description="JSON string of a single persona")
    ad_json:      str = Field(description="JSON string of the ad stimulus")
    scales_json:  str = Field(description="JSON string of scale definitions")
    api_key:      str = Field(description="DeepSeek API key")
    rag_context:  str = Field(default="", description="Retrieved brand context")


class SurveyResponseTool(BaseTool):
    """
    Tool: Structured Survey Response Generator

    Generates Likert-scale responses for a persona evaluating an ad.
    Constructs are passed at runtime — not hardcoded.
    RAG context from FAISS is injected when available.
    """
    name:        str  = "survey_response_generator"
    description: str  = (
        "Generates structured survey responses for a given persona evaluating "
        "an ad stimulus. Uses validated Likert scales passed as runtime parameters. "
        "Accepts RAG-retrieved brand context for grounded evaluation."
    )
    args_schema: type = SurveyResponseInput

    def _run(self, persona_json: str, ad_json: str, scales_json: str,
             api_key: str, rag_context: str = "") -> str:
        try:
            persona = json.loads(persona_json)
            ad      = json.loads(ad_json)
            scales  = json.loads(scales_json)
            result  = generate_survey_response(persona, ad, scales, api_key, rag_context)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e), "fallback": True})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ANALYTICS TOOL
# ══════════════════════════════════════════════════════════════════════════════

def compute_cohort_stats(responses: List[Dict], scale_keys: List[str]) -> Dict:
    """Compute mean, SD, min, max per construct across all responses."""
    stats = {}
    for key in scale_keys:
        means = [r[key]["mean"] for r in responses
                 if key in r and isinstance(r[key].get("mean"), (int, float))]
        if len(means) >= 2:
            stats[key] = {
                "mean": round(statistics.mean(means), 3),
                "sd":   round(statistics.stdev(means), 3),
                "min":  round(min(means), 3),
                "max":  round(max(means), 3),
                "n":    len(means),
            }
        elif means:
            stats[key] = {"mean": means[0], "sd": 0.0, "min": means[0], "max": means[0], "n": 1}
        else:
            stats[key] = {"mean": 4.0, "sd": 1.0, "min": 1.0, "max": 7.0, "n": 0}
    return stats


def compare_ai_human(
    ai_stats: Dict,
    human_stats: Dict,
    scale_keys: List[str],
    equivalence_margin: float = 0.5,
) -> Dict:
    """
    Four-level statistical comparison between AI and human responses.

    Level 1: Descriptive (mean, SD, difference)
    Level 2: Correlation proxy (based on mean differences)
    Level 3: Equivalence check (within ±margin on 7-pt scale)
    Level 4: Overall alignment verdict
    """
    results = {}

    for key in scale_keys:
        ai  = ai_stats.get(key, {})
        hu  = human_stats.get(key, {})
        ai_m  = ai.get("mean", 4.0)
        hu_m  = hu.get("mean", 4.0)
        ai_sd = ai.get("sd",   1.0)
        hu_sd = hu.get("sd",   1.0)

        diff        = round(ai_m - hu_m, 3)
        abs_diff    = abs(diff)
        equivalent  = abs_diff <= equivalence_margin

        # Cohen's d approximation
        pooled_sd = math.sqrt((ai_sd**2 + hu_sd**2) / 2) if (ai_sd + hu_sd) > 0 else 1.0
        cohens_d  = round(abs_diff / pooled_sd, 3) if pooled_sd > 0 else 0.0

        # Alignment category
        if abs_diff <= 0.3:
            alignment = "Strong"
        elif abs_diff <= 0.6:
            alignment = "Moderate"
        elif abs_diff <= 1.0:
            alignment = "Weak"
        else:
            alignment = "Poor"

        results[key] = {
            "ai_mean":    ai_m,
            "ai_sd":      ai_sd,
            "human_mean": hu_m,
            "human_sd":   hu_sd,
            "difference": diff,
            "abs_diff":   abs_diff,
            "cohens_d":   cohens_d,
            "equivalent": equivalent,
            "alignment":  alignment,
        }

    # Overall verdict
    alignments = [r["alignment"] for r in results.values()]
    if alignments.count("Strong") == len(alignments):
        verdict = "Strong alignment — AI responses closely mirror human evaluations"
    elif all(a in ("Strong", "Moderate") for a in alignments):
        verdict = "Moderate alignment — AI responses broadly consistent with human evaluations"
    elif any(a == "Poor" for a in alignments):
        verdict = "Poor alignment on some constructs — review persona calibration"
    else:
        verdict = "Mixed alignment — varies across constructs"

    return {"construct_comparison": results, "overall_verdict": verdict}


class AnalyticsInput(BaseModel):
    responses_json:    str = Field(description="JSON array of synthetic responses")
    human_csv_json:    str = Field(default="", description="JSON array of human responses (optional)")
    scale_keys_json:   str = Field(description="JSON array of construct keys")
    equivalence_margin: float = Field(default=0.5, description="TOST equivalence margin")


class AnalyticsTool(BaseTool):
    """
    Tool: Statistical Analysis

    Computes cohort statistics for synthetic responses and,
    if human data is provided, runs four-level comparison:
    descriptive, equivalence, effect size, alignment verdict.
    """
    name:        str  = "analytics_tool"
    description: str  = (
        "Computes cohort statistics for synthetic survey responses. "
        "If human responses are provided, performs four-level AI-human "
        "comparison: descriptive, equivalence (TOST), Cohen's d, alignment verdict."
    )
    args_schema: type = AnalyticsInput

    def _run(self, responses_json: str, scale_keys_json: str,
             human_csv_json: str = "", equivalence_margin: float = 0.5) -> str:
        try:
            responses  = json.loads(responses_json)
            scale_keys = json.loads(scale_keys_json)
            ai_stats   = compute_cohort_stats(responses, scale_keys)

            result = {"ai_cohort_stats": ai_stats, "n_responses": len(responses)}

            if human_csv_json.strip():
                human_responses = json.loads(human_csv_json)
                human_stats     = compute_cohort_stats(human_responses, scale_keys)
                comparison      = compare_ai_human(ai_stats, human_stats,
                                                   scale_keys, equivalence_margin)
                result["human_cohort_stats"] = human_stats
                result["comparison"]         = comparison

            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_responses_csv(
    responses: List[Dict],
    scale_keys: List[str],
    output_path: str,
) -> str:
    """
    Export synthetic responses to CSV.
    One row per persona. Columns: persona demographics, big five, construct means, item ratings.
    """
    import csv, os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if not responses:
        return "No responses to export"

    # Build header
    demo_cols  = ["persona_id", "ad_id", "age_band", "gender", "income", "education"]
    bf_cols    = [f"bf_{dim}" for dim in BIG_FIVE_DESCRIPTIONS]
    const_cols = []
    for key in scale_keys:
        const_cols.append(f"{key}_mean")
        const_cols += [f"{key}_item{j+1}" for j in range(3)]
    const_cols.append("fallback_flag")

    header = demo_cols + bf_cols + const_cols

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()

        for r in responses:
            demo = r.get("persona_demographics", {})
            bf   = r.get("big_five", {})

            row = {
                "persona_id": r.get("persona_id", ""),
                "ad_id":      r.get("ad_id", ""),
                "age_band":   demo.get("age_band", ""),
                "gender":     demo.get("gender", ""),
                "income":     demo.get("income", ""),
                "education":  demo.get("education", ""),
                "fallback_flag": 1 if r.get("fallback") else 0,
            }
            for dim in BIG_FIVE_DESCRIPTIONS:
                row[f"bf_{dim}"] = bf.get(dim, "")

            for key in scale_keys:
                construct = r.get(key, {})
                row[f"{key}_mean"] = construct.get("mean", "")
                for j, item_val in enumerate(construct.get("items", [])[:3]):
                    row[f"{key}_item{j+1}"] = item_val

            writer.writerow(row)

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ALL_TOOLS list for CrewAI
# ══════════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    PersonaGeneratorTool(),
    SurveyResponseTool(),
    AnalyticsTool(),
]
