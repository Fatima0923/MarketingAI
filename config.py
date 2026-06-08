# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# ── Provider registry ─────────────────────────────────────────────────────────
# All supported LLM providers. UI dropdown reads from this dict.
# api_url     : endpoint for direct requests (tools/tools.py call_llm)
# base_url    : OpenAI-compatible base URL for LangChain ChatOpenAI
# default_model: pre-filled model name in UI
# env_key     : environment variable name for the API key
# note        : shown in UI as a helper hint

PROVIDERS = {
    "DeepSeek": {
        "api_url":       "https://api.deepseek.com/v1/chat/completions",
        "base_url":      "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key":       "DEEPSEEK_API_KEY",
        "note":          "Models: deepseek-chat, deepseek-reasoner",
    },
    "OpenAI": {
        "api_url":       "https://api.openai.com/v1/chat/completions",
        "base_url":      "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_key":       "OPENAI_API_KEY",
        "note":          "Models: gpt-4o, gpt-4o-mini, gpt-4-turbo",
    },
    "Google Gemini": {
        "api_url":       "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "base_url":      "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "env_key":       "GEMINI_API_KEY",
        "note":          "Models: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash",
    },
}

# Pre-load any keys from .env so UI fields are pre-filled
PROVIDER_KEYS = {
    "DeepSeek":      os.getenv("DEEPSEEK_API_KEY", ""),
    "OpenAI":        os.getenv("OPENAI_API_KEY",   ""),
    "Google Gemini": os.getenv("GEMINI_API_KEY",   ""),
}

# ── Langfuse ──────────────────────────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
UPLOADS_DIR = "data/uploads"
OUTPUTS_DIR = "outputs"

# ── Default validated scales ──────────────────────────────────────────────────
DEFAULT_SCALES = {
    "brand_attitude": {
        "label":      "Brand Attitude",
        "source":     "MacKenzie & Lutz (1989)",
        "anchor_lo":  "Very unfavourable",
        "anchor_hi":  "Very favourable",
        "items": [
            "My overall attitude toward this brand is positive.",
            "I think this brand is good.",
            "I have favourable feelings toward this brand.",
        ],
    },
    "purchase_intention": {
        "label":      "Purchase Intention",
        "source":     "Dodds et al. (1991)",
        "anchor_lo":  "Very unlikely",
        "anchor_hi":  "Very likely",
        "items": [
            "I would consider buying this product.",
            "The probability that I would purchase this product is high.",
            "I am likely to purchase this product.",
        ],
    },
    "brand_fit": {
        "label":      "Brand Fit",
        "source":     "Becker-Olsen et al. (2006)",
        "anchor_lo":  "Strongly disagree",
        "anchor_hi":  "Strongly agree",
        "items": [
            "This advertisement is consistent with the brand's image.",
            "The message in this ad fits what I know about the brand.",
            "The ad and the brand are a natural match.",
        ],
    },
    "ad_credibility": {
        "label":      "Ad Credibility",
        "source":     "Newell & Goldsmith (2001)",
        "anchor_lo":  "Not at all credible",
        "anchor_hi":  "Extremely credible",
        "items": [
            "This advertisement is truthful.",
            "I believe the claims made in this advertisement.",
            "This advertisement is credible.",
        ],
    },
}

# ── Outlier threshold ─────────────────────────────────────────────────────────
OUTLIER_SD_THRESHOLD = 2.0
