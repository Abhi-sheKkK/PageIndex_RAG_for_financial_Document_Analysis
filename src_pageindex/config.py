"""
Configuration module for the PageIndex pipeline — paths and API key loading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PDF_DIR = BASE_DIR / "pdfs"
PAGEINDEX_CACHE_DIR = BASE_DIR / "pageindex_cache"
GROUND_TRUTH_PATH = BASE_DIR / "ground_truth_table.xlsx"

# Ensure output directories exist
PAGEINDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Model Configuration ────────────────────────────────────────────────────
OPENAI_MODEL_NAME = "gpt-4o-mini"

# ── API Keys ────────────────────────────────────────────────────────────────
# Standard OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Unified PageIndex API Key for all documents
PAGEINDEX_API_KEY = os.environ.get("PAGEINDEX_API_KEY")

if not PAGEINDEX_API_KEY:
    print("WARNING: PAGEINDEX_API_KEY is not set in the environment.")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY is not set in the environment.")
