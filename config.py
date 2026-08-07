"""
Configuration settings for AI Study Assistant
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# API KEYS
# ============================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env file!")

# ============================================
# MODEL SETTINGS
# ============================================

LLM_MODEL = "meta/llama-3.1-8b-instruct"
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"

TEMPERATURE = 0.2

# ============================================
# DOCUMENT PROCESSING
# ============================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_UPLOAD_SIZE_MB = 50

# ============================================
# VECTOR STORE SETTINGS
# ============================================

CHROMA_PERSIST_DIR = "./data/chroma_db"
RETRIEVAL_K = 4

# ============================================
# PATHS
# ============================================

UPLOAD_DIR = "./data/uploads"
DATA_DIR = "./data"

# ============================================
# UI SETTINGS
# ============================================

PAGE_TITLE = "AI Study Assistant"
PAGE_ICON = "📚"