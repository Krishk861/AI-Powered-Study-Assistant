"""
Configuration settings for AI Study Assistant
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================
# API KEYS
# ============================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API key
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# ============================================
# MODEL SETTINGS
# ============================================
LLM_MODEL = "gemini-2.0-flash-exp"  # Fast and free
EMBEDDING_MODEL = "models/embedding-001"  # For embeddings
TEMPERATURE = 0.2  # Lower = more focused, Higher = more creative

# ============================================
# DOCUMENT PROCESSING
# ============================================
CHUNK_SIZE = 1000  # Size of text chunks
CHUNK_OVERLAP = 200  # Overlap between chunks
MAX_UPLOAD_SIZE_MB = 50  # Maximum file size

# ============================================
# VECTOR STORE SETTINGS
# ============================================
CHROMA_PERSIST_DIR = "./data/chroma_db"  # Where to save vector database
RETRIEVAL_K = 4  # Number of chunks to retrieve

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