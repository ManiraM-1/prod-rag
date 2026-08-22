import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    QDRANT_URL: str = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY: str = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    GROQ_SLUG =  "prod-rag"     # primary: @prod-rag/openai/gpt-oss-120b
    GROQ_SLUG_2 = "prod-rag1"  # fallback: @prod-rag1/openai/gpt-oss-20b

settings = Settings()