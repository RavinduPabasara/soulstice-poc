import os
from dotenv import load_dotenv

load_dotenv()

# --- Model Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Using GPT-4o as it's generally stronger, but GPT-3.5-turbo is also an option
LLM_MODEL_NAME = "gpt-4o-mini"
# As specified in the paper
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Memory Configuration ---
VECTORSTORE_PATH = "./chroma_db" # Directory to persist ChromaDB data
COLLECTION_NAME = "soulstice_memory"

# --- Agent Configuration ---
MAX_CONVERSATION_HISTORY_TOKENS = 3000 # Approximate token limit for history in prompts

# --- Ethical Configuration ---
# Keywords that might trigger an escalation check (simple example)
POTENTIAL_RISK_KEYWORDS = ["suicide", "kill myself", "hopeless", "can't go on", "self-harm", "abuse"]
ESCALATION_MESSAGE = """
I understand you're going through immense pain right now. As an AI, I'm not equipped to handle crises like this, but you don't have to face this alone. Please reach out to a crisis hotline or mental health professional immediately. Here are some resources: [Include relevant local/international hotline numbers/links]. They can offer the support you need.
"""

# LangSmith Tracing (Optional but Recommended)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "Soulstice POC")

if LANGCHAIN_TRACING_V2 and not LANGCHAIN_API_KEY:
    print("Warning: LangSmith tracing is enabled, but LANGCHAIN_API_KEY is not set.")

# Basic validation
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")