from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_model = None

def get_embedding_model():
    """Initializes and returns the Sentence Transformer model."""
    global _model
    if _model is None:
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise
    return _model

def get_embedding(text: str):
    """Generates embedding for the given text."""
    model = get_embedding_model()
    if not isinstance(text, str):
        logger.warning(f"Input text is not a string: {type(text)}. Attempting conversion.")
        text = str(text)
    if not text.strip():
        logger.warning("Input text is empty or whitespace. Returning None.")
        return None # Or return a zero vector of correct dimensionality if required downstream

    try:
        # Normalize to avoid issues with certain characters/formats if necessary
        # text = text.replace("\n", " ") # Example normalization
        embedding = model.encode(text, convert_to_tensor=False) # Get numpy array
        # Ensure embedding is a list of floats for ChromaDB compatibility if needed
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding for text: '{text[:100]}...': {e}", exc_info=True)
        return None # Or handle appropriately