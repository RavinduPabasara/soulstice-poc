import chromadb
from chromadb.config import Settings
from config import VECTORSTORE_PATH, COLLECTION_NAME
import logging
import uuid
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_client = None
_collection = None

def get_chroma_collection():
    """Initializes and returns the ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        try:
            logger.info(f"Initializing ChromaDB client with persistence path: {VECTORSTORE_PATH}")
            _client = chromadb.PersistentClient(path=VECTORSTORE_PATH, settings=Settings(anonymized_telemetry=False)) # Disable telemetry

            logger.info(f"Getting or creating ChromaDB collection: {COLLECTION_NAME}")
            _collection = _client.get_or_create_collection(
                name=COLLECTION_NAME,
                # Optional: Specify embedding function if not providing embeddings directly
                # embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
                # Metadata validation can be added here if needed
                # metadata={"hnsw:space": "cosine"} # Example: specifying distance metric
            )
            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise
    return _collection

def add_memory(embedding: List[float], document: str, metadata: Dict, memory_id: Optional[str] = None):
    """Adds a memory item (document and metadata) with its embedding to the collection."""
    collection = get_chroma_collection()
    if embedding is None:
        logger.warning(f"Skipping add_memory due to None embedding for document: {document[:100]}...")
        return

    if memory_id is None:
        memory_id = str(uuid.uuid4())

    try:
        collection.add(
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
            ids=[memory_id]
        )
        logger.debug(f"Added memory with ID: {memory_id}")
    except Exception as e:
        logger.error(f"Failed to add memory (ID: {memory_id}): {e}", exc_info=True)
        # Consider retry logic or raising the exception

def query_memories(query_embedding: List[float], n_results: int = 5, where_filter: Optional[Dict] = None):
    """Queries the collection for similar memories based on the query embedding."""
    collection = get_chroma_collection()
    if query_embedding is None:
        logger.warning("Skipping query_memories due to None query embedding.")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]} # Match Chroma return structure for empty

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter, # Optional filter based on metadata
            include=['documents', 'metadatas', 'distances'] # Include distances for relevance scoring
        )
        logger.debug(f"Query returned {len(results.get('documents', [[]])[0])} results.")
        return results
    except Exception as e:
        logger.error(f"Failed to query memories: {e}", exc_info=True)
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]} # Return empty structure on error