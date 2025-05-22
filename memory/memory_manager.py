import datetime
import logging
from typing import List, Dict, Optional

from .embedding import get_embedding
from .vector_store import add_memory, query_memories
from config import EMBEDDING_MODEL_NAME # If needed for metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryManager:
    """Handles adding and retrieving memories from the vector store."""

    def add_interaction(self, user_input: str, ai_response: str, session_id: str):
        """Adds both user input and AI response as separate episodic memories."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Add User Input
        user_doc = f"User said: {user_input}"
        user_embedding = get_embedding(user_doc)
        user_metadata = {
            "type": "episodic",
            "role": "user",
            "timestamp": timestamp,
            "session_id": session_id,
            "embedding_model": EMBEDDING_MODEL_NAME
        }
        if user_embedding:
            add_memory(user_embedding, user_doc, user_metadata)
            logger.info(f"Added user interaction memory for session {session_id}")
        else:
             logger.warning(f"Could not generate embedding for user input: {user_input[:100]}...")

        # Add AI Response
        ai_doc = f"AI responded: {ai_response}"
        ai_embedding = get_embedding(ai_doc)
        ai_metadata = {
            "type": "episodic",
            "role": "ai",
            "timestamp": timestamp, # Same timestamp links the turn
            "session_id": session_id,
            "embedding_model": EMBEDDING_MODEL_NAME
        }
        if ai_embedding:
            add_memory(ai_embedding, ai_doc, ai_metadata)
            logger.info(f"Added AI interaction memory for session {session_id}")
        else:
            logger.warning(f"Could not generate embedding for AI response: {ai_response[:100]}...")


    def retrieve_relevant_memories(self, query_text: str, session_id: str, n_results: int = 5):
        """Retrieves memories relevant to the query text, potentially filtered by session."""
        query_embedding = get_embedding(query_text)
        if query_embedding is None:
            logger.warning(f"Cannot retrieve memories, failed to embed query: {query_text[:100]}...")
            return [] # Return empty list

        # Example filter: retrieve only from the current session
        # where_filter = {"session_id": session_id}
        # Or retrieve from all sessions (no filter)
        where_filter = None # Adjust as needed

        try:
            results = query_memories(query_embedding, n_results=n_results, where_filter=where_filter)

            # Process results (combine docs, metadata, distance)
            retrieved = []
            docs = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

            for doc, meta, dist in zip(docs, metadatas, distances):
                retrieved.append({
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                })

            # Sort by distance (ascending) - Chroma usually returns sorted, but good practice
            retrieved.sort(key=lambda x: x['distance'])

            logger.info(f"Retrieved {len(retrieved)} memories for query: '{query_text[:50]}...'")
            return retrieved

        except Exception as e:
            logger.error(f"Error during memory retrieval: {e}", exc_info=True)
            return []

    # TODO: Add methods for adding/querying 'semantic' memories if needed separately
    # def add_semantic_knowledge(self, concept: str, description: str): ...