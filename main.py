import uuid
import logging
from typing import List, Dict

from agent.graph import soulstice_app
from agent.state import AgentState
from config import OPENAI_API_KEY # To ensure it's loaded
from memory.vector_store import get_chroma_collection # To initialize DB early

# Configure basic logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SoulsticeCLI")

def run_chat():
    """Runs the command-line interface for interacting with Soulstice."""
    session_id = str(uuid.uuid4())
    conversation_history: List[Dict[str, str]] = []
    logger.info(f"Starting new chat session: {session_id}")
    print("\n--- Soulstice POC ---")
    print("Type 'quit' or 'exit' to end the session.")
    print("Welcome! How are you feeling today?")

    # Initialize database connection early
    try:
        get_chroma_collection()
        logger.info("Vector database connection verified.")
    except Exception as e:
        logger.critical(f"Failed to initialize vector database: {e}. Exiting.", exc_info=True)
        return

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Soulstice: Take care. Remember support is available if you need it.")
                break
            if not user_input.strip():
                continue

            # Prepare initial state for the graph invocation
            current_state: AgentState = {
                "session_id": session_id,
                "user_input": user_input,
                "conversation_history": list(conversation_history), # Pass copy
                "input_analysis": None,
                "retrieved_memories": None,
                "response_strategy": None,
                "generated_response": None,
                "needs_escalation": False,
                "error": None
            }

            # Invoke the graph
            logger.info(f"Invoking graph for session {session_id}...")
            final_state = soulstice_app.invoke(current_state)
            logger.info(f"Graph invocation complete for session {session_id}.")

            # Handle potential errors from the graph execution
            if final_state.get('error'):
                logger.error(f"Error encountered during graph execution: {final_state['error']}")
                # Use the potentially generic error response generated, or a default
                ai_response = final_state.get('generated_response', "I'm sorry, something went wrong on my end.")
            else:
                ai_response = final_state.get('generated_response', "I'm not sure how to respond to that right now.") # Fallback

            print(f"Soulstice: {ai_response}")

            # Update conversation history *after* successful turn
            # Only add if no critical error prevented response generation
            if not final_state.get('error') or "Failed to generate response" not in (final_state.get('error') or ""):
                 conversation_history.append({"role": "user", "content": user_input})
                 conversation_history.append({"role": "ai", "content": ai_response})
                 # Note: The interaction is added to the vector store *within* the generate_response_node now

        except KeyboardInterrupt:
            print("\nSoulstice: Session interrupted. Goodbye!")
            break
        except Exception as e:
            logger.critical(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
            print("Soulstice: I've encountered a critical error. Ending session.")
            break

if __name__ == "__main__":
    if not OPENAI_API_KEY:
       print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
    else:
        run_chat()