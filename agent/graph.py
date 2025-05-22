import logging
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    process_input_node,
    retrieve_memory_node,
    ethical_check_node,
    generate_response_node,
    should_continue
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_soulstice_graph() -> StateGraph:
    """Creates and compiles the LangGraph for the Soulstice agent."""
    logger.info("Creating Soulstice agent graph...")

    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("process_input", process_input_node)
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("ethical_check", ethical_check_node)
    # Potential future node: workflow.add_node("decide_strategy", decide_response_strategy_node)
    workflow.add_node("generate_response", generate_response_node)

    # Define the edges
    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "ethical_check")

    # Conditional edge after ethical check
    workflow.add_conditional_edges(
        "ethical_check",
        should_continue,
        {
            "generate_response": "generate_response", # Route for both normal and escalation
            "end": END # Route if error occurred
            # Add other paths if needed
        }
    )

    # workflow.add_edge("decide_strategy", "generate_response") # If strategy node is added

    # Final response generation leads to the end of this turn
    workflow.add_edge("generate_response", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("Soulstice agent graph compiled successfully.")
    return app

# --- Get the compiled graph instance ---
soulstice_app = create_soulstice_graph()