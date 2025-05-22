import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import LLM_MODEL_NAME, POTENTIAL_RISK_KEYWORDS, ESCALATION_MESSAGE, MAX_CONVERSATION_HISTORY_TOKENS
from prompts.system_prompts import (
    INPUT_ANALYSIS_PROMPT,
    RESPONSE_GENERATION_PROMPT,
    ETHICAL_RISK_ASSESSMENT_PROMPT,
    MEMORY_RETRIEVAL_QUERY_PROMPT,
    simple_keyword_check
)
from memory.memory_manager import MemoryManager
from agent.state import AgentState
import tiktoken # For token counting
from typing import List,Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Initialize LLM ---
# Consider adding temperature, max_tokens etc.
llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.7)

# --- Initialize Memory Manager ---
# Note: This assumes a single MemoryManager instance is shared or accessible
# In a real app, dependency injection or context passing might be better
memory_manager = MemoryManager()

# --- Helper for Token Counting & History Truncation ---
try:
    tokenizer = tiktoken.encoding_for_model(LLM_MODEL_NAME)
except KeyError:
    logger.warning(f"Tokenizer for {LLM_MODEL_NAME} not found, using default cl100k_base.")
    tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def format_conversation_history(history: List[Dict[str, str]], max_tokens: int) -> str:
    formatted_history = []
    current_tokens = 0
    # Iterate from most recent to oldest
    for turn in reversed(history):
        turn_text = f"{turn['role'].capitalize()}: {turn['content']}"
        turn_tokens = count_tokens(turn_text)
        if current_tokens + turn_tokens <= max_tokens:
            formatted_history.append(turn_text)
            current_tokens += turn_tokens
        else:
            break # Stop adding older turns if limit exceeded
    # Reverse back to chronological order for the prompt
    return "\n".join(reversed(formatted_history))

# --- Graph Nodes ---

def process_input_node(state: AgentState) -> AgentState:
    """Analyzes the user input using an LLM."""
    logger.info("--- Node: process_input ---")
    user_input = state['user_input']
    history_str = format_conversation_history(state['conversation_history'], MAX_CONVERSATION_HISTORY_TOKENS // 2) # Allocate half for history

    prompt = ChatPromptTemplate.from_template(INPUT_ANALYSIS_PROMPT)
    chain = prompt | llm | StrOutputParser()

    logger.info("Calling LLM for input analysis...")
    try:
        analysis_str = chain.invoke({
            "user_input": user_input,
            "conversation_history": history_str
        })
        logger.debug(f"Raw analysis string: {analysis_str}")
        # Basic cleaning if LLM includes markdown etc.
        analysis_str = analysis_str.strip().removeprefix("```json").removesuffix("```").strip()
        analysis_json = json.loads(analysis_str)
        logger.info(f"Input analysis successful: {analysis_json}")
        state['input_analysis'] = analysis_json
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM analysis output as JSON: {e}\nOutput: {analysis_str}", exc_info=True)
        state['error'] = "Failed to analyze input."
        state['input_analysis'] = {"error": "JSON parsing failed"} # Provide some default error structure
    except Exception as e:
        logger.error(f"Error during input analysis LLM call: {e}", exc_info=True)
        state['error'] = "An error occurred during input analysis."
        state['input_analysis'] = {"error": "LLM call failed"}

    return state

def retrieve_memory_node(state: AgentState) -> AgentState:
    """Retrieves relevant memories based on the input analysis."""
    logger.info("--- Node: retrieve_memory ---")
    if state.get('error') or not state.get('input_analysis') or state['input_analysis'].get('error'):
        logger.warning("Skipping memory retrieval due to previous error or missing analysis.")
        state['retrieved_memories'] = []
        return state

    analysis = state['input_analysis']
    user_input = state['user_input'] # Use raw input too?
    session_id = state['session_id']
    history_str = format_conversation_history(state['conversation_history'], MAX_CONVERSATION_HISTORY_TOKENS // 4) # Smaller history context for query gen

    # Option 1: Use analysis components directly for query
    # query_text = f"Topics: {analysis.get('key_topics', [])}, Emotion: {analysis.get('dominant_emotion', 'neutral')}. User said: {user_input}"

    # Option 2: Generate a specific query using LLM
    query_gen_prompt = ChatPromptTemplate.from_template(MEMORY_RETRIEVAL_QUERY_PROMPT)
    query_gen_chain = query_gen_prompt | llm | StrOutputParser()
    try:
        query_text = query_gen_chain.invoke({
            "user_input": user_input,
            "key_topics": analysis.get('key_topics', 'None'),
            "dominant_emotion": analysis.get('dominant_emotion', 'neutral'),
            "conversation_history": history_str
        })
        logger.info(f"Generated memory retrieval query: {query_text}")
    except Exception as e:
        logger.error(f"Failed to generate memory query: {e}. Falling back to basic query.", exc_info=True)
        query_text = f"Topics: {analysis.get('key_topics', [])}, Emotion: {analysis.get('dominant_emotion', 'neutral')}. User input: {user_input}"

    # Perform retrieval
    try:
        retrieved = memory_manager.retrieve_relevant_memories(query_text, session_id, n_results=5)
        state['retrieved_memories'] = retrieved
        logger.info(f"Retrieved {len(retrieved)} memories.")
    except Exception as e:
        logger.error(f"Error during memory retrieval: {e}", exc_info=True)
        state['error'] = "Failed to retrieve memories."
        state['retrieved_memories'] = []

    return state

def ethical_check_node(state: AgentState) -> AgentState:
    """Performs safety and escalation checks."""
    logger.info("--- Node: ethical_check ---")
    if state.get('error'):
        logger.warning("Skipping ethical check due to previous error.")
        state['needs_escalation'] = False
        return state

    user_input = state['user_input']
    history_str = format_conversation_history(state['conversation_history'], MAX_CONVERSATION_HISTORY_TOKENS // 4)
    needs_escalation = False

    # Method 1: Simple Keyword Check (Fast)
    if simple_keyword_check(user_input, POTENTIAL_RISK_KEYWORDS):
        logger.warning(f"Potential risk keyword detected in input: '{user_input[:100]}...'")
        needs_escalation = True # Default to escalate if keyword found, LLM can refine

    # Method 2: LLM-based Assessment (More Nuance, Slower, Costlier)
    # Can run this *if* keyword check is false, or always run it for confirmation
    # if not needs_escalation: # Only run if keywords didn't trigger
    try:
        prompt = ChatPromptTemplate.from_template(ETHICAL_RISK_ASSESSMENT_PROMPT)
        chain = prompt | llm | StrOutputParser()
        logger.info("Calling LLM for ethical risk assessment...")
        assessment = chain.invoke({
            "user_input": user_input,
            "conversation_history": history_str
        })
        assessment = assessment.strip().upper()
        logger.info(f"LLM Risk Assessment Result: {assessment}")
        if assessment == "YES":
            needs_escalation = True
        # If LLM says NO but keywords said YES, maybe keep needs_escalation=True as safer default? Or trust LLM? Policy decision.
        # Current logic: Either keyword or LLM saying YES triggers escalation.
    except Exception as e:
        logger.error(f"Error during ethical risk assessment LLM call: {e}", exc_info=True)
        # Safety default: Escalate if assessment fails? Or assume safe? Assume safe for now, log error.
        # needs_escalation = True # Safer default?
        state['error'] = (state.get('error') or "") + " Ethical check LLM failed."


    state['needs_escalation'] = needs_escalation
    logger.info(f"Ethical Check Result: Needs Escalation = {needs_escalation}")
    return state

def generate_response_node(state: AgentState) -> AgentState:
    """Generates the final AI response based on the state."""
    logger.info("--- Node: generate_response ---")
    if state.get('error'):
        logger.warning("Skipping response generation due to previous error.")
        state['generated_response'] = "I'm sorry, I encountered an error. Could you please rephrase?"
        return state

    if state.get('needs_escalation', False):
        logger.warning("Generating escalation response.")
        state['generated_response'] = ESCALATION_MESSAGE
        # Optionally, add this escalation turn to memory?
        # memory_manager.add_interaction(state['user_input'], ESCALATION_MESSAGE, state['session_id'])
        return state

    # Prepare context for the LLM
    user_input = state['user_input']
    analysis = state.get('input_analysis', {})
    memories = state.get('retrieved_memories', [])
    history_str = format_conversation_history(state['conversation_history'], MAX_CONVERSATION_HISTORY_TOKENS) # Use larger history for generation

    # Format memories for the prompt
    formatted_memories = "\n".join([
        f"- {m['document']} (Relevance: {1 - m['distance']:.2f})"
        for m in memories[:3] # Limit number of memories shown?
    ]) if memories else "No relevant memories found."

    prompt = ChatPromptTemplate.from_template(RESPONSE_GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    logger.info("Calling LLM for final response generation...")
    try:
        response = chain.invoke({
            "user_input": user_input,
            "input_analysis": json.dumps(analysis, indent=2),
            "retrieved_memories": formatted_memories,
            "conversation_history": history_str
        })
        state['generated_response'] = response
        logger.info(f"Generated response: {response[:150]}...")

        # Add the successful interaction to memory *after* generation
        try:
             memory_manager.add_interaction(user_input, response, state['session_id'])
             logger.info("Interaction added to memory.")
        except Exception as mem_e:
             logger.error(f"Failed to add interaction to memory: {mem_e}", exc_info=True)
             # Don't let memory failure stop the response flow, but log it.
             state['error'] = (state.get('error') or "") + " Failed to save interaction to memory."

    except Exception as e:
        logger.error(f"Error during response generation LLM call: {e}", exc_info=True)
        state['error'] = (state.get('error') or "") + " Failed to generate response."
        state['generated_response'] = "I'm having trouble formulating a response right now. Could you try saying that differently?"

    return state

# --- Conditional Edge Logic ---

def should_continue(state: AgentState) -> str:
    """Determines the next step after ethical check."""
    logger.info("--- Edge: should_continue ---")
    if state.get('error'):
        logger.warning(f"Routing to end due to error: {state['error']}")
        return "end" # Or an error handling node
    if state.get('needs_escalation', False):
        logger.info("Routing to generate_response (for escalation message).")
        return "generate_response" # Go to generate to output the standard escalation message
    else:
        logger.info("Routing to generate_response (normal path).")
        # TODO: Add a 'decide_response_strategy' node here if needed
        return "generate_response" # Normal path