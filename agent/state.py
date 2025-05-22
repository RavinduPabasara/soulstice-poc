from typing import TypedDict, List, Dict, Optional, Any
import datetime

class AgentState(TypedDict):
    """Defines the state structure for the Soulstice agent graph."""
    session_id: str
    user_input: str
    input_analysis: Optional[Dict[str, Any]] # Parsed JSON from analysis node
    retrieved_memories: Optional[List[Dict[str, Any]]] # List of dicts {document, metadata, distance}
    conversation_history: List[Dict[str, str]] # List of {"role": "user/ai", "content": "message"}
    response_strategy: Optional[str] # Guidance for the generation node
    generated_response: Optional[str] # The final AI response
    needs_escalation: bool # Flag set by ethical check
    error: Optional[str] # To capture errors during execution