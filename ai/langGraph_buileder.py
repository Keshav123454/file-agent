"""
LangGraph agent builder for RAG and chat functionality.
Handles both general chat and RAG-based responses with proper async initialization.
"""

from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage, AIMessage
from typing_extensions import TypedDict, Annotated
import operator
import logging

logger = logging.getLogger(__name__)


class MessagesState(TypedDict):
    """State for the agent graph."""
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    file_id: str | None


from .models import get_gemini_llm
from .prompt import get_chat_prompt, get_rag_prompt
from .embedding import search_similar


# ============ AGENT NODES ============

async def get_response(state: MessagesState):
    """
    Process user message and generate LLM response.
    
    Supports two modes:
    1. General Chat: LLM responds with general knowledge
    2. RAG: LLM responds using retrieved context from file
    
    Args:
        state: Current agent state with messages and context
        
    Returns:
        dict: Updated state with LLM response and metadata
    """
    try:
        # Get latest user message
        last_message = state["messages"][-1]
        query = last_message.content
        file_id = state.get("file_id")
        
        # ✅ FIXED: Now properly awaits get_gemini_llm()
        llm = await get_gemini_llm()
        logger.info(f"Processing query: {query[:50]}...")
        
        # Case 1: Normal chat (no file context)
        if not file_id:
            logger.info("Mode: General chat")
            response = await (get_chat_prompt(query) | llm).ainvoke({
                "question": query
            })
            content = response.content
        
        # Case 2: RAG (with file context)
        else:
            logger.info(f"Mode: RAG with file {file_id}")
            matches = await search_similar(file_id, query)
            
            # Build context from retrieved chunks
            context = "\n\n".join([
                f"Chunk {i+1}:\n{m['text']}"
                for i, m in enumerate(matches)
                if m.get("text") and m["text"].strip()
            ])
            
            if not context:
                content = "❌ No relevant information found in the document."
                logger.warning(f"No context found for query: {query}")
            else:
                logger.info(f"Retrieved {len(matches)} relevant chunks")
                response = await (get_rag_prompt() | llm).ainvoke({
                    "context": context,
                    "question": query
                })
                content = response.content
        
        logger.info("✅ Response generated successfully")
        
        # Return updated state
        return {
            "messages": [AIMessage(content=content)],
            "llm_calls": state.get("llm_calls", 0) + 1
        }
        
    except Exception as e:
        logger.error(f"Error in get_response: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content=f"❌ Error: {str(e)}")],
            "llm_calls": state.get("llm_calls", 0) + 1
        }


# ============ AGENT BUILDER ============

def build_agent():
    """
    Build and compile the LangGraph agent.
    
    Graph structure:
    START → llm_call → END
    
    Returns:
        CompiledGraph: Compiled LangGraph agent
    """
    try:
        logger.info("Building LangGraph agent...")
        
        # Create state graph
        agent_builder = StateGraph(MessagesState)
        
        # Add node for LLM call
        agent_builder.add_node("llm_call", get_response)
        
        # Connect nodes
        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_edge("llm_call", END)
        
        # Compile graph
        agent = agent_builder.compile()
        
        logger.info("✅ LangGraph agent built successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Error building LangGraph agent: {e}", exc_info=True)
        raise


# Initialize agent at module load (not during startup)
try:
    agent = build_agent()
    logger.info("✅ Agent initialized")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    agent = None
