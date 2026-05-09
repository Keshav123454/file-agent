"""
LangGraph agent builder for RAG and chat functionality.
Handles both general chat and RAG-based responses with proper async initialization.
"""

from langsmith import traceable
from langchain.schema import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from typing_extensions import TypedDict, Annotated
import operator
import logging

from constants import MAX_RETRIES
from constants import MAX_RETRIES
from db.mongodb import get_db

logger = logging.getLogger(__name__)


from .models import get_gemini_llm
from .prompt import get_chat_prompt, get_rag_prompt
from .embedding import search_similar


class MessagesState(TypedDict):
    """
    Shared state passed between LangGraph nodes.

    Attributes:
        messages:
            Conversation history including user and AI messages.

        llm_calls:
            Total number of LLM generations performed
            during the current workflow execution.

        file_id:
            Optional uploaded file identifier used
            for RAG-based retrieval.
    """

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    file_id: str | None
    is_valid_response: bool 
    thread_id: str


@traceable(name="handle_general_chat")
async def handle_general_chat(llm, query, file_id=None, llm_calls=0):
    """
    Handle general chat response generation.

    Args:
        llm:
            Initialized language model instance.

        query:
            User's input message to respond to.

    Returns:
        str:
            Generated response content.
    """
    context = ""
    if not file_id:
        chain = get_chat_prompt(query) | llm | StrOutputParser()
    else:
        matches = await search_similar(file_id, query)
        for m in matches:
            context = "\n\n".join([
                f"Chunk {i+1}:\n{m['text']}"
                for i, m in enumerate(matches)
                if m.get("text") and m["text"].strip()
            ])
        
        chain = get_rag_prompt() | llm | StrOutputParser()


    response = await chain.ainvoke({
        "question": query,
        "context": context
    })
    return response

@traceable(name="get_response")
async def get_response(state: MessagesState):
    """
    Generate a response using either general chat
    or Retrieval-Augmented Generation (RAG).

    Workflow:
        1. Extract user query from state
        2. Initialize LLM
        3. Decide between:
            - General chat
            - RAG response
        4. Return updated graph state

    Args:
        state:
            Current LangGraph state.

    Returns:
        dict:
            Updated state containing:
            - AI response message
            - Incremented llm_calls counter

    Raises:
        Exceptions are caught internally and converted
        into safe fallback responses.
    """

    try:
        query = state["messages"][-1].content
        file_id = state.get("file_id")
        llm_calls = state.get("llm_calls", 0)

        llm = await get_gemini_llm()

        if not file_id:
            content = await handle_general_chat(llm, query)

        else:
            content = await handle_general_chat(
                llm=llm,
                query=query,
                file_id=file_id,
                llm_calls=llm_calls
            )
        return {
            "messages": [
                AIMessage(content=content)
            ],
            "llm_calls": state["llm_calls"] + 1
        }
    
    except Exception:
        logger.exception("Error in get_response")

        return {
            "messages": [
                AIMessage(content="Something went wrong.")
            ],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

@traceable(name="verify_response")
async def verify_response(state: MessagesState):
    """
    Validate the generated LLM response quality.

    This node checks whether the generated response:
        - is empty
        - contains low-quality fallback phrases
        - indicates retrieval failure

    Args:
        state:
            Current LangGraph state.

    Returns:
        dict:
            Partial state update containing:
            - is_valid_response (bool)

    Notes:
        This function only validates responses.
        It does NOT decide graph routing.
    """

    try:
        content = state["messages"][-1].content.strip()

        if not content:
            return {
                "is_valid_response": False
            }

        bad_phrases = [
            "not found",
            "i don't know",
            "no relevant information"
        ]

        content_lower = content.lower()

        is_valid = not any(
            phrase in content_lower
            for phrase in bad_phrases
        )
        return {
            "is_valid_response": is_valid
        }

    except Exception:
        logger.exception("Error in verify_response")

        return {
            "is_valid_response": False
        }

@traceable(name="route_response")
def route_response(state: MessagesState):
    """
    Decide the next graph step after response verification.

    Routing Rules:
        - Retry response generation if:
            - response is invalid
            - retry limit not reached

        - End workflow otherwise

    Args:
        state:
            Current LangGraph state.

    Returns:
        str:
            Graph route label:
            - "retry"
            - "end"
    """

    is_valid = state.get(
        "is_valid_response",
        False
    )

    if (
        not is_valid
        and state["llm_calls"] < MAX_RETRIES
    ):
        return "retry"

    return "end"

# ============ AGENT BUILDER ============
@traceable(name="build_agent")
def build_agent():
    """
    Build and compile the LangGraph workflow.

    Graph Flow:
        START
          ↓
        llm_call
          ↓
        verify_response
          ↓
        route_response
          ├── retry → llm_call
          └── end → END

    Returns:
        CompiledStateGraph:
            Compiled LangGraph agent instance.

    Raises:
        Exception:
            Re-raises graph compilation failures.
    """

    try:
        logger.info("Building LangGraph agent...")
        agent_builder = StateGraph(MessagesState)

        # Register nodes
        agent_builder.add_node(
            "llm_call",
            get_response
        )

        agent_builder.add_node(
            "verify_response",
            verify_response
        )

        # Standard graph edges
        agent_builder.add_edge(
            START,
            "llm_call"
        )

        agent_builder.add_edge(
            "llm_call",
            "verify_response"
        )

        # Conditional routing
        agent_builder.add_conditional_edges(
            "verify_response",
            route_response,
            {
                "retry": "llm_call",
                "end": END
            }
        )

        agent = agent_builder.compile()

        logger.info(
            "LangGraph agent built successfully"
        )

        return agent

    except Exception:
        logger.exception("Error building agent")
        raise

# Initialize agent at module load (not during startup)
try:
    agent = build_agent()
    logger.info("✅ Agent initialized")
except Exception as e:
    logger.error(f"Failed to initialize agent: {e}")
    agent = None
