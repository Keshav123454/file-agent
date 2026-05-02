from langgraph.graph import StateGraph, START, END
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator



class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    file_id: str | None


from .models import get_gemini_llm
from .prompt import get_chat_prompt, get_rag_prompt
from .embedding import search_similar

from langchain.messages import AIMessage
from .models import get_gemini_llm

llm = get_gemini_llm()


async def get_response(state: MessagesState):
    # 👉 Get latest user message
    last_message = state["messages"][-1]
    query = last_message.content
    file_id = state.get("file_id")

    # 👉 Case 1: Normal chat
    if not file_id:
        response = await (get_chat_prompt(query) | llm).ainvoke({
            "question": query
        })
        content = response.content

    # 👉 Case 2: RAG
    else:
        matches = await search_similar(file_id, query)

        context = "\n\n".join([
            f"Chunk {i+1}:\n{m['text']}"
            for i, m in enumerate(matches)
            if m["text"].strip()
        ])

        if not context:
            content = "Not found in context"
        else:
            response = await (get_rag_prompt() | llm).ainvoke({
                "context": context,
                "question": query
            })
            content = response.content

    # 👉 Final return (common for both cases)
    return {
        "messages": [AIMessage(content=content)],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    
agent_builder = StateGraph(MessagesState)

# Add node
agent_builder.add_node("llm_call", get_response)

# Connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("llm_call", END)

# Compile
agent = agent_builder.compile()