from typer import prompt

from .models import get_gemini_llm
from .prompt import get_chat_prompt, get_rag_prompt
from .embedding import search_similar

# global (good practice)
llm = get_gemini_llm()



async def get_response(query, file_id=None):
    if not file_id:
        prompt = get_chat_prompt(query)
        chain = prompt | llm
        response = chain.invoke({
            "question": query
        })
    
    else:
    # ✅ Step 1: vector search
        prompt = get_rag_prompt()
        chain = prompt | llm
        matches = await search_similar(file_id, query)
        print(matches, file_id, query)
        # ✅ Step 2: build context
        context = "\n\n".join([
            f"Chunk {i+1}:\n{m['text']}"
            for i, m in enumerate(matches)
            if m["text"].strip()
        ])
        # ✅ Step 3: LLM call
        if not context:
            return "Not found in context"
        response = chain.invoke({
            "context": context,
            "question": query
        })

    return response.content