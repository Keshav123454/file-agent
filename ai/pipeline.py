from .models import get_gemini_llm
from .prompt import get_rag_prompt
from .embedding import search_similar

# global (good practice)
llm = get_gemini_llm()
prompt = get_rag_prompt()
chain = prompt | llm


async def get_response(query, file_id):
    # ✅ Step 1: vector search
    matches = await search_similar(file_id, query)
    print(matches, file_id, query)
    # ✅ Step 2: build context
    context = "\n\n".join([
        f"Chunk {i+1}:\n{m['text']}"
        for i, m in enumerate(matches)
        if m["text"].strip()
    ])
    print("@"*50)
    print(context)
    print("#"*50)
    # ✅ Step 3: LLM call
    if not context:
        return "Not found in context"
    response = chain.invoke({
        "context": context,
        "question": query
    })

    return response.content