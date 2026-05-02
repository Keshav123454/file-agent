from langchain_core.prompts import PromptTemplate

def get_rag_prompt():
    """
    Returns a LangChain PromptTemplate for RAG (context-based QA)
    """

    template = """
You are a precise AI assistant.

Answer the question using ONLY the context.

Rules:
- Do not use external knowledge
- If answer is missing, say: "Not found in context"
- Include the exact supporting sentence from the context

Context:
---------------------
{context}
---------------------

Question:
{question}

Answer:
Supporting Text:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template.strip()
    )

    return prompt

