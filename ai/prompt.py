from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

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


def get_chat_prompt(question):
    """
    Returns a LangChain PromptTemplate for general chat
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("human", "{question}")
    ])

    return prompt