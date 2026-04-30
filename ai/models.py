from google import genai
from sentence_transformers import SentenceTransformer

_embd_model = None
_gemini_client = None

async def initialize_sentence_splitter_model():
    global _embd_model

    if _embd_model is None:
        print("🔄 Loading embedding model...")
        _embd_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return _embd_model


async def embed_chunks(chunks):
    model = await initialize_sentence_splitter_model() 
    if isinstance(chunks, str):
        chunks = [chunks]

    return model.encode(chunks).tolist() 


async def gemini_embed_model(chunk):
    
    global _gemini_client

    if _gemini_client is None:
        _gemini_client = genai.Client()

    embed_model = _gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    return embed_model.embeddings
