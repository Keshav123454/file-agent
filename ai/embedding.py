from .models import gemini_embed_model, embed_chunks
from db.utils import get_file_by_id
import numpy as np


async def get_chunks(file_id):
    file = await get_file_by_id(file_id)
    if not file:
        raise ValueError("File not found")
    chunks = file['extracted_text']
    return chunks

async def embed_chunks_gemini(chunks):
    embeddings = []

    for chunk in chunks:
        result = await gemini_embed_model(chunk)
        embeddings.append(result[0].values)

    return embeddings


async def generate_hybrid_embeddings(file_id):
    
    chunks = await get_chunks(file_id)

    chunk_embeddings = await embed_chunks(chunks)

    doc_embedding = np.mean(chunk_embeddings, axis=0).tolist()

    return {
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "doc_embedding": doc_embedding,
        "doc_id": file_id
    }

def store_vec(vectors, file_id):
    from db.pincone_db import _pincone_index
    _pincone_index.upsert(
        namespace=file_id,
        vectors=vectors   
    )

async def upsert_document(file_id):

    chunks = await get_chunks(file_id)

    chunk_embeddings = await embed_chunks(chunks)


    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, chunk_embeddings)):

        if not isinstance(emb, (list, tuple)):
            raise ValueError(f"Embedding is not a vector: {emb}")

        vectors.append({
            "id": f"{file_id}_{i}",
            "values": emb,
            "metadata": {
                "text": chunk,
                "file_id": file_id
            }
        })

    store_vec(vectors, file_id)

    return {
        "file_id": file_id,
        "num_chunks": len(chunks)
    }

