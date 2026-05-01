from .models import gemini_embed_model, embed_chunks
from db.utils import get_file_by_id
import numpy as np


from .models import embed_chunks
from db.utils import get_file_by_id


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


async def generate_hybrid_embeddings(file_id=None):
    if file_id:
        chunks = await get_chunks(file_id)

    chunk_embeddings = await embed_chunks(chunks)

    doc_embedding = np.mean(chunk_embeddings, axis=0).tolist()

    return {
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "doc_embedding": doc_embedding,
        "doc_id": file_id
    }

def delete_vec(file_id):
    from db.pincone_db import _pincone_index

    _pincone_index.delete(
        namespace=file_id,
        delete_all=True
    )

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


async def search_similar(file_id: str, query: str, top_k: int = 2):
    from db.pincone_db import _pincone_index
    stats = _pincone_index.describe_index_stats()
    print("1"*50)
    print(stats)
    print("2"*50)

    query_embedding = (await embed_chunks([query]))[0]

    # convert to list (important)
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    # Step 2: search in Pinecone
    results = _pincone_index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=file_id,   # 🔥 VERY IMPORTANT
        include_metadata=True
    )

    # Step 3: extract results
    matches = []
    for match in results.get("matches", []):
        matches.append({
            "score": match["score"],
            "text": match["metadata"].get("text"),
            "file_id": match["metadata"].get("file_id")
        })

    return matches

