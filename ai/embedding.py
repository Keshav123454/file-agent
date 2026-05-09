from pinecone.core.openapi.shared.exceptions import PineconeApiException
from .models import gemini_embed_model, embed_chunks
from db.utils import get_file_by_id
import numpy as np


from .models import embed_chunks
from db.utils import get_file_by_id

async def get_chunks(file_id):
    file = await get_file_by_id(file_id)
    if not file:
        raise ValueError("File not found")

    chunks = file["extracted_text"]
    # LongRAG structure
    if isinstance(chunks, list) and "children" in chunks[0]:

        all_child_chunks = []
        parent_ids = []

        for item in chunks:
            parent_id = item.get("parent_id")

            for child in item["children"]:
                all_child_chunks.append(child)
                parent_ids.append(parent_id)

        return all_child_chunks, parent_ids

    return chunks, None


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
    try:
        _pincone_index.delete(
            namespace=file_id,
            delete_all=True
        )
    except PineconeApiException as e:

        return {
            "success": False,
            "message": str(e)
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }


def store_vec(vectors, file_id):
    from db.pincone_db import _pincone_index
    batch_size = 100
    try:
        for i in range(0, len(vectors), batch_size):

            batch = vectors[i:i + batch_size]

            _pincone_index.upsert(
                vectors=batch,
                namespace=file_id,
            )

        return {
            "success": True,
            "message": "Vectors stored successfully"
        }

    except PineconeApiException as e:

        return {
            "success": False,
            "message": str(e)
        }

    except Exception as e:

        return {
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }


async def upsert_document(file_id):

    chunks, parent_ids = await get_chunks(file_id)
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
                "parent_id": parent_ids[i] if parent_ids else "",
                "file_id": file_id
            }
        })
    pinecone_res = store_vec(vectors, file_id)

    return {
        "file_id": file_id,
        "num_chunks": len(chunks),
        "response": pinecone_res   
    }


async def search_similar(file_id: str, query: str, top_k: int = 3):
    from db.pincone_db import _pincone_index

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
            "file_id": match["metadata"].get("file_id"),
            "parent_id": match["metadata"].get("parent_id")
        })

    return matches

