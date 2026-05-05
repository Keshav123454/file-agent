# Import the Pinecone library
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
_pincone_index = None
async def initialize_pinecone():

    index_name = "file-chunks-index"
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,   # 🔥 MiniLM output size
            metric="cosine",
            spec = ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)
    global _pincone_index
    _pincone_index = index