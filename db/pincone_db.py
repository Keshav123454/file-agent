# Import the Pinecone library
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY

pc = Pinecone(api_key=PINECONE_API_KEY)
_pincone_index = None
async def initialize_pinecone():

    index_name = "file-chunks-index"

    # Create index (only once)
    if not pc.has_index(index_name):
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