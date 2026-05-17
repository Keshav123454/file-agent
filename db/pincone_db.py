import time
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME_GRPC

# 1. Initialize the global gRPC client instance
pc = PineconeGRPC(api_key=PINECONE_API_KEY)
_pincone_index = None

def initialize_pinecone():
    """
    Handles control plane operations synchronously (Index creation and configuration).
    This runs once at application startup.
    """
    global _pincone_index
    index_name = PINECONE_INDEX_NAME_GRPC
    
    # 2. Correct syntax to check if an index exists
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,   # MiniLM output dimension size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # 3. Wait for a Serverless Index to fully provision before connecting
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    # 4. Target the index using the gRPC interface
    _pincone_index = pc.Index(index_name)
