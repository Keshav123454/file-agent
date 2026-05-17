import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME_GRPC = os.getenv("PINECONE_INDEX_NAME_GRPC")
LANGCACHE_API_KEY = os.getenv("LANGCACHE_API_KEY")
LANGCACHE_ID = os.getenv("LANGCACHE_ID")
LANGCACHE_URL = os.getenv("LANGCACHE_URL")