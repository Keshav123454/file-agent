import uuid

from langsmith import traceable
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BaseChunker:
    async def chunk(self, text: str) -> list:
        raise NotImplementedError

class RecursiveChunker(BaseChunker):
    @traceable(name="recursive_chunking")
    async def chunk(self, text: str) -> list:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        texts = text_splitter.split_text(text)
        return texts


class LongRAGChunker(BaseChunker):
    @traceable(name="longrag_chunking")
    async def chunk(self, text: str) -> list:
        # Step 1: Create large parent chunks
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

        parent_chunks = parent_splitter.split_text(text)

        final_chunks = []

        # Step 2: Split each parent chunk into smaller child chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        for parent in parent_chunks:

            child_chunks = child_splitter.split_text(parent)
    
            final_chunks.append({
                "parent_id": str(uuid.uuid4()),
                "parent_context": parent,
                "children": child_chunks
            })

        return final_chunks