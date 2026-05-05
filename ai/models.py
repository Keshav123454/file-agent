"""
AI Models Management - Handles initialization and access to all ML models.
Includes proper async initialization, thread safety, and non-blocking operations.
"""

import asyncio
import logging
from typing import Optional
from google import genai
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from ai.suggestion import SuggestionModel  
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages all ML model instances with thread-safe initialization.
    
    Features:
    - Async initialization with asyncio.Lock for thread safety
    - Run blocking operations in executor threads
    - Prevents duplicate model loading
    - Lazy loading with double-check pattern
    """
    
    def __init__(self):
        self._embd_model: Optional[SentenceTransformer] = None
        self._gemini_client: Optional[genai.Client] = None
        self._gemini_llm: Optional[ChatGoogleGenerativeAI] = None
        self._suggestion_model: Optional[SuggestionModel] = None
        self._lock = asyncio.Lock()
        self._all_initialized = False
    
    async def get_embedding_model(self) -> SentenceTransformer:
        """
        Get or initialize the sentence embedding model asynchronously.
        
        Uses thread executor to avoid blocking the event loop.
        
        Returns:
            SentenceTransformer: The initialized embedding model
        """
        if self._embd_model is not None:
            return self._embd_model

        async with self._lock:
            if self._embd_model is not None:
                return self._embd_model

        logger.info("🔄 Loading embedding model...")
        loop = asyncio.get_event_loop()
        embd_model = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        async with self._lock:
            if self._embd_model is None:
                self._embd_model = embd_model

        logger.info("✅ Embedding model loaded successfully")
        return self._embd_model
    
    async def get_gemini_client(self) -> genai.Client:
        """
        Get or initialize the Gemini API client asynchronously.
        
        Uses thread executor to avoid blocking the event loop.
        
        Returns:
            genai.Client: The initialized Gemini client
        """
        if self._gemini_client is not None:
            return self._gemini_client

        async with self._lock:
            if self._gemini_client is not None:
                return self._gemini_client

        logger.info("🔄 Initializing Gemini API client...")
        loop = asyncio.get_event_loop()
        gemini_client = await loop.run_in_executor(
            None,
            lambda: genai.Client()
        )

        async with self._lock:
            if self._gemini_client is None:
                self._gemini_client = gemini_client

        logger.info("✅ Gemini API client initialized")
        return self._gemini_client
    
    async def get_gemini_llm(self) -> ChatGoogleGenerativeAI:
        """
        Get or initialize the Gemini LLM asynchronously.
        
        Uses thread executor to avoid blocking the event loop.
        
        Returns:
            ChatGoogleGenerativeAI: The initialized Gemini LLM
        """
        if self._gemini_llm is not None:
            return self._gemini_llm

        async with self._lock:
            if self._gemini_llm is not None:
                return self._gemini_llm

        logger.info("🔄 Initializing Gemini LLM...")
        loop = asyncio.get_event_loop()
        gemini_llm = await loop.run_in_executor(
            None,
            lambda: ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=GEMINI_API_KEY,
                temperature=0.2
            )
        )

        async with self._lock:
            if self._gemini_llm is None:
                self._gemini_llm = gemini_llm

        logger.info("✅ Gemini LLM initialized")
        return self._gemini_llm
    
    
    
    async def initialize_all(self) -> None:
        """
        Initialize all models in parallel for faster startup.
        
        Uses asyncio.gather() to load models concurrently.
        """
        if self._all_initialized:
            logger.info("⏭️  Models already initialized, skipping...")
            return
        
        try:
            logger.info("🔄 Loading all AI models in parallel...")
            
            # Load all models concurrently
            await asyncio.gather(
                self.get_embedding_model(),
                self.get_gemini_client(),
                self.get_gemini_llm(),
                self.get_suggestion_model()
                )
            
            self._all_initialized = True
            logger.info("✅ All AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing models: {e}", exc_info=True)
            raise

    async def get_suggestion_model(self) -> SuggestionModel:
        if self._suggestion_model is not None:
            return self._suggestion_model

        async with self._lock:
            if self._suggestion_model is not None:
                return self._suggestion_model

        logger.info("🔄 Loading suggestion model...")
        loop = asyncio.get_event_loop()
        suggestion_model = await loop.run_in_executor(
            None,
            lambda: SuggestionModel()
        )

        async with self._lock:
            if self._suggestion_model is None:
                self._suggestion_model = suggestion_model

        logger.info("✅ Suggestion model loaded")
        return self._suggestion_model

# Global instance
_model_manager = ModelManager()


async def get_model_manager() -> ModelManager:
    """
    FastAPI dependency to get the ModelManager instance.
    
    Returns:
        ModelManager: The global model manager instance
    """
    return _model_manager


async def initialize_all_models() -> None:
    """
    Initialize all models during app startup.
    
    Call this function in the FastAPI lifespan startup event.
    """
    await _model_manager.initialize_all()


async def initialize_sentence_splitter_model() -> SentenceTransformer:
    """
    Get or initialize the embedding model asynchronously.
    
    This is a convenience function for backward compatibility.
    
    Returns:
        SentenceTransformer: The initialized embedding model
    """
    return await _model_manager.get_embedding_model()


async def embed_chunks(chunks) -> list:
    """
    Embed text chunks using the sentence transformer model.
    
    Handles both single strings and lists of strings.
    Runs the blocking encode operation in a thread executor.
    
    Args:
        chunks: String or list of strings to embed
        
    Returns:
        list: List of embeddings (vectors)
    """
    model = await _model_manager.get_embedding_model()
    
    if isinstance(chunks, str):
        chunks = [chunks]
    
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: model.encode(chunks).tolist()
    )
    
    return embeddings


async def gemini_embed_model(chunk: str) -> list:
    """
    Generate embedding using Gemini API.
    
    Args:
        chunk: Text to embed
        
    Returns:
        list: Embedding vector from Gemini
    """
    client = await _model_manager.get_gemini_client()
    
    embed_model = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk
    )
    
    return embed_model.embeddings


async def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Get the initialized Gemini LLM.
    
    Returns:
        ChatGoogleGenerativeAI: The Gemini LLM instance
    """
    return await _model_manager.get_gemini_llm()

async def get_suggestion_model() -> SuggestionModel:
    return await _model_manager.get_suggestion_model()