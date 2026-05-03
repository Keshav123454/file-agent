"""
FastAPI application for file processing with RAG (Retrieval-Augmented Generation)
"""

import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.concurrency import asynccontextmanager
from langchain_core.messages import HumanMessage

# Database imports
from db.mongodb import close_mongo_connection, connect_to_mongo
from db.utils import save_file_to_mongo, get_all_files, get_file_by_id
from db.pincone_db import initialize_pinecone

# File processing imports
from file_reader import extract_text_from_file
from ai.text_splitter import split_document

# AI/ML imports
from ai.models import initialize_all_models, get_model_manager, ModelManager
from ai.embedding import upsert_document, search_similar
from ai.langGraph_buileder import agent

# Utils
from utils import validate_file, validate_file_id
from fastapi.middleware.cors import CORSMiddleware
from db.utils import delete_file_by_id

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage app lifecycle - startup and shutdown events.
    Initializes all models, databases, and services at startup.
    """
    try:
        # 🔥 STARTUP
        logger.info("🚀 Starting application initialization...")
        
        # Initialize database connections
        await connect_to_mongo()
        logger.info("✅ MongoDB connected")
        
        await initialize_pinecone()
        logger.info("✅ Pinecone initialized")
        
        # Initialize all AI models (async, non-blocking)
        await initialize_all_models()
        logger.info("✅ All AI models initialized")
        
        logger.info("🚀 Application ready to handle requests")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}", exc_info=True)
        raise

    yield

    # 🔥 SHUTDOWN
    try:
        await close_mongo_connection()
        logger.info("🛑 Application stopped successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}", exc_info=True)


app = FastAPI(
    title="File Agent API",
    description="Upload files, embed them, and chat with RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ HEALTH CHECK ============

@app.get("/health")
async def health_check(models: ModelManager = Depends(get_model_manager)):
    """
    Health check endpoint to verify all services are running.
    
    Returns:
        dict: Status of all services
    """
    try:
        emb_model = await models.get_embedding_model()
        gemini_client = await models.get_gemini_client()
        gemini_llm = await models.get_gemini_llm()
        
        return {
            "status": "healthy",
            "services": {
                "embedding_model": "loaded",
                "gemini_client": "initialized",
                "gemini_llm": "initialized"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# ============ ROOT ENDPOINT ============

@app.get("/")
def read_root():
    """Root endpoint - welcome message."""
    return {
        "message": "Welcome to File Agent API",
        "docs": "/docs",
        "health": "/health"
    }


# ============ FILE UPLOAD ============

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file and extract text content.
    
    Supported formats: TXT, PDF, DOCX
    
    Args:
        file: The file to upload
        
    Returns:
        dict: File ID, size, and chunk count
        
    Raises:
        HTTPException: If file validation fails or processing error
    """
    try:
        # Validate file
        is_valid = await validate_file(file)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported: TXT, PDF, DOCX"
            )

        # Extract text from file
        text = await extract_text_from_file(file)
        
        if not text or text.startswith("❌"):
            raise HTTPException(status_code=400, detail=text)

        # Split document into chunks
        chunks = await split_document(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from file")

        # Save to MongoDB
        file_id = await save_file_to_mongo(file, chunks)

        logger.info(f"✅ File uploaded successfully: {file.filename} (ID: {file_id})")

        return {
            "message": "File uploaded successfully",
            "file_id": file_id,
            "filename": file.filename,
            "file_size": file.size,
            "chunk_count": len(chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /upload-file endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during file upload")


# ============ FILE RETRIEVAL ============

@app.get("/files")
async def get_files():
    """
    Get all uploaded files.
    
    Returns:
        dict: List of files with metadata
        
    Raises:
        HTTPException: If database error occurs
    """
    try:
        files = await get_all_files()
        logger.info(f"Retrieved {len(files)} files")
        
        return {
            "count": len(files),
            "files": files
        }

    except Exception as e:
        logger.error(f"Error in /files endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving files")


@app.get("/files/{file_id}")
async def get_file(file_id: str):
    """
    Get a specific file by ID.
    
    Args:
        file_id: MongoDB ObjectId of the file
        
    Returns:
        dict: File data with metadata
        
    Raises:
        HTTPException: If file not found or invalid ID
    """
    try:
        # Validate file ID format
        is_valid_id = await validate_file_id(file_id)
        if not is_valid_id:
            raise HTTPException(status_code=400, detail="Invalid file ID format")

        # Retrieve file
        file = await get_file_by_id(file_id)
        
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        logger.info(f"Retrieved file: {file_id}")
        return {"file": file}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /files/{{file_id}} endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving file")

# ============ DELETE FILE ============
@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    result = await delete_file_by_id(file_id)

    if result["message"] == "File not found":
        raise HTTPException(status_code=404, detail="File not found")

    return result

# ============ EMBEDDING & VECTOR SEARCH ============

@app.post("/embed/{file_id}")
async def embed_file(file_id: str, models: ModelManager = Depends(get_model_manager)):
    """
    Generate embeddings for a file and store in vector database.
    
    Args:
        file_id: MongoDB ObjectId of the file
        models: ModelManager dependency
        
    Returns:
        dict: Number of chunks embedded
        
    Raises:
        HTTPException: If file not found or embedding fails
    """
    try:
        # Validate file ID
        is_valid_id = await validate_file_id(file_id)
        if not is_valid_id:
            raise HTTPException(status_code=400, detail="Invalid file ID format")

        # Check if file exists
        file = await get_file_by_id(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        # Generate embeddings
        embeddings = await upsert_document(file_id)
        
        logger.info(f"✅ Embeddings created for file: {file_id}")
        return embeddings

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /embed/{{file_id}} endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error creating embeddings")


@app.get("/search")
async def search_vec(query: str, file_id: str):
    """
    Search for similar content in a file using vector embeddings.
    
    Args:
        query: Search query string
        file_id: MongoDB ObjectId of the file
        
    Returns:
        dict: List of similar chunks with scores
        
    Raises:
        HTTPException: If search fails or file not found
    """
    try:
        # Validate inputs
        if not query or len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        is_valid_id = await validate_file_id(file_id)
        if not is_valid_id:
            raise HTTPException(status_code=400, detail="Invalid file ID format")

        # Check if file exists
        file = await get_file_by_id(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        # Search similar content
        results = await search_similar(file_id, query)
        
        logger.info(f"Search performed for file: {file_id}, found {len(results)} results")
        return {
            "query": query,
            "file_id": file_id,
            "result_count": len(results),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error performing search")


# ============ LLM CHAT & RAG ============



from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    query: str
    file_id: Optional[str] = None


@app.post("/chat")
async def get_llm_response(
    request: ChatRequest,
    models: ModelManager = Depends(get_model_manager)
):
    query = request.query
    file_id = request.file_id
    """
    Chat with the LLM using RAG (Retrieval-Augmented Generation).
    
    If file_id is provided, the LLM uses content from that file as context.
    Otherwise, it responds with general knowledge.
    
    Args:
        query: User's question
        file_id: Optional MongoDB ObjectId for RAG context
        models: ModelManager dependency
        
    Returns:
        dict: LLM response and metadata
        
    Raises:
        HTTPException: If query is empty or processing fails
    """
    try:
        # Validate query
        if not query or len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Validate file ID if provided
        print(query, file_id, "@"*100)
        if file_id:
            is_valid_id = await validate_file_id(file_id)
            if not is_valid_id:
                raise HTTPException(status_code=400, detail="Invalid file ID format")

            # Check if file exists
            file = await get_file_by_id(file_id)
            if not file:
                raise HTTPException(status_code=404, detail="File not found")

        # Prepare state for agent
        state = {
            "messages": [HumanMessage(content=query)],
            "llm_calls": 0,
            "file_id": file_id
        }

        # Get LLM response
        result = await agent.ainvoke(state)

        logger.info(f"✅ Chat response generated (LLM calls: {result['llm_calls']})")

        return {
            "query": query,
            "response": result["messages"][-1].content,
            "llm_calls": result["llm_calls"],
            "mode": "RAG" if file_id else "General"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing chat request")


# ============ ERROR HANDLERS ============

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle any unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
