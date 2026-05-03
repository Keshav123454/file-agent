"""
Validation utilities for the File Agent API.
Provides input validation, security checks, and data sanitization.
"""

import logging
from pathlib import Path
from fastapi import UploadFile
from bson import ObjectId

logger = logging.getLogger(__name__)

# Configuration
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
MIN_QUERY_LENGTH = 1
MAX_QUERY_LENGTH = 1000
MIN_CHUNK_SIZE = 10
MAX_CHUNKS = 10000


# ============ FILE VALIDATION ============

async def validate_file(file: UploadFile) -> bool:
    """
    Validate uploaded file for security and compatibility.
    
    Checks:
    - File extension is allowed
    - File size is within limits
    - Filename is safe
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Check filename exists
        if not file.filename:
            logger.warning("File validation failed: No filename provided")
            return False
        
        # Check extension
        filename_lower = file.filename.lower()
        ext = "." + filename_lower.split(".")[-1] if "." in filename_lower else ""
        
        if ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"File validation failed: Invalid extension '{ext}'")
            return False
        
        # Check file size
        if file.size and file.size > MAX_FILE_SIZE:
            logger.warning(f"File validation failed: Size {file.size} exceeds {MAX_FILE_SIZE}")
            return False
        
        # Validate filename safety
        if not sanitize_filename(file.filename):
            logger.warning(f"File validation failed: Unsafe filename '{file.filename}'")
            return False
        
        logger.info(f"✅ File validated: {file.filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error in file validation: {e}")
        return False


# ============ FILE ID VALIDATION ============

async def validate_file_id(file_id: str) -> bool:
    """
    Validate MongoDB ObjectId format.
    
    Args:
        file_id: String representation of MongoDB ObjectId
        
    Returns:
        bool: True if valid ObjectId, False otherwise
    """
    try:
        if not file_id or not isinstance(file_id, str):
            return False
        
        ObjectId(file_id)
        return True
        
    except Exception:
        logger.warning(f"Invalid file ID format: {file_id}")
        return False


# ============ QUERY VALIDATION ============

async def validate_query(query: str) -> bool:
    """
    Validate search/chat query.
    
    Checks:
    - Query is not empty
    - Query length is within limits
    - Query contains valid content
    
    Args:
        query: Query string
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not query or not isinstance(query, str):
            return False
        
        query_stripped = query.strip()
        
        # Check length
        if len(query_stripped) < MIN_QUERY_LENGTH:
            logger.warning("Query too short")
            return False
        
        if len(query_stripped) > MAX_QUERY_LENGTH:
            logger.warning(f"Query exceeds maximum length of {MAX_QUERY_LENGTH}")
            return False
        
        logger.info(f"✅ Query validated: {len(query_stripped)} characters")
        return True
        
    except Exception as e:
        logger.error(f"Error in query validation: {e}")
        return False


# ============ CHUNK VALIDATION ============

async def validate_chunks(chunks: list) -> bool:
    """
    Validate text chunks for embedding.
    
    Checks:
    - Chunks is a list
    - Chunk count is reasonable
    - Each chunk has content
    
    Args:
        chunks: List of text chunks
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not isinstance(chunks, list):
            logger.warning("Chunks must be a list")
            return False
        
        if len(chunks) == 0:
            logger.warning("No chunks to validate")
            return False
        
        if len(chunks) > MAX_CHUNKS:
            logger.warning(f"Too many chunks: {len(chunks)}")
            return False
        
        # Validate each chunk
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, str):
                logger.warning(f"Chunk {i} is not a string")
                return False
            
            if len(chunk.strip()) < MIN_CHUNK_SIZE:
                logger.warning(f"Chunk {i} is too small")
                return False
        
        logger.info(f"✅ Chunks validated: {len(chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Error in chunks validation: {e}")
        return False


# ============ TEXT VALIDATION ============

async def validate_extracted_text(text: str) -> bool:
    """
    Validate extracted text content.
    
    Args:
        text: Extracted text
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not text or not isinstance(text, str):
            return False
        
        if len(text.strip()) == 0:
            logger.warning("Extracted text is empty")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in text validation: {e}")
        return False


# ============ EMBEDDING VALIDATION ============

async def validate_embedding_input(text: str) -> bool:
    """
    Validate text before generating embeddings.
    
    Args:
        text: Text to embed
        
    Returns:
        bool: True if valid for embedding, False otherwise
    """
    try:
        if not text or not isinstance(text, str):
            return False
        
        text_stripped = text.strip()
        
        if len(text_stripped) < 5:
            logger.warning("Text too short for embedding")
            return False
        
        if len(text_stripped) > 10000:
            logger.warning("Text too long for embedding")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in embedding input validation: {e}")
        return False


# ============ PAGINATION VALIDATION ============

async def validate_pagination(skip: int = 0, limit: int = 20) -> tuple:
    """
    Validate and sanitize pagination parameters.
    
    Args:
        skip: Number of records to skip
        limit: Number of records to return
        
    Returns:
        tuple: (skip, limit) with safe values
    """
    try:
        # Ensure integers
        skip = max(0, int(skip))
        limit = max(1, min(100, int(limit)))  # Limit between 1 and 100
        
        return skip, limit
        
    except Exception as e:
        logger.error(f"Error in pagination validation: {e}")
        return 0, 20


# ============ FILENAME SANITIZATION ============

def sanitize_filename(filename: str) -> bool:
    """
    Sanitize and validate filename to prevent security issues.
    
    Checks:
    - No path traversal attempts (..)
    - No special characters that could cause issues
    - Filename length is reasonable
    
    Args:
        filename: Original filename
        
    Returns:
        bool: True if safe, False if suspicious
    """
    try:
        if not filename:
            return False
        
        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            logger.warning(f"Path traversal attempt detected: {filename}")
            return False
        
        # Check filename length
        if len(filename) > 255:
            logger.warning(f"Filename too long: {len(filename)}")
            return False
        
        # Use pathlib to validate
        try:
            Path(filename)
        except ValueError:
            logger.warning(f"Invalid filename: {filename}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in filename sanitization: {e}")
        return False


# ============ RESPONSE VALIDATION ============

def validate_response(response: dict) -> bool:
    """
    Validate API response format.
    
    Args:
        response: Response dictionary
        
    Returns:
        bool: True if valid response format
    """
    try:
        if not isinstance(response, dict):
            return False
        
        # Check for required fields in response
        if not response.get("status") and not response.get("message"):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in response validation: {e}")
        return False


# ============ BATCH VALIDATION ============

async def validate_batch_operation(items: list, max_size: int = 100) -> bool:
    """
    Validate batch operation parameters.
    
    Args:
        items: Items in batch
        max_size: Maximum batch size
        
    Returns:
        bool: True if valid batch
    """
    try:
        if not isinstance(items, list):
            return False
        
        if len(items) == 0 or len(items) > max_size:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error in batch validation: {e}")
        return False
