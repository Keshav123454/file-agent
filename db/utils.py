from asyncio.log import logger
from datetime import datetime
from .mongodb import get_db


async def save_file_to_mongo(file, extracted_text: str):
    db = await get_db()  # 🔥 get fresh reference
    try:
        contents = await file.read()
        file_size = len(contents)
        await file.seek(0)
        document = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": file_size,
            "extracted_text": extracted_text,
            "created_at": datetime.utcnow()
        }
        result = await db.files.insert_one(document)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving file to MongoDB: {e}")
        raise 

async def get_all_files():
    
    db = await get_db()
    try:
        print("Fetching all files from MongoDB...")
        files = []
        async for doc in db.files.find():
            doc["_id"] = str(doc["_id"])  # 🔥 convert ObjectId → string
            files.append(doc)
        print(f"Fetched {len(files)} files")
        return files
    except Exception as e:
        print(f"Error fetching files: {e}")
        raise 

async def get_file_by_id(file_id: str):
    from bson import ObjectId
    db = await get_db()
    try:
        print(f"Fetching file with ID: {file_id}")
        doc = await db.files.find_one({"_id": ObjectId(file_id)})
        if doc:
            doc["_id"] = str(doc["_id"])  # 🔥 convert ObjectId → string
            print(f"File found: {doc['filename']}")
            return doc
        else:
            print("File not found")
            return None
    except Exception as e:
        print(f"Error fetching file by ID: {e}")
        raise


async def delete_file_by_id(file_id: str):
    from bson import ObjectId
    db = await get_db()

    try:
        result = await db.files.delete_one({"_id": ObjectId(file_id)})

        if result.deleted_count == 1:
            return {"message": "File deleted successfully"}
        else:
            return {"message": "File not found"}

    except Exception as e:
        print(f"Error deleting file: {e}")
        return {"message": "Error deleting file"}  # ✅ NEVER return None
    


async def search_data(search_cache_results):
    try:

        if (
            search_cache_results
            and hasattr(search_cache_results, "data")
            and search_cache_results.data
        ):

            first_result = search_cache_results.data[0]

            cached_response = {
                "id": first_result.id,
                "query": first_result.prompt,
                "response": first_result.response,
                "attributes": first_result.attributes,
                "similarity": first_result.similarity,
                "search_strategy": str(first_result.search_strategy)
            }

        else:
            cached_response = None
        
        return cached_response

    except Exception as e:
        logger.exception(f"Error while processing cache results: {e}")
        cached_response = None

    return cached_response
