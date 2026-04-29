from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import asynccontextmanager
from db.mongodb import close_mongo_connection, connect_to_mongo
from db.utils import save_file_to_mongo
from file_reader import extract_text_from_file

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔥 Startup
    await connect_to_mongo()
    print("🚀 App started")

    yield

    # 🔥 Shutdown
    await close_mongo_connection()
    print("🛑 App stopped")
    


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    text = await extract_text_from_file(file)

    file_id = await save_file_to_mongo(file, text)

    return {
        "message": "File uploaded successfully",
        "file_id": file_id,
        "file_size": file.size
    }

@app.get("/files")
async def get_files():
    from db.utils import get_all_files
    files = await get_all_files()
    return {"files": files}

@app.get("/files/{file_id}")
async def get_file(file_id: str):
    from db.utils import get_file_by_id
    file = await get_file_by_id(file_id)
    if file:
        return {"file": file}
    else:
        return {"message": "File not found"}, 404