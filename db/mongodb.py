# app/db/mongodb.py

from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGO_URI, DB_NAME
import certifi

client: AsyncIOMotorClient = None


async def get_db():
    try:
        return client[DB_NAME]
    except Exception as e:
        print(f"Error occurred while fetching DB reference: {e}")
        raise 


async def connect_to_mongo():
    try:
        global client
        client = AsyncIOMotorClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
        print(MONGO_URI, DB_NAME, "✅ Connected to MongoDB----")
        print("✅ Connected to MongoDB")

    except Exception as e:
        print(f"Error occurred while connecting to MongoDB: {e}")
        raise 

async def close_mongo_connection():
    try:
        global client
        client.close()
        print("❌ MongoDB closed")
    except Exception as e:
        print(f"Error occurred while closing MongoDB connection: {e}")
        raise 