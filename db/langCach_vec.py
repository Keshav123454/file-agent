from langcache import LangCache
from langcache.models import SearchStrategy
from config import LANGCACHE_API_KEY, LANGCACHE_ID, LANGCACHE_URL

class LangCacheClient:
    def __init__(self):
        try:
            self.lang_cache = LangCache(
                server_url=LANGCACHE_URL,
                cache_id=LANGCACHE_ID,
                api_key=LANGCACHE_API_KEY
            )
            print("✅ Connected to LangCache")
        except Exception as e:
            print("❌ Error connecting to LangCache:", e)
            self.lang_cache = None

    def save_entry(self, prompt, response, file_id=None):
        if self.lang_cache:
            try:
                save_response = self.lang_cache.set(prompt=prompt, response=response, attributes={"file_id": file_id})
                print("\n✅ Save Response:")
                return save_response
            except Exception as e:
                print("❌ Error saving entry:", e)

    def search_entry(self, prompt, file_id=None):
        if self.lang_cache:
            try:
                search_response = self.lang_cache.search(prompt=prompt, 
                                                         search_strategies=[SearchStrategy.EXACT, SearchStrategy.SEMANTIC],
                                                         similarity_threshold=0.9,
                                                         attributes={"file_id": file_id} if file_id else None)
                print("\n✅ Search Response:")
                print(search_response)
                return search_response
            except Exception as e:
                print("❌ Error searching entry:", e)

    def delete_entry(self, prompt, file_id=None):
        if self.lang_cache:
            try:
                delete_response = self.lang_cache.delete(prompt=prompt, attributes={"file_id": file_id} if file_id else None)
                print("\n✅ Delete Response:")
                print(delete_response)
            except Exception as e:
                print("❌ Error deleting entry:", e)
