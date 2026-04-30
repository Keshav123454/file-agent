from langchain_text_splitters import RecursiveCharacterTextSplitter

async def split_document(document: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_text(document)
    return texts


