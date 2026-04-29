from fastapi import UploadFile
import os
from PyPDF2 import PdfReader
from docx import Document


async def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename.lower()

    # Read file content once
    content = await file.read()

    # TXT
    if filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")

    # PDF
    elif filename.endswith(".pdf"):
        from io import BytesIO
        pdf = PdfReader(BytesIO(content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text

    # DOCX
    elif filename.endswith(".docx"):
        from io import BytesIO
        doc = Document(BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs])

    # DOC (old format)
    elif filename.endswith(".doc"):
        return "❌ .doc not supported directly. Convert to .docx first."

    else:
        return "❌ Unsupported file format"