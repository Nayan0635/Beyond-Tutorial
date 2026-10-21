from google import genai
from dotenv import load_dotenv
import os
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

pdf_path = "./documents/machine learning.pdf"
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

if not text.strip():
    raise ValueError("No extractable text found in the PDF (scanned image?).")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(text)

if not chunks:
    raise ValueError("No text chunks were created.")

chroma_client = chromadb.PersistentClient(path="./database")
collection = chroma_client.get_or_create_collection(
    name="pdf_doc",
    metadata={"hnsw:space": "cosine"}
)

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks
)
embeddings = [item.values for item in response.embeddings]

source = "machine learning.pdf"

try:
    collection.delete(where={"source": source})
except Exception:
    pass  # nothing to delete on first run

collection.upsert(
    ids=[f"{source}_{i}" for i in range(len(chunks))],
    embeddings=embeddings,
    documents=chunks,
    metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))]
)
print("PDF chunks stored")

# from google import genai
# from dotenv import load_dotenv
# import os
# import chromadb
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# load_dotenv()
# client = genai.Client(
#     api_key=os.getenv("GEMINI_API_KEY")
# )
# "./documents/machine learning.pdf" = "./documents/machine learning.pdf"
# reader = PdfReader("./documents/machine learning.pdf")
# text = ""
# for page in reader.pages:
#     text += page.extract_text() + "\n"

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=50
# )
# chunks = text_splitter.split_text(text)
# # if not chunks:
# #     raise ValueError("The PDF contains no extractable text; no chunks were created.")

# chroma_client = chromadb.PersistentClient(path="./database")
# collection = chroma_client.get_or_create_collection(name="pdf_doc")

# response = client.models.embed_content(
#     model="gemini-embedding-001",
#     contents=chunks
# )
# embeddings = [item.values for item in response.embeddings]
# source = "machine learning.pdf"
# collection.delete(where={"source": "machine learning.pdf"})
# collection.upsert(
#     ids=[f"pdf_{i}" for i in range(len(chunks))],
#     embeddings=embeddings,
#     documents=chunks,
#     metadatas=[{"source": source, "chunk_index": i} for i in range(len(chunks))]
# )
# print("PDF chunks stored")
