from google import genai
from dotenv import load_dotenv
import os
import chromadb

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

chroma_client = chromadb.PersistentClient(path="./database")
collection = chroma_client.get_collection(name="pdf_doc")

while True:
    user_input = input("Ask about the Ejob India: ").strip()

    if user_input.lower() == "exit":
        print("Agent: Cya!")
        break

    # if not user_input:
    #     continue

    try:
        # Create embedding
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=user_input
        )
        query_embedding = response.embeddings[0].values

        # Search ChromaDB
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )

        # Get relevant documents
        documents = result.get("documents") or [[]]
        context = (
            "\n\n".join(documents[0])
            if documents and documents[0]
            else "No relevant context found."
        )

        # Send context to Gemini
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=f"""
                You are a helpful PDF analysis assistant.
                Answer using only the information in the retrieved context.
                If the answer is not available in the context, say "I don't know".
                Do not invent facts or use outside knowledge.
                Context: {context}
                Question: {user_input}
            """
        )

        print("\nAgent:")
        print(response.text, "\n")

    except Exception as e:
        print(f"[error] {e}\n")