import os 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def retrieve_legal_context(query: str) -> str:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    db = FAISS.load_local('./rag_faiss_store', embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    query = "can the landlord increase the rent during the lease term?"
    context = retrieve_legal_context(query)
    print("Retrieved Legal Context:\n", context)