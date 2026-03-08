import fitz 
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def build_index_from_pdf(pdf_path: str, persist_dir: str= './rag_faiss_store'):
    full_text = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents([Document(page_content=full_text, metadata={"source": pdf_path})])
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = FAISS.from_documents(documents, embeddings)
    db.save_local(persist_dir)
    

if __name__ == "__main__":
    build_index_from_pdf('docs/sample_rental_agreement.pdf')

