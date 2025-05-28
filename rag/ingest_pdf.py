import fitz #PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "./Dream_Textbook.pdf"
VECTOR_PATH = "./vector_store"

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc: 
        full_text += page.get_text()
    return full_text

def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap =50)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(VECTOR_PATH)
    print(f"Vector store saved to {VECTOR_PATH}")

if __name__ == "__main__":
    print("Loading PDF...")
    raw_text = extract_text_from_pdf(PDF_PATH)
    print("Splitting text...")
    create_vector_store(raw_text)