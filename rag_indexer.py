# backend/rag_indexer.py
import os
import tempfile
import pandas as pd
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAl, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "config.env"))

BASE_URL = os.getenv("GENAI_BASE_URL")
API_KEY = os.getenv("GENAI_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_index")

# Create httpx client with verify=False per your template
http_client = httpx.Client(verify=False)

# Initialize LLM + Embeddings (use your template names)
llm = ChatOpenAl(
    base_url=BASE_URL,
    model=DEEPSEEK_MODEL,
    api_key=API_KEY,
    http_client=http_client,
)

embedding_model = OpenAIEmbeddings(
    base_url=BASE_URL,
    model=EMBED_MODEL,
    api_key=API_KEY,
    http_client=http_client,
)

# Utility: convert Excel rows into LangChain Documents (simple)
def excel_to_documents(excel_path: str, text_columns: List[str]) -> List[str]:
    df = pd.read_excel(excel_path)
    # Fill NaN to empty strings
    df = df.fillna("")
    docs = []
    for _, row in df.iterrows():
        # Build a textual representation of the row
        pieces = []
        for col in text_columns:
            pieces.append(f"{col}: {row.get(col, '')}")
        # Optionally include other columns
        docs.append("\n".join(pieces))
    return docs

def build_rag_index(excel_path: str, text_columns: List[str], persist_dir: str = CHROMA_PERSIST_DIR, chunk_size: int = 1000, chunk_overlap: int = 200):
    os.makedirs(persist_dir, exist_ok=True)
    raw_docs = excel_to_documents(excel_path, text_columns)
    # chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for d in raw_docs:
        chunks.extend(text_splitter.split_text(d))

    # If no chunks, return error
    if len(chunks) == 0:
        raise ValueError("No text extracted from excel: check the text_columns provided.")

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        client_settings={"chroma_db_impl": "duckdb+parquet"},
    )
    vectordb.persist()
    return vectordb

def make_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa

def query_rag(question: str, persist_dir: str = CHROMA_PERSIST_DIR):
    # Load existing chroma
    if not os.path.exists(persist_dir):
        raise ValueError("Chroma index directory not found. Index data first.")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    qa = make_qa_chain(vectordb)
    result = qa.run(question)
    # Note: RetrievalQA.run returns the string answer; to get sources we can call qa() instead
    detailed = qa({"query": question})
    answer = detailed.get("result", result)
    sources = []
    if "source_documents" in detailed:
        for doc in detailed["source_documents"]:
            sources.append({"page_content": doc.page_content[:500]})
    return {"answer": answer, "sources": sources}
