import os
import tempfile
import chromadb
import ollama
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


# This prompt forces the LLM to answer ONLY from document context
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context.

context will be passed as "Context:"
user question will be passed as "Question:"

Rules:
- Use ONLY the given context
- Do NOT hallucinate
- If answer not found, say clearly
- Write structured, readable responses
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """
    Converts uploaded PDF into small text chunks
    so embeddings work accurately.
    """

    # 1. Create temp file and CLOSE it immediately
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    try:
        # 2. Load PDF after file is closed
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()

        # 3. Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )

        return text_splitter.split_documents(docs)

    finally:
        # 4. Safely delete temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)



def get_vector_collection() -> chromadb.Collection:
    """
    Creates or loads a ChromaDB collection
    using Ollama embedding model.
    """

    # Use your INSTALLED embedding model
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text",
    )

    # Persistent vector database
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")

    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )




def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """
    Stores document chunks into ChromaDB
    with embeddings.
    """

    collection = get_vector_collection()

    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    st.success("✅ Document stored successfully!")



def query_collection(prompt: str, n_results: int = 10):
    """
    Retrieves top matching document chunks
    from vector database.
    """

    collection = get_vector_collection()

    return collection.query(
        query_texts=[prompt],
        n_results=n_results
    )



def re_rank_cross_encoders(prompt: str, documents: list[str]):
    """
    Improves relevance using cross-encoder reranking.
    """

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Rank documents based on query
    ranks = encoder_model.rank(prompt, documents, top_k=3)

    relevant_text = ""
    relevant_text_ids = []

    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids




def call_llm(context: str, prompt: str):
    """
    Sends context + question to Ollama LLM
    and streams response.
    """

    response = ollama.chat(
        model="llama3.2:1b",   # ✅ YOUR INSTALLED MODEL
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{prompt}",
            },
        ],
    )

    for chunk in response:
        if not chunk["done"]:
            yield chunk["message"]["content"]



if __name__ == "__main__":

    # ---------- PAGE CONFIG ----------
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="📄",
        layout="wide"
    )

    # ---------- HEADER ----------
    st.markdown(
        """
        <h1 style='text-align: center;'>📄 RAG Model Chat Assistant - Offline</h1>
        <p style='text-align: center; color: gray;'>
        Upload a PDF • Ask questions • Get accurate answers
        </p>
        <hr>
        """,
        unsafe_allow_html=True
    )

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("📤 Document Upload")

        uploaded_file = st.file_uploader(
            "Upload your PDF",
            type=["pdf"],
            help="Upload a PDF to build knowledge base"
        )

        process_btn = st.button("⚡ Process Document", use_container_width=True)

        st.markdown("---")
        st.caption("⚙️ Powered by Ollama + ChromaDB")

        if uploaded_file and process_btn:
            with st.spinner("Processing document..."):
                clean_name = uploaded_file.name.replace(".", "_").replace(" ", "_")
                splits = process_document(uploaded_file)
                add_to_vector_collection(splits, clean_name)

    # ---------- MAIN LAYOUT ----------
    col1, col2 = st.columns([2, 1])

    # ---------- CHAT SECTION ----------
    with col1:
        st.subheader("💬 Ask Questions")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_question = st.text_input(
            "Type your question here",
            placeholder="What is this document about?"
        )

        ask_btn = st.button("🚀 Ask")

        if ask_btn and user_question:
            with st.spinner("Searching relevant context..."):
                results = query_collection(user_question)
                docs = results["documents"][0]
                relevant_text, ids = re_rank_cross_encoders(user_question, docs)

            with st.spinner("Generating answer..."):
                response_stream = call_llm(relevant_text, user_question)
                answer = st.write_stream(response_stream)

            st.session_state.chat_history.append(
                {"question": user_question, "answer": answer}
            )

       

    # ---------- CONTEXT DEBUG ----------
    with col2:
        st.subheader("📄 Retrieved Context")

        if 'docs' in locals():
            with st.expander("View Retrieved Chunks"):
                for i, d in enumerate(docs):
                    st.markdown(f"**Chunk {i+1}**")
                    st.write(d[:500] + "...")
                    st.markdown("---")

        if 'ids' in locals():
            with st.expander("🎯 Best Chunk IDs"):
                st.write(ids)
