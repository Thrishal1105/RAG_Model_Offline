# 🧠 Offline RAG Question Answering App


A fully offline **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions based on its content. This project uses semantic search, vector embeddings, cross-encoder reranking, and a local language model (via Ollama) — all without any cloud API dependencies.

![Project Image](https://res.cloudinary.com/dalmvzwgj/image/upload/v1765912453/Screenshot_2025-12-17_002225_rsnhjd.png)
---

## 🚀 Project Overview

This application enables you to:

- 📄 Upload a PDF document
- 🧩 Convert it into searchable text chunks
- 🧠 Store text embeddings in a vector database
- 🔎 Retrieve relevant document chunks via semantic search
- 🎯 Improve retrieval quality using reranking
- 🤖 Generate detailed answers using a local LLM

---

## 📦 Dependencies

### 🛠 Tools & Models

These are required before running the project:

 **Ollama** — Local language model platform  
 - `llama3.2:1b` — for text generation  
 - `nomic-embed-text` — for generating context embeddings


