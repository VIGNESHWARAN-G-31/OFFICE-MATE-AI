# 🤖 OFFICE MATE AI

A smart AI-powered chatbot that lets you upload PDFs and ask natural language questions. It uses **Groq's LLaMA3** model for fast and efficient responses and **FAISS** for semantic search on document chunks. Built with **Streamlit**, this project enables contextual Q&A directly from your documents!

---

## 🚀 Features

- 📄 Upload and process multiple PDF files
- 🧠 Semantic chunking and vector storage using FAISS
- 🔍 Ask context-aware questions about your PDFs
- 💬 Real-time conversational interface with memory
- ⚡ Powered by **Groq (LLaMA3)** and **Hugging Face Embeddings**

---

## 🛠️ Tech Stack

- 🐍 Python
- 🤗 HuggingFace Transformers (`sentence-transformers/all-MiniLM-L6-v2`)
- 🦙 Groq API (`llama3-8b-8192`)
- 🧠 FAISS (Facebook AI Similarity Search)
- 📚 PyPDF2
- 🧩 LangChain
- 🌐 Streamlit (for the interactive UI)
- 🔐 dotenv for secure API key handling

---

## 🧠 How It Works

-PDF Ingestion: Extracts raw text using PyPDF2
-Text Splitting: Breaks text into 10,000-character chunks (with overlap)
-Embeddings: Generates embeddings via sentence-transformers/all-MiniLM-L6-v2
-Vector Search: Uses FAISS for semantic similarity search
-QA Pipeline: LLaMA3 model answers questions based on contextually relevant chunks

---
# 🧪 Example Use Case

-Upload: Employee handbook
-Ask: "What is the policy on sick leave?"
-✅ Bot answers: Detailed policy section from the handbook
