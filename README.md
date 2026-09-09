# Personalized RAG Chatbot

A Generative AI-powered document question-answering chatbot that allows users to upload PDF documents and ask questions about their content.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded document and generate context-aware answers using Google Gemini.

## 🚀 Features

- 📄 Upload and analyze PDF documents
- 🔍 Extract text from PDFs using `pdfplumber`
- ✂️ Split documents into smaller chunks for efficient retrieval
- 🧠 Generate semantic embeddings using Gemini
- ⚡ Store and search document embeddings using FAISS
- 🤖 Generate answers using Google Gemini
- 💬 Interactive chatbot interface built with Streamlit
- 🔐 API key securely managed using environment variables / Streamlit Secrets

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Google Gemini API**
- **FAISS**
- **PDFPlumber**
- **RAG (Retrieval-Augmented Generation)**

## 🔄 How It Works

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Gemini Embeddings
    ↓
FAISS Vector Store
    ↓
Similarity Retrieval
    ↓
Relevant Context
    ↓
Gemini LLM
    ↓
Generated Answer
