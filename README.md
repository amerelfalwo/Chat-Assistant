# 🤖 Chat Assistant API (RAG-based)

**Chat Assistant** is an advanced AI system designed to assist users in interacting with, researching, and generating articles from their uploaded documents. By leveraging **Retrieval-Augmented Generation (RAG)**, the application combines the power of modern LLMs with dynamic document analysis.

---

## 🚀 Key Features

* **Intelligent Document Processing**: Automatically parses and understands complex PDF documents from any field.
* **Semantic Search**: Uses advanced embeddings to understand domain-specific terminology across various subjects (e.g., technology, history, law).
* **Vector Memory**: Powered by **Pinecone**, allowing the assistant to "remember" and reference multi-document contexts across sessions.
* **High Accuracy**: Optimized with a modern Gemini/Embedding architecture for precise information retrieval and content generation.

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Backend API:** FastAPI & Uvicorn
* **LLM & Embeddings:** Google Gemini (Generative AI)
* **Vector Database:** Pinecone (Serverless)
* **Framework:** LangChain 
* **Data Processing:** PyPDF & Recursive Character Splitting
* **Package Management:** UV

---

## 🏗️ Architecture Flow

1. **Ingestion**: Document PDFs are uploaded via the `/upload` endpoint.
2. **Chunking**: Documents are split into optimized context chunks via the backend.
3. **Embedding**: Text is converted into vectors using `GoogleGenerativeAIEmbeddings`.
4. **Indexing**: Vectors are upserted into Pinecone with associated metadata bound to a `session_id`.
5. **Retrieval**: Sending a question and a `session_id` to the `/ask` endpoint retrieves the most relevant medical context to generate a mathematically grounded response.

---

## 🔧 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/amerelfalwo/Chat-Assistant.git
cd "Chat Assistant"
```

2. **Install dependencies using [uv](https://docs.astral.sh/uv/):**
```bash
uv sync
```

3. **Set up environment variables:**
Create a `.env` file in the root directory and configure the following keys:
```env
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key  # if applicable
```

4. **Run the Backend API:**
```bash
uv run main.py
```
*The FastAPI backend will start on `http://127.0.0.1:8000`*
*You can access the Swagger UI documentation at `http://127.0.0.1:8000/docs`*

---

## 📂 Project Structure

```text
├── app/                  # FastAPI Backend Core Logic
│   ├── api/              # Endpoints for chat & PDF uploading
│   ├── core/             # Configuration and environment setup
│   └── services/         # Vectorstore, memory mgmt, and RAG pipelines
├── upload_pdfs/          # Local staging for uploaded documents
├── DATA/                 # Directory containing default guidelines and materials
├── main.py               # Application engine (Backend entrypoint)
├── pyproject.toml        # Application dependencies and package resolution
└── README.md             # Project documentation
```

---

## 🛡️ AI Disclaimer

*This project is intended for educational and research purposes only. AI-generated content can be incorrect; always verify critical information from primary sources.*

---

**Developed with ❤️ for the Graduation Project 2026.**
