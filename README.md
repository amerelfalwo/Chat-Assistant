بناءً على مشروعك **ThyraX**، ده ملف `README.md` احترافي ومنظم، مناسب جداً لرفعه على GitHub عشان يوضح قوة المشروع والتقنيات المستخدمة فيه.

---

# 🩺 ThyraX: AI Medical Assistant Chatbot (RAG-based)

**ThyraX** is an advanced medical AI system designed to assist radiologists and clinicians in detecting and analyzing **Thyroid Cancer**. By leveraging **Retrieval-Augmented Generation (RAG)**, the application combines the power of Google's Gemini models with specialized medical metadata and ultrasound report analysis.

## 🚀 Key Features

* **Intelligent Document Processing**: Automatically parses complex medical PDFs and ultrasound reports.
* **Semantic Search**: Uses Google Generative AI Embeddings to understand medical terminology (e.g., Hypoechoic, Microcalcifications).
* **Vector Memory**: Powered by **Pinecone**, allowing the chatbot to "remember" and reference clinical guidelines and past cases.
* **High Accuracy**: Optimized with the latest `text-embedding-004` model for precise classification between benign and malignant nodules.

## 🛠️ Tech Stack

* **Language:** Python
* **LLM & Embeddings:** Google Gemini (Generative AI)
* **Vector Database:** Pinecone (Serverless)
* **Framework:** LangChain
* **Data Processing:** PyPDF & Recursive Character Splitting

---

## 🏗️ Architecture Flow

1. **Ingestion**: Clinical PDFs are uploaded and stored locally.
2. **Chunking**: Documents are split into optimized medical context chunks.
3. **Embedding**: Text is converted into 768-dimensional vectors using `GoogleGenerativeAIEmbeddings`.
4. **Indexing**: Vectors are upserted into Pinecone with associated metadata.
5. **Retrieval**: When a user asks a clinical question, the system retrieves the most relevant medical context to generate a grounded response.

---

## 🔧 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/amerelfalwo/MEDICAL ASSISTANT.git
cd ThyraX

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Set up environment variables:**
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key

```


4. **Run the application:**
```python
python main.py

```



---

## 📂 Project Structure

```text
├── upload_pdfs/          # Storage for uploaded medical reports
├── main.py               # Main application logic
├── .env                  # Environment variables (ignored by git)
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

```

## 🛡️ Medical Disclaimer

*ThyraX is a graduation project intended for educational and decision-support purposes only. It should not be used as a replacement for professional medical diagnosis or clinical judgment.*

---

**Developed with ❤️ for the Graduation Project 2026.**

---
