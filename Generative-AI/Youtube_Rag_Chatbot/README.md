# 🎥 YouTube RAG Chatbot with Gemini

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built with **FastAPI**, **LangChain**, **Google Gemini**, and **Streamlit**. It allows users to ask questions based on the **transcript of any YouTube video** using state-of-the-art language models and vector search!

---

## 🚀 Features

- 🔍 Automatically extracts transcripts from YouTube videos
- 🧠 Uses **Gemini** to answer user questions with context-aware responses
- 🗂️ Vector search over video transcript using **FAISS**
- 📺 Supports both standard and short YouTube video URLs
- ⚡ Built with FastAPI for API backend and Streamlit for UI
- 🎨 Sleek and dark-themed interface for better readability

---

## 🧱 Tech Stack

| Component      | Technology                    |
|----------------|-------------------------------|
| Backend        | FastAPI, LangChain            |
| Frontend       | Streamlit                     |
| Embeddings     | HuggingFace Transformers      |
| Vector Store   | FAISS                         |
| LLM            | Google Gemini                 |
| Transcripts    | youtube-transcript-api        |
| Environment    | Python                        |

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/UTSAVS26/PyVerse.git
cd PyVerse/Generative-AI/Youtube_Rag_Chatbot
```


### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your environment variables
Create a .env file in the root and add your Gemini API key:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key_here
```

## 🛠️ Running the App
### ➤ Start the FastAPI backend
```bash
uvicorn main:app --reload
```
It will be available at: http://127.0.0.1:8000/docs

### ➤ Start the Streamlit frontend
```bash
streamlit run app.py
```
### 💡 How It Works

1.Input YouTube URL
2.Transcript is extracted using youtube-transcript-api
3.Transcript is chunked and embedded using HuggingFaceEmbeddings
4.FAISS indexes the chunks for similarity search
5.User asks a question → Similar transcript chunks retrieved
6.Prompt is passed to Gemini → Answer generated and displayed

### 🖼️ Screenshots

![App Screenshot](screenshots\Screenshot.png)

### 🧪 Example Prompts

1."What are the key points discussed in the video?"

2."Summarize the video in bullet points."

3."What are the challenges mentioned in India's economic growth?"

### 📁 Project Structure

```bash
YouTube_Rag_ChatBot
|
├── main.py          # FastAPI backend
├── app.py            # Streamlit frontend
├── requirements.txt
├── .env             # Gemini and HuggingFace API key
├── screenshots/     # UI images for README
└── README.md
```

### 🙋‍♂️ Contributor
👤 Divyanshu Giri
GitHub: [Divyanshu-hash](https://github.com/Divyanshu-hash)
Email: [rishugiri056@gmail.com](mailto:rishugiri056@gmail.com)

