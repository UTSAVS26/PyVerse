from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI(
    title="YouTube RAG Chatbot API",
    description="An API that lets you ask questions about YouTube videos using transcript and Gemini.",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global models and variables
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store transcript vectors per video
video_indices = {}

# Pydantic request models
class ExtractRequest(BaseModel):
    video_id: Annotated[str, Field(..., description='ID of the video')]

class AskRequest(BaseModel):
    video_id: Annotated[str, Field(..., description='ID of the video')]
    question: Annotated[str, Field(..., description='Question to ask')]

@app.get("/")
async def root():
    return {
        "message": "Welcome to the YouTube RAG Chatbot API!",
        "instructions": "Go to /docs to try out the API interactively."
    }

# Endpoint to extract transcript and build vector index
@app.post("/extract")
async def extract_transcript(data: ExtractRequest):
    video_id = data.video_id

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except VideoUnavailable:
        raise HTTPException(status_code=404, detail="Video is unavailable.")
    except TranscriptsDisabled:
        raise HTTPException(status_code=403, detail="Transcripts are disabled for this video.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    full_text = " ".join(entry['text'] for entry in transcript)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Create vector index and store it
    vector_store = FAISS.from_documents(docs, embeddings)
    video_indices[video_id] = vector_store

    return {"status": "Transcript processed", "chunks": len(chunks)}

# Endpoint to ask questions
@app.post("/ask")
async def ask_question(data: AskRequest):
    video_id = data.video_id
    question = data.question

    # Retrieve the FAISS index for this video
    vector_store = video_indices.get(video_id)
    if vector_store is None:
        raise HTTPException(status_code=404, detail="Video ID not found. Please extract transcript first.")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Prompt template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided video transcript context.
        If the context is insufficient, say "I don't know".

        Context:
        {context}

        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    try:
        # Chain setup
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        chain = parallel_chain | prompt | gemini | StrOutputParser()
        result = chain.invoke(question)
        return {"answer": result.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while processing question: {str(e)}")
