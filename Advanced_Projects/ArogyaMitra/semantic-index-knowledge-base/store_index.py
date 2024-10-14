from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY= os.environ.get('PINECONE_APE_KEY')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

print(PINECONE_API_KEY)
print(PINECONE_API_ENV)

extracted_data=load_pdf("Data/")
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()

from pinecone import Pinecone

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name='medical--chatbot'
index = pinecone.Index(index_name)

vectors = {}


for i, chunk in enumerate(text_chunks):
    embedding = embeddings.embed_query(chunk.page_content)
    vector_id = str(i)
    vectors[vector_id] = embedding
    index.upsert(vectors=[(vector_id, embedding, {"text": chunk.page_content})])
