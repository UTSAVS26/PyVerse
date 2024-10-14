from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import PineconeHybridSearchRetriever
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY= os.environ.get('PINECONE_APE_KEY')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV')

embeddings= download_hugging_face_embeddings()

from pinecone import Pinecone

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name='medical--chatbot'
index = pinecone.Index(index_name)

docsearch = LangChainPinecone(index, embeddings.embed_query, "text")

PROMPT=PromptTemplate(template=prompt_template,input_variables=['context','question'])
chain_type_kwargs={"prompt":PROMPT}

llm = CTransformers(
    model="Model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={
        'max_new_tokens': 256, 
        'temperature': 0.1,
        'top_p': 0.9,
        'top_k': 40,
    }
)

base_retriever = docsearch.as_retriever(search_kwargs={"k": 5})

compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    result=qa({"query":input})
    print("Response: ",result["result"])
    return str(result["result"])

if __name__ =='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)